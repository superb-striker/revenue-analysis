-- ============================================================
-- 07_churn_features.sql  (v4)
-- One row per customer with features known 30 days after first delivery.
--
-- churned = 1 if no second delivered order is purchased in the next
-- 180 days. Customers who already reordered before scoring are excluded.
-- A full 180-day observation window after scoring is required.
--
-- Features include:
--   - seller_late_rate / seller_order_count: seller orders delivered
--     strictly before this customer's first purchase.
--   - customer_seller_distance_km: straight-line distance between
--     customer and seller zip-code centroids (geolocation table),
--     since a Sao Paulo <-> Sao Paulo order behaves differently than
--     a cross-country one
--   - purchase_hour: time-of-day the order was placed
-- ============================================================

WITH dataset_bounds AS (
    SELECT MAX(order_purchase_timestamp)::date AS max_date FROM orders
),

-- Pick each customer's first delivered order. DISTINCT ON exposes the
-- customer cardinality to PostgreSQL's planner, unlike row_number() = 1.
first_order AS MATERIALIZED (
    SELECT DISTINCT ON (c.customer_unique_id)
        c.customer_unique_id,
        c.customer_state,
        c.customer_zip_code_prefix,
        o.order_id,
        o.order_purchase_timestamp            AS first_order_ts,
        o.order_purchase_timestamp::date      AS first_order_date,
        o.order_delivered_customer_date::date AS delivered_date,
        o.order_estimated_delivery_date::date AS estimated_date,
        (o.order_delivered_customer_date::date + 30) AS scoring_date
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.order_status = 'delivered'
      AND o.order_delivered_customer_date IS NOT NULL
    ORDER BY c.customer_unique_id, o.order_purchase_timestamp, o.order_id
),

-- Item-level details of that first order: total price/freight, item/seller
-- counts, dominant category, and a representative seller for that order
-- (MODE -- most frequent seller_id on the order; most orders have exactly one)
first_order_items AS MATERIALIZED (
    SELECT
        oi.order_id,
        SUM(oi.price)         AS price,
        SUM(oi.freight_value) AS freight_value,
        COUNT(*)                        AS num_items,
        COUNT(DISTINCT oi.seller_id)    AS num_sellers,
        MODE() WITHIN GROUP (ORDER BY p.product_category_name) AS product_category_name,
        MODE() WITHIN GROUP (ORDER BY oi.seller_id)            AS representative_seller_id
    FROM order_items oi
    LEFT JOIN products p ON p.product_id = oi.product_id
    GROUP BY oi.order_id
),

-- Payment details of that first order
first_order_payment AS (
    SELECT
        order_id,
        (ARRAY_AGG(payment_type ORDER BY payment_sequential))[1] AS payment_type,
        MAX(payment_installments) AS payment_installments,
        SUM(payment_value)        AS payment_value
    FROM order_payments
    GROUP BY order_id
),

-- Review score of that first order (average, in case of multiple review rows)
first_order_review AS (
    SELECT r.order_id, AVG(r.review_score) AS review_score
    FROM order_reviews r
    JOIN first_order fo ON fo.order_id = r.order_id
    WHERE r.review_creation_date < fo.scoring_date::timestamp + INTERVAL '1 day'
    GROUP BY r.order_id
),

-- Zip-code-prefix centroids from the geolocation table (it has many raw
-- lat/lng pings per prefix; we average them into one representative point
-- per prefix -- an approximation, not exact geocoding, but fine for a
-- "roughly how far apart" distance feature)
zip_centroids AS (
    SELECT
        geolocation_zip_code_prefix AS zip_prefix,
        AVG(geolocation_lat) AS lat,
        AVG(geolocation_lng) AS lng
    FROM geolocation
    GROUP BY geolocation_zip_code_prefix
)

SELECT
    fo.customer_unique_id,
    fo.customer_state,
    fo.first_order_date,
    fo.scoring_date,
    foi.price,
    foi.freight_value,
    ROUND(foi.freight_value / NULLIF(foi.price + foi.freight_value, 0), 4) AS freight_ratio,
    foi.num_items,
    foi.num_sellers,
    COALESCE(t.product_category_name_english, foi.product_category_name)  AS product_category,
    fp.payment_type,
    fp.payment_installments,
    fr.review_score,
    (fo.delivered_date - fo.estimated_date)   AS delivery_delay_days,
    (fo.estimated_date - fo.first_order_date) AS estimated_lead_days,
    (fo.delivered_date - fo.first_order_date) AS actual_delivery_days,
    CASE WHEN fo.delivered_date > fo.estimated_date THEN 1 ELSE 0 END      AS is_late_delivery,
    EXTRACT(DOW   FROM fo.first_order_date)::int                          AS purchase_dow,
    EXTRACT(MONTH FROM fo.first_order_date)::int                          AS purchase_month,
    EXTRACT(HOUR  FROM fo.first_order_ts)::int                            AS purchase_hour,
    sp.seller_late_rate,
    sp.seller_order_count,
    -- Haversine distance in km between customer and seller zip centroids.
    -- LEAST/GREATEST clamp guards against acos() domain errors from
    -- floating-point rounding pushing the argument fractionally past +-1.
    CASE WHEN cz.lat IS NOT NULL AND sz.lat IS NOT NULL THEN
        6371 * acos(
            LEAST(1.0, GREATEST(-1.0,
                cos(radians(cz.lat)) * cos(radians(sz.lat)) * cos(radians(sz.lng) - radians(cz.lng))
                + sin(radians(cz.lat)) * sin(radians(sz.lat))
            ))
        )
    ELSE NULL END AS customer_seller_distance_km,
    CASE WHEN next_order.next_order_ts <= fo.scoring_date::timestamp + INTERVAL '180 days'
         THEN 0 ELSE 1 END AS churned
FROM first_order fo
JOIN first_order_items foi               ON foi.order_id = fo.order_id
LEFT JOIN first_order_payment fp         ON fp.order_id = fo.order_id
LEFT JOIN first_order_review fr          ON fr.order_id = fo.order_id
LEFT JOIN product_category_translation t ON t.product_category_name = foi.product_category_name
LEFT JOIN LATERAL (
    SELECT MIN(o2.order_purchase_timestamp) AS next_order_ts
    FROM customers c2
    JOIN orders o2 ON o2.customer_id = c2.customer_id
    WHERE c2.customer_unique_id = fo.customer_unique_id
      AND o2.order_status = 'delivered'
      AND o2.order_purchase_timestamp > fo.first_order_ts
) next_order ON TRUE
LEFT JOIN LATERAL (
    SELECT AVG(h.was_late) AS seller_late_rate,
           COUNT(h.order_id) AS seller_order_count
    FROM (
        SELECT DISTINCT oi.order_id,
               CASE WHEN o.order_delivered_customer_date::date > o.order_estimated_delivery_date::date
                    THEN 1.0 ELSE 0.0 END AS was_late
        FROM order_items oi
        JOIN orders o ON o.order_id = oi.order_id
        WHERE oi.seller_id = foi.representative_seller_id
          AND o.order_status = 'delivered'
          AND o.order_delivered_customer_date < fo.first_order_ts
    ) h
) sp ON TRUE
LEFT JOIN zip_centroids cz               ON cz.zip_prefix = fo.customer_zip_code_prefix
LEFT JOIN sellers s                      ON s.seller_id = foi.representative_seller_id
LEFT JOIN zip_centroids sz               ON sz.zip_prefix = s.seller_zip_code_prefix
CROSS JOIN dataset_bounds db
WHERE fo.scoring_date <= (db.max_date - 180)
  AND (next_order.next_order_ts IS NULL OR next_order.next_order_ts > fo.scoring_date::timestamp);
