-- ============================================================
-- 05_rfm_segmentation.sql
-- Recency / Frequency / Monetary segmentation using NTILE(5) quintiles.
-- "Analysis date" = one day after the last order in the dataset,
-- since this is historical data rather than a live feed.
-- ============================================================

WITH analysis_date AS (
    SELECT (MAX(order_purchase_timestamp)::date + 1) AS as_of_date  -- adding an int to a date keeps it a date, not an interval
    FROM orders
),

customer_orders AS (
    SELECT
        c.customer_unique_id,
        o.order_id,
        o.order_purchase_timestamp::date AS order_date,
        SUM(oi.price + oi.freight_value) AS order_revenue
    FROM orders o
    JOIN customers c    ON c.customer_id = o.customer_id
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id, o.order_id, o.order_purchase_timestamp
),

-- A customer_unique_id can technically show multiple customer_id rows
-- (e.g. if Olist recorded a different state on a later order) -- take the
-- most frequently seen state as "the" state for filtering purposes.
customer_state_lookup AS (
    SELECT customer_unique_id,
           MODE() WITHIN GROUP (ORDER BY customer_state) AS customer_state
    FROM customers
    GROUP BY customer_unique_id
),

rfm_base AS (
    SELECT
        co.customer_unique_id,
        ((SELECT as_of_date FROM analysis_date) - MAX(co.order_date))::integer AS recency_days,
        COUNT(DISTINCT co.order_id)      AS frequency,
        SUM(co.order_revenue)            AS monetary
    FROM customer_orders co
    GROUP BY co.customer_unique_id
),

rfm_scored AS (
    SELECT
        customer_unique_id,
        recency_days,
        frequency,
        monetary,

        -- Recency: SMALLER days-since-last-order is better, so invert the tile
        -- (NTILE puts smallest values in bucket 1; we want smallest -> score 5)
        6 - NTILE(5) OVER (ORDER BY recency_days ASC)  AS r_score,

        -- Frequency: NTILE(5) breaks down here because ~97% of customers
        -- share the identical value (frequency=1) -- with that much tied
        -- mass, NTILE's positional quintile boundaries land INSIDE the tie
        -- block, so "top 40% by frequency" ends up mostly one-time buyers
        -- who got arbitrarily bucketed high, not genuine repeat customers.
        -- Since order count only takes a handful of distinct values here,
        -- map it directly instead of forcing artificial quintiles:
        LEAST(frequency, 5) AS f_score,

        -- Monetary: bigger is better
        NTILE(5) OVER (ORDER BY monetary ASC)           AS m_score
    FROM rfm_base
)

SELECT
    rs.customer_unique_id,
    csl.customer_state,
    rs.recency_days,
    rs.frequency,
    rs.monetary,
    rs.r_score,
    rs.f_score,
    rs.m_score,
    (rs.r_score + rs.f_score + rs.m_score)          AS rfm_total,
    CAST(rs.r_score AS TEXT) || CAST(rs.f_score AS TEXT) || CAST(rs.m_score AS TEXT) AS rfm_code,

    -- Human-readable segment labels for the Tableau quadrant view
    CASE
        WHEN rs.r_score >= 4 AND rs.f_score >= 4 AND rs.m_score >= 4 THEN 'Champions'
        WHEN rs.r_score >= 4 AND rs.f_score >= 3                  THEN 'Loyal Customers'
        WHEN rs.r_score >= 4 AND rs.f_score <= 2                  THEN 'New / Promising'
        WHEN rs.r_score = 3                                    THEN 'Needs Attention'
        WHEN rs.r_score <= 2 AND rs.f_score >= 4 AND rs.m_score >= 4  THEN 'At Risk (High Value)'
        WHEN rs.r_score <= 2 AND rs.f_score >= 3                  THEN 'At Risk'
        WHEN rs.r_score <= 2 AND rs.f_score <= 2 AND rs.m_score <= 2  THEN 'Hibernating / Lost'
        ELSE 'Other'
    END AS rfm_segment
FROM rfm_scored rs
JOIN customer_state_lookup csl ON csl.customer_unique_id = rs.customer_unique_id
ORDER BY rfm_total DESC;
