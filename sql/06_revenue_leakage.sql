-- ============================================================
-- 06_revenue_leakage.sql  (v2 - optimized)
-- Same logic as before, but the churned/retained/revenue split is now
-- computed with a single JOIN + GROUP BY per friction type instead of
-- correlated subqueries -- avoids re-scanning all orders once per
-- customer, which is what made the first version slow.
-- ============================================================

WITH customer_orders AS (
    SELECT
        c.customer_unique_id,
        o.order_id,
        o.order_purchase_timestamp::date AS order_date,
        o.order_delivered_customer_date::date AS delivered_date,
        o.order_estimated_delivery_date::date AS estimated_date,
        SUM(oi.price + oi.freight_value) AS order_revenue
    FROM orders o
    JOIN customers c    ON c.customer_id = o.customer_id
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY c.customer_unique_id, o.order_id, o.order_purchase_timestamp,
             o.order_delivered_customer_date, o.order_estimated_delivery_date
),

order_friction AS (
    SELECT
        co.*,
        CASE WHEN co.delivered_date > co.estimated_date THEN 1 ELSE 0 END AS is_late,
        CASE WHEN r.review_score <= 2 THEN 1 ELSE 0 END AS is_bad_review
    FROM customer_orders co
    LEFT JOIN order_reviews r ON r.order_id = co.order_id
),

-- First friction date per customer, per friction type (unpivoted via UNION ALL)
friction_first_date AS (
    SELECT customer_unique_id, 'Late Delivery' AS friction_type,
           MIN(order_date) FILTER (WHERE is_late = 1) AS friction_date
    FROM order_friction
    GROUP BY customer_unique_id
    HAVING MIN(order_date) FILTER (WHERE is_late = 1) IS NOT NULL

    UNION ALL

    SELECT customer_unique_id, 'Bad Review (<=2 stars)' AS friction_type,
           MIN(order_date) FILTER (WHERE is_bad_review = 1) AS friction_date
    FROM order_friction
    GROUP BY customer_unique_id
    HAVING MIN(order_date) FILTER (WHERE is_bad_review = 1) IS NOT NULL
),

-- Single join between each customer's friction date and their FULL order
-- history, then aggregate in one pass -- no per-row correlated subqueries.
customer_post_friction AS (
    SELECT
        ffd.customer_unique_id,
        ffd.friction_type,
        ffd.friction_date,
        BOOL_OR(of.order_date > ffd.friction_date)                              AS reordered_after_friction,
        SUM(CASE WHEN of.order_date <= ffd.friction_date THEN of.order_revenue ELSE 0 END) AS revenue_before,
        SUM(CASE WHEN of.order_date >  ffd.friction_date THEN of.order_revenue ELSE 0 END) AS revenue_after
    FROM friction_first_date ffd
    JOIN order_friction of ON of.customer_unique_id = ffd.customer_unique_id
    GROUP BY ffd.customer_unique_id, ffd.friction_type, ffd.friction_date
)

SELECT
    friction_type,
    COUNT(*)                                                         AS customers_with_friction,
    SUM(CASE WHEN reordered_after_friction THEN 0 ELSE 1 END)        AS customers_churned,
    SUM(CASE WHEN reordered_after_friction THEN 1 ELSE 0 END)        AS customers_retained,
    ROUND(100.0 * SUM(CASE WHEN reordered_after_friction THEN 0 ELSE 1 END)
          / COUNT(*), 2)                                             AS churn_rate_pct,
    SUM(revenue_before)                                              AS revenue_before_friction,
    SUM(revenue_after)                                               AS revenue_after_friction,
    SUM(CASE WHEN NOT reordered_after_friction THEN revenue_before ELSE 0 END)
                                                                      AS estimated_leaked_revenue
FROM customer_post_friction
GROUP BY friction_type
ORDER BY estimated_leaked_revenue DESC;
