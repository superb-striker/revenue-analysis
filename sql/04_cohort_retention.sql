WITH dataset_bounds AS (
    SELECT DATE_TRUNC('month', MAX(order_purchase_timestamp))::date AS max_month
    FROM orders
),

customer_orders AS (
    SELECT
        c.customer_unique_id,
        o.order_id,
        DATE_TRUNC('month', o.order_purchase_timestamp)::date AS order_month
    FROM orders o
    JOIN customers c ON c.customer_id = o.customer_id
    WHERE o.order_status = 'delivered'
),

first_purchase AS (
    SELECT customer_unique_id, MIN(order_month) AS cohort_month
    FROM customer_orders
    GROUP BY customer_unique_id
),

cohort_activity AS (
    SELECT
        fp.cohort_month,
        co.customer_unique_id,
        (
            (DATE_PART('year', co.order_month) - DATE_PART('year', fp.cohort_month)) * 12
            + (DATE_PART('month', co.order_month) - DATE_PART('month', fp.cohort_month))
        )::int AS month_offset
    FROM customer_orders co
    JOIN first_purchase fp ON fp.customer_unique_id = co.customer_unique_id
),

cohort_size AS (
    SELECT cohort_month, COUNT(DISTINCT customer_unique_id) AS num_customers
    FROM first_purchase
    GROUP BY cohort_month
),

retention_counts AS (
    SELECT cohort_month, month_offset, COUNT(DISTINCT customer_unique_id) AS active_customers
    FROM cohort_activity
    GROUP BY cohort_month, month_offset
),

offset_scaffold AS (
    SELECT cs.cohort_month, gs.month_offset
    FROM cohort_size cs
    CROSS JOIN dataset_bounds db
    CROSS JOIN LATERAL generate_series(
        0,
        (
            (DATE_PART('year', db.max_month) - DATE_PART('year', cs.cohort_month)) * 12
            + (DATE_PART('month', db.max_month) - DATE_PART('month', cs.cohort_month))
        )::int
    ) AS gs(month_offset)
)

SELECT
    os.cohort_month,
    os.month_offset,
    cs.num_customers                         AS cohort_size,
    COALESCE(rc.active_customers, 0)         AS active_customers,
    ROUND(100.0 * COALESCE(rc.active_customers, 0) / cs.num_customers, 2) AS retention_pct
FROM offset_scaffold os
JOIN cohort_size cs             ON cs.cohort_month = os.cohort_month
LEFT JOIN retention_counts rc   ON rc.cohort_month = os.cohort_month
                                AND rc.month_offset = os.month_offset
ORDER BY os.cohort_month, os.month_offset;
