-- Running customer lifetime value + inter-purchase timing via LAG/LEAD.


-- Base: one row per delivered order, at the PERSON level (customer_unique_id), with total order revenue (items + freight).
WITH order_revenue AS (
    SELECT
        o.order_id,
        c.customer_unique_id,
        o.order_purchase_timestamp::date AS order_date,
        SUM(oi.price + oi.freight_value) AS order_revenue
    FROM orders o
    JOIN customers c   ON c.customer_id = o.customer_id
    JOIN order_items oi ON oi.order_id = o.order_id
    WHERE o.order_status = 'delivered'
    GROUP BY o.order_id, c.customer_unique_id, o.order_purchase_timestamp
)

-- Running CLV per customer + gap-to-previous/next order via LAG/LEAD.
SELECT
    customer_unique_id,
    order_id,
    order_date,
    order_revenue,

    -- Running (cumulative) lifetime value, order by order
    SUM(order_revenue) OVER (
        PARTITION BY customer_unique_id
        ORDER BY order_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_clv,

    -- Which purchase number is this for the customer (1st, 2nd, 3rd order...)
    ROW_NUMBER() OVER (
        PARTITION BY customer_unique_id ORDER BY order_date
    ) AS purchase_sequence,

    -- Days since this customer's PREVIOUS order (NULL on their first order)
    order_date - LAG(order_date) OVER (
        PARTITION BY customer_unique_id ORDER BY order_date
    ) AS days_since_prev_order,

    -- Days until this customer's NEXT order (NULL if this was their last order, i.e. a strong churn signal once this gap exceeds your churn threshold)
    LEAD(order_date) OVER (
        PARTITION BY customer_unique_id ORDER BY order_date
    ) - order_date AS days_until_next_order,

    -- Total number of orders this customer ever placed (window, not aggregate, so it can sit alongside the row-level columns above)
    COUNT(*) OVER (PARTITION BY customer_unique_id) AS lifetime_order_count

FROM order_revenue
ORDER BY customer_unique_id, order_date;
