\copy customers FROM 'data/raw/olist_customers_dataset.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');

\copy products FROM 'data/raw/olist_products_dataset.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');

\copy sellers FROM 'data/raw/olist_sellers_dataset.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');

\copy product_category_translation FROM 'data/raw/product_category_name_translation.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');

\copy orders FROM 'data/raw/olist_orders_dataset.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');

\copy order_items FROM 'data/raw/olist_order_items_dataset.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');

\copy order_payments FROM 'data/raw/olist_order_payments_dataset.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');

\copy order_reviews FROM 'data/raw/olist_order_reviews_dataset.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');

\copy geolocation FROM 'data/raw/olist_geolocation_dataset.csv' WITH (FORMAT csv, HEADER true, ENCODING 'UTF8');

-- Sanity check row counts after loading
SELECT 'customers' AS tbl, COUNT(*) FROM customers
UNION ALL SELECT 'orders', COUNT(*) FROM orders
UNION ALL SELECT 'order_items', COUNT(*) FROM order_items
UNION ALL SELECT 'order_payments', COUNT(*) FROM order_payments
UNION ALL SELECT 'order_reviews', COUNT(*) FROM order_reviews
UNION ALL SELECT 'products', COUNT(*) FROM products
UNION ALL SELECT 'sellers', COUNT(*) FROM sellers;
