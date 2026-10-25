-- migrate:up
CREATE INDEX IF NOT EXISTS idx_order_items_seller_id ON order_items(seller_id);

-- migrate:down
DROP INDEX IF EXISTS idx_order_items_seller_id;
