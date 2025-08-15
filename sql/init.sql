-- Database initialization script for Telco Customer Intelligence Platform
-- This script sets up the basic database schema and initial data

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS analytics;
CREATE SCHEMA IF NOT EXISTS ml_models;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Create basic tables for customer data
CREATE TABLE IF NOT EXISTS customers (
    customer_id VARCHAR(50) PRIMARY KEY,
    gender VARCHAR(10),
    senior_citizen INTEGER,
    partner VARCHAR(10),
    dependents VARCHAR(10),
    tenure INTEGER,
    phone_service VARCHAR(10),
    multiple_lines VARCHAR(50),
    internet_service VARCHAR(50),
    online_security VARCHAR(50),
    online_backup VARCHAR(50),
    device_protection VARCHAR(50),
    tech_support VARCHAR(50),
    streaming_tv VARCHAR(50),
    streaming_movies VARCHAR(50),
    contract VARCHAR(50),
    paperless_billing VARCHAR(10),
    payment_method VARCHAR(50),
    monthly_charges DECIMAL(10,2),
    total_charges DECIMAL(10,2),
    churn VARCHAR(10),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create model predictions table
CREATE TABLE IF NOT EXISTS ml_models.predictions (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    customer_id VARCHAR(50) REFERENCES customers(customer_id),
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    prediction_type VARCHAR(50),
    prediction_value DECIMAL(10,4),
    prediction_probability DECIMAL(10,4),
    features JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create model performance table
CREATE TABLE IF NOT EXISTS ml_models.model_performance (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    model_name VARCHAR(100),
    model_version VARCHAR(50),
    metric_name VARCHAR(50),
    metric_value DECIMAL(10,4),
    evaluation_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    dataset_type VARCHAR(20) -- 'train', 'validation', 'test'
);

-- Create analytics views
CREATE OR REPLACE VIEW analytics.customer_summary AS
SELECT 
    COUNT(*) as total_customers,
    AVG(tenure) as avg_tenure,
    AVG(monthly_charges) as avg_monthly_charges,
    COUNT(CASE WHEN churn = 'Yes' THEN 1 END) as churned_customers,
    ROUND(
        COUNT(CASE WHEN churn = 'Yes' THEN 1 END)::decimal / COUNT(*)::decimal * 100, 
        2
    ) as churn_rate
FROM customers;

-- Create monitoring tables
CREATE TABLE IF NOT EXISTS monitoring.api_metrics (
    id UUID DEFAULT uuid_generate_v4() PRIMARY KEY,
    endpoint VARCHAR(200),
    method VARCHAR(10),
    status_code INTEGER,
    response_time_ms INTEGER,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_customers_churn ON customers(churn);
CREATE INDEX IF NOT EXISTS idx_customers_tenure ON customers(tenure);
CREATE INDEX IF NOT EXISTS idx_predictions_customer_id ON ml_models.predictions(customer_id);
CREATE INDEX IF NOT EXISTS idx_predictions_created_at ON ml_models.predictions(created_at);
CREATE INDEX IF NOT EXISTS idx_api_metrics_timestamp ON monitoring.api_metrics(timestamp);

-- Insert sample data if table is empty
INSERT INTO customers (customer_id, gender, senior_citizen, partner, dependents, tenure, phone_service, multiple_lines, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, contract, paperless_billing, payment_method, monthly_charges, total_charges, churn)
SELECT 
    'SAMPLE_' || generate_series(1, 10),
    CASE WHEN random() > 0.5 THEN 'Male' ELSE 'Female' END,
    CASE WHEN random() > 0.8 THEN 1 ELSE 0 END,
    CASE WHEN random() > 0.5 THEN 'Yes' ELSE 'No' END,
    CASE WHEN random() > 0.7 THEN 'Yes' ELSE 'No' END,
    (random() * 72)::integer + 1,
    'Yes',
    CASE WHEN random() > 0.5 THEN 'Yes' ELSE 'No' END,
    CASE WHEN random() > 0.3 THEN 'Fiber optic' ELSE 'DSL' END,
    CASE WHEN random() > 0.5 THEN 'Yes' ELSE 'No' END,
    CASE WHEN random() > 0.5 THEN 'Yes' ELSE 'No' END,
    CASE WHEN random() > 0.5 THEN 'Yes' ELSE 'No' END,
    CASE WHEN random() > 0.5 THEN 'Yes' ELSE 'No' END,
    CASE WHEN random() > 0.5 THEN 'Yes' ELSE 'No' END,
    CASE WHEN random() > 0.5 THEN 'Yes' ELSE 'No' END,
    CASE 
        WHEN random() > 0.66 THEN 'Month-to-month'
        WHEN random() > 0.33 THEN 'One year'
        ELSE 'Two year'
    END,
    CASE WHEN random() > 0.5 THEN 'Yes' ELSE 'No' END,
    CASE 
        WHEN random() > 0.75 THEN 'Electronic check'
        WHEN random() > 0.5 THEN 'Mailed check'
        WHEN random() > 0.25 THEN 'Bank transfer (automatic)'
        ELSE 'Credit card (automatic)'
    END,
    (random() * 100 + 20)::decimal(10,2),
    (random() * 5000 + 100)::decimal(10,2),
    CASE WHEN random() > 0.8 THEN 'Yes' ELSE 'No' END
WHERE NOT EXISTS (SELECT 1 FROM customers WHERE customer_id LIKE 'SAMPLE_%');

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO telco_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA analytics TO telco_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA ml_models TO telco_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA monitoring TO telco_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO telco_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA analytics TO telco_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA ml_models TO telco_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA monitoring TO telco_user;
