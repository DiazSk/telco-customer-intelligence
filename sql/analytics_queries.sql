-- Advanced SQL Analytics for Telco Churn
-- Demonstrates DS/DA SQL proficiency

-- ============================================
-- 1. COHORT RETENTION ANALYSIS
-- ============================================
WITH cohort_data AS (
    SELECT 
        customerID,
        DATE_TRUNC('month', 
            CURRENT_DATE - INTERVAL '1 month' * tenure) as cohort_month,
        tenure,
        CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END as churned
    FROM telco_customers
),
cohort_size AS (
    SELECT 
        cohort_month,
        COUNT(DISTINCT customerID) as cohort_customers
    FROM cohort_data
    GROUP BY cohort_month
),
retention_data AS (
    SELECT 
        c.cohort_month,
        c.tenure as months_since_joining,
        COUNT(DISTINCT c.customerID) as customers_retained,
        cs.cohort_customers,
        ROUND(100.0 * COUNT(DISTINCT c.customerID) / cs.cohort_customers, 2) as retention_rate
    FROM cohort_data c
    JOIN cohort_size cs ON c.cohort_month = cs.cohort_month
    WHERE c.churned = 0
    GROUP BY c.cohort_month, c.tenure, cs.cohort_customers
)
SELECT 
    cohort_month,
    months_since_joining,
    customers_retained,
    cohort_customers,
    retention_rate,
    LAG(retention_rate) OVER (PARTITION BY cohort_month ORDER BY months_since_joining) as prev_retention,
    retention_rate - LAG(retention_rate) OVER (PARTITION BY cohort_month ORDER BY months_since_joining) as retention_change
FROM retention_data
ORDER BY cohort_month, months_since_joining;

-- ============================================
-- 2. CUSTOMER LIFETIME VALUE CALCULATION
-- ============================================
WITH customer_revenue AS (
    SELECT 
        customerID,
        tenure,
        MonthlyCharges,
        TotalCharges,
        CASE WHEN Churn = 'Yes' THEN tenure ELSE tenure + 12 END as projected_tenure,
        Churn
    FROM telco_customers
),
clv_calculation AS (
    SELECT 
        customerID,
        tenure,
        MonthlyCharges,
        TotalCharges as historical_value,
        CASE 
            WHEN Churn = 'Yes' THEN TotalCharges
            ELSE TotalCharges + (MonthlyCharges * 12 * 0.8) -- 80% probability of 12 more months
        END as estimated_clv,
        CASE 
            WHEN Churn = 'Yes' THEN 0
            ELSE MonthlyCharges * 12 * 0.8 
        END as future_value,
        Churn
    FROM customer_revenue
),
clv_segments AS (
    SELECT 
        *,
        NTILE(5) OVER (ORDER BY estimated_clv DESC) as clv_quintile,
        PERCENT_RANK() OVER (ORDER BY estimated_clv) as clv_percentile
    FROM clv_calculation
)
SELECT 
    clv_quintile,
    COUNT(*) as customer_count,
    ROUND(AVG(estimated_clv), 2) as avg_clv,
    ROUND(SUM(estimated_clv), 2) as total_clv,
    ROUND(AVG(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100, 2) as churn_rate,
    ROUND(SUM(future_value), 2) as revenue_at_risk
FROM clv_segments
GROUP BY clv_quintile
ORDER BY clv_quintile;

-- ============================================
-- 3. CHURN PREDICTION FEATURES
-- ============================================
WITH feature_engineering AS (
    SELECT 
        customerID,
        
        -- Tenure features
        tenure,
        CASE 
            WHEN tenure <= 6 THEN 'New'
            WHEN tenure <= 24 THEN 'Regular'
            WHEN tenure <= 48 THEN 'Established'
            ELSE 'Loyal'
        END as customer_stage,
        
        -- Revenue features
        MonthlyCharges,
        TotalCharges,
        TotalCharges / NULLIF(tenure, 0) as avg_monthly_spend,
        MonthlyCharges - (TotalCharges / NULLIF(tenure, 0)) as spending_trend,
        
        -- Service adoption
        CASE WHEN PhoneService = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN InternetService != 'No' THEN 1 ELSE 0 END +
        CASE WHEN OnlineSecurity = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN OnlineBackup = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN DeviceProtection = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN TechSupport = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN StreamingTV = 'Yes' THEN 1 ELSE 0 END +
        CASE WHEN StreamingMovies = 'Yes' THEN 1 ELSE 0 END as service_count,
        
        -- Contract risk score
        CASE 
            WHEN Contract = 'Month-to-month' THEN 3
            WHEN Contract = 'One year' THEN 2
            WHEN Contract = 'Two year' THEN 1
        END as contract_risk,
        
        -- Payment method risk
        CASE 
            WHEN PaymentMethod = 'Electronic check' THEN 3
            WHEN PaymentMethod = 'Mailed check' THEN 2
            ELSE 1
        END as payment_risk,
        
        Churn
    FROM telco_customers
),
risk_scoring AS (
    SELECT 
        *,
        -- Composite risk score
        (contract_risk * 0.3 + 
         payment_risk * 0.2 + 
         CASE WHEN customer_stage = 'New' THEN 3 
              WHEN customer_stage = 'Regular' THEN 2 
              ELSE 1 END * 0.3 +
         CASE WHEN service_count <= 2 THEN 3
              WHEN service_count <= 4 THEN 2
              ELSE 1 END * 0.2) as risk_score,
        
        -- Percentile ranks for key metrics
        PERCENT_RANK() OVER (ORDER BY tenure DESC) as tenure_percentile,
        PERCENT_RANK() OVER (ORDER BY MonthlyCharges) as price_percentile,
        PERCENT_RANK() OVER (ORDER BY service_count DESC) as service_percentile
    FROM feature_engineering
)
SELECT 
    customerID,
    customer_stage,
    service_count,
    risk_score,
    CASE 
        WHEN risk_score >= 2.5 THEN 'High Risk'
        WHEN risk_score >= 1.5 THEN 'Medium Risk'
        ELSE 'Low Risk'
    END as risk_category,
    tenure_percentile,
    price_percentile,
    service_percentile,
    Churn
FROM risk_scoring
ORDER BY risk_score DESC
LIMIT 100;

-- ============================================
-- 4. A/B TEST ANALYSIS FOR RETENTION CAMPAIGNS
-- ============================================
WITH campaign_assignment AS (
    SELECT 
        customerID,
        -- Simulate random assignment to test/control
        CASE WHEN RANDOM() < 0.5 THEN 'Test' ELSE 'Control' END as test_group,
        MonthlyCharges,
        tenure,
        Churn
    FROM telco_customers
    WHERE Contract = 'Month-to-month'  -- Target high-risk segment
),
campaign_results AS (
    SELECT 
        test_group,
        COUNT(*) as sample_size,
        AVG(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churn_rate,
        STDDEV(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churn_stddev,
        AVG(MonthlyCharges * tenure) as avg_revenue
    FROM campaign_assignment
    GROUP BY test_group
),
statistical_test AS (
    SELECT 
        t.churn_rate as test_churn_rate,
        c.churn_rate as control_churn_rate,
        t.churn_rate - c.churn_rate as lift,
        (t.churn_rate - c.churn_rate) / c.churn_rate * 100 as lift_percentage,
        t.sample_size as test_size,
        c.sample_size as control_size,
        -- Z-score calculation
        ABS(t.churn_rate - c.churn_rate) / 
        SQRT((t.churn_stddev^2 / t.sample_size) + (c.churn_stddev^2 / c.sample_size)) as z_score,
        -- Simplified p-value interpretation
        CASE 
            WHEN ABS(t.churn_rate - c.churn_rate) / 
                 SQRT((t.churn_stddev^2 / t.sample_size) + (c.churn_stddev^2 / c.sample_size)) > 1.96 
            THEN 'Significant (p < 0.05)'
            ELSE 'Not Significant'
        END as significance
    FROM campaign_results t, campaign_results c
    WHERE t.test_group = 'Test' AND c.test_group = 'Control'
)
SELECT 
    ROUND(test_churn_rate * 100, 2) as test_churn_pct,
    ROUND(control_churn_rate * 100, 2) as control_churn_pct,
    ROUND(lift * 100, 2) as absolute_lift_pct,
    ROUND(lift_percentage, 2) as relative_lift_pct,
    test_size,
    control_size,
    ROUND(z_score, 3) as z_score,
    significance,
    -- Business impact calculation
    ROUND(ABS(lift) * test_size * 1350, 2) as revenue_impact  -- Assuming $1350 CLV
FROM statistical_test;

-- ============================================
-- 5. CUSTOMER JOURNEY ANALYSIS
-- ============================================
WITH customer_events AS (
    SELECT 
        customerID,
        tenure,
        Contract,
        PaymentMethod,
        InternetService,
        -- Create synthetic event timeline
        UNNEST(ARRAY[
            ROW(0, 'Acquisition', Contract),
            ROW(GREATEST(1, tenure/4), 'Service_Check', InternetService),
            ROW(GREATEST(2, tenure/2), 'Payment_Update', PaymentMethod),
            ROW(tenure, CASE WHEN Churn = 'Yes' THEN 'Churned' ELSE 'Active' END, '')
        ]) as event(month, event_type, event_detail)
    FROM telco_customers
),
journey_paths AS (
    SELECT 
        customerID,
        STRING_AGG(event_type || ':' || event_detail, ' -> ' 
                  ORDER BY month) as journey,
        MAX(CASE WHEN event_type = 'Churned' THEN 1 ELSE 0 END) as churned
    FROM customer_events
    GROUP BY customerID
),
path_analysis AS (
    SELECT 
        journey,
        COUNT(*) as customer_count,
        AVG(churned) as churn_rate,
        RANK() OVER (ORDER BY COUNT(*) DESC) as path_frequency_rank,
        RANK() OVER (ORDER BY AVG(churned) DESC) as churn_risk_rank
    FROM journey_paths
    GROUP BY journey
    HAVING COUNT(*) >= 10  -- Minimum sample size
)
SELECT 
    journey as customer_journey_path,
    customer_count,
    ROUND(churn_rate * 100, 2) as churn_rate_pct,
    path_frequency_rank,
    churn_risk_rank,
    CASE 
        WHEN churn_risk_rank <= 10 THEN 'High Risk Path'
        WHEN churn_rate > 0.3 THEN 'Medium Risk Path'
        ELSE 'Low Risk Path'
    END as path_risk_category
FROM path_analysis
ORDER BY churn_risk_rank
LIMIT 20;

-- ============================================
-- 6. REVENUE IMPACT DASHBOARD QUERY
-- ============================================
WITH monthly_metrics AS (
    SELECT 
        DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month' * tenure) as month,
        COUNT(*) as new_customers,
        COUNT(CASE WHEN Churn = 'Yes' THEN 1 END) as churned_customers,
        SUM(MonthlyCharges) as monthly_revenue,
        AVG(MonthlyCharges) as avg_monthly_charge,
        SUM(CASE WHEN Churn = 'Yes' THEN MonthlyCharges ELSE 0 END) as lost_revenue
    FROM telco_customers
    GROUP BY DATE_TRUNC('month', CURRENT_DATE - INTERVAL '1 month' * tenure)
),
running_totals AS (
    SELECT 
        month,
        new_customers,
        churned_customers,
        SUM(new_customers) OVER (ORDER BY month) as cumulative_customers,
        SUM(churned_customers) OVER (ORDER BY month) as cumulative_churned,
        monthly_revenue,
        lost_revenue,
        SUM(monthly_revenue) OVER (ORDER BY month) as cumulative_revenue,
        AVG(monthly_revenue) OVER (ORDER BY month ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) as revenue_3m_avg,
        churned_customers::FLOAT / NULLIF(new_customers, 0) as churn_ratio
    FROM monthly_metrics
)
SELECT 
    TO_CHAR(month, 'YYYY-MM') as month,
    new_customers,
    churned_customers,
    cumulative_customers - cumulative_churned as active_customers,
    ROUND(churned_customers::NUMERIC / NULLIF(cumulative_customers - cumulative_churned, 0) * 100, 2) as monthly_churn_rate,
    ROUND(monthly_revenue, 2) as monthly_revenue,
    ROUND(lost_revenue, 2) as revenue_lost_to_churn,
    ROUND(revenue_3m_avg, 2) as revenue_3m_moving_avg,
    ROUND(cumulative_revenue, 2) as total_revenue_to_date,
    CASE 
        WHEN churn_ratio > 0.3 THEN 'Critical'
        WHEN churn_ratio > 0.2 THEN 'Warning'
        ELSE 'Healthy'
    END as business_health
FROM running_totals
ORDER BY month DESC
LIMIT 12;