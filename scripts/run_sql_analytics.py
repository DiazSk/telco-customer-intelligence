"""
Run SQL Analytics on SQLite Database
Works with SQLite-compatible queries
"""

import os
import sqlite3
from datetime import datetime, timedelta

import pandas as pd


def create_database():
    """Create SQLite database from processed data"""
    print("Creating SQLite database...")

    # Load processed data
    df = pd.read_csv("data/processed/processed_telco_data.csv")

    # Create connection
    conn = sqlite3.connect("data/telco.db")

    # Save to database
    df.to_sql("telco_customers", conn, if_exists="replace", index=False)

    print(f"✅ Database created with {len(df)} records")

    return conn


def run_analytics_queries(conn):
    """Run SQLite-compatible analytics queries"""

    queries = {
        "1. Churn Rate Analysis": """
            SELECT
                Churn,
                COUNT(*) as customer_count,
                ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM telco_customers), 2) as percentage
            FROM telco_customers
            GROUP BY Churn
            ORDER BY Churn DESC
        """,
        "2. Churn by Contract Type": """
            SELECT
                Contract,
                COUNT(*) as total_customers,
                SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned_customers,
                ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate
            FROM telco_customers
            GROUP BY Contract
            ORDER BY churn_rate DESC
        """,
        "3. Revenue Impact Analysis": """
            SELECT
                Churn,
                COUNT(*) as customer_count,
                ROUND(AVG(MonthlyCharges), 2) as avg_monthly_charges,
                ROUND(SUM(MonthlyCharges), 2) as total_monthly_revenue,
                ROUND(AVG(TotalCharges), 2) as avg_total_charges,
                ROUND(SUM(TotalCharges), 2) as total_lifetime_value
            FROM telco_customers
            GROUP BY Churn
        """,
        "4. Tenure Analysis": """
            SELECT
                CASE
                    WHEN tenure <= 6 THEN '0-6 months'
                    WHEN tenure <= 12 THEN '7-12 months'
                    WHEN tenure <= 24 THEN '13-24 months'
                    WHEN tenure <= 48 THEN '25-48 months'
                    ELSE '49+ months'
                END as tenure_group,
                COUNT(*) as customer_count,
                SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
                ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate
            FROM telco_customers
            GROUP BY tenure_group
            ORDER BY
                CASE
                    WHEN tenure_group = '0-6 months' THEN 1
                    WHEN tenure_group = '7-12 months' THEN 2
                    WHEN tenure_group = '13-24 months' THEN 3
                    WHEN tenure_group = '25-48 months' THEN 4
                    ELSE 5
                END
        """,
        "5. Service Usage Impact": """
            SELECT
                total_services,
                COUNT(*) as customer_count,
                SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
                ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate,
                ROUND(AVG(MonthlyCharges), 2) as avg_monthly_charges
            FROM telco_customers
            GROUP BY total_services
            ORDER BY total_services
        """,
        "6. Payment Method Risk": """
            SELECT
                PaymentMethod,
                COUNT(*) as customer_count,
                SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
                ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate,
                ROUND(AVG(MonthlyCharges), 2) as avg_monthly_charges
            FROM telco_customers
            GROUP BY PaymentMethod
            ORDER BY churn_rate DESC
        """,
        "7. High-Value Customers at Risk": """
            SELECT
                customerID,
                MonthlyCharges,
                TotalCharges,
                tenure,
                Contract,
                Churn
            FROM telco_customers
            WHERE TotalCharges > (SELECT AVG(TotalCharges) FROM telco_customers)
                AND Churn = 'Yes'
            ORDER BY TotalCharges DESC
            LIMIT 20
        """,
        "8. Customer Segmentation": """
            SELECT
                CASE
                    WHEN TotalCharges < 500 THEN 'Low Value'
                    WHEN TotalCharges < 2000 THEN 'Medium Value'
                    WHEN TotalCharges < 5000 THEN 'High Value'
                    ELSE 'Premium'
                END as value_segment,
                COUNT(*) as customer_count,
                SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
                ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate,
                ROUND(SUM(MonthlyCharges), 2) as monthly_revenue_at_risk
            FROM telco_customers
            GROUP BY value_segment
            ORDER BY
                CASE
                    WHEN value_segment = 'Low Value' THEN 1
                    WHEN value_segment = 'Medium Value' THEN 2
                    WHEN value_segment = 'High Value' THEN 3
                    ELSE 4
                END
        """,
        "9. Internet Service Impact": """
            SELECT
                InternetService,
                COUNT(*) as customer_count,
                SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) as churned,
                ROUND(SUM(CASE WHEN Churn = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 2) as churn_rate,
                ROUND(AVG(MonthlyCharges), 2) as avg_charges
            FROM telco_customers
            GROUP BY InternetService
            ORDER BY churn_rate DESC
        """,
        "10. Business Metrics Summary": """
            SELECT
                (SELECT COUNT(*) FROM telco_customers) as total_customers,
                (SELECT COUNT(*) FROM telco_customers WHERE Churn = 'Yes') as churned_customers,
                (SELECT ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM telco_customers), 2)
                 FROM telco_customers WHERE Churn = 'Yes') as churn_rate,
                (SELECT ROUND(SUM(MonthlyCharges), 2) FROM telco_customers) as total_monthly_revenue,
                (SELECT ROUND(SUM(MonthlyCharges), 2)
                 FROM telco_customers WHERE Churn = 'Yes') as monthly_revenue_at_risk,
                (SELECT ROUND(AVG(tenure), 2) FROM telco_customers) as avg_customer_tenure,
                (SELECT ROUND(AVG(TotalCharges), 2) FROM telco_customers) as avg_customer_lifetime_value
        """,
    }

    results = {}

    for name, query in queries.items():
        print(f"\n{'='*60}")
        print(f"{name}")
        print("=" * 60)

        try:
            df_result = pd.read_sql_query(query, conn)
            results[name] = df_result

            # Display results
            if len(df_result) <= 20:
                print(df_result.to_string(index=False))
            else:
                print(df_result.head(10).to_string(index=False))
                print(f"... ({len(df_result) - 10} more rows)")

        except Exception as e:
            print(f"Error executing query: {e}")

    return results


def save_results_to_excel(results):
    """Save all query results to Excel file"""
    output_file = "data/processed/sql_analytics_results.xlsx"

    with pd.ExcelWriter(output_file, engine="openpyxl") as writer:
        for sheet_name, df in results.items():
            # Clean sheet name (Excel has restrictions)
            clean_name = sheet_name.split(". ")[-1][:31]  # Max 31 chars
            df.to_excel(writer, sheet_name=clean_name, index=False)

    print(f"\n✅ Results saved to {output_file}")


def generate_insights(results):
    """Generate business insights from SQL results"""
    print("\n" + "=" * 60)
    print("📊 KEY BUSINESS INSIGHTS")
    print("=" * 60)

    # Get churn rate
    churn_df = results.get("1. Churn Rate Analysis")
    if churn_df is not None:
        churn_rate = churn_df[churn_df["Churn"] == "Yes"]["percentage"].values[0]
        print(f"\n1. Overall Churn Rate: {churn_rate}%")

    # Contract type insights
    contract_df = results.get("2. Churn by Contract Type")
    if contract_df is not None:
        highest_churn = contract_df.iloc[0]
        print(f"\n2. Highest Risk Contract Type: {highest_churn['Contract']}")
        print(f"   - Churn Rate: {highest_churn['churn_rate']}%")
        print(f"   - Affected Customers: {highest_churn['churned_customers']}")

    # Revenue impact
    revenue_df = results.get("3. Revenue Impact Analysis")
    if revenue_df is not None:
        churned_revenue = revenue_df[revenue_df["Churn"] == "Yes"]["total_monthly_revenue"].values[
            0
        ]
        print(f"\n3. Monthly Revenue at Risk: ${churned_revenue:,.2f}")
        print(f"   - Annual Impact: ${churned_revenue * 12:,.2f}")

    # Tenure insights
    tenure_df = results.get("4. Tenure Analysis")
    if tenure_df is not None:
        highest_risk_tenure = tenure_df.iloc[0]
        print(f"\n4. Highest Risk Tenure Group: {highest_risk_tenure['tenure_group']}")
        print(f"   - Churn Rate: {highest_risk_tenure['churn_rate']}%")

    # Payment method risk
    payment_df = results.get("6. Payment Method Risk")
    if payment_df is not None:
        riskiest_payment = payment_df.iloc[0]
        print(f"\n5. Riskiest Payment Method: {riskiest_payment['PaymentMethod']}")
        print(f"   - Churn Rate: {riskiest_payment['churn_rate']}%")
        print(f"   - Customers: {riskiest_payment['customer_count']}")

    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS")
    print("=" * 60)
    print(
        """
1. IMMEDIATE ACTIONS:
   - Target month-to-month contract customers for upgrade campaigns
   - Implement retention program for customers in first 6 months
   - Convert electronic check users to automatic payments

2. REVENUE PROTECTION:
   - Focus on high-value customers showing churn signals
   - Implement early warning system for tenure < 12 months
   - Bundle services to increase switching costs

3. OPERATIONAL IMPROVEMENTS:
   - Improve fiber optic service quality (highest churn)
   - Incentivize 1-2 year contracts with discounts
   - Enhance onboarding for new customers
    """
    )


def main():
    """Main execution function"""
    print("🔍 Telco Customer Churn - SQL Analytics")
    print("=" * 60)

    # Check if database exists
    db_path = "data/telco.db"
    if not os.path.exists(db_path):
        conn = create_database()
    else:
        print("✅ Using existing database")
        conn = sqlite3.connect(db_path)

    # Run analytics queries
    results = run_analytics_queries(conn)

    # Generate insights
    generate_insights(results)

    # Save results to Excel
    try:
        save_results_to_excel(results)
    except ImportError:
        print("\n⚠️ openpyxl not installed. Skipping Excel export.")
        print("Install with: pip install openpyxl")

    # Close connection
    conn.close()

    print("\n✅ SQL Analytics Complete!")
    print("Check data/processed/sql_analytics_results.xlsx for detailed results")


if __name__ == "__main__":
    main()
