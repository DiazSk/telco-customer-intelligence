"""
Test Dashboard Components
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.dashboard.dashboard_utils import *
import pandas as pd

def test_utilities():
    """Test utility functions"""
    print("Testing Dashboard Utilities...")
    
    # Test currency formatting
    assert format_currency(1500000) == "$1.5M"
    assert format_currency(25000) == "$25.0K"
    assert format_currency(999) == "$999"
    print("✅ Currency formatting works")
    
    # Test percentage formatting
    assert format_percentage(0.256) == "25.6%"
    assert format_percentage(0.256, 2) == "25.60%"
    print("✅ Percentage formatting works")
    
    # Load test data
    df = pd.read_csv("data/processed/processed_telco_data.csv")
    print(f"✅ Loaded {len(df)} records")
    
    # Test business metrics
    metrics = calculate_business_metrics(df)
    print(f"✅ Calculated {len(metrics)} metrics")
    print(f"   - Churn Rate: {metrics['churn_rate']*100:.1f}%")
    print(f"   - Monthly Revenue: ${metrics['monthly_revenue']:,.0f}")
    
    # Test ROI calculation
    roi = calculate_intervention_roi(
        num_customers=100,
        avg_churn_prob=0.3,
        intervention_cost=25,
        success_rate=0.4,
        customer_lifetime_value=2000
    )
    print(f"✅ ROI Calculation: {roi['roi_percentage']:.1f}%")
    
    print("\n✅ All utility tests passed!")

if __name__ == "__main__":
    test_utilities()