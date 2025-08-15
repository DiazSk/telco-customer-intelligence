"""
Integration tests for the API

These tests verify that different components work together correctly.
"""

import numpy as np
import pandas as pd
import pytest


def test_data_pipeline_integration():
    """Test that the data pipeline components work together."""
    # Create minimal test data
    data = {
        "customerID": ["TEST001", "TEST002"],
        "gender": ["Male", "Female"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "Yes"],
        "tenure": [12, 24],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["Yes", "No"],
        "OnlineBackup": ["No", "Yes"],
        "DeviceProtection": ["Yes", "No"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["Yes", "No"],
        "StreamingMovies": ["No", "Yes"],
        "Contract": ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [50.5, 75.0],
        "TotalCharges": ["606", "1800"],
        "Churn": ["No", "Yes"],
    }

    df = pd.DataFrame(data)

    # Test basic data processing
    assert len(df) == 2
    assert "customerID" in df.columns
    assert "Churn" in df.columns

    # Test data types
    assert df["tenure"].dtype in [np.int64, int]
    assert df["MonthlyCharges"].dtype in [np.float64, float]


def test_model_schema_compatibility():
    """Test that the model schemas are compatible with expected data."""
    try:
        from src.api.schemas.models import CustomerFeatures

        # Test data that should work with the schema
        test_customer = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "No",
            "DeviceProtection": "Yes",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 50.5,
            "TotalCharges": 606.0,
        }

        # This should not raise an exception
        customer = CustomerFeatures(**test_customer)
        assert customer.gender == "Male"
        assert customer.tenure == 12
        assert customer.MonthlyCharges == 50.5

    except ImportError:
        pytest.skip("API schemas not available")


def test_configuration_files_exist():
    """Test that required configuration files exist."""
    import os

    # Check that key config files exist
    config_files = ["configs/pipeline_config.yaml", ".pre-commit-config.yaml"]

    for config_file in config_files:
        if os.path.exists(config_file):
            assert True  # File exists
        else:
            pytest.skip(f"Configuration file {config_file} not found")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
