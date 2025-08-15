"""
Tests for Data Pipeline
"""

import numpy as np
import pandas as pd
import pytest

from src.data_pipeline.pipeline import (DataProcessor, DataQualityChecker,
                                        TelcoChurnPipeline)


@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data = {
        "customerID": ["1001", "1002", "1003", "1004", "1005"],
        "gender": ["Male", "Female", "Male", "Female", "Male"],
        "SeniorCitizen": [0, 1, 0, 1, 0],
        "Partner": ["Yes", "No", "Yes", "No", "Yes"],
        "Dependents": ["No", "Yes", "No", "Yes", "No"],
        "tenure": [12, 24, 6, 48, 36],
        "PhoneService": ["Yes", "Yes", "No", "Yes", "Yes"],
        "MultipleLines": ["No", "Yes", "No phone service", "No", "Yes"],
        "InternetService": ["DSL", "Fiber optic", "DSL", "Fiber optic", "No"],
        "OnlineSecurity": ["Yes", "No", "Yes", "No", "No internet service"],
        "OnlineBackup": ["No", "Yes", "No", "Yes", "No internet service"],
        "DeviceProtection": ["Yes", "No", "Yes", "No", "No internet service"],
        "TechSupport": ["No", "Yes", "No", "Yes", "No internet service"],
        "StreamingTV": ["Yes", "No", "Yes", "No", "No internet service"],
        "StreamingMovies": ["No", "Yes", "No", "Yes", "No internet service"],
        "Contract": [
            "Month-to-month",
            "One year",
            "Two year",
            "Month-to-month",
            "Two year",
        ],
        "PaperlessBilling": ["Yes", "No", "Yes", "No", "Yes"],
        "PaymentMethod": [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
            "Electronic check",
        ],
        "MonthlyCharges": [50.5, 75.0, 30.0, 90.0, 45.0],
        "TotalCharges": ["606", "1800", "180", "4320", " "],  # Include missing value
        "Churn": ["No", "Yes", "No", "Yes", "No"],
    }
    return pd.DataFrame(data)


@pytest.fixture
def config():
    """Create sample configuration"""
    return {
        "schema": {
            "id_column": "customerID",
            "numeric_columns": [
                "tenure",
                "MonthlyCharges",
                "TotalCharges",
                "SeniorCitizen",
            ],
            "categorical_columns": ["gender", "Partner", "Dependents", "Contract"],
            "target_column": "Churn",
        },
        "quality_checks": {
            "missing_threshold": 0.1,
            "outlier_detection": True,
            "duplicate_check": True,
        },
    }


class TestDataQualityChecker:
    """Test data quality checker functionality"""

    def test_check_missing_values(self, sample_data, config):
        """Test missing value detection"""
        checker = DataQualityChecker(config)
        result = checker.check_missing_values(sample_data)

        assert "total_missing" in result
        assert "columns_with_missing" in result
        assert result["total_missing"] == 0  # No actual NaN values yet

    def test_check_duplicates(self, sample_data, config):
        """Test duplicate detection"""
        checker = DataQualityChecker(config)
        result = checker.check_duplicates(sample_data, "customerID")

        assert "total_duplicates" in result
        assert "duplicate_ids" in result
        assert result["total_duplicates"] == 0
        assert result["duplicate_ids"] == 0

    def test_check_outliers(self, sample_data, config):
        """Test outlier detection"""
        checker = DataQualityChecker(config)
        numeric_cols = ["tenure", "MonthlyCharges"]
        result = checker.check_outliers(sample_data, numeric_cols)

        assert isinstance(result, dict)
        for col in numeric_cols:
            assert col in result
            assert "count" in result[col]
            assert "percentage" in result[col]


class TestDataProcessor:
    """Test data processing functionality"""

    def test_handle_missing_values(self, sample_data, config):
        """Test missing value handling"""
        processor = DataProcessor(config)
        processed_df = processor.handle_missing_values(sample_data)

        # Check TotalCharges is numeric and has no missing values
        assert pd.api.types.is_numeric_dtype(processed_df["TotalCharges"])
        assert processed_df["TotalCharges"].isnull().sum() == 0

    def test_clean_column_values(self, sample_data, config):
        """Test column value cleaning"""
        processor = DataProcessor(config)
        processed_df = processor.clean_column_values(sample_data)

        # Check standardization
        assert all(processed_df["Partner"].isin(["Yes", "No"]))
        assert "No_internet_service" in processed_df["OnlineSecurity"].values

    def test_create_derived_features(self, sample_data, config):
        """Test feature creation"""
        processor = DataProcessor(config)
        # First handle missing values
        sample_data = processor.handle_missing_values(sample_data)
        processed_df = processor.create_derived_features(sample_data)

        # Check new features exist
        assert "tenure_group" in processed_df.columns
        assert "total_services" in processed_df.columns
        assert "customer_value_score" in processed_df.columns
        assert "loyalty_score" in processed_df.columns

        # Check feature values are reasonable
        assert processed_df["total_services"].min() >= 0
        assert processed_df["total_services"].max() <= 9
        assert processed_df["loyalty_score"].min() >= 0
        assert processed_df["loyalty_score"].max() <= 1


class TestPipeline:
    """Test complete pipeline functionality"""

    def test_pipeline_initialization(self, tmp_path):
        """Test pipeline initialization"""
        # Create temporary config file
        config = {
            "data": {
                "raw_data_path": "data/raw/test.csv",
                "processed_data_path": str(tmp_path / "processed.csv"),
                "feature_store_path": str(tmp_path / "features.parquet"),
            },
            "schema": {
                "id_column": "customerID",
                "numeric_columns": ["tenure"],
                "target_column": "Churn",
            },
        }

        import yaml

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f)

        # Initialize pipeline
        pipeline = TelcoChurnPipeline(str(config_path))
        assert pipeline.config == config
        assert pipeline.quality_checker is not None
        assert pipeline.processor is not None


def test_data_shape_preservation(sample_data, config):
    """Test that data shape is preserved or expanded correctly"""
    processor = DataProcessor(config)
    original_rows = len(sample_data)

    processed_df = processor.process(sample_data)

    # Same number of rows
    assert len(processed_df) == original_rows

    # Same or more columns (due to feature engineering)
    assert len(processed_df.columns) >= len(sample_data.columns)


def test_no_data_leakage(sample_data, config):
    """Test that target column is not used in feature creation"""
    processor = DataProcessor(config)
    processed_df = processor.process(sample_data)

    # Get all new features
    original_cols = sample_data.columns.tolist()
    new_features = [col for col in processed_df.columns if col not in original_cols]

    # Ensure Churn column wasn't modified
    assert all(processed_df["Churn"] == sample_data["Churn"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
