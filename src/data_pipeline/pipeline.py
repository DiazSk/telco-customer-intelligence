"""
Data Pipeline for Telco Customer Churn Analysis
Author: Zaid Shaikh
Date: 2025
"""

import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class DataQualityChecker:
    """Handles data quality validation and reporting"""

    def __init__(self, config: Dict):
        self.config = config
        self.quality_report: Dict[str, Any] = {}

    def check_missing_values(self, df: pd.DataFrame) -> Dict:
        """Check for missing values in the dataset"""
        missing_stats = {
            "total_missing": df.isnull().sum().sum(),
            "columns_with_missing": {},
            "missing_percentage": {},
        }

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                missing_stats["columns_with_missing"][col] = missing_count
                missing_stats["missing_percentage"][col] = round(missing_pct, 2)

        return missing_stats

    def check_duplicates(self, df: pd.DataFrame, id_column: str) -> Dict:
        """Check for duplicate records"""
        duplicate_stats = {
            "total_duplicates": df.duplicated().sum(),
            "duplicate_ids": 0,
        }

        if id_column in df.columns:
            duplicate_stats["duplicate_ids"] = df[id_column].duplicated().sum()

        return duplicate_stats

    def check_data_types(self, df: pd.DataFrame, schema: Dict) -> Dict:
        """Validate data types against expected schema"""
        type_issues = []

        for col in df.columns:
            if col in schema.get("numeric_columns", []):
                if not pd.api.types.is_numeric_dtype(df[col]):
                    type_issues.append(
                        f"{col} should be numeric but is {df[col].dtype}"
                    )

        return {"type_issues": type_issues}

    def check_outliers(self, df: pd.DataFrame, numeric_cols: List[str]) -> Dict:
        """Detect outliers using IQR method"""
        outlier_stats = {}

        for col in numeric_cols:
            if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
                outlier_stats[col] = {
                    "count": len(outliers),
                    "percentage": round((len(outliers) / len(df)) * 100, 2),
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                }

        return outlier_stats

    def run_all_checks(self, df: pd.DataFrame) -> Dict:
        """Run all quality checks"""
        logger.info("Running data quality checks...")

        self.quality_report["missing_values"] = self.check_missing_values(df)
        self.quality_report["duplicates"] = self.check_duplicates(
            df, self.config["schema"]["id_column"]
        )
        self.quality_report["data_types"] = self.check_data_types(
            df, self.config["schema"]
        )
        self.quality_report["outliers"] = self.check_outliers(
            df, self.config["schema"]["numeric_columns"]
        )
        self.quality_report["shape"] = {"rows": len(df), "columns": len(df.columns)}
        self.quality_report["timestamp"] = datetime.now().isoformat()

        return self.quality_report


class DataProcessor:
    """Handles data cleaning and preprocessing"""

    def __init__(self, config: Dict):
        self.config = config

    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        logger.info("Handling missing values...")

        # Handle TotalCharges - contains empty strings
        if "TotalCharges" in df.columns:
            df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
            # Fill missing TotalCharges with MonthlyCharges * tenure
            mask = df["TotalCharges"].isna()
            df.loc[mask, "TotalCharges"] = (
                df.loc[mask, "MonthlyCharges"] * df.loc[mask, "tenure"]
            )

        # Handle any other missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=["object"]).columns

        # Fill numeric columns with median
        for col in numeric_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)

        # Fill categorical columns with mode
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(
                    df[col].mode()[0] if not df[col].mode().empty else "Unknown",
                    inplace=True,
                )

        return df

    def clean_column_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize column values"""
        logger.info("Cleaning column values...")

        # Standardize Yes/No values
        yes_no_columns = [
            "Partner",
            "Dependents",
            "PhoneService",
            "PaperlessBilling",
            "Churn",
        ]
        for col in yes_no_columns:
            if col in df.columns:
                df[col] = df[col].str.strip().str.title()

        # Clean service columns
        service_columns = [
            "MultipleLines",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]
        for col in service_columns:
            if col in df.columns:
                df[col] = df[col].str.strip()
                # Standardize 'No internet service' and 'No phone service'
                df[col] = df[col].replace(
                    {
                        "No internet service": "No_internet_service",
                        "No phone service": "No_phone_service",
                    }
                )

        return df

    def create_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create derived features for better model performance"""
        logger.info("Creating derived features...")

        # Tenure groups
        df["tenure_group"] = pd.cut(
            df["tenure"],
            bins=[0, 12, 24, 48, 60, 72],
            labels=["0-12", "13-24", "25-48", "49-60", "61-72"],
            include_lowest=True,
        )

        # Average charges per month
        df["avg_charges_per_tenure"] = df["TotalCharges"] / (df["tenure"] + 1)

        # Charge difference from monthly
        df["charge_difference"] = df["avg_charges_per_tenure"] - df["MonthlyCharges"]

        # Service count (number of services subscribed)
        service_columns = [
            "PhoneService",
            "MultipleLines",
            "InternetService",
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
            "StreamingTV",
            "StreamingMovies",
        ]

        df["total_services"] = 0
        for col in service_columns:
            if col in df.columns:
                df["total_services"] += (df[col] == "Yes").astype(int)

        # Has online services
        online_services = [
            "OnlineSecurity",
            "OnlineBackup",
            "DeviceProtection",
            "TechSupport",
        ]
        df["has_online_services"] = df[online_services].apply(
            lambda x: 1 if "Yes" in x.values else 0, axis=1
        )

        # Has streaming services
        streaming_services = ["StreamingTV", "StreamingMovies"]
        df["has_streaming"] = df[streaming_services].apply(
            lambda x: 1 if "Yes" in x.values else 0, axis=1
        )

        # Contract type score (longer contracts = higher score)
        contract_score = {"Month-to-month": 0, "One year": 1, "Two year": 2}
        df["contract_score"] = df["Contract"].map(contract_score)

        # Payment method risk (based on churn correlation)
        payment_risk = {
            "Electronic check": 3,
            "Mailed check": 1,
            "Bank transfer (automatic)": 0,
            "Credit card (automatic)": 0,
        }
        df["payment_risk_score"] = df["PaymentMethod"].map(payment_risk).fillna(2)

        # Customer lifetime value proxy
        df["customer_value_score"] = (
            df["tenure"] * 0.3
            + df["MonthlyCharges"] * 0.3
            + df["total_services"] * 0.2
            + df["contract_score"] * 0.2
        )

        # Loyalty score
        df["loyalty_score"] = (
            (df["tenure"] / df["tenure"].max()) * 0.4
            + (df["contract_score"] / 2) * 0.3
            + (1 - df["payment_risk_score"] / 3) * 0.3
        )

        return df

    def process(self, df: pd.DataFrame) -> pd.DataFrame:
        """Main processing pipeline"""
        df = self.handle_missing_values(df)
        df = self.clean_column_values(df)
        df = self.create_derived_features(df)
        return df


class TelcoChurnPipeline:
    """Main pipeline orchestrator"""

    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.quality_checker = DataQualityChecker(self.config)
        self.processor = DataProcessor(self.config)

    def load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file"""
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def load_data(self) -> pd.DataFrame:
        """Load raw data"""
        logger.info(f"Loading data from {self.config['data']['raw_data_path']}")
        df = pd.read_csv(self.config["data"]["raw_data_path"])
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        return df

    def save_processed_data(self, df: pd.DataFrame):
        """Save processed data"""
        # Save as CSV
        processed_path = self.config["data"]["processed_data_path"]
        logger.info(f"Saving processed data to {processed_path}")
        df.to_csv(processed_path, index=False)

        # Save as Parquet (more efficient)
        feature_store_path = self.config["data"]["feature_store_path"]
        logger.info(f"Saving feature store to {feature_store_path}")
        df.to_parquet(feature_store_path, index=False)

    def save_quality_report(self, report: Dict):
        """Save data quality report"""
        report_path = Path("data/processed/quality_report.yaml")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        with open(report_path, "w") as f:
            yaml.dump(report, f, default_flow_style=False)
        logger.info(f"Quality report saved to {report_path}")

    def run(self) -> pd.DataFrame:
        """Run the complete pipeline"""
        logger.info("=" * 50)
        logger.info("Starting Telco Churn Data Pipeline")
        logger.info("=" * 50)

        # Load data
        df = self.load_data()

        # Run quality checks on raw data
        logger.info("Running quality checks on raw data...")
        quality_report_raw = self.quality_checker.run_all_checks(df)

        # Process data
        logger.info("Processing data...")
        df_processed = self.processor.process(df)

        # Run quality checks on processed data
        logger.info("Running quality checks on processed data...")
        quality_report_processed = self.quality_checker.run_all_checks(df_processed)

        # Save quality reports
        final_report = {
            "raw_data": quality_report_raw,
            "processed_data": quality_report_processed,
        }
        self.save_quality_report(final_report)

        # Save processed data
        self.save_processed_data(df_processed)

        # Print summary
        logger.info("=" * 50)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Original shape: {quality_report_raw['shape']}")
        logger.info(f"Processed shape: {quality_report_processed['shape']}")
        logger.info(
            f"New features created: {len(df_processed.columns) - len(df.columns)}"
        )
        logger.info("=" * 50)

        return df_processed


def main():
    """Main execution function"""
    import argparse

    parser = argparse.ArgumentParser(description="Run Telco Churn Data Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/pipeline_config.yaml",
        help="Path to configuration file",
    )

    args = parser.parse_args()

    # Run pipeline
    pipeline = TelcoChurnPipeline(args.config)
    df_processed = pipeline.run()

    # Display sample of processed data
    print("\nSample of processed data:")
    print(df_processed.head())

    print("\nNew features created:")
    original_cols = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn",
    ]
    new_features = [col for col in df_processed.columns if col not in original_cols]
    for feature in new_features:
        print(f" - {feature}")


if __name__ == "__main__":
    main()
