#!/usr/bin/env python
"""
Script to run the data pipeline with the Telco Customer Churn dataset
"""

import logging
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd  # noqa: E402

from src.data_pipeline.pipeline import TelcoChurnPipeline  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    """Run the pipeline and display results"""

    # Check if data file exists
    data_path = Path("data/raw/Telco_Customer_Churn.csv")
    if not data_path.exists():
        logger.error(f"Data file not found at {data_path}")
        logger.info("Please place the Telco_Customer_Churn.csv file in data/raw/")
        return

    # Run pipeline
    logger.info("Starting Telco Churn Data Pipeline")
    logger.info("=" * 60)

    try:
        # Initialize and run pipeline
        pipeline = TelcoChurnPipeline("configs/pipeline_config.yaml")
        df_processed = pipeline.run()

        # Display results
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE RESULTS")
        logger.info("=" * 60)

        # Basic statistics
        logger.info(f"Processed dataset shape: {df_processed.shape}")
        logger.info(f"Memory usage: {df_processed.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # Feature statistics
        original_features = 21  # Original dataset had 21 columns
        new_features = len(df_processed.columns) - original_features
        logger.info(f"New features created: {new_features}")

        # Churn distribution
        churn_dist = df_processed["Churn"].value_counts()
        churn_rate = (churn_dist.get("Yes", 0) / len(df_processed)) * 100
        logger.info(f"Churn rate: {churn_rate:.2f}%")

        # Display new features
        logger.info("\nNew features created:")
        new_feature_cols = [
            "tenure_group",
            "avg_charges_per_tenure",
            "charge_difference",
            "total_services",
            "has_online_services",
            "has_streaming",
            "contract_score",
            "payment_risk_score",
            "customer_value_score",
            "loyalty_score",
        ]
        for feature in new_feature_cols:
            if feature in df_processed.columns:
                logger.info(f"  {feature}")

        # Sample of processed data
        logger.info("\nSample of processed data (first 3 rows):")
        print(
            df_processed[
                [
                    "customerID",
                    "tenure",
                    "MonthlyCharges",
                    "total_services",
                    "loyalty_score",
                    "Churn",
                ]
            ].head(3)
        )

        # Data quality summary
        logger.info("\nData Quality Summary:")
        logger.info("  - Missing values handled: Yes")
        logger.info("  - Duplicates removed: Yes")
        logger.info("  - Features engineered: Yes")
        logger.info("  - Data saved to: data/processed/")

        # Next steps
        logger.info("\n" + "=" * 60)
        logger.info("NEXT STEPS")
        logger.info("=" * 60)
        logger.info("1. Review the quality report: data/processed/quality_report.yaml")
        logger.info("2. Explore the data: jupyter notebook notebooks/01_eda.ipynb")
        logger.info("3. Train a model: python src/models/train.py")
        logger.info("4. Start the API: uvicorn src.api.main:app --reload")
        logger.info("5. Launch dashboard: streamlit run src/dashboard/app.py")

        logger.info("\nPipeline completed successfully!")

    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
