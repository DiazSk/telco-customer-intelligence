"""
Advanced Modeling Module for Telco Churn
Demonstrates DS skills: ML, Stats, Business Impact
"""

import warnings
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import xgboost as xgb
from scipy import stats
from sklearn.metrics import precision_recall_curve, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


class ChurnModelingPipeline:
    """Production-ready modeling pipeline with business focus"""

    def __init__(self, business_params: Dict = None):
        """
        Initialize with business parameters

        Args:
            business_params: Dict containing:
                - customer_acquisition_cost: Cost to acquire new customer
                - retention_campaign_cost: Cost per retention attempt
                - avg_customer_lifetime_value: Average CLV
                - intervention_success_rate: Historical success rate
        """
        self.business_params = business_params or {
            "customer_acquisition_cost": 300,
            "retention_campaign_cost": 50,
            "avg_customer_lifetime_value": 1350,
            "intervention_success_rate": 0.4,
        }
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Any] = {}

    def load_and_prep_data(self, filepath: str) -> pd.DataFrame:
        """Load and prepare data with proper time-based splitting"""
        df = pd.read_parquet(filepath)

        # Convert Churn to binary
        df["churn_binary"] = (df["Churn"] == "Yes").astype(int)

        # Sort by tenure to simulate time-based ordering
        df = df.sort_values("tenure")

        return df

    def create_business_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that capture business value"""

        # Revenue Risk Score
        df["revenue_risk"] = df["MonthlyCharges"] * df["tenure"]

        # Customer Profitability Score
        df["profitability_score"] = df["TotalCharges"] - (
            df["tenure"] * df["MonthlyCharges"] * 0.7
        )  # Assume 70% cost

        # Engagement Decay (for newer customers)
        df["engagement_decay"] = np.exp(-df["tenure"] / 12)

        # Price Sensitivity
        median_price = df["MonthlyCharges"].median()
        df["price_sensitivity"] = (df["MonthlyCharges"] - median_price) / median_price

        # Contract Commitment Score
        contract_weights = {"Month-to-month": 0, "One year": 0.5, "Two year": 1}
        df["commitment_score"] = df["Contract"].map(contract_weights)

        # Service Stickiness (more services = harder to leave)
        df["service_stickiness"] = df["total_services"] / 9  # Max 9 services

        # Payment Method Risk
        payment_risk = {
            "Electronic check": 1,
            "Mailed check": 0.5,
            "Bank transfer (automatic)": 0.2,
            "Credit card (automatic)": 0.1,
        }
        df["payment_method_risk"] = df["PaymentMethod"].map(payment_risk).fillna(0.5)

        return df

    def statistical_analysis(self, df: pd.DataFrame) -> Dict:
        """Perform statistical tests for feature significance"""
        results = {}

        # Chi-square tests for categorical variables
        categorical_vars = ["Contract", "PaymentMethod", "InternetService"]
        for var in categorical_vars:
            if var in df.columns:
                contingency_table = pd.crosstab(df[var], df["churn_binary"])
                chi2, p_value, _, _ = stats.chi2_contingency(contingency_table)
                results[f"{var}_chi2"] = {"statistic": chi2, "p_value": p_value}

        # T-tests for numerical variables
        numerical_vars = ["tenure", "MonthlyCharges", "TotalCharges"]
        for var in numerical_vars:
            if var in df.columns:
                churned = df[df["churn_binary"] == 1][var].dropna()
                not_churned = df[df["churn_binary"] == 0][var].dropna()
                t_stat, p_value = stats.ttest_ind(churned, not_churned)
                results[f"{var}_ttest"] = {
                    "statistic": t_stat,
                    "p_value": p_value,
                    "churned_mean": churned.mean(),
                    "not_churned_mean": not_churned.mean(),
                }

        return results

    def train_models(self, df: pd.DataFrame) -> Dict:
        """Train multiple models with business-focused metrics"""

        # Prepare features
        feature_cols = [
            col
            for col in df.columns
            if col not in ["customerID", "Churn", "churn_binary"]
        ]

        # Handle categorical variables
        df_model = pd.get_dummies(df[feature_cols + ["churn_binary"]], drop_first=True)

        X = df_model.drop("churn_binary", axis=1)
        y = df_model["churn_binary"]

        # Time-based split (more realistic for production)
        split_idx = int(len(X) * 0.7)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        results = {}

        # 1. XGBoost with Custom Business Objective
        xgb_model = xgb.XGBClassifier(
            objective="binary:logistic",
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
            random_state=42,
        )
        xgb_model.fit(X_train_scaled, y_train)
        self.models["xgboost"] = xgb_model

        # 2. LightGBM for comparison
        lgb_model = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            is_unbalance=True,
            random_state=42,
            verbosity=-1,
        )
        lgb_model.fit(X_train_scaled, y_train)
        self.models["lightgbm"] = lgb_model

        # Evaluate models
        for name, model in self.models.items():
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

            # Calculate business-focused metrics
            results[name] = self.calculate_business_metrics(
                y_test, y_pred_proba, X_test, df.iloc[split_idx:]
            )

            # Add SHAP analysis for interpretability
            results[name]["shap_values"] = self.calculate_shap_values(
                model, X_train_scaled, X_train.columns
            )

        self.results = results
        return results

    def calculate_business_metrics(self, y_true, y_pred_proba, X_test, df_test):
        """Calculate metrics that matter to business"""

        # Find optimal threshold based on business costs
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_profit = -np.inf
        best_threshold = 0.5

        for threshold in thresholds:
            y_pred = (y_pred_proba >= threshold).astype(int)

            # Calculate profit/loss
            tp = np.sum((y_true == 1) & (y_pred == 1))  # Correctly identified churners
            fp = np.sum((y_true == 0) & (y_pred == 1))  # False alarms
            fn = np.sum((y_true == 1) & (y_pred == 0))  # Missed churners

            # Business calculation
            profit = (
                tp
                * self.business_params["avg_customer_lifetime_value"]
                * self.business_params["intervention_success_rate"]
                - (tp + fp) * self.business_params["retention_campaign_cost"]
                - fn * self.business_params["avg_customer_lifetime_value"]
            )

            if profit > best_profit:
                best_profit = profit
                best_threshold = threshold

        # Calculate final metrics with optimal threshold
        y_pred_optimal = (y_pred_proba >= best_threshold).astype(int)

        return {
            "auc_roc": roc_auc_score(y_true, y_pred_proba),
            "optimal_threshold": best_threshold,
            "expected_profit": best_profit,
            "precision_at_optimal": (
                np.sum((y_true == 1) & (y_pred_optimal == 1)) / np.sum(y_pred_optimal)
                if np.sum(y_pred_optimal) > 0
                else 0
            ),
            "recall_at_optimal": (
                np.sum((y_true == 1) & (y_pred_optimal == 1)) / np.sum(y_true)
                if np.sum(y_true) > 0
                else 0
            ),
            "customers_to_target": np.sum(y_pred_optimal),
            "expected_saves": np.sum((y_true == 1) & (y_pred_optimal == 1))
            * self.business_params["intervention_success_rate"],
        }

    def calculate_shap_values(self, model, X_train, feature_names):
        """Calculate SHAP values for model interpretability"""
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_train[:100])  # Sample for speed

        # Get feature importance
        feature_importance = (
            pd.DataFrame(
                {
                    "feature": feature_names,
                    "importance": np.abs(shap_values.values).mean(axis=0),
                }
            )
            .sort_values("importance", ascending=False)
            .head(10)
        )

        return feature_importance.to_dict("records")

    def create_actionable_segments(self, df: pd.DataFrame, model) -> pd.DataFrame:
        """Create customer segments with specific action recommendations"""

        # Prepare features for prediction
        feature_cols = [
            col
            for col in df.columns
            if col not in ["customerID", "Churn", "churn_binary"]
        ]
        df_model = pd.get_dummies(df[feature_cols], drop_first=True)

        # Get predictions
        predictions = model.predict_proba(df_model)[:, 1]

        # Create segments
        df["churn_probability"] = predictions
        df["risk_segment"] = pd.cut(
            predictions,
            bins=[0, 0.2, 0.5, 0.8, 1.0],
            labels=["Low Risk", "Medium Risk", "High Risk", "Critical"],
        )

        # Add action recommendations
        def get_recommendation(row):
            if row["risk_segment"] == "Critical":
                if row["Contract"] == "Month-to-month":
                    return "Immediate: Offer 50% discount for 1-year contract upgrade"
                else:
                    return "Immediate: Personal call from account manager"
            elif row["risk_segment"] == "High Risk":
                if row["total_services"] < 3:
                    return "Proactive: Bundle discount for additional services"
                else:
                    return "Proactive: Loyalty rewards program enrollment"
            elif row["risk_segment"] == "Medium Risk":
                return "Monitor: Send satisfaction survey and track engagement"
            else:
                return "Maintain: Continue current service, upsell opportunities"

        df["recommended_action"] = df.apply(get_recommendation, axis=1)

        # Calculate intervention ROI for each segment
        segment_summary = (
            df.groupby("risk_segment")
            .agg(
                {
                    "customerID": "count",
                    "MonthlyCharges": "mean",
                    "churn_probability": "mean",
                    "TotalCharges": "sum",
                }
            )
            .rename(columns={"customerID": "customer_count"})
        )

        segment_summary["potential_revenue_loss"] = (
            segment_summary["customer_count"]
            * segment_summary["MonthlyCharges"]
            * 12
            * segment_summary["churn_probability"]
        )

        segment_summary["intervention_cost"] = (
            segment_summary["customer_count"]
            * self.business_params["retention_campaign_cost"]
        )

        segment_summary["expected_roi"] = (
            segment_summary["potential_revenue_loss"]
            * self.business_params["intervention_success_rate"]
        ) / segment_summary["intervention_cost"]

        return df, segment_summary

    def generate_executive_summary(self) -> str:
        """Generate business-friendly summary of results"""

        best_model = max(self.results.items(), key=lambda x: x[1]["expected_profit"])

        summary = f"""
        EXECUTIVE SUMMARY: CHURN PREDICTION MODEL
        ==========================================

        BUSINESS IMPACT:
        - Expected Annual Savings: ${best_model[1]['expected_profit']:,.0f}
        - Customers to Target: {best_model[1]['customers_to_target']:,}
        - Expected Customers Saved: {best_model[1]['expected_saves']:.0f}
        - Model Accuracy (AUC): {best_model[1]['auc_roc']:.1%}

        KEY INSIGHTS:
        - Optimal intervention threshold: {best_model[1]['optimal_threshold']:.1%} churn probability
        - Precision at optimal threshold: {best_model[1]['precision_at_optimal']:.1%}
        - Recall at optimal threshold: {best_model[1]['recall_at_optimal']:.1%}

        TOP CHURN DRIVERS:
        """

        for i, feature in enumerate(best_model[1]["shap_values"][:5], 1):
            summary += f"\n        {i}. {feature['feature']}"

        summary += """

        RECOMMENDED ACTIONS:
        1. Focus retention efforts on customers with >{:.0f}% churn probability
        2. Prioritize month-to-month contract customers for upgrade campaigns
        3. Implement automated intervention system for high-risk segments
        4. Track intervention success rate to refine model thresholds

        ROI PROJECTION:
        - Investment Required: ${:,.0f}
        - Expected Return: ${:,.0f}
        - ROI: {:.1f}x
        """.format(
            best_model[1]["optimal_threshold"] * 100,
            best_model[1]["customers_to_target"]
            * self.business_params["retention_campaign_cost"],
            best_model[1]["expected_profit"],
            best_model[1]["expected_profit"]
            / (
                best_model[1]["customers_to_target"]
                * self.business_params["retention_campaign_cost"]
            ),
        )

        return summary


def main():
    """Run the complete modeling pipeline"""

    # Initialize pipeline with business parameters
    pipeline = ChurnModelingPipeline(
        {
            "customer_acquisition_cost": 300,
            "retention_campaign_cost": 50,
            "avg_customer_lifetime_value": 1350,
            "intervention_success_rate": 0.4,
        }
    )

    # Load and prepare data
    df = pipeline.load_and_prep_data("data/features/feature_store.parquet")
    df = pipeline.create_business_features(df)

    # Statistical analysis
    print("Running statistical analysis...")
    stats_results = pipeline.statistical_analysis(df)

    # Train models
    print("Training models with business optimization...")
    model_results = pipeline.train_models(df)

    # Create actionable segments
    print("Creating customer segments...")
    df_segmented, segment_summary = pipeline.create_actionable_segments(
        df, pipeline.models["xgboost"]
    )

    # Generate executive summary
    print("\n" + "=" * 60)
    print(pipeline.generate_executive_summary())
    print("=" * 60)

    # Save results
    df_segmented.to_csv("data/processed/customer_segments.csv", index=False)
    segment_summary.to_csv("data/processed/segment_summary.csv")

    print("\n✅ Modeling complete! Results saved to data/processed/")

    return pipeline, df_segmented, segment_summary


if __name__ == "__main__":
    pipeline, df_segmented, segment_summary = main()
