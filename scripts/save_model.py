"""
Script to train and save the model for API serving.

Run this first: python scripts/save_model.py
"""

import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))


def prepare_and_save_model():
    """Train and save the model with all necessary components."""
    print("🚀 Training and saving model for API...")

    # Load processed data
    df = pd.read_csv("data/processed/processed_telco_data.csv")
    print(f"✅ Loaded {len(df)} records")

    # Prepare features (same as in advanced_modeling.py)
    df["churn_binary"] = (df["Churn"] == "Yes").astype(int)

    # Select features
    exclude_cols = ["customerID", "Churn", "churn_binary"]

    # Get feature columns
    numeric_features = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = df.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_features = [col for col in numeric_features if col not in exclude_cols]
    categorical_features = [col for col in categorical_features if col not in exclude_cols]

    # Create feature dataframe
    X = pd.DataFrame()

    # Add numeric features
    for col in numeric_features:
        if col in df.columns:
            X[col] = df[col].fillna(df[col].median())

    # Create label encoders dictionary
    label_encoders = {}

    # Encode categorical features
    for col in categorical_features:
        if col in df.columns:
            col_data = df[col].astype(str).fillna("Unknown")
            le = LabelEncoder()
            X[col + "_encoded"] = le.fit_transform(col_data)
            label_encoders[col] = le  # Save encoder

    # Add interaction features
    if "tenure" in X.columns and "MonthlyCharges" in X.columns:
        X["tenure_monthly_interaction"] = X["tenure"] * X["MonthlyCharges"]

    if "tenure" in X.columns and "TotalCharges" in X.columns:
        X["charges_per_tenure"] = X["TotalCharges"] / (X["tenure"] + 1)

    # Get target
    y = df["churn_binary"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost model (primary model)
    print("🔧 Training XGBoost model...")
    xgb_model = xgb.XGBClassifier(
        objective="binary:logistic",
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=3,
        random_state=42,
        eval_metric="auc",
    )

    xgb_model.fit(X_train_scaled, y_train)

    # Calculate metrics
    y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    auc_score = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"✅ Model trained - AUC: {auc_score:.3f}, Accuracy: {accuracy:.3f}")

    # Create models directory
    os.makedirs("models/saved", exist_ok=True)

    # Save all components
    model_artifacts = {
        "model": xgb_model,
        "scaler": scaler,
        "label_encoders": label_encoders,
        "feature_columns": X.columns.tolist(),
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "model_metrics": {
            "auc": auc_score,
            "accuracy": accuracy,
            "optimal_threshold": 0.5,
            "training_date": pd.Timestamp.now().isoformat(),
        },
    }

    # Save with joblib
    joblib.dump(model_artifacts, "models/saved/churn_model_artifacts.pkl")
    print("✅ Model artifacts saved to models/saved/churn_model_artifacts.pkl")

    # Also save feature importance
    feature_importance = pd.DataFrame(
        {"feature": X.columns, "importance": xgb_model.feature_importances_}
    ).sort_values("importance", ascending=False)

    feature_importance.to_csv("models/saved/feature_importance.csv", index=False)
    print("✅ Feature importance saved")

    # Save sample data for testing
    test_sample = df.sample(5)[numeric_features + categorical_features]
    test_sample.to_csv("models/saved/test_sample.csv", index=False)
    print("✅ Test sample saved")

    print("\n📊 Model Summary:")
    print(f"  - Features: {len(X.columns)}")
    print(f"  - Training samples: {len(X_train)}")
    print(f"  - Test samples: {len(X_test)}")
    print(f"  - Model AUC: {auc_score:.3f}")
    print("  - Top 5 Features:")
    for idx, row in feature_importance.head(5).iterrows():
        print(f"    {idx+1}. {row['feature']}: {row['importance']:.3f}")

    return model_artifacts


if __name__ == "__main__":
    prepare_and_save_model()
    print("\n✅ Model preparation complete! Ready for API serving.")
