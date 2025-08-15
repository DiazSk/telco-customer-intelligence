#!/usr/bin/env python3
"""
Simple MLflow Demo Script
Demonstrates basic MLflow functionality without complex dependencies
"""

import os
import sys
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import warnings

warnings.filterwarnings('ignore')

def create_sample_data():
    """Create sample telco-like dataset"""
    print("📊 Creating sample dataset...")
    
    np.random.seed(42)
    n_samples = 1000
    
    # Create features
    data = {
        'tenure': np.random.randint(1, 72, n_samples),
        'monthly_charges': np.random.normal(65, 20, n_samples),
        'total_charges': np.random.normal(2000, 1000, n_samples),
        'contract_monthly': np.random.binomial(1, 0.4, n_samples),
        'payment_electronic': np.random.binomial(1, 0.3, n_samples),
        'fiber_optic': np.random.binomial(1, 0.3, n_samples),
        'senior_citizen': np.random.binomial(1, 0.16, n_samples),
        'partner': np.random.binomial(1, 0.5, n_samples),
        'dependents': np.random.binomial(1, 0.3, n_samples),
        'phone_service': np.random.binomial(1, 0.9, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic churn target
    churn_prob = (
        0.1 +  # Base rate
        0.3 * df['contract_monthly'] +
        0.2 * (df['tenure'] < 12) +
        0.1 * df['payment_electronic'] +
        0.1 * (df['monthly_charges'] > 80)
    )
    
    y = np.random.binomial(1, churn_prob.clip(0, 1), n_samples)
    
    print(f"✅ Created {len(df)} samples with {y.mean():.1%} churn rate")
    return df.values, y

def run_mlflow_demo():
    """Run MLflow demonstration"""
    print("🎯 MLflow Demonstration")
    print("=" * 60)
    
    # Set tracking URI to local file storage
    mlflow.set_tracking_uri("file:./mlruns")
    print("🔧 MLflow tracking URI set to: ./mlruns")
    
    # Create sample data
    X, y = create_sample_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create/set experiment
    experiment_name = "Telco_Churn_Demo"
    mlflow.set_experiment(experiment_name)
    print(f"🧪 Using experiment: {experiment_name}")
    
    models_to_test = [
        ("RandomForest", RandomForestClassifier(n_estimators=50, random_state=42)),
        ("LogisticRegression", LogisticRegression(random_state=42, max_iter=1000))
    ]
    
    results = {}
    
    for model_name, model in models_to_test:
        print(f"\\n{'='*50}")
        print(f"🚀 Training {model_name}")
        print(f"{'='*50}")
        
        with mlflow.start_run(run_name=f"{model_name}_Demo"):
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            # Log parameters
            if hasattr(model, 'n_estimators'):
                mlflow.log_param("n_estimators", model.n_estimators)
            if hasattr(model, 'max_iter'):
                mlflow.log_param("max_iter", model.max_iter)
            
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("features", 10)
            mlflow.log_param("train_samples", len(X_train))
            mlflow.log_param("test_samples", len(X_test))
            
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            
            # Log model
            mlflow.sklearn.log_model(
                model, 
                "model",
                registered_model_name=f"telco_churn_{model_name.lower()}"
            )
            
            # Add tags
            mlflow.set_tag("project", "telco_churn")
            mlflow.set_tag("team", "data_science")
            mlflow.set_tag("model_category", "classification")
            
            # Store results
            results[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'run_id': mlflow.active_run().info.run_id
            }
            
            print(f"✅ {model_name} Results:")
            print(f"   Accuracy: {accuracy:.3f}")
            print(f"   Precision: {precision:.3f}")
            print(f"   Recall: {recall:.3f}")
            print(f"   F1-Score: {f1:.3f}")
            print(f"   Run ID: {mlflow.active_run().info.run_id}")
    
    # Summary
    print(f"\\n{'='*60}")
    print("📊 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1-Score: {metrics['f1_score']:.3f}")
        print(f"  Run ID: {metrics['run_id'][:8]}")
    
    # Find best model
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\\n🏆 Best Model: {best_model[0]} (Accuracy: {best_model[1]['accuracy']:.3f})")
    
    print(f"\\n🎉 MLflow Demo Complete!")
    print(f"\\n📁 Experiment data saved to: ./mlruns")
    print(f"🌐 To view in MLflow UI:")
    print(f"   1. Run: mlflow ui --backend-store-uri file:./mlruns")
    print(f"   2. Open: http://localhost:5000")
    print(f"\\n💡 Your mlruns folder should now contain:")
    print(f"   - Experiment metadata")
    print(f"   - Run information")
    print(f"   - Model artifacts") 
    print(f"   - Metrics and parameters")
    
    return results

if __name__ == "__main__":
    try:
        results = run_mlflow_demo()
        
        # Try to list mlruns contents
        if os.path.exists("mlruns"):
            print(f"\\n📂 MLruns folder contents:")
            for root, dirs, files in os.walk("mlruns"):
                level = root.replace("mlruns", "").count(os.sep)
                indent = " " * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 2 * (level + 1)
                for file in files[:5]:  # Show first 5 files only
                    print(f"{subindent}{file}")
                if len(files) > 5:
                    print(f"{subindent}... and {len(files) - 5} more files")
        
    except Exception as e:
        print(f"❌ Error running MLflow demo: {e}")
        sys.exit(1)
