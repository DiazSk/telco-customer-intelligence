"""
Model Training Script with MLflow Integration
Integrates the existing advanced modeling with MLflow tracking
"""

import os
import sys
import warnings
from pathlib import Path
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.models.advanced_modeling import ChurnModelingPipeline

warnings.filterwarnings('ignore')

class MLflowModelTrainer:
    """Model trainer with MLflow integration"""
    
    def __init__(self, mlflow_uri: str = "http://localhost:5000"):
        self.mlflow_uri = mlflow_uri
        self.setup_mlflow()
        
    def setup_mlflow(self):
        """Setup MLflow tracking"""
        try:
            mlflow.set_tracking_uri(self.mlflow_uri)
            print(f"✅ Connected to MLflow at {self.mlflow_uri}")
        except Exception as e:
            print(f"⚠️ Could not connect to MLflow server: {e}")
            print("Using local file tracking...")
            mlflow.set_tracking_uri("file:./mlruns")
    
    def train_production_model(self, data_path: str = "data/processed/processed_telco_data.csv"):
        """Train production model with MLflow tracking"""
        
        # Set experiment
        mlflow.set_experiment("Production_Model_Training")
        
        with mlflow.start_run(run_name=f"Production_XGBoost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            
            print("🚀 Starting production model training with MLflow...")
            
            # Initialize modeling pipeline
            pipeline = ChurnModelingPipeline()
            
            # Log business parameters
            for key, value in pipeline.business_params.items():
                mlflow.log_param(f"business_{key}", value)
            
            # Load and prepare data
            try:
                df = pipeline.load_and_prep_data(data_path)
                print(f"✅ Loaded data: {len(df)} records")
            except Exception as e:
                print(f"⚠️ Could not load data from {data_path}: {e}")
                print("Creating sample data for demonstration...")
                df = self.create_sample_data()
            
            mlflow.log_param("data_source", data_path)
            mlflow.log_param("dataset_size", len(df))
            
            # Prepare features and train models
            feature_summary = pipeline.prepare_features(df)
            
            # Log feature engineering metrics
            mlflow.log_param("num_features", feature_summary['total_features'])
            mlflow.log_param("num_categorical", feature_summary['categorical_features'])
            mlflow.log_param("num_numerical", feature_summary['numerical_features'])
            
            # Train XGBoost model (primary)
            print("🔧 Training XGBoost model...")
            xgb_results = pipeline.train_xgboost(df)
            
            # Log XGBoost parameters and metrics
            self.log_model_results("XGBoost", xgb_results, pipeline)
            
            # Train LightGBM model (comparison)
            print("🔧 Training LightGBM model...")
            lgb_results = pipeline.train_lightgbm(df)
            
            # Log LightGBM parameters and metrics
            self.log_model_results("LightGBM", lgb_results, pipeline)
            
            # Generate and log business insights
            business_insights = pipeline.calculate_business_impact(xgb_results)
            
            # Log business metrics
            for key, value in business_insights.items():
                if isinstance(value, (int, float)):
                    mlflow.log_metric(f"business_{key}", value)
                else:
                    mlflow.log_param(f"business_{key}", str(value))
            
            # Log model artifacts
            self.log_model_artifacts(xgb_results, pipeline, df)
            
            # Set tags
            mlflow.set_tag("model_purpose", "production_churn_prediction")
            mlflow.set_tag("team", "data_science")
            mlflow.set_tag("project", "telco_customer_intelligence")
            mlflow.set_tag("primary_model", "XGBoost")
            
            print(f"✅ Training complete! MLflow run: {mlflow.active_run().info.run_id}")
            
            return xgb_results, lgb_results, business_insights
    
    def log_model_results(self, model_name: str, results: dict, pipeline):
        """Log model results to MLflow"""
        
        model = results.get('model')
        metrics = results.get('performance_metrics', {})
        params = results.get('model_params', {})
        
        # Log model parameters
        for key, value in params.items():
            mlflow.log_param(f"{model_name.lower()}_{key}", value)
        
        # Log performance metrics
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"{model_name.lower()}_{key}", value)
        
        # Log model
        if model is not None:
            if model_name == "XGBoost":
                mlflow.xgboost.log_model(
                    model, 
                    f"{model_name.lower()}_model",
                    registered_model_name=f"telco_churn_{model_name.lower()}"
                )
            elif model_name == "LightGBM":
                mlflow.lightgbm.log_model(
                    model, 
                    f"{model_name.lower()}_model",
                    registered_model_name=f"telco_churn_{model_name.lower()}"
                )
            else:
                mlflow.sklearn.log_model(
                    model, 
                    f"{model_name.lower()}_model",
                    registered_model_name=f"telco_churn_{model_name.lower()}"
                )
    
    def log_model_artifacts(self, results: dict, pipeline, df: pd.DataFrame):
        """Log model artifacts and visualizations"""
        
        try:
            # Feature importance
            if 'feature_importance' in results:
                importance_df = pd.DataFrame(results['feature_importance'])
                importance_df.to_csv('feature_importance.csv', index=False)
                mlflow.log_artifact('feature_importance.csv')
                os.remove('feature_importance.csv')
            
            # Model performance summary
            if 'performance_metrics' in results:
                metrics_df = pd.DataFrame([results['performance_metrics']])
                metrics_df.to_csv('performance_metrics.csv', index=False)
                mlflow.log_artifact('performance_metrics.csv')
                os.remove('performance_metrics.csv')
            
            # Data summary
            data_summary = {
                'total_customers': len(df),
                'churn_rate': df['Churn'].value_counts(normalize=True).get('Yes', 0) if 'Churn' in df.columns else 0,
                'features': list(df.columns),
                'data_types': df.dtypes.to_dict()
            }
            
            import json
            with open('data_summary.json', 'w') as f:
                json.dump(data_summary, f, indent=2, default=str)
            mlflow.log_artifact('data_summary.json')
            os.remove('data_summary.json')
            
            print("✅ Artifacts logged successfully")
            
        except Exception as e:
            print(f"⚠️ Could not log some artifacts: {e}")
    
    def create_sample_data(self):
        """Create sample data if real data not available"""
        np.random.seed(42)
        n_samples = 1000
        
        data = {
            'customerID': [f'C{i:04d}' for i in range(n_samples)],
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
            'tenure': np.random.randint(1, 72, n_samples),
            'PhoneService': np.random.choice(['Yes', 'No'], n_samples),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], n_samples),
            'MonthlyCharges': np.random.normal(65, 20, n_samples),
            'TotalCharges': np.random.normal(2000, 1000, n_samples),
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic churn target
        churn_prob = (
            0.1 +  # Base rate
            0.3 * (df['Contract'] == 'Month-to-month') +
            0.2 * (df['tenure'] < 12) +
            0.1 * (df['PaymentMethod'] == 'Electronic check') +
            0.1 * (df['MonthlyCharges'] > 80)
        )
        
        df['Churn'] = np.where(np.random.binomial(1, churn_prob.clip(0, 1), n_samples), 'Yes', 'No')
        
        print(f"✅ Created sample dataset: {len(df)} records, churn rate: {(df['Churn']=='Yes').mean():.1%}")
        
        return df

def main():
    """Main training function"""
    print("🎯 Training Models with MLflow Integration")
    print("=" * 50)
    
    # Initialize trainer
    trainer = MLflowModelTrainer()
    
    # Train production model
    xgb_results, lgb_results, business_insights = trainer.train_production_model()
    
    print("\\n" + "="*50)
    print("🎉 Training Complete!")
    print("="*50)
    print(f"🔍 XGBoost AUC: {xgb_results.get('performance_metrics', {}).get('roc_auc', 'N/A')}")
    print(f"🔍 LightGBM AUC: {lgb_results.get('performance_metrics', {}).get('roc_auc', 'N/A')}")
    print(f"💰 Potential Annual Savings: ${business_insights.get('potential_annual_savings', 0):,.0f}")
    print("\\n🌐 Check MLflow UI: http://localhost:5000")

if __name__ == "__main__":
    main()
