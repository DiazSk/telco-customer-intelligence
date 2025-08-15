#!/usr/bin/env python3
"""
MLflow Test Script for Telco Customer Intelligence Platform
Demonstrates experiment tracking, model management, and artifact storage

This script will:
1. Set up MLflow tracking
2. Run multiple model experiments
3. Log parameters, metrics, and artifacts
4. Register models in the model registry
5. Demonstrate model serving capabilities
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# ML Libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
import xgboost as xgb
import lightgbm as lgb

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

warnings.filterwarnings('ignore')

class MLflowExperimentRunner:
    """Comprehensive MLflow experiment runner"""
    
    def __init__(self, mlflow_uri: str = "http://localhost:5000"):
        """Initialize MLflow tracking"""
        self.mlflow_uri = mlflow_uri
        self.setup_mlflow()
        self.data_path = "data/processed/processed_telco_data.csv"
        
    def setup_mlflow(self):
        """Configure MLflow tracking"""
        print(f"🔧 Setting up MLflow tracking...")
        print(f"   URI: {self.mlflow_uri}")
        
        try:
            mlflow.set_tracking_uri(self.mlflow_uri)
            # Test connection
            experiments = mlflow.search_experiments()
            print(f"✅ Connected to MLflow! Found {len(experiments)} existing experiments.")
        except Exception as e:
            print(f"⚠️  Warning: Could not connect to MLflow server: {e}")
            print("   Using local file tracking instead...")
            mlflow.set_tracking_uri("file:./mlruns")
    
    def load_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Load and prepare the telco dataset"""
        print(f"📊 Loading data from {self.data_path}...")
        
        if not os.path.exists(self.data_path):
            print(f"❌ Data file not found: {self.data_path}")
            print("   Creating sample data for testing...")
            return self.create_sample_data()
        
        try:
            df = pd.read_csv(self.data_path)
            print(f"✅ Loaded {len(df)} records with {len(df.columns)} features")
            
            # Prepare features and target
            target_col = 'Churn'
            if target_col not in df.columns:
                print(f"⚠️  Target column '{target_col}' not found. Available columns:")
                print(f"   {list(df.columns)}")
                # Try to find churn-like column
                churn_cols = [col for col in df.columns if 'churn' in col.lower()]
                if churn_cols:
                    target_col = churn_cols[0]
                    print(f"   Using '{target_col}' as target column")
                else:
                    return self.create_sample_data()
            
            # Encode target if needed
            y = df[target_col].copy()
            if y.dtype == 'object':
                le = LabelEncoder()
                y = le.fit_transform(y)
                print(f"   Encoded target: {dict(zip(le.classes_, range(len(le.classes_))))}")
            
            # Prepare features
            X = df.drop(columns=[target_col])
            X = self.preprocess_features(X)
            
            return X, y
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return self.create_sample_data()
    
    def create_sample_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Create sample telco-like data for testing"""
        print("🔧 Creating sample telco dataset...")
        
        np.random.seed(42)
        n_samples = 1000
        
        # Create sample features
        data = {
            'tenure': np.random.randint(1, 72, n_samples),
            'MonthlyCharges': np.random.normal(65, 20, n_samples),
            'TotalCharges': np.random.normal(2000, 1000, n_samples),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples),
            'PaymentMethod': np.random.choice(['Electronic check', 'Credit card', 'Bank transfer'], n_samples),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples),
            'gender': np.random.choice(['Male', 'Female'], n_samples),
            'SeniorCitizen': np.random.choice([0, 1], n_samples),
            'Partner': np.random.choice(['Yes', 'No'], n_samples),
            'Dependents': np.random.choice(['Yes', 'No'], n_samples),
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
        
        y = np.random.binomial(1, churn_prob.clip(0, 1), n_samples)
        
        # Preprocess features
        X = self.preprocess_features(df)
        
        print(f"✅ Created sample dataset: {len(X)} samples, {len(X.columns)} features")
        print(f"   Churn rate: {y.mean():.1%}")
        
        return X, y
    
    def preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess features for modeling"""
        X = df.copy()
        
        # Handle numeric columns
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X[numeric_cols] = X[numeric_cols].fillna(X[numeric_cols].median())
        
        # Handle categorical columns
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            X[col] = X[col].fillna('Unknown')
            # One-hot encode
            dummies = pd.get_dummies(X[col], prefix=col, drop_first=True)
            X = pd.concat([X, dummies], axis=1)
            X = X.drop(columns=[col])
        
        return X
    
    def run_experiment(self, experiment_name: str, model_name: str, model, 
                      params: Dict[str, Any], X_train, X_test, y_train, y_test):
        """Run a single MLflow experiment"""
        
        # Set experiment
        mlflow.set_experiment(experiment_name)
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            print(f"🚀 Running experiment: {model_name}")
            
            # Log parameters
            for key, value in params.items():
                mlflow.log_param(key, value)
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("train_size", len(X_train))
            mlflow.log_param("test_size", len(X_test))
            
            # Train model
            start_time = datetime.now()
            model.fit(X_train, y_train)
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'f1_score': f1_score(y_test, y_pred, average='weighted'),
                'training_time_seconds': training_time
            }
            
            if y_pred_proba is not None:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Create and log artifacts
            self.create_artifacts(model, X_test, y_test, y_pred, y_pred_proba)
            
            # Log model with signature
            if hasattr(model, 'predict_proba'):
                signature = infer_signature(X_train, model.predict_proba(X_train))
            else:
                signature = infer_signature(X_train, model.predict(X_train))
            
            if model_name == "XGBoost":
                mlflow.xgboost.log_model(model, "model", signature=signature)
            elif model_name == "LightGBM":
                mlflow.lightgbm.log_model(model, "model", signature=signature)
            else:
                mlflow.sklearn.log_model(model, "model", signature=signature)
            
            # Log tags
            mlflow.set_tag("team", "data-science")
            mlflow.set_tag("project", "telco-churn")
            mlflow.set_tag("stage", "experiment")
            
            print(f"   ✅ Accuracy: {metrics['accuracy']:.3f}")
            if 'roc_auc' in metrics:
                print(f"   ✅ ROC-AUC: {metrics['roc_auc']:.3f}")
            print(f"   ✅ Training time: {training_time:.2f}s")
            
            return mlflow.active_run().info.run_id, metrics
    
    def create_artifacts(self, model, X_test, y_test, y_pred, y_pred_proba):
        """Create and log various artifacts"""
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
        plt.close()
        mlflow.log_artifact('confusion_matrix.png')
        os.remove('confusion_matrix.png')
        
        # 2. Feature Importance (if available)
        if hasattr(model, 'feature_importances_'):
            feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False).head(20)
            
            plt.figure(figsize=(10, 8))
            sns.barplot(data=importance_df, x='importance', y='feature')
            plt.title('Top 20 Feature Importance')
            plt.xlabel('Importance')
            plt.tight_layout()
            plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact('feature_importance.png')
            os.remove('feature_importance.png')
            
            # Save as CSV
            importance_df.to_csv('feature_importance.csv', index=False)
            mlflow.log_artifact('feature_importance.csv')
            os.remove('feature_importance.csv')
        
        # 3. ROC Curve (if probabilities available)
        if y_pred_proba is not None:
            from sklearn.metrics import roc_curve
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            auc = roc_auc_score(y_test, y_pred_proba)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
            plt.plot([0, 1], [0, 1], 'k--', label='Random')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve')
            plt.legend()
            plt.savefig('roc_curve.png', dpi=150, bbox_inches='tight')
            plt.close()
            mlflow.log_artifact('roc_curve.png')
            os.remove('roc_curve.png')
        
        # 4. Classification Report
        report = classification_report(y_test, y_pred, output_dict=True)
        with open('classification_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        mlflow.log_artifact('classification_report.json')
        os.remove('classification_report.json')
    
    def run_all_experiments(self):
        """Run multiple experiments with different models"""
        print("🧪 Starting MLflow Experiments...")
        
        # Load data
        X, y = self.load_data()
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features for some models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        experiments_results = {}
        
        # Experiment 1: Random Forest
        print("\\n" + "="*50)
        print("EXPERIMENT 1: Random Forest")
        print("="*50)
        
        rf_params = {
            'n_estimators': 100,
            'max_depth': 10,
            'min_samples_split': 5,
            'random_state': 42
        }
        rf_model = RandomForestClassifier(**rf_params)
        
        run_id_1, metrics_1 = self.run_experiment(
            "Telco_Churn_Models", "RandomForest", rf_model, 
            rf_params, X_train, X_test, y_train, y_test
        )
        experiments_results['RandomForest'] = {'run_id': run_id_1, 'metrics': metrics_1}
        
        # Experiment 2: XGBoost
        print("\\n" + "="*50)
        print("EXPERIMENT 2: XGBoost")
        print("="*50)
        
        xgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'random_state': 42
        }
        xgb_model = xgb.XGBClassifier(**xgb_params)
        
        run_id_2, metrics_2 = self.run_experiment(
            "Telco_Churn_Models", "XGBoost", xgb_model,
            xgb_params, X_train, X_test, y_train, y_test
        )
        experiments_results['XGBoost'] = {'run_id': run_id_2, 'metrics': metrics_2}
        
        # Experiment 3: LightGBM
        print("\\n" + "="*50)
        print("EXPERIMENT 3: LightGBM")
        print("="*50)
        
        lgb_params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'num_leaves': 31,
            'random_state': 42,
            'verbosity': -1
        }
        lgb_model = lgb.LGBMClassifier(**lgb_params)
        
        run_id_3, metrics_3 = self.run_experiment(
            "Telco_Churn_Models", "LightGBM", lgb_model,
            lgb_params, X_train, X_test, y_train, y_test
        )
        experiments_results['LightGBM'] = {'run_id': run_id_3, 'metrics': metrics_3}
        
        # Experiment 4: Logistic Regression
        print("\\n" + "="*50)
        print("EXPERIMENT 4: Logistic Regression")
        print("="*50)
        
        lr_params = {
            'max_iter': 1000,
            'random_state': 42,
            'solver': 'liblinear'
        }
        lr_model = LogisticRegression(**lr_params)
        
        run_id_4, metrics_4 = self.run_experiment(
            "Telco_Churn_Models", "LogisticRegression", lr_model,
            lr_params, X_train_scaled, X_test_scaled, y_train, y_test
        )
        experiments_results['LogisticRegression'] = {'run_id': run_id_4, 'metrics': metrics_4}
        
        # Hyperparameter Tuning Experiment
        print("\\n" + "="*50)
        print("EXPERIMENT 5: Hyperparameter Tuning")
        print("="*50)
        
        self.run_hyperparameter_tuning(X_train, X_test, y_train, y_test)
        
        return experiments_results
    
    def run_hyperparameter_tuning(self, X_train, X_test, y_train, y_test):
        """Demonstrate hyperparameter tuning with MLflow"""
        
        mlflow.set_experiment("Hyperparameter_Tuning")
        
        # Try different XGBoost configurations
        param_grid = [
            {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1},
            {'n_estimators': 50, 'max_depth': 6, 'learning_rate': 0.1},
            {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
            {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
            {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.05},
        ]
        
        best_auc = 0
        best_run_id = None
        
        for i, params in enumerate(param_grid):
            with mlflow.start_run(run_name=f"XGB_Tuning_{i+1}"):
                print(f"   Trying params: {params}")
                
                # Log parameters
                for key, value in params.items():
                    mlflow.log_param(key, value)
                
                # Train model
                model = xgb.XGBClassifier(**params, random_state=42)
                model.fit(X_train, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                accuracy = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                # Log metrics
                mlflow.log_metric("accuracy", accuracy)
                mlflow.log_metric("roc_auc", auc)
                
                print(f"      AUC: {auc:.3f}, Accuracy: {accuracy:.3f}")
                
                if auc > best_auc:
                    best_auc = auc
                    best_run_id = mlflow.active_run().info.run_id
                    
                    # Log the best model
                    mlflow.xgboost.log_model(model, "model")
        
        print(f"✅ Best hyperparameter tuning result: AUC = {best_auc:.3f}")
        return best_run_id
    
    def register_best_model(self, experiments_results):
        """Register the best model in MLflow Model Registry"""
        print("\\n" + "="*50)
        print("MODEL REGISTRATION")
        print("="*50)
        
        # Find best model by ROC-AUC
        best_model = None
        best_auc = 0
        best_run_id = None
        
        for model_name, result in experiments_results.items():
            metrics = result['metrics']
            if 'roc_auc' in metrics and metrics['roc_auc'] > best_auc:
                best_auc = metrics['roc_auc']
                best_model = model_name
                best_run_id = result['run_id']
        
        if best_run_id:
            print(f"🏆 Best model: {best_model} (AUC: {best_auc:.3f})")
            
            try:
                # Register model
                model_uri = f"runs:/{best_run_id}/model"
                model_name = "telco_churn_predictor"
                
                mv = mlflow.register_model(model_uri, model_name)
                print(f"✅ Registered model: {model_name}")
                print(f"   Version: {mv.version}")
                print(f"   Run ID: {best_run_id}")
                
                # Transition to staging
                from mlflow.tracking import MlflowClient
                client = MlflowClient()
                
                client.transition_model_version_stage(
                    name=model_name,
                    version=mv.version,
                    stage="Staging"
                )
                print(f"✅ Transitioned model to Staging")
                
                return model_name, mv.version
                
            except Exception as e:
                print(f"⚠️  Could not register model: {e}")
                return None, None
        
        return None, None
    
    def demonstrate_model_serving(self, model_name: str, model_version: str):
        """Demonstrate model serving capabilities"""
        print("\\n" + "="*50)
        print("MODEL SERVING DEMO")
        print("="*50)
        
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            # Load model from registry
            model_uri = f"models:/{model_name}/{model_version}"
            model = mlflow.pyfunc.load_model(model_uri)
            
            print(f"✅ Loaded model: {model_name} v{model_version}")
            
            # Create sample prediction data
            sample_data = pd.DataFrame({
                'tenure': [12, 36, 6],
                'MonthlyCharges': [70.0, 85.0, 95.0],
                'TotalCharges': [840.0, 3060.0, 570.0]
            })
            
            # Add dummy columns to match training data shape
            X, _ = self.load_data()
            for col in X.columns:
                if col not in sample_data.columns:
                    sample_data[col] = 0
            
            # Reorder columns to match training data
            sample_data = sample_data[X.columns]
            
            # Make predictions
            predictions = model.predict(sample_data)
            print(f"✅ Sample predictions: {predictions}")
            
            # Save prediction example
            prediction_example = {
                'input': sample_data.iloc[0].to_dict(),
                'output': float(predictions[0]),
                'timestamp': datetime.now().isoformat()
            }
            
            with open('prediction_example.json', 'w') as f:
                json.dump(prediction_example, f, indent=2)
            
            # Log to current run if any
            if mlflow.active_run():
                mlflow.log_artifact('prediction_example.json')
            
            os.remove('prediction_example.json')
            
            return True
            
        except Exception as e:
            print(f"⚠️  Model serving demo failed: {e}")
            return False
    
    def generate_summary_report(self, experiments_results):
        """Generate and log experiment summary"""
        print("\\n" + "="*50)
        print("EXPERIMENT SUMMARY")
        print("="*50)
        
        # Create summary dataframe
        summary_data = []
        for model_name, result in experiments_results.items():
            metrics = result['metrics']
            summary_data.append({
                'Model': model_name,
                'Run_ID': result['run_id'][:8],
                'Accuracy': f"{metrics['accuracy']:.3f}",
                'ROC_AUC': f"{metrics.get('roc_auc', 'N/A')}",
                'F1_Score': f"{metrics['f1_score']:.3f}",
                'Training_Time': f"{metrics['training_time_seconds']:.2f}s"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        # Save summary
        summary_df.to_csv('experiment_summary.csv', index=False)
        
        # Create a final summary run
        mlflow.set_experiment("Experiment_Summary")
        with mlflow.start_run(run_name=f"Summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_artifact('experiment_summary.csv')
            
            # Log best metrics
            best_accuracy = max([r['metrics']['accuracy'] for r in experiments_results.values()])
            best_auc = max([r['metrics'].get('roc_auc', 0) for r in experiments_results.values()])
            
            mlflow.log_metric('best_accuracy', best_accuracy)
            mlflow.log_metric('best_roc_auc', best_auc)
            mlflow.log_param('total_experiments', len(experiments_results))
            
            # Tags
            mlflow.set_tag('experiment_type', 'model_comparison')
            mlflow.set_tag('dataset', 'telco_churn')
            
        os.remove('experiment_summary.csv')
        
        return summary_df

def main():
    """Main execution function"""
    print("🎯 MLflow Comprehensive Test Script")
    print("=" * 60)
    
    # Initialize experiment runner
    runner = MLflowExperimentRunner()
    
    # Run all experiments
    results = runner.run_all_experiments()
    
    # Register best model
    model_name, model_version = runner.register_best_model(results)
    
    # Demonstrate model serving
    if model_name and model_version:
        runner.demonstrate_model_serving(model_name, model_version)
    
    # Generate summary
    summary = runner.generate_summary_report(results)
    
    print("\\n" + "="*60)
    print("🎉 MLflow Test Complete!")
    print("="*60)
    print(f"📊 Total experiments run: {len(results)}")
    print(f"🏆 Best model registered: {model_name} v{model_version}" if model_name else "❌ No model registered")
    print("\\n🌐 Next steps:")
    print("1. Open MLflow UI: http://localhost:5000")
    print("2. Explore experiments and compare models")
    print("3. Check the model registry")
    print("4. Review artifacts and visualizations")
    print("\\n📁 Check your mlruns/ folder - it should now contain experiment data!")

if __name__ == "__main__":
    main()
