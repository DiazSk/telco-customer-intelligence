#!/usr/bin/env python3
"""
MLflow Utilities Script
Provides helper functions for MLflow operations
"""

import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
from datetime import datetime
import json

class MLflowManager:
    """Utility class for MLflow operations"""
    
    def __init__(self, tracking_uri: str = "http://localhost:5000"):
        self.tracking_uri = tracking_uri
        mlflow.set_tracking_uri(tracking_uri)
        self.client = MlflowClient()
    
    def list_experiments(self):
        """List all experiments"""
        experiments = self.client.search_experiments()
        
        print("📊 Available Experiments:")
        print("-" * 50)
        for exp in experiments:
            print(f"ID: {exp.experiment_id}")
            print(f"Name: {exp.name}")
            print(f"Lifecycle Stage: {exp.lifecycle_stage}")
            print(f"Artifact Location: {exp.artifact_location}")
            print("-" * 50)
        
        return experiments
    
    def list_runs(self, experiment_name: str = None, max_results: int = 10):
        """List runs in an experiment"""
        if experiment_name:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if not experiment:
                print(f"❌ Experiment '{experiment_name}' not found")
                return []
            experiment_id = experiment.experiment_id
        else:
            experiment_id = None
        
        runs = self.client.search_runs(
            experiment_ids=[experiment_id] if experiment_id else None,
            max_results=max_results,
            order_by=["start_time DESC"]
        )
        
        print(f"🏃 Recent Runs{f' in {experiment_name}' if experiment_name else ''}:")
        print("-" * 80)
        
        for run in runs:
            print(f"Run ID: {run.info.run_id}")
            print(f"Name: {run.data.tags.get('mlflow.runName', 'Unnamed')}")
            print(f"Status: {run.info.status}")
            print(f"Start Time: {datetime.fromtimestamp(run.info.start_time/1000)}")
            
            # Show key metrics
            if run.data.metrics:
                metrics_str = ", ".join([f"{k}: {v:.3f}" for k, v in list(run.data.metrics.items())[:3]])
                print(f"Key Metrics: {metrics_str}")
            
            print("-" * 80)
        
        return runs
    
    def compare_runs(self, run_ids: list):
        """Compare multiple runs"""
        runs_data = []
        
        for run_id in run_ids:
            try:
                run = self.client.get_run(run_id)
                run_data = {
                    'run_id': run_id[:8],
                    'name': run.data.tags.get('mlflow.runName', 'Unnamed'),
                    'status': run.info.status
                }
                
                # Add metrics
                run_data.update(run.data.metrics)
                
                # Add key parameters
                for param_name in ['model_type', 'n_estimators', 'max_depth', 'learning_rate']:
                    if param_name in run.data.params:
                        run_data[param_name] = run.data.params[param_name]
                
                runs_data.append(run_data)
                
            except Exception as e:
                print(f"⚠️ Could not fetch run {run_id}: {e}")
        
        if runs_data:
            df = pd.DataFrame(runs_data)
            print("📊 Run Comparison:")
            print("=" * 80)
            print(df.to_string(index=False))
            return df
        else:
            print("❌ No valid runs to compare")
            return None
    
    def list_registered_models(self):
        """List all registered models"""
        try:
            models = self.client.search_registered_models()
            
            print("📦 Registered Models:")
            print("-" * 50)
            
            for model in models:
                print(f"Name: {model.name}")
                print(f"Description: {model.description or 'No description'}")
                
                # List versions
                versions = self.client.get_latest_versions(model.name)
                for version in versions:
                    print(f"  Version {version.version} ({version.current_stage})")
                
                print("-" * 50)
            
            return models
            
        except Exception as e:
            print(f"⚠️ Could not fetch registered models: {e}")
            return []
    
    def get_model_info(self, model_name: str, version: str = None):
        """Get detailed model information"""
        try:
            if version:
                model_version = self.client.get_model_version(model_name, version)
                print(f"📦 Model: {model_name} v{version}")
                print(f"Stage: {model_version.current_stage}")
                print(f"Source Run: {model_version.run_id}")
                print(f"Creation Time: {datetime.fromtimestamp(model_version.creation_timestamp/1000)}")
                
                # Get run details
                run = self.client.get_run(model_version.run_id)
                print(f"\\nRun Metrics:")
                for key, value in run.data.metrics.items():
                    print(f"  {key}: {value}")
                
                return model_version
            else:
                latest_versions = self.client.get_latest_versions(model_name)
                for version in latest_versions:
                    self.get_model_info(model_name, version.version)
                return latest_versions
                
        except Exception as e:
            print(f"❌ Model not found: {e}")
            return None
    
    def promote_model(self, model_name: str, version: str, stage: str):
        """Promote model to different stage"""
        try:
            self.client.transition_model_version_stage(
                name=model_name,
                version=version,
                stage=stage
            )
            print(f"✅ Promoted {model_name} v{version} to {stage}")
        except Exception as e:
            print(f"❌ Could not promote model: {e}")
    
    def delete_experiment(self, experiment_name: str):
        """Delete an experiment"""
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if experiment:
                self.client.delete_experiment(experiment.experiment_id)
                print(f"✅ Deleted experiment: {experiment_name}")
            else:
                print(f"❌ Experiment not found: {experiment_name}")
        except Exception as e:
            print(f"❌ Could not delete experiment: {e}")
    
    def export_experiment(self, experiment_name: str, output_file: str = None):
        """Export experiment data to CSV"""
        try:
            experiment = self.client.get_experiment_by_name(experiment_name)
            if not experiment:
                print(f"❌ Experiment '{experiment_name}' not found")
                return
            
            runs = self.client.search_runs([experiment.experiment_id])
            
            export_data = []
            for run in runs:
                run_data = {
                    'run_id': run.info.run_id,
                    'name': run.data.tags.get('mlflow.runName', 'Unnamed'),
                    'status': run.info.status,
                    'start_time': datetime.fromtimestamp(run.info.start_time/1000),
                    'end_time': datetime.fromtimestamp(run.info.end_time/1000) if run.info.end_time else None
                }
                
                # Add metrics and parameters
                run_data.update(run.data.metrics)
                run_data.update({f"param_{k}": v for k, v in run.data.params.items()})
                
                export_data.append(run_data)
            
            df = pd.DataFrame(export_data)
            
            if output_file is None:
                output_file = f"{experiment_name}_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            df.to_csv(output_file, index=False)
            print(f"✅ Exported {len(export_data)} runs to {output_file}")
            
            return df
            
        except Exception as e:
            print(f"❌ Could not export experiment: {e}")
            return None

def main():
    """CLI interface for MLflow utilities"""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python mlflow_utils.py <command> [args]")
        print("Commands:")
        print("  list-experiments")
        print("  list-runs [experiment_name]")
        print("  list-models")
        print("  model-info <model_name> [version]")
        print("  promote <model_name> <version> <stage>")
        print("  compare <run_id1> <run_id2> [run_id3] ...")
        print("  export <experiment_name> [output_file]")
        return
    
    manager = MLflowManager()
    command = sys.argv[1]
    
    if command == "list-experiments":
        manager.list_experiments()
    
    elif command == "list-runs":
        experiment_name = sys.argv[2] if len(sys.argv) > 2 else None
        manager.list_runs(experiment_name)
    
    elif command == "list-models":
        manager.list_registered_models()
    
    elif command == "model-info":
        if len(sys.argv) < 3:
            print("❌ Model name required")
            return
        model_name = sys.argv[2]
        version = sys.argv[3] if len(sys.argv) > 3 else None
        manager.get_model_info(model_name, version)
    
    elif command == "promote":
        if len(sys.argv) < 5:
            print("❌ Usage: promote <model_name> <version> <stage>")
            return
        model_name, version, stage = sys.argv[2:5]
        manager.promote_model(model_name, version, stage)
    
    elif command == "compare":
        if len(sys.argv) < 4:
            print("❌ At least 2 run IDs required")
            return
        run_ids = sys.argv[2:]
        manager.compare_runs(run_ids)
    
    elif command == "export":
        if len(sys.argv) < 3:
            print("❌ Experiment name required")
            return
        experiment_name = sys.argv[2]
        output_file = sys.argv[3] if len(sys.argv) > 3 else None
        manager.export_experiment(experiment_name, output_file)
    
    else:
        print(f"❌ Unknown command: {command}")

if __name__ == "__main__":
    main()
