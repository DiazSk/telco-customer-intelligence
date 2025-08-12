"""
Secure configuration loader
Separates sensitive and non-sensitive configuration
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class ConfigLoader:
    """Securely load configuration from yaml and environment variables"""
    
    def __init__(self, config_path: str = "configs/pipeline_config.yaml"):
        self.config_path = config_path
        self.config = self.load_config()
        self.inject_secrets()
    
    def load_config(self) -> Dict[str, Any]:
        """Load non-sensitive configuration from YAML"""
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def inject_secrets(self):
        """Inject sensitive data from environment variables"""
        # Only add database config if enabled
        if self.config.get('database', {}).get('enabled', False):
            db_url = os.getenv('DATABASE_URL')
            if db_url:
                self.config['database']['connection_string'] = db_url
            else:
                print("Warning: DATABASE_URL not set, using local files only")
                self.config['database']['enabled'] = False
        
        # Add MLflow tracking URI if set
        mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
        self.config['mlflow'] = {'tracking_uri': mlflow_uri}
    
    def get(self, key: str, default=None):
        """Get configuration value"""
        return self.config.get(key, default)
    
    def get_nested(self, *keys, default=None):
        """Get nested configuration value"""
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
        return value if value is not None else default


# Singleton instance
config = ConfigLoader()


def get_data_paths():
    """Get all data paths from configuration"""
    return {
        'raw': config.get_nested('data', 'raw_data_path'),
        'processed': config.get_nested('data', 'processed_data_path'),
        'features': config.get_nested('data', 'feature_store_path')
    }


def get_database_config():
    """Get database configuration (only if enabled)"""
    if config.get_nested('database', 'enabled'):
        return {
            'connection_string': config.get_nested('database', 'connection_string'),
            'enabled': True
        }
    return {'enabled': False}


def is_production():
    """Check if running in production environment"""
    return os.getenv('ENV', 'development').lower() == 'production'