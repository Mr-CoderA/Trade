"""Configuration module for trading system."""

import os
import yaml
from pathlib import Path
from typing import Dict, Any

CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIR.parent


def load_config() -> Dict[str, Any]:
    """Load configuration from YAML file."""
    config_path = CONFIG_DIR / "settings.yaml"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


class Config:
    """Configuration class for trading system."""
    
    _instance = None
    _config = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
            cls._config = load_config()
        return cls._instance
    
    def __getitem__(self, key: str) -> Any:
        """Get configuration value by key."""
        return self._config.get(key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value with default."""
        return self._config.get(key, default)
    
    def get_nested(self, path: str, default: Any = None) -> Any:
        """Get nested configuration value using dot notation (e.g., 'model.type')."""
        keys = path.split('.')
        value = self._config
        
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        
        return value
    
    def __repr__(self) -> str:
        return f"Config({self._config})"


# Create global config instance
config = Config()
