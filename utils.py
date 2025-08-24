# utils.py
import yaml
from typing import Any, Dict

class NestedDict:
    """Convert nested dict to object with dot notation access"""
    def __init__(self, data: Dict):
        for key, value in data.items():
            if isinstance(value, dict):
                setattr(self, key, NestedDict(value))
            else:
                setattr(self, key, self._convert_value(value))
    
    def _convert_value(self, value):
        """Convert string numbers to appropriate types"""
        if isinstance(value, str):
            # Try to convert scientific notation strings to float
            try:
                if 'e' in value.lower() or '.' in value:
                    return float(value)
                elif value.isdigit():
                    return int(value)
            except ValueError:
                pass
        return value
    
    def __getitem__(self, key):
        """Enable subscript access like dict[key]"""
        return getattr(self, key)
    
    def __setitem__(self, key, value):
        """Enable subscript assignment like dict[key] = value"""
        setattr(self, key, value)
    
    def __contains__(self, key):
        """Enable 'in' operator"""
        return hasattr(self, key)
    
    def get(self, key, default=None):
        """Dict-like get method"""
        return getattr(self, key, default)
    
    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items()}
        return f"NestedDict({attrs})"

class Configuration:
    """Simple configuration class with nested YAML support and type conversion"""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self._load_config()
    
    def _load_config(self):
        """Load YAML config and set nested attributes"""
        try:
            with open(self.config_path, 'r') as f:
                config_data = yaml.safe_load(f) or {}
            
            # Set all config keys as attributes (with nested support)
            for key, value in config_data.items():
                if isinstance(value, dict):
                    setattr(self, key, NestedDict(value))
                else:
                    setattr(self, key, self._convert_value(value))
            
            print(f"✓ Loaded config from {self.config_path}")
            
        except FileNotFoundError:
            print(f"❌ Config file not found: {self.config_path}")
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML: {e}")
    
    def _convert_value(self, value):
        """Convert string numbers to appropriate types"""
        if isinstance(value, str):
            # Try to convert scientific notation strings to float
            try:
                if 'e' in value.lower() or '.' in value:
                    return float(value)
                elif value.isdigit():
                    return int(value)
            except ValueError:
                pass
        return value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get attribute with default value (supports dot notation)"""
        if '.' in key:
            # Handle nested access like 'qos.embb.min_rate'
            keys = key.split('.')
            obj = self
            for k in keys:
                if hasattr(obj, k):
                    obj = getattr(obj, k)
                else:
                    return default
            return obj
        else:
            return getattr(self, key, default)
    
    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith('_')}
        return f"Configuration({attrs})"


if __name__ == "__main__":
    # Example usage
    config = Configuration('config/environment/default.yaml')

    
    # Access nested attributes
    num_uavs = config.slicing.slice_weights
    print(num_uavs)
    
    # Pretty print the config
    # print(config)
    
    # Convert to dict
    # config_dict = config.__dict__
    # print("Config as dict:", config_dict)
    
    # Check if a key exists
    if hasattr(config, 'system'):
        print("UAVs configured")