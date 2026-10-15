import os
import pyjson5

class ConfigManager:
    _instance = None
    _data = None

    def __new__(cls):
        """Ensures only one instance of ConfigManager exists."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance

    def _load_config(self):
        # config.json5 lives next to this file, so the config is found from any working directory
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.json5')
        with open(config_path, 'r') as f:
            self._data = pyjson5.load(f)

    def get(self, key, default=None):
        return self._data.get(key, default)

# Create the instance immediately
global_config = ConfigManager()
