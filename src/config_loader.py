import yaml

def load_config(config_path):
    """Loads the YAML config file."""
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"Configuration loaded from {config_path}")
        return config
    except FileNotFoundError:
        print(f"Error: Configuration file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        exit(1)