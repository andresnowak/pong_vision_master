from pathlib import Path
import time
import sys
import json


def create_run_directory(base_path):
    """Create a unique run directory with timestamp and return its path."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    run_path = Path(base_path) / run_id
    run_path.mkdir(parents=True, exist_ok=True)
    return run_path

def save_run_metadata(run_path, config, results, device):
    """Save configuration and results to a metadata file."""
    metadata = {
        'config': config,
        'results': results,
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'system': {
            'device': device,
            'python_version': sys.version,
        }
    }
    
    with open(run_path / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)