import joblib
import yaml
from pathlib import Path


def save_model(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)


def load_model(path):
    return joblib.load(path)


def load_config(path='config.yaml'):
    with open(path, 'r') as f:
        return yaml.safe_load(f)
