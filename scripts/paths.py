"""
Path configuration for the project.
Defines all file paths to avoid hardcoding and path errors.
"""
from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data Paths
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
DATASET_PATH = RAW_DATA_DIR / "airlines.csv"
PROCESSED_DATASET_PATH = PROCESSED_DATA_DIR / "airline-predict-g4-cleaning.csv"

# Model Paths
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"
METRICS_PATH = MODELS_DIR / "metrics.joblib"

# Ensure directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("í³‚ Project Paths:")
    print(f"   PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"   DATASET_PATH: {DATASET_PATH}")
    print(f"   MODEL_PATH: {MODEL_PATH}")
    print(f"   PREPROCESSOR_PATH: {PREPROCESSOR_PATH}")
