# scripts/paths.py

"""
Path configuration for the project.
Defines all file paths to avoid hardcoding and path errors.
All scripts import from here so paths are consistent across the project.
"""
from pathlib import Path
import os

# Project Root (one level up from scripts/)

PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data Pathspython

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

# Neural Network (optional - Day 5)

NN_MODEL_PATH = MODELS_DIR / "model_nn.keras"

# Database

DB_PATH = PROCESSED_DATA_DIR / "airline_monitoring.db"

# Logs

LOGS_DIR = PROJECT_ROOT / "logs"

# Reports

REPORTS_DIR = PROJECT_ROOT / "reports"

# Ensure critical directories exist

for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, LOGS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    print("Project Paths Configuration")
print("=" * 50)
print(f"   PROJECT_ROOT:      {PROJECT_ROOT}")
print(f"   DATASET_PATH:      {DATASET_PATH}")
print(f"   MODEL_PATH:        {MODEL_PATH}")
print(f"   PREPROCESSOR_PATH: {PREPROCESSOR_PATH}")
print(f"   METRICS_PATH:      {METRICS_PATH}")
print(f"   DB_PATH:           {DB_PATH}")
print(f"\n   Dataset exists: {DATASET_PATH.exists()}")