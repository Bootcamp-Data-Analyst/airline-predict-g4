# scripts/model_utils.py

"""
Model utilities for saving and loading models and encoders.
Used by train_model.py to save artifacts and by predict.py/app.py to load them.
"""
import joblib
from pathlib import Path
import os

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from scripts.paths import MODEL_PATH, PREPROCESSOR_PATH
except ImportError:
    from paths import MODEL_PATH, PREPROCESSOR_PATH


def save_model(model, filepath: str = None):
    """
    Saves a trained model to disk using joblib.

    Args:
        model: Trained sklearn model.
        filepath (str): Path where to save. Defaults to MODEL_PATH from paths.py.
    """
    if filepath is None:
        filepath = str(MODEL_PATH)
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, filepath)
    print(f"   Model saved to: {filepath}")


def load_model(filepath: str = None):
    """
    Loads a trained model from disk.

    Args:
        filepath (str): Path to the model file. Defaults to MODEL_PATH.

    Returns:
        Loaded sklearn model ready for prediction.
    """
    if filepath is None:
        filepath = str(MODEL_PATH)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"   Model not found at: {filepath}")
    model = joblib.load(filepath)
    print(f"   Model loaded from: {filepath}")
    return model


def save_encoders(encoders, filepath: str = None):
    """
    Saves the preprocessing encoders (LabelEncoder + OrdinalEncoder).
    Needed for preprocess_single_input() to work in production.

    Args:
        encoders (dict): Dictionary with all encoders from training.
        filepath (str): Path where to save. Defaults to PREPROCESSOR_PATH.
    """
    if filepath is None:
        filepath = str(PREPROCESSOR_PATH)
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(encoders, filepath)
    print(f"   Encoders saved to: {filepath}")


def load_encoders(filepath: str = None):
    """
    Loads the preprocessing encoders from disk.

    Args:
        filepath (str): Path to the encoders file. Defaults to PREPROCESSOR_PATH.

    Returns:
        dict: Dictionary with all encoders.
    """
    if filepath is None:
        filepath = str(PREPROCESSOR_PATH)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"   Encoders not found at: {filepath}")
    encoders = joblib.load(filepath)
    print(f"   Encoders loaded from: {filepath}")
    return encoders


if __name__ == "__main__":
    print("Model Utilities Module")
    print("=" * 50)
    print(f"   Default model path: {MODEL_PATH}")
    print(f"   Default encoders path: {PREPROCESSOR_PATH}")
    print(f"   Model exists: {os.path.exists(MODEL_PATH)}")
    print(f"   Encoders exist: {os.path.exists(PREPROCESSOR_PATH)}")