"""
Model utilities for saving and loading models.
"""
import joblib
from pathlib import Path


def save_model(model, filepath: str):
    """
    Saves a trained model to disk.
    
    Args:
        model: Trained sklearn model
        filepath (str): Path where to save the model
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, filepath)
    print(f"✅ Model saved to: {filepath}")


def load_model(filepath: str):
    """
    Loads a trained model from disk.
    
    Args:
        filepath (str): Path to the saved model
        
    Returns:
        Loaded model
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")
    
    model = joblib.load(filepath)
    print(f"✅ Model loaded from: {filepath}")
    return model


if __name__ == "__main__":
    print("Model utilities module loaded successfully")
