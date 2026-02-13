"""
Preprocessing pipeline for the Airlines Dataset.
Handles data cleaning, transformation, and feature engineering.
"""

import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import joblib
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from scripts.paths import PREPROCESSOR_PATH
except ImportError:
    from paths import PREPROCESSOR_PATH

# Configuration
COLUMNS_TO_DROP = ['Unnamed: 0', 'id']
TARGET_COLUMN = 'satisfaction'

NUMERIC_FEATURES = [
    'Age', 'Flight Distance', 
    'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink',
    'Online boarding', 'Seat comfort', 'Inflight entertainment',
    'On-board service', 'Leg room service', 'Baggage handling',
    'Checkin service', 'Inflight service', 'Cleanliness',
    'Departure Delay in Minutes', 'Arrival Delay in Minutes'
]

CATEGORICAL_FEATURES = [
    'Gender', 'Customer Type', 'Type of Travel', 'Class'
]


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the DataFrame by removing unnecessary columns and duplicates.
    
    Args:
        df (pd.DataFrame): Raw dataframe
        
    Returns:
        pd.DataFrame: Cleaned dataframe
    """
    df_clean = df.copy()
    
    # Drop columns
    cols_to_drop = [col for col in COLUMNS_TO_DROP if col in df_clean.columns]
    if cols_to_drop:
        df_clean = df_clean.drop(columns=cols_to_drop)
        print(f"✅ Dropped columns: {cols_to_drop}")
    
    # Drop duplicates
    before = len(df_clean)
    df_clean = df_clean.drop_duplicates()
    n_duplicates = before - len(df_clean)
    if n_duplicates > 0:
        print(f"✅ Dropped duplicates: {n_duplicates}")
    
    return df_clean


def create_preprocessing_pipeline() -> ColumnTransformer:
    """
    Creates the sklearn preprocessing pipeline.
    
    Returns:
        ColumnTransformer: Configured preprocessor
    """
    # Numeric Pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Categorical Pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    # Combine Transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )
    
    return preprocessor


def encode_target(y: pd.Series) -> np.ndarray:
    """
    Encodes target variable to numeric.
    
    Args:
        y (pd.Series): Target series
        
    Returns:
        np.ndarray: Encoded target (1=satisfied, 0=neutral or dissatisfied)
    """
    mapping = {'satisfied': 1, 'neutral or dissatisfied': 0}
    return y.map(mapping).values


def decode_target(y: np.ndarray) -> list:
    """
    Decodes target numeric values to original labels.
    
    Args:
        y (np.ndarray): Encoded target values
        
    Returns:
        list: Decoded labels
    """
    mapping = {1: 'satisfied', 0: 'neutral or dissatisfied'}
    return [mapping.get(val, "unknown") for val in y]


def preprocess_for_training(df: pd.DataFrame, preprocessor: ColumnTransformer = None):
    """
    Preprocesses data for training (fit_transform).
    
    Args:
        df (pd.DataFrame): Training data
        preprocessor (ColumnTransformer): Optional existing preprocessor
        
    Returns:
        tuple: (X_transformed, y_encoded, preprocessor)
    """
    df_clean = clean_data(df)
    
    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN]
    
    if preprocessor is None:
        preprocessor = create_preprocessing_pipeline()
    
    X_transformed = preprocessor.fit_transform(X)
    y_encoded = encode_target(y)
    
    print(f"✅ Preprocessing complete. X shape: {X_transformed.shape}")
    
    return X_transformed, y_encoded, preprocessor


def preprocess_for_prediction(df: pd.DataFrame, preprocessor: ColumnTransformer) -> np.ndarray:
    """
    Preprocesses data for prediction (transform only).
    
    Args:
        df (pd.DataFrame): Input data
        preprocessor (ColumnTransformer): Fitted preprocessor
        
    Returns:
        np.ndarray: Transformed data
    """
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    
    # Ensure correct columns exist before transform (basic check)
    # Note: ColumnTransformer handles cleaning implicitly if configured, 
    # but clean_data is manually called in training. For prediction we pass raw input.
    # To correspond exactly, we should apply clean_data logic if needed, 
    # but input dict usually doesn't have ID columns.
    
    return preprocessor.transform(df)


def save_preprocessor(preprocessor: ColumnTransformer, filepath: str = str(PREPROCESSOR_PATH)):
    """Saves the preprocessor artifact."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessor, filepath)
    print(f"✅ Preprocessor saved to: {filepath}")


def load_preprocessor(filepath: str = str(PREPROCESSOR_PATH)) -> ColumnTransformer:
    """Loads the preprocessor artifact."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocessor not found at: {filepath}")
    return joblib.load(filepath)


if __name__ == "__main__":
    print("🧪 Testing preprocessing pipeline...")
    # Placeholder for unit testing logic
