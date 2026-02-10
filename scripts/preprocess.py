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


def load_data():
    """Load the raw airlines dataset."""
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    csv_path = os.path.join(data_dir, 'airlines.csv')
    return pd.read_csv(csv_path)


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
    
    return preprocessor.transform(df)


def preprocess_single_input(sample: dict, encoders) -> pd.DataFrame:
    """
    Preprocesses a single input sample for prediction.
    
    Args:
        sample (dict): Single input sample
        encoders: Fitted encoders/preprocessor
        
    Returns:
        pd.DataFrame: Preprocessed sample
    """
    return preprocess_for_prediction(sample, encoders)


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


def decode_prediction(prediction, encoders):
    """
    Converts numeric prediction (0/1) back to readable text.
    Args:
        prediction: Array with predictions (0 or 1).
        encoders (dict): Encoders from training.
    Returns:
        Array with text: "satisfied" or "neutral or dissatisfied".
    """
    if hasattr(encoders, 'named_transformers_'):
        # It's a ColumnTransformer, decode using decode_target
        return decode_target(prediction)
    return prediction


if __name__ == "__main__":
    print("=" * 60)
    print("PREPROCESSING MODULE TEST")
    print("=" * 60)
    # Test 1: Full dataset preprocessing
    print("\n--- Test 1: Full dataset ---")
    df = load_data()
    X, y, encoders = preprocess_for_training(df)
    print(f"Features shape: {X.shape}")
    print(f"Target values: {np.unique(y)}")
    # Test 2: Save and load preprocessor
    print("\n--- Test 2: Save/Load preprocessor ---")
    save_preprocessor(encoders)
    loaded_encoders = load_preprocessor()
    print(f"Encoders keys: {list(loaded_encoders.named_transformers_.keys())}")
    # Test 3: Single input (simulating Streamlit form)
    print("\n--- Test 3: Single input ---")
    sample = {
        'Gender': 'Male', 'Customer Type': 'Loyal Customer', 'Age': 35,
        'Type of Travel': 'Business travel', 'Class': 'Business',
        'Flight Distance': 1500, 'Inflight wifi service': 4,
        'Departure/Arrival time convenient': 3, 'Ease of Online booking': 4,
        'Gate location': 3, 'Food and drink': 4, 'Online boarding': 5,
        'Seat comfort': 4, 'Inflight entertainment': 5, 'On-board service': 4,
        'Leg room service': 4, 'Baggage handling': 4, 'Checkin service': 3,
        'Inflight service': 4, 'Cleanliness': 4,
        'Departure Delay in Minutes': 10, 'Arrival Delay in Minutes': 5
    }
    X_single = preprocess_single_input(sample, encoders)
    print(f"Processed shape: {X_single.shape}")
    print(f"Class value: {X_single[0, 4]}")
    print("\n" + "=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)