"""
Preprocessing module for training and prediction.
Handles data cleaning, encoding, and feature engineering.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path


# Target column
TARGET = "satisfaction"

# Columns to drop
COLUMNS_TO_DROP = ["Unnamed: 0", "id"]

# Ordinal encoding for Class (Eco < Eco Plus < Business)
CLASS_ORDER = [["Eco", "Eco Plus", "Business"]]


def clean_data(df):
    """
    Cleans the dataset: removes unnecessary columns and handles nulls.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        pd.DataFrame: Cleaned dataset
    """
    df = df.copy()
    
    # Drop unnecessary columns
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    print(f"  ✓ Columns dropped: {COLUMNS_TO_DROP}")
    
    # Handle nulls in 'Arrival Delay in Minutes'
    if 'Arrival Delay in Minutes' in df.columns:
        median_value = df['Arrival Delay in Minutes'].median()
        df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(median_value)
        print(f"  ✓ Nulls filled with median: {median_value}")
    
    # Drop remaining nulls
    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        df = df.dropna()
        print(f"  ✓ Dropped {remaining_nulls} remaining null rows")
    
    print(f"  ✓ Clean dataset shape: {df.shape}")
    return df


def encode_features(df):
    """
    Encodes categorical features for training.
    
    Args:
        df (pd.DataFrame): Cleaned dataset (without target)
        
    Returns:
        tuple: (X_encoded, encoders_dict)
    """
    df = df.copy()
    encoders = {}
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    print(f"  ✓ Categorical columns: {categorical_cols}")
    
    # Encode 'Class' with OrdinalEncoder
    if 'Class' in categorical_cols:
        ord_enc = OrdinalEncoder(categories=CLASS_ORDER)
        df['Class'] = ord_enc.fit_transform(df[['Class']]).astype(int)
        encoders['Class'] = ord_enc
        print(f"    - Class (OrdinalEncoder): Eco=0, Eco Plus=1, Business=2")
        categorical_cols.remove('Class')
    
    # Encode remaining categorical columns with LabelEncoder
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
        print(f"    - {col} (LabelEncoder): {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    return df, encoders


def preprocess_for_training(df):
    """
    Full preprocessing pipeline for training.
    
    Args:
        df (pd.DataFrame): Raw dataset
        
    Returns:
        tuple: (X, y, preprocessor)
            - X: Encoded features
            - y: Encoded target
            - preprocessor: Dictionary with encoders and target encoder
    """
    print("\n��� Starting preprocessing...")
    print("=" * 50)
    
    # Clean data
    df_clean = clean_data(df)
    
    # Separate features and target
    if TARGET not in df_clean.columns:
        raise ValueError(f"Target column '{TARGET}' not found in dataset")
    
    X = df_clean.drop(TARGET, axis=1)
    y = df_clean[TARGET]
    
    # Encode features
    X_encoded, feature_encoders = encode_features(X)
    
    # Encode target
    target_encoder = LabelEncoder()
    y_encoded = target_encoder.fit_transform(y)
    print(f"  ✓ Target encoded: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")
    
    # Create preprocessor object
    preprocessor = {
        'feature_encoders': feature_encoders,
        'target_encoder': target_encoder,
        'feature_columns': list(X_encoded.columns)
    }
    
    print("=" * 50)
    print(f"✅ Preprocessing complete:")
    print(f"   Features (X): {X_encoded.shape}")
    print(f"   Target (y): {len(y_encoded)} values")
    
    return X_encoded, y_encoded, preprocessor


def save_preprocessor(preprocessor, filepath: str):
    """
    Saves the preprocessor to disk.
    
    Args:
        preprocessor (dict): Preprocessor with encoders
        filepath (str): Path to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(preprocessor, filepath)
    print(f"✅ Preprocessor saved to: {filepath}")


def load_preprocessor(filepath: str):
    """
    Loads a saved preprocessor.
    
    Args:
        filepath (str): Path to preprocessor file
        
    Returns:
        dict: Preprocessor with encoders
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
    
    preprocessor = joblib.load(filepath)
    print(f"✅ Preprocessor loaded from: {filepath}")
    return preprocessor


if __name__ == "__main__":
    print("Preprocessing module loaded successfully")
