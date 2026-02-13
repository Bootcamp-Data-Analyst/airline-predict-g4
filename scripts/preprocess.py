import os
import sys
import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from scripts.paths import PREPROCESSOR_PATH
except ImportError:
    from paths import PREPROCESSOR_PATH

TARGET_COLUMN = 'satisfaction'
COLS_TO_DROP = ['Unnamed: 0', 'id']

NUMERIC_FEATURES = [
    'Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
    'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding',
    'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
    'Departure Delay in Minutes', 'Arrival Delay in Minutes'
]

CATEGORICAL_FEATURES = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    # Drop unnecessary columns
    existing_cols = [c for c in COLS_TO_DROP if c in df.columns]
    if existing_cols:
        df.drop(columns=existing_cols, inplace=True)
    
    # Remove duplicates
    df.drop_duplicates(inplace=True)
    
    return df

def create_preprocessing_pipeline() -> ColumnTransformer:
    numeric_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ],
        remainder='drop'
    )

def encode_target(y: pd.Series) -> np.ndarray:
    return y.map({'satisfied': 1, 'neutral or dissatisfied': 0}).values

def decode_target(y: np.ndarray) -> list:
    mapping = {1: 'satisfied', 0: 'neutral or dissatisfied'}
    return [mapping.get(val, "unknown") for val in y]

def load_data():
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')
    return pd.read_csv(os.path.join(data_dir, 'airlines.csv'))

def preprocess_for_training(df: pd.DataFrame, preprocessor: ColumnTransformer = None):
    df_clean = clean_data(df)
    
    X = df_clean.drop(columns=[TARGET_COLUMN])
    y = df_clean[TARGET_COLUMN]
    
    if preprocessor is None:
        preprocessor = create_preprocessing_pipeline()
    
    X_transformed = preprocessor.fit_transform(X)
    y_encoded = encode_target(y)
    
    return X_transformed, y_encoded, preprocessor

def preprocess_for_prediction(df: pd.DataFrame, preprocessor: ColumnTransformer) -> np.ndarray:
    if isinstance(df, dict):
        df = pd.DataFrame([df])
    return preprocessor.transform(df)

def save_preprocessor(preprocessor, filepath=str(PREPROCESSOR_PATH)):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(preprocessor, filepath)

def load_preprocessor(filepath=str(PREPROCESSOR_PATH)):
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Preprocessor artifact not found: {filepath}")
    return joblib.load(filepath)

if __name__ == "__main__":
    # Quick test logic
    df = load_data()
    X, y, preprocessor = preprocess_for_training(df)
    print(f"Data processed. X shape: {X.shape}")
