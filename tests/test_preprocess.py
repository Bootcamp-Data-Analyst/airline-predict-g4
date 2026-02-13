import pytest
import pandas as pd
import numpy as np
import os
import sys
from sklearn.compose import ColumnTransformer

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.preprocess import (
    clean_data, 
    create_preprocessing_pipeline, 
    encode_target, 
    decode_target,
    preprocess_for_training,
    preprocess_for_prediction,
    COLUMNS_TO_DROP
)

class TestPreprocess:
    
    @pytest.fixture
    def sample_raw_data(self):
        return pd.DataFrame({
            'Unnamed: 0': [0, 1],
            'id': [100, 101],
            'Gender': ['Male', 'Female'],
            'Customer Type': ['Loyal Customer', 'disloyal Customer'],
            'Age': [30, 25],
            'Type of Travel': ['Business travel', 'Personal Travel'],
            'Class': ['Business', 'Eco'],
            'Flight Distance': [1000, 500],
            'Inflight wifi service': [5, 3],
            'Departure/Arrival time convenient': [4, 4],
            'Ease of Online booking': [3, 3],
            'Gate location': [2, 2],
            'Food and drink': [5, 4],
            'Online boarding': [5, 3],
            'Seat comfort': [5, 4],
            'Inflight entertainment': [5, 4],
            'On-board service': [5, 4],
            'Leg room service': [5, 4],
            'Baggage handling': [5, 4],
            'Checkin service': [5, 4],
            'Inflight service': [5, 4],
            'Cleanliness': [5, 4],
            'Departure Delay in Minutes': [0, 10],
            'Arrival Delay in Minutes': [0, 5],
            'satisfaction': ['satisfied', 'neutral or dissatisfied']
        })

    def test_clean_data(self, sample_raw_data):
        """Test that unwanted columns are dropped."""
        df_clean = clean_data(sample_raw_data)
        
        for col in COLUMNS_TO_DROP:
            assert col not in df_clean.columns
            
        assert 'Gender' in df_clean.columns

    def test_clean_data_duplicates(self, sample_raw_data):
        """Test validation of duplicate removal."""
        # Create duplicate row
        df_dup = pd.concat([sample_raw_data, sample_raw_data.iloc[[0]]], ignore_index=True)
        assert len(df_dup) == 3
        
        df_clean = clean_data(df_dup)
        # Should be 2 (original length) because cleanup handles id/Unnamed drop first
        # Note: clean_data logic drops columns THEN drops duplicates. 
        # If rows are identical after dropping columns, they are dups.
        assert len(df_clean) == 2

    def test_create_preprocessing_pipeline(self):
        """Test that the pipeline is created correctly."""
        pipeline = create_preprocessing_pipeline()
        assert isinstance(pipeline, ColumnTransformer)
        
        # Check transformers exist
        transformers = [t[0] for t in pipeline.transformers]
        assert 'num' in transformers
        assert 'cat' in transformers

    def test_target_encoding(self):
        """Test target encoding and decoding logic."""
        y = pd.Series(['satisfied', 'neutral or dissatisfied', 'satisfied'])
        y_encoded = encode_target(y)
        
        expected = np.array([1, 0, 1])
        np.testing.assert_array_equal(y_encoded, expected)
        
        y_decoded = decode_target(y_encoded)
        assert y_decoded == ['satisfied', 'neutral or dissatisfied', 'satisfied']

    def test_preprocess_for_training(self, sample_raw_data):
        """Test the full training preprocessing flow."""
        X, y, preprocessor = preprocess_for_training(sample_raw_data)
        
        assert len(X) == 2
        assert len(y) == 2
        # Check that preprocessor is fitted
        assert hasattr(preprocessor, 'transformers_')
        # Target should be numeric
        assert y.dtype in [np.int32, np.int64]

    def test_preprocess_for_prediction(self, sample_raw_data):
        """Test prediction preprocessing with single dict input."""
        # Train first to get fitted preprocessor
        _, _, preprocessor = preprocess_for_training(sample_raw_data)
        
        # Create single sample (dict)
        sample = sample_raw_data.iloc[0].to_dict()
        # Remove target and dropped columns to simulate real inference input
        for k in ['satisfaction', 'Unnamed: 0', 'id']:
            if k in sample: del sample[k]
            
        X_pred = preprocess_for_prediction(sample, preprocessor)
        
        assert X_pred.shape[0] == 1
        assert X_pred.shape[1] > 0
