import pytest
import pandas as pd
import numpy as np
import os
import sys
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.predict import predict_with_probability

class TestPredict:

    @pytest.fixture
    def mock_model(self):
        model = MagicMock()
        # Mock predict classes (0 or 1)
        model.predict.return_value = np.array([1]) 
        # Mock predict_proba (prob_0, prob_1)
        model.predict_proba.return_value = np.array([[0.2, 0.8]]) 
        return model

    @pytest.fixture
    def mock_preprocessor(self):
        prep = MagicMock()
        # Mock transform to return a dummy array
        prep.transform.return_value = np.array([[0.5, -0.1, 1.0, 0.0]])
        return prep

    @pytest.fixture
    def sample_input(self):
        return {
            'Gender': 'Female',
            'Age': 30,
            'Class': 'Eco',
            'Flight Distance': 500,
            # Add other necessary fields...
            'Inflight wifi service': 3
        }

    def test_predict_with_probability_structure(self, mock_model, mock_preprocessor, sample_input):
        """Test that the prediction returns the correct dictionary structure."""
        
        # Mock preprocess_for_prediction inside scripts.predict to avoid import issues or real logic
        with patch('scripts.predict.preprocess_for_prediction') as mock_prep_func:
            mock_prep_func.return_value = np.array([[0.1, 0.2]]) # Dummy transformed data
            
            result = predict_with_probability(
                sample_input, 
                model=mock_model, 
                preprocessor=mock_preprocessor
            )
            
            assert isinstance(result, dict)
            assert 'prediction' in result
            assert 'prediction_numeric' in result
            assert 'probability_satisfied' in result
            assert 'probability_dissatisfied' in result
            
            assert result['prediction_numeric'] == 1
            assert result['probability_satisfied'] == 0.8
            assert result['prediction'] == 'satisfied'

    def test_predict_with_probability_dissatisfied(self, mock_model, mock_preprocessor, sample_input):
        """Test prediction for dissatisfied case."""
        mock_model.predict.return_value = np.array([0])
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
        
        with patch('scripts.predict.preprocess_for_prediction') as mock_prep_func:
            mock_prep_func.return_value = np.array([[0.1, 0.2]])
            
            result = predict_with_probability(
                sample_input, 
                model=mock_model, 
                preprocessor=mock_preprocessor
            )
            
            assert result['prediction_numeric'] == 0
            assert result['probability_dissatisfied'] == 0.9
            assert result['prediction'] == 'neutral or dissatisfied'
