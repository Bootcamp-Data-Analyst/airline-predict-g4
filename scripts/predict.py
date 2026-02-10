# scripts/predict.py
"""
Inference engine: receives data and returns predictions with probabilities.
Used by the Streamlit App (Rocio P) to predict passenger satisfaction.
"""
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from scripts.model_utils import load_model, load_encoders
    from scripts.preprocess import preprocess_for_prediction, decode_target, load_preprocessor
    from scripts.paths import MODEL_PATH, PREPROCESSOR_PATH
except ImportError:
    from model_utils import load_model, load_encoders
    from preprocess import preprocess_for_prediction, decode_target, load_preprocessor
    from paths import MODEL_PATH, PREPROCESSOR_PATH


def predict_with_probability(user_data, model=None, preprocessor=None):
    """
    Predicts passenger satisfaction and returns probability.
    This is the MAIN FUNCTION that Rocio P's Streamlit App will call.

    Args:
        user_data (dict): Data from the Streamlit form.
        model: Loaded model (optional, loads from disk if not provided).
        preprocessor: Loaded preprocessor (optional, loads from disk if not provided).

    Returns:
        dict: {
            'prediction': 'satisfied' or 'neutral or dissatisfied',
            'prediction_numeric': 0 or 1,
            'probability_satisfied': float (0.0 to 1.0),
            'probability_dissatisfied': float (0.0 to 1.0)
        }
    """
    try:
        # Load model and preprocessor if not provided
        if model is None:
            model = load_model()
        if preprocessor is None:
            preprocessor = load_preprocessor()

        # Preprocess the user input
        X = preprocess_for_prediction(user_data, preprocessor)

        # Predict class
        prediction_numeric = model.predict(X)[0]

        # Predict probabilities
        probabilities = model.predict_proba(X)[0]

        # Decode to text
        prediction_text = decode_target(np.array([prediction_numeric]))[0]

        result = {
            'prediction': prediction_text,
            'prediction_numeric': int(prediction_numeric),
            'probability_satisfied': float(probabilities[1]),
            'probability_dissatisfied': float(probabilities[0])
        }

        print(
            f" Prediction: {prediction_text} "
            f"(satisfied: {probabilities[1]:.1%}, dissatisfied: {probabilities[0]:.1%})"
        )
        return result
    except Exception as e:
        print(f" Error in prediction: {e}")
        return {
            'prediction': 'error',
            'prediction_numeric': -1,
            'probability_satisfied': 0.0,
            'probability_dissatisfied': 0.0
        }


def predict_batch(X, model=None):
    """
    Predicts for multiple rows (used in evaluation and monitoring).

    Args:
        X: Preprocessed data (already transformed).
        model: Loaded model.

    Returns:
        np.ndarray: Array of predictions.
    """
    if model is None:
        model = load_model()
    return model.predict(X)


if __name__ == "__main__":
    # Check if model exists, train if not
    if not os.path.exists(str(MODEL_PATH)):
        print(" No model found. Training first...")
        from scripts.train_model import train_model
        train_model()

    # Test with a sample passenger
    sample = {
        'Gender': 'Female', 'Customer Type': 'Loyal Customer', 'Age': 40,
        'Type of Travel': 'Business travel', 'Class': 'Business',
        'Flight Distance': 2000, 'Inflight wifi service': 5,
        'Departure/Arrival time convenient': 4, 'Ease of Online booking': 5,
        'Gate location': 3, 'Food and drink': 5, 'Online boarding': 5,
        'Seat comfort': 5, 'Inflight entertainment': 5, 'On-board service': 5,
        'Leg room service': 5, 'Baggage handling': 5, 'Checkin service': 4,
        'Inflight service': 5, 'Cleanliness': 5,
        'Departure Delay in Minutes': 0, 'Arrival Delay in Minutes': 0
    }

    print("\n--- Test: Single prediction ---")
    result = predict_with_probability(sample)
    print(f" Full result: {result}")

    # Test with a dissatisfied passenger
    sample_bad = {
        'Gender': 'Male', 'Customer Type': 'disloyal Customer', 'Age': 22,
        'Type of Travel': 'Personal Travel', 'Class': 'Eco',
        'Flight Distance': 300, 'Inflight wifi service': 1,
        'Departure/Arrival time convenient': 1, 'Ease of Online booking': 1,
        'Gate location': 1, 'Food and drink': 1, 'Online boarding': 1,
        'Seat comfort': 1, 'Inflight entertainment': 1, 'On-board service': 1,
        'Leg room service': 1, 'Baggage handling': 1, 'Checkin service': 1,
        'Inflight service': 1, 'Cleanliness': 1,
        'Departure Delay in Minutes': 120, 'Arrival Delay in Minutes': 130
    }

    print("\n--- Test: Dissatisfied passenger ---")
    result_bad = predict_with_probability(sample_bad)
    print(f" Full result: {result_bad}")