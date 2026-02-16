import os
import sys
import logging
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.model_utils import load_model
from scripts.preprocess import preprocess_for_prediction, decode_target, load_preprocessor
from scripts.paths import MODEL_PATH

def predict_satisfaction(user_data, model=None, preprocessor=None):
    try:
        if model is None:
            model = load_model()
        if preprocessor is None:
            preprocessor = load_preprocessor()

        logger.debug(f"Preprocessing data: {user_data}")
        X = preprocess_for_prediction(user_data, preprocessor)

        # Get prediction and probabilities
        pred_numeric = model.predict(X)[0]
        probs = model.predict_proba(X)[0]
        pred_text = decode_target(np.array([pred_numeric]))[0]

        result = {
            'prediction': pred_text,
            'prediction_numeric': int(pred_numeric),
            'probability_satisfied': float(probs[1]),
            'probability_dissatisfied': float(probs[0])
        }

        logger.info(f"Prediction: {pred_text} (sat: {probs[1]:.1%}, dissat: {probs[0]:.1%})")
        return result

    except Exception as e:
        logger.error(f"Error in prediction: {e}", exc_info=True)
        return {
            'prediction': 'error',
            'prediction_numeric': -1,
            'probability_satisfied': 0.0,
            'probability_dissatisfied': 0.0
        }

if __name__ == "__main__":
    if not os.path.exists(str(MODEL_PATH)):
        logger.warning("No model found. Training first...")
        from scripts.train_model import train_model
        train_model()

    # Test sample
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

    print("\n--- Test Prediction ---")
    print(predict_satisfaction(sample))
