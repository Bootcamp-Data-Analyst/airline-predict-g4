# scripts/logging_utils.py
"""
Connects predictions with the SQLite database.
Every time the model predicts, it gets logged here automatically.
"""
import json
from datetime import datetime
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from scripts.database import get_connection
except ImportError:
    from database import get_connection


def log_prediction(input_data, prediction_result, db_path=None):
    """
    Logs a prediction to the SQLite database.
    Args:
        input_data (dict): User data used for prediction.
        prediction_result (dict): Result from predict_with_probability().
            Must contain: prediction, probability_satisfied, probability_dissatisfied.
        db_path (str): Path to DB (optional).
    """
    try:
        conn = get_connection(db_path)
        cursor = conn.cursor()
        # Convert input dictionary to JSON text for storage
        input_json = json.dumps(input_data, ensure_ascii=False)
        cursor.execute(
            '''
            INSERT INTO predictions (
                timestamp, input_data, prediction,
                probability_satisfied, probability_dissatisfied
            ) VALUES (?, ?, ?, ?, ?)
            ''',
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                input_json,
                prediction_result.get('prediction', 'unknown'),
                prediction_result.get('probability_satisfied', 0.0),
                prediction_result.get('probability_dissatisfied', 0.0),
            ),
        )
        conn.commit()
        conn.close()
        print(f" Log recorded: {prediction_result.get('prediction', 'unknown')}")
    except Exception as e:
        print(f" Error logging prediction: {e}")


def get_prediction_logs(n=50, db_path=None):
    """
    Retrieves the last n prediction logs.
    Args:
        n (int): Number of records.
        db_path (str): Path to DB (optional).
    Returns:
        list: List of dictionaries with the records.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        '''
        SELECT id, timestamp, input_data, prediction,
               probability_satisfied, probability_dissatisfied
        FROM predictions
        ORDER BY timestamp DESC
        LIMIT ?
        ''',
        (n,),
    )
    rows = cursor.fetchall()
    conn.close()
    logs = []
    for row in rows:
        logs.append({
            'id': row[0],
            'timestamp': row[1],
            'input_data': json.loads(row[2]) if row[2] else {},
            'prediction': row[3],
            'probability_satisfied': row[4],
            'probability_dissatisfied': row[5],
        })
    return logs


if __name__ == "__main__":
    print("=" * 50)
    print("LOGGING UTILS TEST")
    print("=" * 50)
    # Check if model exists
    from scripts.paths import MODEL_PATH
    if not os.path.exists(str(MODEL_PATH)):
        print(" No model found. Training first...")
        from scripts.train_model import train_model
        train_model()
    from scripts.predict import predict_with_probability
    # Test: predict and log
    sample = {
        'Gender': 'Male', 'Customer Type': 'Loyal Customer', 'Age': 30,
        'Type of Travel': 'Business travel', 'Class': 'Eco Plus',
        'Flight Distance': 1000, 'Inflight wifi service': 3,
        'Departure/Arrival time convenient': 3, 'Ease of Online booking': 3,
        'Gate location': 3, 'Food and drink': 3, 'Online boarding': 3,
        'Seat comfort': 3, 'Inflight entertainment': 3, 'On-board service': 3,
        'Leg room service': 3, 'Baggage handling': 3, 'Checkin service': 3,
        'Inflight service': 3, 'Cleanliness': 3,
        'Departure Delay in Minutes': 5, 'Arrival Delay in Minutes': 0
    }
    print("\n--- Predicting and logging ---")
    result = predict_with_probability(sample)
    log_prediction(sample, result)
    # Read logs
    print("\n--- Reading logs ---")
    logs = get_prediction_logs(n=5)
    print(f" Last {len(logs)} logs:")
    for log in logs:
        print(
            f" {log['timestamp']} -> {log['prediction']} "
            f"(satisfied: {log['probability_satisfied']:.1%})"
        )
    print("\n Logging test complete")