# scripts/monitor.py
"""
Monitoring module for model predictions.

Responsibilities:
- Run predictions (single or batch)
- Log predictions to SQLite
- Serve as the monitoring entry point for future drift / metrics analysis
"""

import os
import sys
import pandas as pd

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from scripts.predict import predict_with_probability
    from scripts.logging_utils import log_prediction, get_prediction_logs
    from scripts.paths import DATASET_PATH
except ImportError:
    from predict import predict_with_probability
    from logging_utils import log_prediction, get_prediction_logs
    from paths import DATASET_PATH


def monitor_single_prediction(input_data: dict) -> dict:
    """
    Runs a single prediction and logs it to the database.

    Args:
        input_data (dict): Passenger data.

    Returns:
        dict: Prediction result.
    """
    result = predict_with_probability(input_data)
    log_prediction(input_data, result)
    return result


def monitor_batch_predictions(df: pd.DataFrame, max_rows: int = None):
    """
    Runs predictions on a batch of rows and logs each result.

    Args:
        df (pd.DataFrame): Input dataframe.
        max_rows (int): Optional limit for rows.
    """
    if max_rows:
        df = df.head(max_rows)

    print(f" Monitoring batch of {len(df)} rows")

    for _, row in df.iterrows():
        input_data = row.to_dict()
        result = predict_with_probability(input_data)
        log_prediction(input_data, result)


def show_recent_logs(n: int = 10):
    """
    Prints the most recent prediction logs.

    Args:
        n (int): Number of logs to show.
    """
    logs = get_prediction_logs(n=n)
    print("\n Recent prediction logs")
    print("=" * 50)
    for log in logs:
        print(
            f"{log['timestamp']} | {log['prediction']} | "
            f"satisfied: {log['probability_satisfied']:.1%}"
        )


if __name__ == "__main__":
    print("=" * 60)
    print("MODEL MONITORING TEST")
    print("=" * 60)

    # --- Test 1: Single prediction ---
    sample = {
        'Gender': 'Male', 'Customer Type': 'Loyal Customer', 'Age': 45,
        'Type of Travel': 'Business travel', 'Class': 'Eco Plus',
        'Flight Distance': 1200, 'Inflight wifi service': 4,
        'Departure/Arrival time convenient': 3, 'Ease of Online booking': 4,
        'Gate location': 3, 'Food and drink': 4, 'Online boarding': 4,
        'Seat comfort': 4, 'Inflight entertainment': 4,
        'On-board service': 4, 'Leg room service': 4,
        'Baggage handling': 4, 'Checkin service': 4,
        'Inflight service': 4, 'Cleanliness': 4,
        'Departure Delay in Minutes': 5, 'Arrival Delay in Minutes': 0
    }

    print("\n--- Single prediction ---")
    result = monitor_single_prediction(sample)
    print(result)

    # --- Test 2: Batch monitoring (optional) ---
    if os.path.exists(DATASET_PATH):
        print("\n--- Batch monitoring (5 rows) ---")
        df = pd.read_csv(DATASET_PATH)
        monitor_batch_predictions(df.drop(columns=['satisfaction']), max_rows=5)

    # --- Show logs ---
    show_recent_logs(n=5)

    print("\n Monitoring test complete")