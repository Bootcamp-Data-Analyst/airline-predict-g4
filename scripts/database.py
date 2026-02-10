# scripts/database.py
"""
SQLite database management for prediction logging.
SQLite is a lightweight database stored as a single .db file.
No server needed: everything is managed from Python with the sqlite3 module.
"""
import sqlite3
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
try:
    from scripts.paths import DB_PATH
except ImportError:
    from paths import DB_PATH


def init_db(db_path=None):
    """
    Initializes the SQLite database and creates the predictions table.
    The 'predictions' table stores:
    - id: auto-incremented unique identifier
    - timestamp: date and time of the prediction
    - input_data: user input data as JSON text
    - prediction: model prediction (text)
    - probability_satisfied: probability of satisfied (0.0 to 1.0)
    - probability_dissatisfied: probability of dissatisfied (0.0 to 1.0)

    Args:
        db_path (str): Path to the .db file. Defaults to paths.py config.

    Returns:
        sqlite3.Connection: Database connection.
    """
    if db_path is None:
        db_path = str(DB_PATH)
    
    # Create directory if it does not exist
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    # Connect (creates the file if it does not exist)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it does not exist
    cursor.execute(
        '''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            input_data TEXT,
            prediction TEXT,
            probability_satisfied REAL,
            probability_dissatisfied REAL
        )
        '''
    )
    conn.commit()
    print(f" Database initialized at: {db_path}")
    return conn


def get_connection(db_path=None):
    """
    Gets a connection to the database.
    Creates the table automatically if it does not exist.

    Args:
        db_path (str): Path to the .db file.

    Returns:
        sqlite3.Connection: Active connection.
    """
    if db_path is None:
        db_path = str(DB_PATH)
    return init_db(db_path)


def get_all_predictions(db_path=None):
    """
    Retrieves all logged predictions.

    Returns:
        list: List of tuples with all rows.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM predictions ORDER BY timestamp DESC')
    rows = cursor.fetchall()
    conn.close()
    print(f" {len(rows)} predictions in database")
    return rows


def get_recent_predictions(n=100, db_path=None):
    """
    Retrieves the last n predictions.

    Args:
        n (int): Number of predictions to retrieve.

    Returns:
        list: List of tuples.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    cursor.execute(
        'SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?', (n,)
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


if __name__ == "__main__":
    print("=" * 50)
    print("DATABASE MODULE TEST")
    print("=" * 50)
    # Test: initialize DB
    conn = init_db()
    print(f" Table 'predictions' ready")
    # Verify it is empty (or show count)
    rows = get_all_predictions()
    print(f" Predictions stored: {len(rows)}")
    conn.close()
    print("\n Database test complete")