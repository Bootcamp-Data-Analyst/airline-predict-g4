import pandas as pd
from datetime import datetime
import os

def log_prediction(input_data: pd.DataFrame, prediction: pd.Series, filepath: str = "logs/predictions.csv"):
    """
    Guarda inputs y predicciones del Airlines Dataset con timestamp en CSV.

    Args:
        input_data (pd.DataFrame): Datos de entrada
        prediction (pd.Series): Predicciones del modelo
        filepath (str): Ruta del archivo de logs
    """
    try:
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Crear dataframe de log
        log_entry = input_data.copy()
        log_entry['prediction'] = prediction.values if isinstance(prediction, pd.Series) else prediction
        log_entry['timestamp'] = datetime.now()
        
        # Guardar (append si existe)
        write_header = not os.path.exists(filepath)
        log_entry.to_csv(filepath, mode='a', header=write_header, index=False)
        print(f"Predicción registrada en {filepath}")
        
    except Exception as e:
        print(f"Error al registrar log: {e}")
