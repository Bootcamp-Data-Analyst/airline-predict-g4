# src/pipeline/logging.py

import pandas as pd
from datetime import datetime
import os

# Ruta por defecto del archivo de logs.
DEFAULT_LOG_PATH = "logs/predictions.csv"


def log_prediction(input_data, prediction, filepath=DEFAULT_LOG_PATH):
    """
    Registra inputs + predicciones + timestamp en un CSV.

    Cada vez que el modelo hace una prediccion, se añade una fila al log.
    Si el archivo no existe, lo crea con cabeceras.
    Si ya existe, añade filas al final (append).

    Args:
        input_data (pd.DataFrame): Datos de entrada que recibio el modelo.
        prediction: Predicciones generadas (array, list o Series).
        filepath (str): Ruta del archivo de logs.
    """
    try:
        # Crear directorio "logs/" si no existe.
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Crear DataFrame con los datos de entrada.
        if isinstance(input_data, pd.DataFrame):
            log_entry = input_data.copy()
        else:
            log_entry = pd.DataFrame(input_data)

        # Añadir columna con la prediccion.
        log_entry['prediction'] = prediction if not hasattr(prediction, 'values') else prediction.values

        # Añadir columna con la fecha y hora actual.
        log_entry['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Guardar en CSV.
        # mode='a' -> append (añadir al final del archivo).
        # header: solo escribir cabeceras si el archivo NO existe aun.
        write_header = not os.path.exists(filepath)
        log_entry.to_csv(filepath, mode='a', header=write_header, index=False)

        print(f"  Log registrado: {len(log_entry)} predicciones -> {filepath}")

    except Exception as e:
        # Si falla el logging, mostramos el error pero NO detenemos el programa.
        # El logging es importante pero no critico.
        print(f"  Error al registrar log: {e}")


def read_logs(filepath=DEFAULT_LOG_PATH):
    """
    Lee los logs de predicciones.

    Returns:
        pd.DataFrame: Todas las predicciones registradas, o DataFrame vacio.
    """
    if not os.path.exists(filepath):
        print("  No hay logs registrados aun.")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    print(f"  {len(df)} predicciones registradas en total")
    return df


if __name__ == "__main__":
    from src.pipeline.train_model import train_model
    from src.pipeline.predict import predict

    model, X_train, X_test, y_train, y_test, encoders = train_model(
        "data/reduced/airlines_sample.csv"
    )
    preds = predict(model, X_test)

    # Registrar log
    log_prediction(X_test, preds)

    # Leer logs
    logs = read_logs()
    if not logs.empty:
        print(f"\nUltimas 3 entradas del log:")
        print(logs.tail(3))