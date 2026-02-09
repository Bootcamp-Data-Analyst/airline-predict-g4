# src/pipeline/persistence.py

import joblib
import os
import pandas as pd

# Rutas por defecto donde se guardan los archivos.
# Mariana las usara desde la app para cargar el modelo.
DEFAULT_MODEL_PATH = "models/model.joblib"
DEFAULT_ENCODERS_PATH = "models/encoders.joblib"


def save_model(model, path=DEFAULT_MODEL_PATH):
    """
    Guarda el modelo entrenado en disco.

    Args:
        model: Modelo entrenado de scikit-learn.
        path (str): Ruta donde guardar. Usamos extension .joblib por convencion.
    """
    # Crear el directorio si no existe.
    os.makedirs(os.path.dirname(path), exist_ok=True)

    # joblib.dump() serializa (convierte a bytes) el modelo y lo guarda en disco.
    joblib.dump(model, path)
    print(f"  Modelo guardado en: {path}")


def load_model(path=DEFAULT_MODEL_PATH):
    """
    Carga un modelo previamente guardado desde disco.

    Args:
        path (str): Ruta del archivo del modelo.

    Returns:
        El modelo cargado, listo para hacer predicciones.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"  No se encontro el modelo en: {path}")

    model = joblib.load(path)
    print(f"  Modelo cargado desde: {path}")
    return model


def save_encoders(label_encoders, path=DEFAULT_ENCODERS_PATH):
    """
    Guarda los label encoders para reutilizarlos en produccion.

    ¿Por que guardarlos?
    Si entrenas con Gender: Female=0, Male=1, necesitas el mismo mapeo
    cuando lleguen datos nuevos del formulario. Sin esto, podrias codificar
    "Male" como un numero diferente y el modelo predecira mal.

    Args:
        label_encoders (dict): Diccionario con los encoders.
        path (str): Ruta donde guardarlos.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(label_encoders, path)
    print(f"  Encoders guardados en: {path}")


def load_encoders(path=DEFAULT_ENCODERS_PATH):
    """
    Carga los label encoders guardados.

    Returns:
        dict: Diccionario con los encoders.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"  No se encontraron encoders en: {path}")

    encoders = joblib.load(path)
    print(f"  Encoders cargados desde: {path}")
    return encoders


def save_predictions(df, path="data/processed/predictions.csv"):
    """
    Guarda predicciones en un CSV.

    Args:
        df (pd.DataFrame): DataFrame con las predicciones.
        path (str): Ruta del archivo de salida.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  Predicciones guardadas en: {path}")


if __name__ == "__main__":
    from src.pipeline.train_model import train_model

    # Entrenar
    model, X_train, X_test, y_train, y_test, encoders = train_model(
        "data/reduced/airlines_sample.csv"
    )

    # Guardar modelo y encoders
    save_model(model)
    save_encoders(encoders)

    # Cargar (verificar que funciona)
    model_loaded = load_model()
    encoders_loaded = load_encoders()

    print(f"\n  Verificacion: modelo cargado puede predecir = {hasattr(model_loaded, 'predict')}")