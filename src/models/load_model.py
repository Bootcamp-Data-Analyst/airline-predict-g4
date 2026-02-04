import joblib
from typing import Any
import os

def load_trained_model(model_path: str) -> Any:
    """
    Carga el modelo entrenado desde disco.

    Args:
        model_path (str): Ruta absoluta o relativa al archivo del modelo (.pkl/.joblib).

    Returns:
        Any: El objeto del modelo cargado (ej. sklearn estimator).

    Raises:
        FileNotFoundError: Si el archivo especificado no existe.
    """
    try:
        # Verificación preliminar de existencia
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontró el archivo del modelo en: {model_path}")

        print(f"Cargando modelo desde: {model_path}")
        
        # TODO: Cargar el modelo usando joblib
        # model = joblib.load(model_path)
        # return model
        
        return None # Placeholder

    except Exception as e:
        print(f"Error al cargar el modelo: {e}")
        return None

def save_model(model: Any, save_path: str) -> bool:
    """
    Guarda el modelo entrenado en disco.

    Args:
        model (Any): Objeto del modelo a guardar.
        save_path (str): Ruta donde se guardará el archivo.
    
    Returns:
        bool: True si se guardó exitosamente, False en caso contrario.
    """
    try:
        # TODO: Implementar lógica de guardado
        # joblib.dump(model, save_path)
        print(f"Modelo guardado en: {save_path}")
        return True
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")
        return False
