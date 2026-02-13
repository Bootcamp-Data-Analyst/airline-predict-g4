from typing import Any
import joblib
import os

def load_model(path: str) -> Any:
    """
    Carga un modelo serializado desde disco.

    Args:
        path (str): Ruta del archivo del modelo

    Returns:
        Any: Modelo cargado
    """
    try:
        if not os.path.exists(path):
            raise FileNotFoundError(f"El modelo no existe en la ruta: {path}")
            
        model = joblib.load(path)
        print(f"Modelo cargado exitosamente desde {path}")
        return model
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        return None
