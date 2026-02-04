import pandas as pd
from typing import Any, List, Dict, Union

def make_prediction(model: Any, data: pd.DataFrame) -> Union[List[int], Any]:
    """
    Realiza predicciones usando el modelo entrenado.

    Args:
        model (Any): Modelo (scikit-learn, etc).
        data (pd.DataFrame): Datos preprocesados (misma estructura que X_train).

    Returns:
        List[int]: Predicciones de clase.
    """
    try:
        # TODO: Ejecutar predicción
        # preds = model.predict(data)
        # return preds
        return []
    except Exception as e:
        print(f"Error en inferencia: {e}")
        return []

def get_prediction_probabilities(model: Any, data: pd.DataFrame) -> List[float]:
    """
    Obtiene las probabilidades de las predicciones si el modelo lo soporta.
    """
    try:
        # TODO: Obtener probas
        # probs = model.predict_proba(data)
        return []
    except Exception:
        return []
