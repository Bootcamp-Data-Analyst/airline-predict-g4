from typing import Any
import pandas as pd

def predict(model: Any, X: pd.DataFrame) -> pd.Series:
    """
    Genera predicciones usando el modelo entrenado.

    Args:
        model (Any): Modelo entrenado
        X (pd.DataFrame): Datos de entrada

    Returns:
        pd.Series: Predicciones
    """
    try:
        predictions = model.predict(X)
        return pd.Series(predictions)
    except Exception as e:
        print(f"Error en la predicción: {e}")
        return pd.Series()
