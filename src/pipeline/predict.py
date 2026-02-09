# src/pipeline/predict.py

from typing import Any
import pandas as pd
import numpy as np

def predict(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Genera predicciones usando el modelo entrenado.

    Args:
        model: Modelo entrenado (resultado de train_model).
        X (pd.DataFrame): Datos de entrada.
                          Deben tener las MISMAS columnas que los datos de entrenamiento,
                          en el MISMO orden y con la MISMA codificacion.

    Returns:
        np.ndarray: Array con las predicciones (0 o 1).
                    0 = neutral or dissatisfied
                    1 = satisfied
    """
    try:
        # .predict() toma los datos de entrada y devuelve la clase predicha.
        # Internamente, cada arbol del bosque "vota" y gana la mayoria.
        predictions = model.predict(X)
        print(f"  {len(predictions)} predicciones generadas")
        return predictions

    except Exception as e:
        print(f"  Error en la prediccion: {e}")
        return np.array([])


def predict_proba(model: Any, X: pd.DataFrame) -> np.ndarray:
    """
    Genera probabilidades de cada clase (en vez de solo la clase final).

    Esto es util para entender que tan "seguro" esta el modelo.
    Ej: [0.3, 0.7] significa 30% insatisfecho, 70% satisfecho.

    Args:
        model: Modelo entrenado.
        X (pd.DataFrame): Datos de entrada.

    Returns:
        np.ndarray: Array de shape (n_muestras, 2) con probabilidades.
                    Columna 0 = prob de insatisfecho.
                    Columna 1 = prob de satisfecho.
    """
    try:
        probabilities = model.predict_proba(X)
        return probabilities
    except Exception as e:
        print(f"  Error en predict_proba: {e}")
        return np.array([])


if __name__ == "__main__":
    from src.pipeline.train_model import train_model

    model, X_train, X_test, y_train, y_test, encoders = train_model(
        "data/reduced/airlines_sample.csv"
    )

    # Predecir con datos de test
    preds = predict(model, X_test)
    print(f"\nPrimeras 10 predicciones: {preds[:10]}")
    print(f"Valores reales:          {y_test[:10]}")

    # Ver probabilidades
    probas = predict_proba(model, X_test)
    if len(probas) > 0:
        print(f"\nProbabilidades (primeras 5):")
        for i in range(min(5, len(probas))):
            print(f"   Muestra {i}: insatisfecho={probas[i][0]:.2%}, satisfecho={probas[i][1]:.2%}")