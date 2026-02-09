# src/pipeline/feedback.py

import pandas as pd
import os

# Ruta donde se acumulan los datos de feedback.
FEEDBACK_PATH = "data/feedback/new_data.csv"


def save_feedback(input_data, prediction, real_label, filepath=FEEDBACK_PATH):
    """
    Guarda datos nuevos con la etiqueta real para futuro reentrenamiento.

    ¿Cuando usarlo?
    Cuando un usuario confirma o corrige una prediccion:
    - El modelo predijo "satisfied" -> el usuario dice "en realidad estaba insatisfecho".
    - Ese dato (input + etiqueta real) se guarda para mejorar el modelo.

    Args:
        input_data (pd.DataFrame): Datos de entrada (una o varias filas).
        prediction: Lo que el modelo predijo.
        real_label: La etiqueta real (feedback del usuario).
        filepath (str): Ruta donde guardar los datos de feedback.
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        if isinstance(input_data, pd.DataFrame):
            feedback_entry = input_data.copy()
        else:
            feedback_entry = pd.DataFrame(input_data)

        feedback_entry['model_prediction'] = prediction
        feedback_entry['real_label'] = real_label
        feedback_entry['feedback_date'] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

        write_header = not os.path.exists(filepath)
        feedback_entry.to_csv(filepath, mode='a', header=write_header, index=False)

        print(f"  Feedback guardado: {len(feedback_entry)} muestras -> {filepath}")

    except Exception as e:
        print(f"  Error al guardar feedback: {e}")


def load_feedback(filepath=FEEDBACK_PATH):
    """
    Carga los datos de feedback acumulados.

    Estos datos se pueden combinar con los datos originales
    para reentrenar el modelo con informacion mas reciente.

    Returns:
        pd.DataFrame: Datos de feedback, o DataFrame vacio si no hay.
    """
    if not os.path.exists(filepath):
        print("  No hay datos de feedback aun.")
        return pd.DataFrame()

    df = pd.read_csv(filepath)
    print(f"  {len(df)} muestras de feedback disponibles para reentrenamiento")
    return df


if __name__ == "__main__":
    import numpy as np
    from src.pipeline.train_model import train_model
    from src.pipeline.predict import predict

    model, X_train, X_test, y_train, y_test, encoders = train_model(
        "data/reduced/airlines_sample.csv"
    )
    preds = predict(model, X_test)

    # Simular feedback: el usuario confirma las etiquetas reales
    save_feedback(X_test.head(5), preds[:5], y_test[:5])

    # Ver feedback acumulado
    feedback = load_feedback()
    if not feedback.empty:
        print(feedback.head())