# src/pipeline/monitoring.py

import pandas as pd
import numpy as np

def detect_drift(train_data, new_data, threshold=0.1):
    """
    Compara datos de entrenamiento con datos nuevos para detectar drift.

    Metodo: comparar las medias de cada columna numerica.
    Si la diferencia relativa supera el umbral (threshold), hay posible drift.

    ¿Por que es importante?
    Si los datos de produccion son muy diferentes a los de entrenamiento,
    el modelo puede dar predicciones poco fiables.

    Args:
        train_data (pd.DataFrame): Datos originales de entrenamiento (solo features X).
        new_data (pd.DataFrame): Datos nuevos de produccion.
        threshold (float): Umbral de diferencia relativa.
                           0.1 = 10%. Si la media cambia mas de 10%, hay alerta.

    Returns:
        dict: Reporte con columnas en alerta y estadisticas.
    """
    print("\n  ANALISIS DE DATA DRIFT")
    print("=" * 50)

    # Solo columnas numericas.
    numeric_cols = train_data.select_dtypes(include=[np.number]).columns
    common_cols = [col for col in numeric_cols if col in new_data.columns]

    drift_report = {}
    alerts = []

    for col in common_cols:
        train_mean = train_data[col].mean()
        new_mean = new_data[col].mean()

        # Diferencia relativa: cuanto cambio en proporcion.
        # max(..., 1e-10) para evitar division por cero.
        relative_diff = abs(train_mean - new_mean) / max(abs(train_mean), 1e-10)

        has_drift = relative_diff > threshold

        drift_report[col] = {
            "train_mean": round(train_mean, 4),
            "new_mean": round(new_mean, 4),
            "relative_diff": round(relative_diff, 4),
            "drift_detected": has_drift
        }

        if has_drift:
            alerts.append(col)
            print(f"   DRIFT en '{col}': train={train_mean:.2f} -> new={new_mean:.2f} (diff={relative_diff:.1%})")

    if not alerts:
        print("   No se detecto drift significativo")
    else:
        print(f"\n   {len(alerts)} columna(s) con drift: {alerts}")

    print("=" * 50)

    return {
        "drift_report": drift_report,
        "columns_with_drift": alerts,
        "total_columns_checked": len(common_cols)
    }


if __name__ == "__main__":
    from src.pipeline.train_model import train_model

    # Entrenar para obtener X_train
    model, X_train, X_test, y_train, y_test, encoders = train_model(
        "data/raw/airlines.csv"
    )

    # Simular datos "nuevos" tomando una muestra aleatoria
    # (en produccion real, estos serian datos que llegan de usuarios)
    new_sample = X_test.sample(n=100, random_state=99)

    # Detectar drift
    report = detect_drift(X_train, new_sample, threshold=0.1)