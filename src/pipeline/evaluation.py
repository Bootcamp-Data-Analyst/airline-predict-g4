# src/pipeline/evaluation.py

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

def evaluate_classification(y_true, y_pred, label_encoder=None):
    """
    Evalua el rendimiento del modelo de clasificacion.

    Args:
        y_true: Valores reales (del dataset de test).
        y_pred: Valores predichos por el modelo.
        label_encoder: (Opcional) El encoder del target para mostrar nombres
                       de clases en vez de numeros.

    Returns:
        dict: Diccionario con todas las metricas.
    """
    # --- 1. Accuracy ---
    # Porcentaje de aciertos sobre el total.
    acc = accuracy_score(y_true, y_pred)

    # --- 2. Classification Report ---
    # Muestra precision, recall y f1-score para CADA clase.
    target_names = None
    if label_encoder is not None:
        target_names = label_encoder.classes_.tolist()

    report = classification_report(
        y_true, y_pred,
        target_names=target_names
    )

    # --- 3. Confusion Matrix ---
    # Tabla 2x2 que muestra:
    #   [Verdaderos Negativos, Falsos Positivos]
    #   [Falsos Negativos,     Verdaderos Positivos]
    cm = confusion_matrix(y_true, y_pred)

    # --- 4. Imprimir resultados ---
    print("\n  EVALUACION DEL MODELO")
    print("=" * 50)
    print(f"Accuracy: {acc:.4f} ({acc:.1%})")
    print(f"\nClassification Report:\n{report}")
    print(f"Confusion Matrix:\n{cm}")
    print("=" * 50)

    return {
        "accuracy": acc,
        "report": report,
        "confusion_matrix": cm
    }


if __name__ == "__main__":
    from src.pipeline.train_model import train_model
    from src.pipeline.predict import predict

    model, X_train, X_test, y_train, y_test, encoders = train_model(
        "data/reduced/airlines_sample.csv"
    )
    preds = predict(model, X_test)
    results = evaluate_classification(y_test, preds, encoders.get("satisfaction"))