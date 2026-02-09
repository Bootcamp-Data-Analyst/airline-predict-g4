# src/pipeline/train_model.py

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.pipeline.preprocess import load_data, preprocess_data

def train_model(data_path="data/reduced/airlines_sample.csv"):
    """
    Entrena un modelo de clasificación RandomForest.

    Pasos internos:
    1. Cargar datos
    2. Preprocesar (limpiar + codificar)
    3. Dividir en train/test
    4. Entrenar el modelo
    5. Devolver modelo + datos de test + encoders

    Args:
        data_path (str): Ruta al dataset.
                         "data/reduced/airlines_sample.csv" para pruebas rápidas.
                         "data/raw/airlines.csv" para entrenamiento real.

    Returns:
        tuple: (model, X_train, X_test, y_train, y_test, label_encoders)
               Devolvemos también X_train e y_train porque los necesitaremos
               para monitoring (comparar datos de train vs producción).
    """
    # --- 1. Cargar y preprocesar ---
    df = load_data(data_path)
    X, y, label_encoders = preprocess_data(df)

    # --- 2. Dividir datos en entrenamiento y prueba ---
    # test_size=0.2 -> 20% para test, 80% para train.
    # random_state=42 -> semilla para reproducibilidad.
    #   (Si tu y yo usamos la misma semilla, obtenemos la misma division.)
    # stratify=y -> mantiene la misma proporcion de satisfechos/insatisfechos
    #   en train y test. Sin esto, podria pasar que test tenga 90% satisfechos
    #   y train solo 50%, dando resultados engañosos.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\n  Division de datos:")
    print(f"   Train: {X_train.shape[0]} muestras")
    print(f"   Test:  {X_test.shape[0]} muestras")

    # --- 3. Crear y entrenar el modelo ---
    # n_estimators=100 -> numero de arboles en el bosque.
    #   Mas arboles = mas preciso pero mas lento. 100 es un buen inicio.
    # max_depth=10 -> profundidad maxima de cada arbol.
    #   Limita la complejidad para evitar "sobreajuste" (overfitting).
    #   Overfitting = el modelo memoriza los datos de train pero no generaliza.
    # random_state=42 -> reproducibilidad.
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42
    )

    # .fit() es donde ocurre la "magia": el modelo analiza los datos de train
    # y aprende que patrones llevan a satisfaccion o insatisfaccion.
    model.fit(X_train, y_train)

    print(f"  Modelo entrenado con {X_train.shape[0]} muestras")

    return model, X_train, X_test, y_train, y_test, label_encoders


# --- Bloque para probar ---
if __name__ == "__main__":
    model, X_train, X_test, y_train, y_test, encoders = train_model(
        "data/reduced/airlines_sample.csv"
    )
    print(f"\n  Resumen:")
    print(f"   Muestras de train: {len(y_train)}")
    print(f"   Muestras de test: {len(y_test)}")
    print(f"   Features usadas: {X_test.shape[1]}")