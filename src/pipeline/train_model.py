from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from src.pipeline.preprocess import load_data, preprocess_data

def train_model():
    """
    Entrena un modelo básico con los datos preprocesados
    """
    # Cargar y preprocesar
    df = load_data()
    X, y = preprocess_data(df)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Modelo simple
    model = RandomForestClassifier(n_estimators=10, random_state=42, max_depth=5)
    model.fit(X_train, y_train)
    
    print(f"✅ Modelo entrenado con {len(X_train)} muestras de entrenamiento")
    
    return model, X_test, y_test

if __name__ == "__main__":
    model, X_test, y_test = train_model()
    print(f"✅ {len(X_test)} muestras de test disponibles")