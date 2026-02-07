# src/pipeline/preprocess.py

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Definimos el nombre de la columna que queremos predecir.
# Lo ponemos aquí arriba como constante para no repetirlo en todo el código.
TARGET = "satisfaction"

# Columnas que NO aportan información al modelo.
# "Unnamed: 0" es un índice que se creó al guardar el CSV.
# "id" es un identificador único de cada pasajero, no tiene valor predictivo.
COLUMNS_TO_DROP = ["Unnamed: 0", "id"]


def load_data(path="data/reduced/airlines_sample.csv"):
    """
    Carga el dataset desde un archivo CSV.

    Args:
        path (str): Ruta al archivo CSV.
                     Por defecto usa el dataset reducido (100 filas) para pruebas rápidas.
                     Para entrenar de verdad, usa: "data/raw/airlines.csv"

    Returns:
        pd.DataFrame: El dataset cargado como tabla de pandas.
    """
    # pd.read_csv() lee un archivo CSV y lo convierte en un DataFrame.
    # Un DataFrame es como una tabla de Excel en Python.
    df = pd.read_csv(path)
    print(f"  Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def clean_data(df):
    """
    Limpia el dataset: elimina columnas innecesarias y maneja valores nulos.

    ¿Por qué limpiar?
    - Columnas como 'id' no ayudan al modelo a aprender patrones.
    - Los valores nulos pueden hacer que el modelo falle o aprenda mal.

    Args:
        df (pd.DataFrame): Dataset original.

    Returns:
        pd.DataFrame: Dataset limpio.
    """
    # .copy() crea una copia del DataFrame.
    # Sin esto, modificaríamos el original (puede causar bugs difíciles de encontrar).
    df = df.copy()

    # --- 1. Eliminar columnas inútiles ---
    # axis=1 significa "eliminar columnas" (axis=0 serían filas).
    # errors='ignore' evita que falle si la columna no existe.
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    print(f"  Columnas eliminadas: {COLUMNS_TO_DROP}")

    # --- 2. Manejar valores nulos ---
    # Hay 310 nulos en 'Arrival Delay in Minutes' (de 103,904 filas = 0.3%).
    # Como son muy pocos, los rellenamos con la mediana.
    # ¿Por qué mediana y no media?
    #   - La media es sensible a valores extremos (ej: un retraso de 1000 min la distorsiona).
    #   - La mediana es más robusta: el valor del medio cuando ordenas todos los datos.
    if 'Arrival Delay in Minutes' in df.columns:
        # .median() calcula la mediana de la columna.
        median_value = df['Arrival Delay in Minutes'].median()
        # .fillna() rellena los nulos con el valor que le pases.
        df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(median_value)
        print(f"  Nulos en 'Arrival Delay in Minutes' rellenados con mediana: {median_value}")

    # Verificación: ¿quedaron nulos en algún otro sitio?
    remaining_nulls = df.isnull().sum().sum()
    if remaining_nulls > 0:
        print(f"  Quedan {remaining_nulls} nulos. Eliminando filas afectadas...")
        df = df.dropna()
    else:
        print(f"  Sin valores nulos restantes")

    print(f"  Dataset limpio: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df


def encode_features(df):
    """
    Codifica variables categóricas a números para que el modelo las entienda.

    ¿Qué son variables categóricas?
    Son columnas con texto como "Male"/"Female", "Business"/"Eco".
    Los modelos de ML no entienden texto, solo números.

    Método: LabelEncoder
    Asigna un número a cada categoría. Ej: "Male"=1, "Female"=0.

    Nota para tener en cuenta  el futuro: Me dice Tito GePeTo que LabelEncoder es simple pero tiene una limitación:
    asigna un orden que no existe (ej: "Business"=0 < "Eco"=1 < "Eco Plus"=2).
    Para proyectos más avanzados, usa OneHotEncoder que crea columnas binarias.
    Para este proyecto, LabelEncoder funciona bien con RandomForest.

    Args:
        df (pd.DataFrame): Dataset limpio (sin nulos, sin columnas inútiles).

    Returns:
        tuple: (X, y, label_encoders)
            - X (pd.DataFrame): Features codificadas (todo lo que el modelo usa para predecir).
            - y (np.array): Target codificado (lo que queremos predecir: 0 o 1).
            - label_encoders (dict): Diccionario con los encoders usados.
              Lo necesitaremos después para decodificar predicciones
              o para aplicar la misma transformación a datos nuevos.
    """
    df = df.copy()

    # --- 1. Separar features (X) y target (y) ---
    # X = todas las columnas EXCEPTO 'satisfaction' (son las "pistas" para el modelo).
    # y = solo la columna 'satisfaction' (es la "respuesta" que el modelo debe aprender).
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # --- 2. Identificar columnas categóricas ---
    # select_dtypes(include=['object']) selecciona solo columnas de tipo texto.
    # En nuestro caso: Gender, Customer Type, Type of Travel, Class.
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"  Columnas categóricas a codificar: {categorical_cols}")

    # --- 3. Codificar cada columna categórica ---
    # Guardamos cada encoder en un diccionario para poder reutilizarlos después.
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        # fit_transform() hace dos cosas:
        #   1. fit(): aprende las categorías (ej: "Male", "Female")
        #   2. transform(): las convierte a números (ej: 0, 1)
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le  # Guardamos el encoder para uso futuro
        print(f"      {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # --- 4. Codificar el target ---
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    label_encoders[TARGET] = target_encoder
    print(f"  Target codificado: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")

    return X, y, label_encoders


def preprocess_data(df):
    """
    Pipeline completo de preprocesamiento: limpieza + codificación.

    Esta es la función principal que llamarás desde otros scripts.
    Encadena clean_data() y encode_features() en un solo paso.

    Args:
        df (pd.DataFrame): Dataset crudo.

    Returns:
        tuple: (X, y, label_encoders)
    """
    print("\n  Iniciando preprocesamiento...")
    print("=" * 50)

    # Paso 1: Limpiar datos
    df_clean = clean_data(df)

    # Paso 2: Codificar variables
    X, y, label_encoders = encode_features(df_clean)

    print("=" * 50)
    print(f"   Preprocesamiento completo:")
    print(f"   Features (X): {X.shape}")
    print(f"   Target (y): {len(y)} valores")
    print(f"   Clases del target: {label_encoders[TARGET].classes_}")

    return X, y, label_encoders


# --- Bloque para probar este script de forma independiente ---
# Este bloque solo se ejecuta si corres: python src/pipeline/preprocess.py
# NO se ejecuta cuando otro archivo importa las funciones de este módulo.
if __name__ == "__main__":
    df = load_data("data/raw/airlines.csv")
    X, y, encoders = preprocess_data(df)
    print(f"\nPrimeras 5 filas de X:\n{X.head()}")