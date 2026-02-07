# src/pipeline/preprocess.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

# ============================================================
# CONSTANTES
# ============================================================

# Nombre de la columna que queremos predecir.
# Lo ponemos aquí arriba como constante para no repetirlo en todo el código.
TARGET = "satisfaction"

# Columnas que NO aportan información al modelo.
# "Unnamed: 0" es un índice que se creó al guardar el CSV.
# "id" es un identificador único de cada pasajero, no tiene valor predictivo.
COLUMNS_TO_DROP = ["Unnamed: 0", "id"]

# Orden lógico para la columna 'Class'.
# Eco (0) < Eco Plus (1) < Business (2)
# Usamos OrdinalEncoder para esta columna porque tiene un orden real:
#   - Eco es la clase más básica
#   - Eco Plus es intermedia
#   - Business es la más alta
# Con LabelEncoder el orden sería alfabético (Business=0, Eco=1, Eco Plus=2),
# lo cual no refleja la realidad.
# Este mismo criterio usa Rocío en el EDA, así mantenemos coherencia.
CLASS_ORDER = [["Eco", "Eco Plus", "Business"]]

# Lista de las features que el modelo espera recibir, EN ORDEN.
# Esto es importante para que preprocess_single_input() genere
# los datos en el mismo formato que el entrenamiento.
FEATURE_COLUMNS = [
    'Gender', 'Customer Type', 'Age', 'Type of Travel', 'Class',
    'Flight Distance', 'Inflight wifi service',
    'Departure/Arrival time convenient', 'Ease of Online booking',
    'Gate location', 'Food and drink', 'Online boarding', 'Seat comfort',
    'Inflight entertainment', 'On-board service', 'Leg room service',
    'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
    'Departure Delay in Minutes', 'Arrival Delay in Minutes'
]


# ============================================================
# FUNCIONES PARA ENTRENAMIENTO (datos completos)
# ============================================================

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
    # errors='ignore' evita que falle si la columna no existe.
    df = df.drop(columns=COLUMNS_TO_DROP, errors='ignore')
    print(f"  Columnas eliminadas: {COLUMNS_TO_DROP}")

    # --- 2. Manejar valores nulos ---
    # Hay 310 nulos en 'Arrival Delay in Minutes' (de 103,904 filas = 0.3%).
    # Como son muy pocos, los rellenamos con la mediana.
    # ¿Por qué mediana y no media?
    #   - La media se ve afectada por valores extremos (ej: un retraso de 1000 min la distorsiona).
    #   - La mediana es más robusta: es el valor del medio cuando ordenas todos los datos.
    if 'Arrival Delay in Minutes' in df.columns:
        median_value = df['Arrival Delay in Minutes'].median()
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

    Usamos DOS métodos de codificación:

    1. LabelEncoder (para Gender, Customer Type, Type of Travel):
       Asigna un número a cada categoría. Ej: "Male"=1, "Female"=0.
       Se usa cuando NO hay un orden lógico entre las categorías.

    2. OrdinalEncoder (para Class):
       Asigna números respetando un ORDEN que nosotros definimos.
       Eco=0 < Eco Plus=1 < Business=2
       Se usa cuando SÍ hay un orden lógico (de menos a más).
       Este mismo criterio usa Rocío en el EDA.

    Nota para el futuro: Para proyectos más avanzados, considera usar
    OneHotEncoder para variables sin orden. Para este proyecto,
    LabelEncoder + OrdinalEncoder funciona bien con RandomForest.

    Args:
        df (pd.DataFrame): Dataset limpio (sin nulos, sin columnas inútiles).

    Returns:
        tuple: (X, y, encoders)
            - X (pd.DataFrame): Features codificadas.
            - y (np.array): Target codificado (0 o 1).
            - encoders (dict): Diccionario con TODOS los encoders usados.
              Lo necesitamos para:
              1. Decodificar predicciones (convertir 0/1 de vuelta a texto).
              2. Aplicar la MISMA transformación a datos nuevos (producción).
    """
    df = df.copy()

    # --- 1. Separar features (X) y target (y) ---
    X = df.drop(TARGET, axis=1)
    y = df[TARGET]

    # --- 2. Identificar columnas categóricas ---
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    print(f"  Columnas categoricas a codificar: {categorical_cols}")

    # --- 3. Codificar 'Class' con OrdinalEncoder ---
    # OrdinalEncoder respeta el orden que le indicamos: Eco(0) < Eco Plus(1) < Business(2).
    # Esto es más correcto que LabelEncoder, que ordenaría alfabéticamente
    # (Business=0, Eco=1, Eco Plus=2), lo cual NO refleja la realidad.
    encoders = {}

    if 'Class' in categorical_cols:
        ord_enc = OrdinalEncoder(categories=CLASS_ORDER)
        # fit_transform() aprende las categorías y las transforma.
        # Necesitamos reshape porque OrdinalEncoder espera una matriz 2D (columnas),
        # pero X['Class'] es una serie 1D. Con [['Class']] lo convertimos a DataFrame de 1 columna.
        X['Class'] = ord_enc.fit_transform(X[['Class']]).astype(int)
        encoders['Class'] = ord_enc
        print(f"      Class (OrdinalEncoder): Eco=0, Eco Plus=1, Business=2")

        # Quitamos 'Class' de la lista de categóricas para no procesarla otra vez.
        categorical_cols.remove('Class')

    # --- 4. Codificar el resto con LabelEncoder ---
    # Gender, Customer Type, Type of Travel: no tienen un orden lógico,
    # así que LabelEncoder está bien.
    for col in categorical_cols:
        le = LabelEncoder()
        # fit_transform() hace dos cosas:
        #   1. fit(): aprende las categorías (ej: "Male", "Female")
        #   2. transform(): las convierte a números (ej: 0, 1)
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
        print(f"      {col} (LabelEncoder): {dict(zip(le.classes_, le.transform(le.classes_)))}")

    # --- 5. Codificar el target ---
    target_encoder = LabelEncoder()
    y = target_encoder.fit_transform(y)
    encoders[TARGET] = target_encoder
    print(f"  Target codificado: {dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))}")

    return X, y, encoders


def preprocess_data(df):
    """
    Pipeline completo de preprocesamiento: limpieza + codificación.

    Esta es la función principal que se usa durante el ENTRENAMIENTO.
    Encadena clean_data() y encode_features() en un solo paso.

    Args:
        df (pd.DataFrame): Dataset crudo.

    Returns:
        tuple: (X, y, encoders)
    """
    print("\n  Iniciando preprocesamiento...")
    print("=" * 50)

    df_clean = clean_data(df)
    X, y, encoders = encode_features(df_clean)

    print("=" * 50)
    print(f"  Preprocesamiento completo:")
    print(f"   Features (X): {X.shape}")
    print(f"   Target (y): {len(y)} valores")
    print(f"   Clases del target: {encoders[TARGET].classes_}")

    return X, y, encoders


# ============================================================
# FUNCIÓN PARA PRODUCCIÓN (datos de un usuario individual)
# ============================================================
# ESTA FUNCIÓN ES LA QUE MARIANA NECESITA PARA LA APP STREAMLIT.
# Recibe los datos de UN pasajero (del formulario) y los transforma
# al mismo formato que usó el modelo durante el entrenamiento.

def preprocess_single_input(user_data, encoders):
    """
    Preprocesa los datos de UN solo usuario para hacer una predicción.

    ¿Cuándo se usa?
    Cuando un usuario llena el formulario de la app Streamlit y hace clic
    en "Predecir". Los datos del formulario llegan como un diccionario
    y esta función los convierte al formato que el modelo espera.

    ¿Por qué es necesaria?
    El modelo fue entrenado con datos codificados (ej: "Male"=1, "Female"=0,
    "Eco"=0, "Business"=2). Si le pasamos texto directamente, no funcionará.
    Esta función aplica las MISMAS transformaciones que se usaron
    durante el entrenamiento, garantizando consistencia.

    Args:
        user_data (dict): Datos del usuario en formato diccionario.
            Ejemplo:
            {
                'Gender': 'Male',
                'Customer Type': 'Loyal Customer',
                'Age': 35,
                'Type of Travel': 'Business travel',
                'Class': 'Business',
                'Flight Distance': 1500,
                'Inflight wifi service': 4,
                ...
            }
        encoders (dict): Los mismos encoders que se usaron durante
                         el entrenamiento (se cargan desde el archivo guardado).

    Returns:
        pd.DataFrame: DataFrame de una fila, codificado y listo para predict().
    """
    # 1. Convertir el diccionario a DataFrame de una fila.
    #    [user_data] lo envuelve en una lista porque DataFrame espera una lista de filas.
    df = pd.DataFrame([user_data])

    # 2. Asegurar que solo tenemos las columnas que el modelo espera, en el orden correcto.
    df = df[FEATURE_COLUMNS]

    # 3. Codificar 'Class' con OrdinalEncoder (el mismo que se usó en entrenamiento).
    if 'Class' in encoders and 'Class' in df.columns:
        df['Class'] = encoders['Class'].transform(df[['Class']]).astype(int)

    # 4. Codificar el resto de categóricas con LabelEncoder.
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if col in encoders:
            # .transform() (sin fit) usa el mapeo ya aprendido.
            # NO aprende categorías nuevas, solo aplica las que ya conoce.
            df[col] = encoders[col].transform(df[col])

    # 5. Rellenar nulos si hay alguno (por si el usuario dejó algo vacío).
    df = df.fillna(0)

    return df


def decode_prediction(prediction, encoders):
    """
    Convierte la predicción numérica (0 o 1) de vuelta a texto legible.

    ¿Por qué?
    El modelo devuelve 0 o 1. Pero el usuario quiere ver
    "satisfied" o "neutral or dissatisfied", no números.

    Args:
        prediction (int o array): La predicción del modelo (0 o 1).
        encoders (dict): Los encoders (necesitamos el del target).

    Returns:
        str: La predicción en texto ("satisfied" o "neutral or dissatisfied").
    """
    if TARGET in encoders:
        # .inverse_transform() hace lo contrario de transform():
        # convierte números de vuelta a las categorías originales.
        return encoders[TARGET].inverse_transform(prediction)
    return prediction


# --- Bloque para probar este script de forma independiente ---
if __name__ == "__main__":
    # --- Prueba 1: preprocesamiento completo (entrenamiento) ---
    print("=" * 60)
    print("PRUEBA 1: Preprocesamiento del dataset completo")
    print("=" * 60)
    df = load_data("data/raw/airlines.csv")
    X, y, encoders = preprocess_data(df)
    print(f"\nPrimeras 5 filas de X:\n{X.head()}")

    # --- Prueba 2: preprocesamiento de un solo input (producción) ---
    print("\n" + "=" * 60)
    print("PRUEBA 2: Preprocesamiento de un input individual")
    print("=" * 60)
    sample_input = {
        'Gender': 'Male',
        'Customer Type': 'Loyal Customer',
        'Age': 35,
        'Type of Travel': 'Business travel',
        'Class': 'Business',
        'Flight Distance': 1500,
        'Inflight wifi service': 4,
        'Departure/Arrival time convenient': 3,
        'Ease of Online booking': 4,
        'Gate location': 3,
        'Food and drink': 4,
        'Online boarding': 5,
        'Seat comfort': 4,
        'Inflight entertainment': 5,
        'On-board service': 4,
        'Leg room service': 4,
        'Baggage handling': 4,
        'Checkin service': 3,
        'Inflight service': 4,
        'Cleanliness': 4,
        'Departure Delay in Minutes': 10,
        'Arrival Delay in Minutes': 5
    }
    X_single = preprocess_single_input(sample_input, encoders)
    print(f"Input procesado:\n{X_single}")
    print(f"Shape: {X_single.shape}")
    print(f"\nValor de 'Class': {X_single['Class'].values[0]}  (debe ser 2 = Business)")