import pandas as pd
from typing import Tuple, Union, Optional
import os

def load_data(file_path: str) -> Optional[pd.DataFrame]:
    """
    Carga el Dataset de Airlines desde un archivo CSV.

    Args:
        file_path (str): Ruta al archivo CSV en data/raw.

    Returns:
        pd.DataFrame or None: DataFrame cargado si tiene éxito.
    """
    if not os.path.exists(file_path):
        print(f"Error: Archivo no encontrado en {file_path}")
        return None
        
    try:
        # TODO: Cargar datos
        # df = pd.read_csv(file_path)
        # return df
        print(f"Datos cargados desde {file_path}")
        return pd.DataFrame()
    except Exception as e:
        print(f"Error leyendo el CSV: {e}")
        return None

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Realiza limpieza inicial del dataset (tratamiento de nulos, duplicados).

    Args:
        df (pd.DataFrame): Dataset crudo.

    Returns:
        pd.DataFrame: Dataset limpio.
    """
    # TODO: Tratar valores nulos (ej. rellenar media en 'Arrival Delay in Minutes')
    # TODO: Eliminar columnas irrelavantes (ej. 'Unnamed: 0', 'id')
    return df

def preprocess_features(df: pd.DataFrame, target_column: str = 'satisfaction') -> Union[pd.DataFrame, Tuple[pd.DataFrame, pd.Series]]:
    """
    Aplica ingeniería de características y encoding.

    Args:
        df (pd.DataFrame): Dataset limpio.
        target_column (str): Nombre de la columna objetivo.

    Returns:
        Si existe la columna target, retorna (X, y). Si no, retorna X.
    """
    # TODO: One-Hot Encoding para variables categóricas
    # TODO: Scaling para variables numéricas (si es necesario para el modelo)
    
    return df
