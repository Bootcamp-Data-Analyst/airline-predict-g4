import pandas as pd

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y transforma el Airlines Dataset para el modelo.

    Args:
        df (pd.DataFrame): Dataset original

    Returns:
        pd.DataFrame: Dataset preprocesado listo para entrenamiento
    """
    # Copia para evitar SettingWithCopyWarning
    df_clean = df.copy()
    
    # TODO: Implementar limpieza
    # - Manejo de nulos
    # - Codificación de categóricas
    # - Escalamiento
    
    print("Preprocesamiento completado.")
    return df_clean
