import logging
import os
import sys

def setup_logging(name: str = "AirlineApp", log_file: str = "app.log") -> logging.Logger:
    """
    Configura y retorna un logger estandarizado para la aplicación.

    Args:
        name (str): Nombre del logger.
        log_file (str): Ruta al archivo de log.

    Returns:
        logging.Logger: Instancia de logger configurada.
    """
    # Crear directorio de logs si no existe
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    handler_file = logging.FileHandler(log_file)
    handler_file.setFormatter(formatter)

    handler_console = logging.StreamHandler(sys.stdout)
    handler_console.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Evitar duplicar handlers si la función se llama varias veces
    if not logger.handlers:
        logger.addHandler(handler_file)
        logger.addHandler(handler_console)

    return logger
