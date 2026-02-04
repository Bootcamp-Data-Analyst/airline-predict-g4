# Usar una imagen base ligera de Python
FROM python:3.9-slim

# Evitar que Python genere archivos .pyc y buffer de salida
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Directorio de trabajo
WORKDIR /app

# Instalar dependencias del sistema si son necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copiar requirements e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copiar el código fuente
COPY . .

# Exponer el puerto por el que corre Streamlit (default 8501)
EXPOSE 8501

# Comando para correr la App
CMD ["streamlit", "run", "src/app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
