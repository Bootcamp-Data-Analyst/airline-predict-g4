# Airline Predict G4

Proyecto de clasificación de satisfacción de clientes utilizando el **Airlines Dataset**.

## 🚀 Descripción
Este proyecto implementa un modelo de Machine Learning para predecir la satisfacción de pasajeros, integrando un pipeline completo de datos, aplicación web y despliegue en contenedores.

## 🛠️ Stack Tecnológico
*   Python
*   Scikit-learn
*   Pandas
*   Streamlit
*   Docker

## 🐳 Docker (fase inicial)

> **Nota:** Esta es una dockerización base para desarrollo. El modelo final y el pipeline completo se integrarán en fases posteriores.

### Construir la imagen

```bash
cd docker
docker-compose build
```

### Ejecutar el contenedor

```bash
docker-compose up
```

La aplicación estará disponible en: `http://localhost:8501`

### Detener el contenedor

```bash
docker-compose down
```

