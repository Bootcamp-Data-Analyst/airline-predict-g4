# airline-predict-g4

**Proyecto de Clasificación de Satisfacción de Clientes de Aerolíneas**

Este proyecto implementa un pipeline de Machine Learning completo para predecir si un cliente está satisfecho o no, basándose en el **Airlines Dataset**.

## 🚀 Tecnologías usadas
- **Python**: Lenguaje principal.
- **Scikit-learn**: Modelado y preprocesamiento.
- **Pandas**: Manipulación de datos.
- **Streamlit**: Interfaz de usuario web.
- **Docker**: Contenerización para despliegue reproducible.

## 📂 Estructura del proyecto

```bash
airline-predict-g4/
├── data/               # Datasets (raw/processed)
├── notebooks/          # Análisis Exploratorio (EDA)
├── src/
│   ├── models/         # Entrenamiento y carga de modelos
│   ├── pipeline/       # Scripts de preprocesamiento, predicción y logging
│   └── app/            # Aplicación Frontend (Streamlit)
├── docker/             # Configuración de Docker
├── tests/              # Pruebas unitarias
└── README.md           # Documentación principal
```

## 👥 Roles y Flujo de Trabajo

El desarrollo se realiza siguiendo **Gitflow**:

| Feature Branch | Responsable | Descripción |
|----------------|-------------|-------------|
| `feature-eda-model` | **Rocio L** | EDA, selección y entrenamiento del modelo. |
| `feature-pipeline` | **Thami** | Pipeline de transformación de datos y scripts de predicción. |
| `feature-app` | **Rocio P** | Desarrollo de la aplicación visual en Streamlit. |
| `feature-deployment` | **Mariana** | Configuración de Docker y proceso de despliegue. |

**Flujo:**
1. Crear rama `feature-x` desde `develop`.
2. Implementar cambios.
3. Pull Request hacia `develop`.

## 🛠️ Ejecución con Docker

El proyecto incluye un `Dockerfile` listo para ejecutar la aplicación Streamlit.

1. **Construir la imagen:**
   ```bash
   docker-compose build
   ```

2. **Ejecutar el contenedor:**
   ```bash
   docker-compose up
   ```
   La aplicación estará disponible en `http://localhost:8501`.

## 🎯 Objetivo final
Crear un sistema modular, profesional y listo para producción, elevando el estándar de MLOps del equipo.
