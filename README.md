# Airline Customer Satisfaction Prediction (Group 4)

Este proyecto tiene como objetivo desarrollar un modelo de Machine Learning para predecir la satisfacción de pasajeros de línea aérea utilizando el "Airlines Dataset".

## 👥 Equipo (Grupo 4)

| Miembro | Rol / Feature |
|---------|---------------|
| **Rocio L** | EDA & Model Training (`feature-eda-model`) |
| **Thami** | Pipeline de Datos (`feature-pipeline`) |
| **Rocio P** | Aplicación Web (`feature-app`) |
| **Mariana** | Despliegue & Docker (`feature-deployment`) |

## 🚀 Estructura del Proyecto

```
airline-predict-g4/
├── data/               # Datasets raw y processed
├── notebooks/          # Jupyter notebooks para EDA
├── src/                # Código fuente
│   ├── models/         # Entrenamiento y carga de modelos
│   ├── pipeline/       # Preprocesamiento y predicción
│   └── app/            # Aplicación Streamlit
├── docker/             # Configuración Docker
├── tests/              # Tests unitarios
└── requirements.txt    # Dependencias
```

## 🛠️ Instalación y Uso

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/Bootcamp-Data-Analyst/airline-predict-g4.git
   cd airline-predict-g4
   ```

2. **Crear entorno virtual e instalar dependencias:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Ejecutar la App localmente:**
   ```bash
   streamlit run src/app/app.py
   ```

4. **Ejecutar con Docker:**
   ```bash
   cd docker
   docker-compose up --build
   ```

## 📊 Dataset
El dataset contiene información sobre clientes de una aerolínea, incluyendo detalles de su vuelo y puntuaciones de satisfacción sobre diversos servicios.

## 🤝 Contribución
Las contribuciones se manejan mediante **Gitflow**. Cada miembro trabaja en su rama de `feature` y hace Pull Request hacia `develop`.
