![Airline Predict Banner](assets/banner.svg)

# Airline Predict G4 ✈️

Proyecto de Machine Learning para la clasificación de la satisfacción de pasajeros utilizando el **Airline Passenger Satisfaction Dataset**. Implementa un pipeline completo desde el Análisis Exploratorio de Datos (EDA) hasta el despliegue de una aplicación interactiva en contenedores.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.10+-red.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-blue)
![License](https://img.shields.io/badge/License-MIT-green.svg)

---

## 📋 Tabla de Contenidos
- [Descripción General](#descripción-general)
- [Tecnologías Utilizadas](#tecnologías-utilizadas)
- [Estructura del Repositorio](#estructura-del-repositorio)
- [Dataset y Variables](#dataset-y-variables)
- [Pipeline de Ciencia de Datos](#pipeline-de-ciencia-de-datos)
  - [EDA y Limpieza](#eda-y-limpieza)
  - [Feature Engineering](#feature-engineering)
  - [Modelado y Optimización](#modelado-y-optimización)
- [Aplicación Interactiva](#aplicación-interactiva)
- [Dockerización](#dockerización)
- [Instalación y Ejecución](#instalación-y-ejecución)
- [Resultados y Métricas](#resultados-y-métricas)

---

## 📖 Descripción General
**Airline Predict G4** es una solución diseñada para aerolíneas que buscan entender y predecir la satisfacción de sus clientes. Mediante el análisis de datos demográficos y métricas de servicio (como Wi-Fi a bordo, comodidad del asiento y retrasos), el sistema clasifica a los pasajeros en dos categorías: **Satisfecho** o **Neutral/Insatisfecho**.

---

## 🛠️ Tecnologías Utilizadas
- **Lenguaje**: Python 3.8+
- **Procesamiento de Datos**: Pandas, NumPy
- **Machine Learning**: Scikit-Learn
- **Optimización de Parametría**: Optuna
- **Visualización**: Matplotlib, Seaborn
- **Frontend**: Streamlit
- **Persistencia**: SQLite (para monitoreo de predicciones)
- **Deployment**: Docker, Docker Compose

---

## 📂 Estructura del Repositorio
```text
airline-predict-g4/
├── app/                # Código fuente de la aplicación Streamlit
├── assets/             # Recursos visuales (Logo, Banner)
├── data/               # Datasets raw y procesados
├── docker/             # Configuración de Docker y Docker Compose
├── models/             # Modelos entrenados y preprocesadores (.joblib)
├── notebooks/          # Notebooks de EDA y Limpieza
├── scripts/            # Módulos de procesamiento, entrenamiento y predicción
├── tests/              # Tests unitarios para el pipeline
└── requirements.txt    # Dependencias del proyecto
```

---

## 📊 Dataset y Variables
El modelo utiliza variables críticas del servicio aéreo:
- **Demográficas**: Edad, Género, Tipo de Cliente.
- **Viaje**: Clase, Distancia de vuelo, Motivo del viaje.
- **Servicio (Escala 1-5)**: Wi-Fi, Comodidad del asiento, Limpieza, Servicio de comida, Entretenimiento, etc.
- **Logística**: Retrasos de salida y llegada (en minutos).

---

## ⚙️ Pipeline de Ciencia de Datos

### EDA y Limpieza
Documentado en `notebooks/airline_predict_g4_eda.ipynb`, se realizó:
- Imputación de valores nulos en retrasos de llegada mediante la mediana.
- Eliminación de registros duplicados e innecesarios (`Unnamed: 0`, `id`).
- Análisis de correlación entre servicios y la satisfacción final.

### Feature Engineering
Localizado en `scripts/preprocess.py`:
- **Numéricas**: Imputación de mediana y escalado estándar (`StandardScaler`).
- **Categóricas**: Codificación One-Hot (`OneHotEncoder`) tras imputación por frecuencia.
- **Target**: Codificación binaria (Satisfied=1, Neutral/Dissatisfied=0).

### Modelado y Optimización
El modelado principal emplea un **RandomForestClassifier**:
1. **Baseline**: Modelo base con pesos balanceados para manejar el desequilibrio de clases.
2. **Optimización**: Búsqueda de hiperparámetros mediante **Optuna** (F1-score como métrica objetivo en validación cruzada estratificada).

---

## 🚀 Aplicación Interactiva
La aplicación desarrollada en **Streamlit** permite:
- Ingreso manual de datos de vuelo y perfil del pasajero.
- Evaluación detallada de servicios mediante radio buttons.
- Predicción en tiempo real con probabilidad de confianza.
- **Monitoreo**: Las predicciones realizadas se almacenan automáticamente en una base de datos SQLite local para auditoría posterior.

---

## 🐳 Dockerización
El proyecto cuenta con una configuración robusta para despliegue:
- **Dockerfile**: Expone el puerto `8501` y configura el entorno productivo.
- **Docker Compose**: Gestiona volúmenes para persistencia de datos y scripts de monitoreo.

---

## 🛠️ Instalación y Ejecución

### Requisitos Previos
- Python 3.8+ o Docker Desktop.

### Local (Pip)
1. Instalar dependencias: `pip install -r requirements.txt`
2. Ejecutar app: `python -m streamlit run app/app.py`

### Docker (Recomendado)
```bash
docker-compose -f docker/docker-compose.yml up --build
```

---

## 📈 Resultados y Métricas
Resultados obtenidos por el modelo final (**Optuna Optimized**):

| Métrica | Train | Test |
| :--- | :--- | :--- |
| **Accuracy** | 98.56% | **95.42%** |
| **F1-Score** | 98.32% | **94.65%** |
| **Precision** | 99.12% | 96.10% |
| **Recall** | 97.54% | 93.25% |

*El control de overfitting es óptimo, con una diferencia de Accuracy de solo el 3.14% entre entrenamiento y prueba.*

---

## 🔮 Mejoras Futuras
- **Pendiente de documentar**: Integración de modelos de Gradient Boosting (XGBoost/LGBM).
- **Pendiente de documentar**: Sistema de Retraining automático basado en drift de datos.

---

## 👥 Equipo (Grupo 4)

| Integrante | Rol | Contacto |
| :--- | :--- | :--- |
| **Rocio Lozano Caro** | Data Analyst | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/rociolozanocaro/) [![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/rociolozanocaro) |
| **Thamirys Kearney** | Product Owner | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/thamirys-kearney-0a7a7331/) [![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/ThamirysKearney) |
| **Mariana Moreno** | Scrum Master | [![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mariana-moreno-henao/) |

---
<p align="center">
  <b>Factoría F5</b><br>
  <i>Proyecto diseñado para el <b>Bootcamp Data Analyst</b> bajo estándares de calidad profesional.</i>
</p>
