# tests/test_pipeline.py

import pytest
import pandas as pd
import numpy as np
import os
import sys

# Añadir la raiz del proyecto al path para que las importaciones funcionen.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pipeline.preprocess import (
    load_data, clean_data, encode_features, preprocess_data,
    preprocess_single_input, FEATURE_COLUMNS
)
from src.pipeline.train_model import train_model
from src.pipeline.predict import predict
from src.pipeline.evaluation import evaluate_classification
from src.pipeline.persistence import save_model, load_model, save_encoders, load_encoders


# ============================================================
# TESTS DE PREPROCESAMIENTO
# ============================================================

class TestPreprocess:
    """Tests para verificar que el preprocesamiento funciona correctamente."""

    def test_load_data(self):
        """Verifica que el dataset se carga correctamente."""
        df = load_data("data/reduced/airlines_sample.csv")
        assert isinstance(df, pd.DataFrame), "load_data debe devolver un DataFrame"
        assert len(df) > 0, "El DataFrame no debe estar vacio"

    def test_clean_data_removes_columns(self):
        """Verifica que se eliminan las columnas innecesarias."""
        df = load_data("data/reduced/airlines_sample.csv")
        df_clean = clean_data(df)
        assert "Unnamed: 0" not in df_clean.columns, "Debe eliminar 'Unnamed: 0'"
        assert "id" not in df_clean.columns, "Debe eliminar 'id'"

    def test_clean_data_no_nulls(self):
        """Verifica que no quedan valores nulos despues de la limpieza."""
        df = load_data("data/reduced/airlines_sample.csv")
        df_clean = clean_data(df)
        assert df_clean.isnull().sum().sum() == 0, "No deben quedar valores nulos"

    def test_encode_features_no_object_columns(self):
        """Verifica que no quedan columnas de tipo texto en X."""
        df = load_data("data/reduced/airlines_sample.csv")
        df_clean = clean_data(df)
        X, y, encoders = encode_features(df_clean)
        object_cols = X.select_dtypes(include=['object']).columns
        assert len(object_cols) == 0, f"Quedan columnas de texto: {list(object_cols)}"

    def test_preprocess_data_shapes(self):
        """Verifica que X e y tienen dimensiones coherentes."""
        df = load_data("data/reduced/airlines_sample.csv")
        X, y, encoders = preprocess_data(df)
        assert X.shape[0] == y.shape[0], "X e y deben tener el mismo numero de filas"
        assert X.shape[1] > 0, "X debe tener al menos una columna"
        assert "satisfaction" in encoders, "Debe existir encoder para el target"

    def test_preprocess_single_input(self):
        """Verifica que preprocess_single_input funciona correctamente.
        ESTE TEST ES CRITICO: valida la consistencia train/produccion."""
        # Primero entrenar para obtener encoders
        df = load_data("data/reduced/airlines_sample.csv")
        X, y, encoders = preprocess_data(df)

        # Simular input de un usuario
        sample = {
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
        result = preprocess_single_input(sample, encoders)

        assert isinstance(result, pd.DataFrame), "Debe devolver un DataFrame"
        assert result.shape[0] == 1, "Debe tener exactamente 1 fila"
        assert list(result.columns) == FEATURE_COLUMNS, "Columnas deben coincidir con FEATURE_COLUMNS"
        # Verificar que no hay texto (todo debe estar codificado)
        assert result.select_dtypes(include=['object']).empty, "No debe haber columnas de texto"


# ============================================================
# TESTS DE ENTRENAMIENTO Y PREDICCION
# ============================================================

class TestModelPipeline:
    """Tests para verificar entrenamiento, prediccion y evaluacion."""

    def test_train_model_returns_model(self):
        """Verifica que train_model devuelve un modelo entrenado."""
        model, X_train, X_test, y_train, y_test, encoders = train_model(
            "data/reduced/airlines_sample.csv"
        )
        assert hasattr(model, 'predict'), "El modelo debe tener metodo predict"

    def test_predict_returns_correct_length(self):
        """Verifica que predict devuelve el numero correcto de predicciones."""
        model, X_train, X_test, y_train, y_test, encoders = train_model(
            "data/reduced/airlines_sample.csv"
        )
        preds = predict(model, X_test)
        assert len(preds) == len(y_test), "Una prediccion por cada muestra de test"

    def test_predictions_are_binary(self):
        """Verifica que las predicciones son 0 o 1."""
        model, X_train, X_test, y_train, y_test, encoders = train_model(
            "data/reduced/airlines_sample.csv"
        )
        preds = predict(model, X_test)
        unique_values = set(preds)
        assert unique_values.issubset({0, 1}), f"Predicciones deben ser 0 o 1, son: {unique_values}"

    def test_evaluation_returns_metrics(self):
        """Verifica que evaluate_classification devuelve metricas validas."""
        model, X_train, X_test, y_train, y_test, encoders = train_model(
            "data/reduced/airlines_sample.csv"
        )
        preds = predict(model, X_test)
        results = evaluate_classification(y_test, preds)
        assert "accuracy" in results, "Debe devolver accuracy"
        assert 0 <= results["accuracy"] <= 1, "Accuracy debe estar entre 0 y 1"


# ============================================================
# TESTS DE PERSISTENCIA (GUARDAR/CARGAR)
# ============================================================

class TestPersistence:
    """Tests para verificar que modelo y encoders se guardan/cargan bien."""

    def test_save_and_load_model(self, tmp_path):
        """Verifica que el modelo guardado da las mismas predicciones."""
        model, X_train, X_test, y_train, y_test, encoders = train_model(
            "data/reduced/airlines_sample.csv"
        )

        model_path = str(tmp_path / "test_model.joblib")
        save_model(model, model_path)
        assert os.path.exists(model_path), "El archivo del modelo debe existir"

        model_loaded = load_model(model_path)
        preds_original = predict(model, X_test)
        preds_loaded = predict(model_loaded, X_test)

        np.testing.assert_array_equal(
            preds_original, preds_loaded,
            err_msg="Modelo cargado debe dar mismas predicciones"
        )

    def test_save_and_load_encoders(self, tmp_path):
        """Verifica que los encoders se guardan y cargan correctamente."""
        model, X_train, X_test, y_train, y_test, encoders = train_model(
            "data/reduced/airlines_sample.csv"
        )

        enc_path = str(tmp_path / "test_encoders.joblib")
        save_encoders(encoders, enc_path)
        assert os.path.exists(enc_path), "El archivo de encoders debe existir"

        encoders_loaded = load_encoders(enc_path)
        assert "satisfaction" in encoders_loaded, "Debe contener encoder del target"
        assert "Gender" in encoders_loaded, "Debe contener encoder de Gender"


# ============================================================
# TEST DE INTEGRIDAD: PIPELINE COMPLETO
# ============================================================

class TestIntegration:
    """Test que verifica el flujo completo de principio a fin."""

    def test_full_pipeline(self, tmp_path):
        """Pipeline completo: cargar -> preprocesar -> entrenar -> predecir -> evaluar -> guardar."""
        # 1. Cargar
        df = load_data("data/reduced/airlines_sample.csv")
        assert len(df) > 0

        # 2. Preprocesar
        X, y, encoders = preprocess_data(df)
        assert X.shape[0] > 0

        # 3. Entrenar
        model, X_train, X_test, y_train, y_test, enc = train_model(
            "data/reduced/airlines_sample.csv"
        )

        # 4. Predecir
        preds = predict(model, X_test)
        assert len(preds) > 0

        # 5. Evaluar
        results = evaluate_classification(y_test, preds)
        assert results["accuracy"] > 0

        # 6. Guardar
        model_path = str(tmp_path / "model.joblib")
        save_model(model, model_path)
        assert os.path.exists(model_path)

        # 7. Cargar y predecir de nuevo
        model_loaded = load_model(model_path)
        preds_loaded = predict(model_loaded, X_test)
        np.testing.assert_array_equal(preds, preds_loaded)

        print("\n  Pipeline completo funcionando correctamente!")

    def test_single_input_prediction(self, tmp_path):
        """Simula el flujo completo de la app: entrenar -> guardar -> cargar -> predecir con 1 input.
        ESTE TEST SIMULA EXACTAMENTE LO QUE HARA LA APP DE MARIANA."""
        # 1. Entrenar y guardar
        model, X_train, X_test, y_train, y_test, encoders = train_model(
            "data/reduced/airlines_sample.csv"
        )
        model_path = str(tmp_path / "model.joblib")
        enc_path = str(tmp_path / "encoders.joblib")
        save_model(model, model_path)
        save_encoders(encoders, enc_path)

        # 2. Simular: la app carga modelo y encoders
        model_loaded = load_model(model_path)
        encoders_loaded = load_encoders(enc_path)

        # 3. Simular: un usuario llena el formulario
        user_input = {
            'Gender': 'Female',
            'Customer Type': 'Loyal Customer',
            'Age': 28,
            'Type of Travel': 'Business travel',
            'Class': 'Eco',
            'Flight Distance': 800,
            'Inflight wifi service': 3,
            'Departure/Arrival time convenient': 4,
            'Ease of Online booking': 3,
            'Gate location': 2,
            'Food and drink': 3,
            'Online boarding': 4,
            'Seat comfort': 3,
            'Inflight entertainment': 3,
            'On-board service': 3,
            'Leg room service': 3,
            'Baggage handling': 4,
            'Checkin service': 3,
            'Inflight service': 3,
            'Cleanliness': 3,
            'Departure Delay in Minutes': 0,
            'Arrival Delay in Minutes': 0
        }

        # 4. Preprocesar el input del usuario
        X_user = preprocess_single_input(user_input, encoders_loaded)

        # 5. Predecir
        pred = predict(model_loaded, X_user)
        assert len(pred) == 1, "Debe haber exactamente 1 prediccion"
        assert pred[0] in [0, 1], "La prediccion debe ser 0 o 1"

        print(f"\n  Test app completo: prediccion = {pred[0]}")