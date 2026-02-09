import streamlit as st
import pandas as pd
import sys
import os

# Ajuste de path para importaciones
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.pipeline.predict import predict

def main():
    """
    Aplicación Streamlit básica para interactuar con el modelo.
    """
    st.set_page_config(page_title="Airline Predict G4", page_icon="✈️")
    
    st.title("Airline Predict G4 - Clasificación de Satisfacción")
    
    st.write("### Formulario de inputs del usuario para el Airlines Dataset")
    
    # Placeholder de inputs
    # Ejemplo: flight_dist = st.number_input("Distancia de vuelo")
    
    if st.button("Predecir Satisfacción"):
        st.write("Resultados de predicción aquí")
        
        # TODO: Construir DataFrame con inputs
        # X_input = pd.DataFrame(...)
        
        # TODO: Cargar modelo
        # model = load_model(...)
        
        # TODO: Predecir
        # result = predict(model, X_input)
        # st.success(f"Resultado: {result}")

if __name__ == "__main__":
    main()
