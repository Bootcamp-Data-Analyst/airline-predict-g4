import streamlit as st
import pandas as pd
import sys
import os

# Permitir importaciones relativas añadiendo src al path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# root_dir = os.path.join(current_dir, '..', '..')
# sys.path.append(root_dir)

# from src.pipeline import predict
# from src.models import load_model

def main():
    st.set_page_config(
        page_title="Airline Satisfaction App",
        page_icon="✈️",
        layout="wide"
    )

    st.title("🛫 Predicción de Satisfacción de Clientes")
    st.markdown("---")

    st.markdown("""
    ### Descripción
    Esta aplicación predice si un pasajero estará **Satisfecho** o **Neutral/Insatisfecho** 
    basándose en los parametros del vuelo y servicios recibidos.
    """)

    with st.container():
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.header("📋 Datos del Pasajero")
            
            # TODO: Completar el formulario con los inputs reales del dataset
            flight_distance = st.number_input("Distancia de Vuelo (km)", min_value=0, value=1000)
            seat_comfort = st.slider("Confort del Asiento (1-5)", 1, 5, 3)
            # Agregar resto de features...
            
            predict_btn = st.button("Analizar")

        with col2:
            st.header("📊 Resultado de la Predicción")
            
            if predict_btn:
                # Placeholder de lógica de predicción
                st.info("Procesando datos...")
                
                # TODO: Integrar lógica real
                # input_df = pd.DataFrame([features])
                # result = predict.make_prediction(model, input_df)
                
                # Ejemplo dummy
                prediction = "Satisfied" 
                prob = 0.85
                
                if prediction == "Satisfied":
                    st.success(f"**Cliente Satisfecho** (Probabilidad: {prob:.2%})")
                else:
                    st.warning(f"**Cliente Insatisfecho/Neutral** (Probabilidad: {prob:.2%})")

if __name__ == "__main__":
    main()
