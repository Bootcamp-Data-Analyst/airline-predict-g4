import streamlit as st 
import pandas as pd
import sys
import os
import logging
from typing import Dict, Any, Optional, Tuple

# Configuration
APP_NAME = "Airline Predict"
MODEL_VERSION = "v1.0"

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

# Add project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

# Import prediction logic
try:
    from scripts.predict import predict_satisfaction
    from scripts.database import insert_prediction
except ImportError as e:
    import traceback
    error_details = traceback.format_exc()
    logger.error(f"Failed to import scripts: {e}\n{error_details}")
    # Showing the specific error in the UI to help debugging
    st.error(f"⚠️ Error cargando componentes: {e}")
    with st.expander("Ver detalles técnicos"):
        st.code(error_details)
    
    def predict_satisfaction(data):
        return {"prediction": "Error", "probability_satisfied": 0.0, "probability_dissatisfied": 0.0}
    def insert_prediction(*args, **kwargs):
        pass


# Constants
CSAT_LABELS = {
    1: "Muy insatisfecho", 2: "Insatisfecho", 3: "Neutral",
    4: "Satisfecho", 5: "Muy satisfecho", 0: "No aplicable",
}

SERVICE_BLOCKS = {
    "Digital y Conveniencia": {
        "description": "Experiencia digital y horarios.",
        "items": [
            ("inflight_wifi_service", "Wi-Fi a bordo", "Calidad del Wi-Fi."),
            ("ease_of_online_booking", "Reserva online", "Facilidad de reserva."),
            ("online_boarding", "Embarque online", "Proceso digital."),
            ("departure_arrival_time_convenient", "Horarios", "Conveniencia de salida/llegada."),
        ],
    },
    "Procesos en Aeropuerto": {
        "description": "Puntos de contacto en aeropuerto.",
        "items": [
            ("gate_location", "Ubicación puerta", "Conveniencia de la puerta."),
            ("checkin_service", "Check-in", "Servicio en mostrador/kiosco."),
            ("baggage_handling", "Equipaje", "Manejo de equipaje."),
        ],
    },
    "Confort a Bordo": {
        "description": "Confort físico.",
        "items": [
            ("seat_comfort", "Asiento", "Comodidad del asiento."),
            ("leg_room_service", "Espacio piernas", "Espacio disponible."),
            ("cleanliness", "Limpieza", "Higiene de cabina."),
        ],
    },
    "Servicio y Experiencia": {
        "description": "Atención y entretenimiento.",
        "items": [
            ("food_and_drink", "Comida/Bebida", "Calidad de alimentos."),
            ("inflight_entertainment", "Entretenimiento", "Sistema de entretenimiento."),
            ("onboard_service", "Tripulación", "Atención a bordo."),
            ("inflight_service", "Servicio general", "Calidad general."),
        ],
    },
}

def inject_css():
    st.markdown("""
        <style>
          :root{ --ap-accent: #1F3C88; --ap-radius: 12px; }
          .stApp { font-family: 'Segoe UI', sans-serif; }
          .block-container { max-width: 1000px; padding-top: 2rem; }
          .ap-card { background: #ffffff; padding: 20px; border-radius: var(--ap-radius); box-shadow: 0 4px 12px rgba(0,0,0,0.05); border: 1px solid #eee; margin-bottom: 20px; }
          .ap-card-title { font-size: 1.1rem; font-weight: 600; margin-bottom: 5px; color: #333; }
          .ap-card-subtitle { font-size: 0.9rem; color: #666; margin-bottom: 15px; }
          div.stButton > button { width: 100%; border-radius: 8px; font-weight: 600; }
          @media (prefers-color-scheme: dark) { 
            .ap-card { background: #1e1e1e; border-color: #333; } 
            .ap-card-title { color: #eee; }
            .ap-card-subtitle { color: #aaa; }
          }
        </style>
    """, unsafe_allow_html=True)



def init_state():
    defaults = {
        "gender": None, "customer_type": None, "age": 30, "type_of_travel": None,
        "class": None, "flight_distance": 1000, "departure_delay": 0, "arrival_delay": 0,
        "ratings": {key: None for block in SERVICE_BLOCKS.values() for key, _, _ in block["items"]},
        "nav": "Evaluación"
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def render_csat_input(key, label, help_text):
    st.markdown(f"**{label}**")
    val = st.session_state["ratings"].get(key)
    options = [None, 0, 1, 2, 3, 4, 5]
    
    selected = st.radio(
        label=help_text,
        options=options,
        index=options.index(val) if val in options else 0,
        horizontal=True,
        key=f"radio_{key}",
        format_func=lambda x: "N/A" if x == 0 else ("Seleccione..." if x is None else str(x)),
        label_visibility="collapsed"
    )
    
    if selected is not None:
        st.session_state["ratings"][key] = selected

def get_passenger_data() -> pd.DataFrame:
    ui_map = {
        "Mujer": "Female", "Hombre": "Male",
        "Cliente Leal": "Loyal Customer", "Cliente Desleal": "Disloyal Customer",
        "Viaje de negocios": "Business travel", "Viaje personal": "Personal travel",
        "Business": "Business", "Eco": "Eco", "Eco Plus": "Eco Plus" 
    }

    row = {
        "Gender": ui_map.get(st.session_state["gender"]),
        "Customer Type": ui_map.get(st.session_state["customer_type"]),
        "Age": st.session_state["age"],
        "Type of Travel": ui_map.get(st.session_state["type_of_travel"]),
        "Class": ui_map.get(st.session_state["class"]),
        "Flight Distance": st.session_state["flight_distance"],
        "Departure Delay in Minutes": st.session_state["departure_delay"],
        "Arrival Delay in Minutes": st.session_state["arrival_delay"],
    }

    ui_to_dataset = {
        "inflight_wifi_service": "Inflight wifi service",
        "ease_of_online_booking": "Ease of Online booking",
        "online_boarding": "Online boarding",
        "departure_arrival_time_convenient": "Departure/Arrival time convenient",
        "gate_location": "Gate location",
        "checkin_service": "Checkin service",
        "baggage_handling": "Baggage handling",
        "seat_comfort": "Seat comfort",
        "leg_room_service": "Leg room service",
        "cleanliness": "Cleanliness",
        "food_and_drink": "Food and drink",
        "inflight_entertainment": "Inflight entertainment",
        "onboard_service": "On-board service",
        "inflight_service": "Inflight service",
    }

    for ui, ds in ui_to_dataset.items():
        val = st.session_state["ratings"].get(ui)
        row[ds] = val if val is not None else 3 # Default neutral if missing

    return pd.DataFrame([row])




def render_sidebar():
    # Logo
    try:
        logo_path = os.path.join(project_root, "assets", "logo.svg")
        if os.path.exists(logo_path):
            st.sidebar.image(logo_path, use_container_width=True)
    except Exception:
        pass

    st.sidebar.title(APP_NAME)
    st.sidebar.caption(f"Modelo {MODEL_VERSION}")
    
    opts = ["Datos Vuelo", "Evaluación", "Resultados"]
    nav = st.sidebar.radio("Navegación", opts, index=opts.index(st.session_state.get("nav", "Evaluación")) if st.session_state.get("nav") in opts else 1)
    st.session_state["nav"] = nav
    
    st.sidebar.divider()
    if st.sidebar.button("Reiniciar"):
        st.session_state.clear()
        st.rerun()

def render_flight_data():
    st.markdown("<div class='ap-card'><div class='ap-card-title'>Perfil del Pasajero</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    
    with c1:
        st.session_state["gender"] = st.radio("Género", ["Mujer", "Hombre"], horizontal=True, index=["Mujer", "Hombre"].index(st.session_state["gender"]) if st.session_state["gender"] else 0)
        st.session_state["age"] = st.number_input("Edad", 0, 100, st.session_state["age"])
        st.session_state["class"] = st.selectbox("Clase", ["Business", "Eco", "Eco Plus"], index=["Business", "Eco", "Eco Plus"].index(st.session_state["class"]) if st.session_state["class"] else 0)
        st.session_state["departure_delay"] = st.number_input("Retraso Salida (min)", 0, value=st.session_state["departure_delay"])

    with c2:
        st.session_state["customer_type"] = st.selectbox("Tipo Cliente", ["Cliente Leal", "Cliente Desleal"], index=["Cliente Leal", "Cliente Desleal"].index(st.session_state["customer_type"]) if st.session_state["customer_type"] else 0)
        st.session_state["type_of_travel"] = st.radio("Motivo Viaje", ["Viaje de negocios", "Viaje personal"], horizontal=True, index=["Viaje de negocios", "Viaje personal"].index(st.session_state["type_of_travel"]) if st.session_state["type_of_travel"] else 0)
        st.session_state["flight_distance"] = st.number_input("Distancia", 0, value=st.session_state["flight_distance"])
        st.session_state["arrival_delay"] = st.number_input("Retraso Llegada (min)", 0, value=st.session_state["arrival_delay"])
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("Siguiente: Evaluación de Servicio", type="primary"):
        st.session_state["nav"] = "Evaluación"
        st.rerun()

def render_evaluation():
    st.markdown("<div class='ap-card'><div class='ap-card-title'>Evaluación de Servicio</div>", unsafe_allow_html=True)
    
    done = sum(1 for v in st.session_state["ratings"].values() if v is not None)
    total = len(st.session_state["ratings"])
    st.progress(done/total)
    st.caption(f"{done}/{total} completados")

    for section, data in SERVICE_BLOCKS.items():
        with st.expander(section, expanded=False):
            for k, l, h in data["items"]:
                render_csat_input(k, l, h)
                st.divider()
    st.markdown("</div>", unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("Atrás"):
            st.session_state["nav"] = "Datos Vuelo"
            st.rerun()
    with c2:
        if st.button("Ejecutar Predicción", type="primary"):
            st.session_state["nav"] = "Resultados"
            st.rerun()

def render_results():
    st.markdown("<div class='ap-card'><div class='ap-card-title'>Predicción</div>", unsafe_allow_html=True)
    
    # Auto-run prediction on load
    with st.spinner("Analizando datos..."):
        try:
            data = get_passenger_data()
            result = predict_satisfaction(data)
            
            c1, c2 = st.columns(2)
            pred = result.get("prediction", "Error")
            prob = result.get("probability_satisfied", 0.0)
            
            with c1:
                color = "green" if pred == "satisfied" else "red"
                text = "Satisfecho" if pred == "satisfied" else "Insatisfecho/Neutral"
                st.markdown(f"<h2 style='color: {color};'>{text}</h2>", unsafe_allow_html=True)
            
            with c2:
                st.metric("Probabilidad Satisfacción", f"{prob:.1%}")
                st.progress(prob)
            
            # Save to Database
            if pred != "Error":
                try:
                    insert_prediction(
                        input_data=data.to_dict(orient='records')[0],
                        prediction_text=pred,
                        prob_satisfied=prob,
                        prob_dissatisfied=result.get("probability_dissatisfied", 0.0)
                    )
                except Exception as db_e:
                    logger.error(f"Failed to save to DB: {db_e}")
            
        except Exception as e:
            st.error(f"Error en predicción: {str(e)}")
            logger.error(e)

    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("Reiniciar Formulario"):
        st.session_state.clear()
        st.rerun()


def main():
    st.set_page_config(page_title=APP_NAME, layout="centered")
    inject_css()
    init_state()
    render_sidebar()

    # Main content
    # Banner
    try:
        banner_path = os.path.join(project_root, "assets", "banner.svg")
        if os.path.exists(banner_path):
             st.image(banner_path, use_container_width=True)
    except Exception:
        pass

    nav = st.session_state["nav"]
    # st.title(f"✈️ {APP_NAME}")

    if nav == "Datos Vuelo":
        render_flight_data()
    elif nav == "Evaluación":
        render_evaluation()
    elif nav == "Resultados":
        render_results()

if __name__ == "__main__":
    main()
