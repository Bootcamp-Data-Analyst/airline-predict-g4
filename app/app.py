import streamlit as st 
import pandas as pd
import sys
import os
import logging
from typing import Dict, Any, Optional, Tuple

# NOTE: `predict` is provided by the modeling/pipeline workstream.
# Right now I’m wiring the UI to the expected entry point, and we’ll adapt
# the input schema/return format once the final pipeline contract is confirmed.
# NOTE: `predict` is provided by the modeling/pipeline workstream.
# Right now I’m wiring the UI to the expected entry point, and we’ll adapt
# the input schema/return format once the final pipeline contract is confirmed.
try:
    from scripts.predict import predict_with_probability

    def predict(data):
        """
        Wrapper to adapt the UI contract to the script contract.
        UI expects: {prediction, confidence, average_csat, top_drivers}
        Script returns: {prediction, prediction_numeric, probability_satisfied, probability_dissatisfied}
        """
        # Convert DataFrame to dict if necessary (though script handles both, dict is safer for single row)
        if hasattr(data, "to_dict"):
            data = data.to_dict(orient="records")[0]
            
        result = predict_with_probability(data)
        
        # Map response keys
        pred_label = result.get("prediction", "Unknown")
        
        # Calculate confidence based on the predicted class
        if pred_label == 'satisfied':
            conf = result.get("probability_satisfied", 0.0)
        else:
            conf = result.get("probability_dissatisfied", 0.0)
            
        return {
            "prediction": pred_label,
            "confidence": conf,
            "average_csat": None, # Not calculated by model yet
            "top_drivers": []     # Not returned by model yet
        }

except ImportError:
    # Fallback for development if scripts module is not found or predict is missing
    def predict(data):
        return {"prediction": "Mock Prediction", "confidence": 0.0, "average_csat": 0.0}


# =============================================================================
# UX / Copy constants (kept here to keep the UI consistent and easy to tweak)
# =============================================================================

APP_NAME = "Airline Predict"
MODEL_BADGE = "Model v1 — Classification"

# CSAT semantics (1–5). We also support 0 = Not applicable.
# IMPORTANT: 0 is NOT a “bad score”. In airline survey datasets it typically means:
# “Not used / Not rated / Not applicable”.
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



# =============================================================================
# UI sections (split so we can iterate without breaking everything)
# =============================================================================

def render_sidebar():
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


def render_prediction_result() -> None:
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>Resultado de Predicción</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ap-card-subtitle'>Salida del modelo y factores clave. Use esto para priorizar mejoras de CX.</div>",
        unsafe_allow_html=True,
    )

    valid, msg = validate_inputs()
    if not valid:
        st.warning(msg)
        st.markdown("</div>", unsafe_allow_html=True)
        return

    # Build input row (encoding/scaling/missing handling happens in the pipeline)
    X_input = build_input_dataframe()

    try:
        # NOTE: This is the integration point.
        # When the modeling team finalizes the pipeline, we’ll align:
        #   - expected feature names
        #   - missing value treatment (0 -> N/A, None -> missing)
        #   - return format (label + proba + drivers)
        result = predict(X_input)

        # For now I handle both:
        #  - a dict contract (recommended)
        #  - a simple label (fallback)
        if isinstance(result, dict):
            pred = result.get("prediction", "—")
            conf = result.get("confidence", None)  # expected 0..1
            csat_avg = result.get("average_csat", None)
            drivers = result.get("top_drivers", [])
        else:
            pred = str(result)
            conf = None
            csat_avg, _, _ = compute_live_csat(st.session_state["ratings"])
            drivers = []

        c1, c2, c3 = st.columns([1.2, 1.0, 1.2])
        with c1:
            st.metric("Predicción", pred)
        with c2:
            if conf is None:
                st.metric("Confianza", "—")
            else:
                st.metric("Confianza", f"{int(round(conf * 100))}%")
                st.progress(min(max(conf, 0.0), 1.0))
        with c3:
            if csat_avg is None:
                st.metric("CSAT Promedio", "—")
            else:
                st.metric("CSAT Promedio", f"{csat_avg} / 5")

        st.divider()
        st.subheader("Factores principales (explicabilidad)")
        if drivers:
            for d in drivers[:5]:
                st.write(f"• {d}")
        else:
            st.caption(
                "Los insights de factores aparecerán aquí una vez que el pipeline exponga la importancia de características / salidas SHAP. "
                "La UI ya está preparada para ello."
            )

        st.success("Predicción generada. Puede ajustar las calificaciones para probar diferentes escenarios.")

    except Exception:
        # NOTE: Keeping the error message user-friendly.
        # The technical traceback can be added later to logs if we want.
        st.error(
            "No pudimos generar una predicción con la configuración actual. "
            "Por favor intente de nuevo, o verifique que el pipeline del modelo esté conectado y funcionando."
        )

    st.markdown("</div>", unsafe_allow_html=True)
    
    if st.button("Reiniciar Formulario"):
        st.session_state.clear()
        st.rerun()


def main():
    st.set_page_config(page_title=APP_NAME, layout="centered")
    inject_css()
    init_state()
    render_sidebar()

    nav = st.session_state["nav"]
    st.title(f"✈️ {APP_NAME}")

    if nav == "Datos Vuelo":
        render_flight_data()
    elif nav == "Evaluación":
        render_evaluation()
    elif nav == "Resultados":
        render_results()

if __name__ == "__main__":
    main()

