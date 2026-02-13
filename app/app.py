import streamlit as st 
import pandas as pd
import sys
import os
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
    1: "Muy insatisfecho",
    2: "Insatisfecho",
    3: "Neutral",
    4: "Satisfecho",
    5: "Muy satisfecho",
    0: "No aplicable",
}

# I grouped the service ratings by passenger journey moments to reduce cognitive load.
# This matches how CX teams think about the experience (digital → airport → onboard).
SERVICE_BLOCKS = {
    "Digital y Conveniencia": {
        "description": "Evalúe la experiencia digital y de horarios.",
        "items": [
            ("inflight_wifi_service", "Servicio Wi-Fi a bordo", "Califique la calidad y disponibilidad del Wi-Fi."),
            ("ease_of_online_booking", "Facilidad de reserva online", "Califique qué tan fácil fue reservar en línea."),
            ("online_boarding", "Embarque online", "Califique el proceso de tarjeta de embarque digital y abordaje."),
            ("departure_arrival_time_convenient", "Conveniencia de horarios", "Califique la conveniencia de la hora de salida/llegada."),
        ],
    },
    "Procesos en Aeropuerto": {
        "description": "Evalúe los puntos de contacto en el aeropuerto.",
        "items": [
            ("gate_location", "Ubicación de la puerta", "Califique qué tan conveniente fue la ubicación de la puerta."),
            ("checkin_service", "Servicio de Check-in", "Califique la experiencia y servicio en el check-in."),
            ("baggage_handling", "Manejo de equipaje", "Califique la eficiencia y fiabilidad del manejo de equipaje."),
        ],
    },
    "Confort a Bordo": {
        "description": "Evalúe el confort físico a bordo.",
        "items": [
            ("seat_comfort", "Comodidad del asiento", "Califique la comodidad percibida del asiento."),
            ("leg_room_service", "Espacio para piernas", "Califique el espacio disponible para las piernas."),
            ("cleanliness", "Limpieza", "Califique la limpieza e higiene de la cabina."),
        ],
    },
    "Servicio y Experiencia": {
        "description": "Evalúe la calidad del servicio y entretenimiento.",
        "items": [
            ("food_and_drink", "Comida y bebida", "Califique la calidad de alimentos y bebidas."),
            ("inflight_entertainment", "Entretenimiento a bordo", "Califique el sistema de entretenimiento y contenido."),
            ("onboard_service", "Servicio a bordo", "Califique la atención y soporte de la tripulación."),
            ("inflight_service", "Servicio en vuelo", "Califique la calidad general del servicio durante el vuelo."),
        ],
    },
}


# =============================================================================
# Styling (clean dashboard feel + clear hierarchy)
# =============================================================================

def inject_css() -> None:
    """
    Lightweight CSS to make Streamlit look more like a clean SaaS dashboard.
    I’m keeping it minimal so it doesn’t fight with Streamlit defaults.
    """
    st.markdown(
        """
        <style>
          /* Design tokens (kept small so it's easy to tweak later) */
          :root{
            --ap-bg: #ffffff;
            --ap-surface: #ffffff;
            --ap-text: rgba(0,0,0,.88);
            --ap-text-muted: rgba(0,0,0,.62);
            --ap-border: rgba(0,0,0,.10);
            --ap-border-strong: rgba(0,0,0,.14);
            --ap-shadow: 0 10px 30px rgba(0,0,0,.06);
            --ap-shadow-soft: 0 6px 18px rgba(0,0,0,.06);
            --ap-radius: 16px;
            --ap-radius-sm: 12px;
            --ap-focus: rgba(31,60,136,.28);
            --ap-accent: #1F3C88;
            --ap-accent-soft: rgba(31,60,136,.08);
            --ap-success: rgba(18,125,73,.16);
            --ap-warning: rgba(176,116,0,.16);
            --ap-danger: rgba(176,30,30,.16);
            --ap-font: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
          }

          /* Dark mode support (so the dashboard doesn't break in system dark theme) */
          @media (prefers-color-scheme: dark){
            :root{
              --ap-bg: #0b0f17;
              --ap-surface: rgba(255,255,255,.06);
              --ap-text: rgba(255,255,255,.90);
              --ap-text-muted: rgba(255,255,255,.68);
              --ap-border: rgba(255,255,255,.12);
              --ap-border-strong: rgba(255,255,255,.18);
              --ap-shadow: 0 10px 30px rgba(0,0,0,.35);
              --ap-shadow-soft: 0 6px 18px rgba(0,0,0,.28);
              --ap-focus: rgba(120,155,255,.30);
              --ap-accent: #8FB2FF;
              --ap-accent-soft: rgba(143,178,255,.12);
            }
          }

          /* Base typography + smoothing */
          html, body, [class*="stApp"]{
            font-family: var(--ap-font);
            color: var(--ap-text);
            text-rendering: optimizeLegibility;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
          }

          /* Layout */
          .block-container{
            padding-top: 1.25rem;
            padding-bottom: 2.5rem;
            max-width: 1180px;
          }

          /* Subtle app background so cards feel grounded */
          [data-testid="stAppViewContainer"]{
            background: radial-gradient(1200px 600px at 15% 0%, var(--ap-accent-soft), transparent 60%),
                        radial-gradient(900px 500px at 85% 10%, rgba(0,0,0,.03), transparent 55%),
                        var(--ap-bg);
          }
          @media (prefers-color-scheme: dark){
            [data-testid="stAppViewContainer"]{
              background: radial-gradient(1200px 600px at 15% 0%, rgba(143,178,255,.10), transparent 60%),
                          radial-gradient(900px 500px at 85% 10%, rgba(255,255,255,.04), transparent 55%),
                          var(--ap-bg);
            }
          }

          /* Header badge */
          .ap-badge{
            display:inline-flex;
            align-items:center;
            gap:.35rem;
            padding: 0.28rem 0.62rem;
            border-radius: 999px;
            font-size: 12px;
            border: 1px solid color-mix(in srgb, var(--ap-accent) 22%, transparent);
            background: var(--ap-accent-soft);
            color: var(--ap-accent);
            vertical-align: middle;
            margin-left: 0.5rem;
            line-height: 1;
            letter-spacing: .2px;
          }

          /* Card container */
          .ap-card{
            border: 1px solid var(--ap-border);
            border-radius: var(--ap-radius);
            padding: 16px 16px 14px 16px;
            background: var(--ap-surface);
            box-shadow: var(--ap-shadow-soft);
            backdrop-filter: saturate(120%) blur(6px);
          }
          .ap-card-title{
            font-size: 16px;
            font-weight: 650;
            margin-bottom: 2px;
            letter-spacing: .2px;
          }
          .ap-card-subtitle{
            color: var(--ap-text-muted);
            font-size: 13px;
            margin-bottom: 10px;
          }

          /* Microcopy */
          .ap-microcopy{
            color: var(--ap-text-muted);
            font-size: 12px;
            margin-top: -6px;
            margin-bottom: 8px;
          }

          /* Streamlit dividers: lighter + more consistent rhythm */
          hr{
            border: none !important;
            height: 1px !important;
            background: var(--ap-border) !important;
            margin: .85rem 0 !important;
          }

          /* Labels and captions: keep readable hierarchy */
          [data-testid="stCaptionContainer"]{
            color: var(--ap-text-muted);
          }
          label, .stMarkdown p{
            line-height: 1.45;
          }

          /* Inputs: consistent radius + focus ring (keyboard accessible) */
          .stTextInput input,
          .stNumberInput input,
          .stSelectbox div[data-baseweb="select"] > div,
          .stTextArea textarea{
            border-radius: var(--ap-radius-sm) !important;
            border-color: var(--ap-border-strong) !important;
          }

          /* Focus states: visible, not noisy */
          :is(button, input, textarea, [role="combobox"], [role="radio"], a):focus{
            outline: none !important;
          }
          :is(button, input, textarea, [role="combobox"], a):focus-visible{
            box-shadow: 0 0 0 4px var(--ap-focus) !important;
            border-radius: var(--ap-radius-sm);
          }
          /* Radio pills can be tricky; at least keep group focus readable */
          [data-testid="stRadio"] :focus-visible{
            box-shadow: 0 0 0 4px var(--ap-focus) !important;
            border-radius: 999px;
          }

          /* Primary CTA: modern pill, better hover/active, keeps Streamlit kind="primary" */
          div.stButton > button[kind="primary"]{
            border-radius: 14px !important;
            padding: 0.70rem 1.05rem !important;
            font-weight: 650 !important;
            border: 1px solid color-mix(in srgb, var(--ap-accent) 35%, transparent) !important;
            background: linear-gradient(180deg,
              color-mix(in srgb, var(--ap-accent) 92%, #ffffff 8%),
              color-mix(in srgb, var(--ap-accent) 78%, #000000 22%)
            ) !important;
          }
          div.stButton > button[kind="primary"]:hover{
            transform: translateY(-1px);
            box-shadow: var(--ap-shadow);
          }
          div.stButton > button[kind="primary"]:active{
            transform: translateY(0px);
            box-shadow: var(--ap-shadow-soft);
          }

          /* Secondary: link-style button (keeps hierarchy) */
          .ap-link-btn button{
            background: transparent !important;
            border: none !important;
            color: var(--ap-accent) !important;
            padding: 0.25rem 0.25rem !important;
            font-weight: 650 !important;
            text-decoration: underline;
            text-underline-offset: 3px;
          }
          .ap-link-btn button:hover{
            opacity: .92;
          }

          /* Expander: smoother, clearer hit area */
          details{
            border-radius: var(--ap-radius) !important;
            border: 1px solid var(--ap-border) !important;
            background: color-mix(in srgb, var(--ap-surface) 92%, transparent) !important;
            box-shadow: none !important;
            overflow: hidden;
          }
          details > summary{
            padding: .95rem 1rem !important;
            cursor: pointer;
            color: var(--ap-text) !important;
          }
          details[open] > summary{
            border-bottom: 1px solid var(--ap-border) !important;
          }

          /* Sidebar: slightly softer separation */
          [data-testid="stSidebar"]{
            border-right: 1px solid var(--ap-border) !important;
          }

          /* Reduce motion for accessibility */
          @media (prefers-reduced-motion: reduce){
            *{
              animation: none !important;
              transition: none !important;
              scroll-behavior: auto !important;
            }
            div.stButton > button[kind="primary"]:hover{
              transform: none !important;
            }
          }

          /*
            NOTE: Streamlit doesn’t expose per-option styling for radio inputs in a robust way.
            I avoided semantic colors for scores so low ratings don’t read as “error”.
          */
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# State + helper functions
# =============================================================================

def init_state() -> None:
    """
    Session state makes the multi-page experience stable and prevents losing inputs.

    I’m also intentionally not forcing “perfect defaults” for service ratings:
    the progress bar should reflect real completion, not pre-filled values.
    """
    defaults: Dict[str, Any] = {
        # Passenger & flight context
        "gender": None,
        "customer_type": None,
        "age": None,
        "type_of_travel": None,
        "class": None,
        "flight_distance": None,
        "departure_delay": 0,
        "arrival_delay": 0,

        # Service ratings (CSAT 1–5 + 0 for N/A). None means “not answered yet”.
        "ratings": {key: None for block in SERVICE_BLOCKS.values() for key, _, _ in block["items"]},

        # Navigation (default into Service Ratings because that’s the core signal)
        "nav": "Evaluación de Servicio",
    }

    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def csat_input(key: str, label: str, help_text: str) -> Optional[int]:
    """
    Likert-style CSAT input with an explicit N/A option.
    Returns:
      - 1..5 for a real rating
      - 0 for “Not applicable”
      - None if the user hasn’t selected anything yet
    """
    st.markdown(f"**{label}**")
    st.markdown(f"<div class='ap-microcopy'>{help_text}</div>", unsafe_allow_html=True)

    options = [None, 0, 1, 2, 3, 4, 5]
    option_labels = {
        None: "Seleccione...",
        0: "N/A",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
    }

    current = st.session_state["ratings"].get(key, None)

    selected = st.radio(
        label="",
        options=options,
        index=options.index(current) if current in options else 0,
        horizontal=True,
        key=f"radio_{key}",
        format_func=lambda x: option_labels.get(x, str(x)),
        label_visibility="collapsed",
    )

    if selected is None:
        st.caption("Califique de 1 (muy insatisfecho) a 5 (muy satisfecho). Use N/A si no utilizó el servicio.")
        st.session_state["ratings"][key] = None
        return None

    st.caption(f"Seleccionado: **{option_labels[selected]}** — {CSAT_LABELS.get(selected, '')}")
    st.session_state["ratings"][key] = selected
    return selected


def compute_live_csat(ratings: Dict[str, Optional[int]]) -> Tuple[Optional[float], int, int]:
    """
    Live average CSAT:
      - averages only rated values (1..5)
      - excludes 0 (N/A) and None (not answered)
    """
    values = [v for v in ratings.values() if isinstance(v, int) and v in (1, 2, 3, 4, 5)]
    avg = round(sum(values) / len(values), 2) if values else None
    return avg, len(values), len(ratings)


def compute_progress() -> int:
    """
    Completion logic:
      - Passenger profile fields are “done” if filled
      - Each rating counts as “done” if it’s answered (including N/A)

    This gives a realistic completion indicator and helps reduce form drop-off.
    """
    passenger_fields = ["gender", "customer_type", "age", "type_of_travel", "class", "flight_distance"]
    passenger_done = sum(1 for f in passenger_fields if st.session_state.get(f) not in (None, ""))

    ratings = st.session_state["ratings"]
    ratings_done = sum(1 for v in ratings.values() if v is not None)  # includes 0 = N/A

    total = len(passenger_fields) + len(ratings)
    done = passenger_done + ratings_done
    pct = int(round((done / total) * 100)) if total else 0
    return min(max(pct, 0), 100)


def validate_inputs() -> Tuple[bool, str]:
    """
    Friendly validation with a CX/ops tone.
    I’m validating only what’s needed to run a reliable prediction.
    """
    required_fields = {
        "Gender": st.session_state.get("gender"),
        "Customer type": st.session_state.get("customer_type"),
        "Age": st.session_state.get("age"),
        "Type of travel": st.session_state.get("type_of_travel"),
        "Class": st.session_state.get("class"),
        "Flight distance": st.session_state.get("flight_distance"),
    }

    missing = [name for name, val in required_fields.items() if val in (None, "")]
    if missing:
        return (
            False,
            "Por favor complete el perfil del pasajero primero: "
            + ", ".join(missing)
            + ". Esto ayuda al modelo a dar una predicción más confiable.",
        )

    # Soft requirement: we don’t need every single attribute rated, but we do need some signal.
    avg, rated_count, _ = compute_live_csat(st.session_state["ratings"])
    if rated_count == 0:
        return (
            False,
            "Para ejecutar una predicción, por favor califique al menos un atributo de servicio (o marque N/A). "
            "Las calificaciones de servicio son la señal más fuerte para predecir la satisfacción.",
        )

    # Simple numeric sanity checks
    age = st.session_state.get("age")
    if age is not None and (age < 0 or age > 120):
        return (False, "La edad parece fuera de rango. Por favor ingrese una edad válida.")

    dist = st.session_state.get("flight_distance")
    if dist is not None and dist < 0:
        return (False, "La distancia de vuelo no puede ser negativa. Verifique el valor e intente nuevamente.")

    return (True, "")


def build_input_dataframe() -> pd.DataFrame:
    """
    Creates a single-row DataFrame aligned to the dataset-style column names.

    NOTE: column names may need to be adjusted once the final pipeline expects
    snake_case vs original dataset labels. I’m keeping a clear mapping here so it’s easy to swap.
    """
    # Helper mapping for translation (UI Spanish -> Model English)
    ui_map = {
        "Mujer": "Female", "Hombre": "Male",
        "Cliente Leal": "Loyal Customer", "Cliente Desleal": "Disloyal Customer",
        "Viaje de negocios": "Business travel", "Viaje personal": "Personal travel",
        "Business": "Business", "Eco": "Eco", "Eco Plus": "Eco Plus" 
    }

    row: Dict[str, Any] = {
        "Gender": ui_map.get(st.session_state["gender"], st.session_state["gender"]),
        "Customer Type": ui_map.get(st.session_state["customer_type"], st.session_state["customer_type"]),
        "Age": st.session_state["age"],
        "Type of Travel": ui_map.get(st.session_state["type_of_travel"], st.session_state["type_of_travel"]),
        "Class": ui_map.get(st.session_state["class"], st.session_state["class"]),
        "Flight distance": st.session_state["flight_distance"],
        "Departure Delay in Minutes": st.session_state["departure_delay"],
        "Arrival Delay in Minutes": st.session_state["arrival_delay"],
    }

    # Map UI keys -> dataset-like column names
    ui_to_dataset = {
        "inflight_wifi_service": "Inflight wifi service",
        "ease_of_online_booking": "Ease of Online booking",
        "online_boarding": "Online boarding",
        "departure_arrival_time_convenient": "Departure/Arrival time convenient",
        "gate_location": "Gate location",
        "checkin_service": "Check-in service",
        "baggage_handling": "Baggage handling",
        "seat_comfort": "Seat comfort",
        "leg_room_service": "Leg room service",
        "cleanliness": "Cleanliness",
        "food_and_drink": "Food and drink",
        "inflight_entertainment": "Inflight entertainment",
        "onboard_service": "On-board service",
        "inflight_service": "Inflight service",
    }

    for ui_key, ds_key in ui_to_dataset.items():
        row[ds_key] = st.session_state["ratings"].get(ui_key)

    return pd.DataFrame([row])


def reset_all() -> None:
    """
    Clears current inputs and starts a new scenario.
    (Useful for CX managers when they want to compare “what-if” cases quickly.)
    """
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()


# =============================================================================
# UI sections (split so we can iterate without breaking everything)
# =============================================================================

def render_header() -> None:
    left, right = st.columns([3, 1])
    with left:
        st.markdown(
            f"## ✈️ {APP_NAME} <span class='ap-badge'>{MODEL_BADGE}</span>",
            unsafe_allow_html=True,
        )
        st.caption("Predicción de Satisfacción de Pasajeros — Herramienta CX")
        st.info(
            "Complete este formulario para simular una experiencia de pasajero y predecir su nivel de satisfacción. "
            "Sus calificaciones se utilizan para pronosticar la satisfacción y destacar oportunidades de mejora."
        )
    with right:
        st.metric(label="Completado", value=f"{compute_progress()}%")
        st.progress(compute_progress() / 100.0)


def render_sidebar() -> None:
    st.sidebar.markdown(f"### {APP_NAME}")
    st.sidebar.caption("Pronóstico operativo de CX para aerolíneas")

    options = ["Datos del Vuelo", "Evaluación de Servicio", "Resultado de Predicción", "Info del Modelo"]
    
    # Fallback if session state has an old/invalid value (e.g. from English version)
    if st.session_state["nav"] not in options:
        st.session_state["nav"] = "Evaluación de Servicio"

    nav = st.sidebar.radio(
        "Navegación",
        options=options,
        index=options.index(st.session_state["nav"]),
    )
    st.session_state["nav"] = nav

    st.sidebar.divider()
    st.sidebar.caption("Tip: Use **N/A** cuando no haya utilizado el servicio. No afectará el promedio CSAT.")


def render_flight_inputs() -> None:
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>Perfil del Pasajero</div>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-subtitle'>Datos de contexto del pasajero y vuelo.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.session_state["gender"] = st.radio(
            "Género",
            options=["Mujer", "Hombre"],
            index=0 if st.session_state["gender"] is None else ["Mujer", "Hombre"].index(st.session_state["gender"]),
            horizontal=True,
        )
        st.session_state["age"] = st.number_input(
            "Edad",
            min_value=0,
            max_value=120,
            value=st.session_state["age"] if st.session_state["age"] is not None else 30,
            help="Edad del pasajero en años.",
        )
        st.session_state["class"] = st.selectbox(
            "Clase",
            options=["Business", "Eco", "Eco Plus"],
            index=0 if st.session_state["class"] is None else ["Business", "Eco", "Eco Plus"].index(st.session_state["class"]),
            help="Clase de viaje para este trayecto.",
        )
        st.session_state["departure_delay"] = st.number_input(
            "Retraso Salida (minutos)",
            min_value=0,
            value=int(st.session_state["departure_delay"] or 0),
            help="Minutos de retraso en la salida.",
        )

    with c2:
        st.session_state["customer_type"] = st.selectbox(
            "Tipo de cliente",
            options=["Cliente Leal", "Cliente Desleal"],
            index=0 if st.session_state["customer_type"] is None else ["Cliente Leal", "Cliente Desleal"].index(st.session_state["customer_type"]),
            help="Estado de relación con el cliente.",
        )
        st.session_state["type_of_travel"] = st.radio(
            "Tipo de viaje",
            options=["Viaje de negocios", "Viaje personal"],
            index=0 if st.session_state["type_of_travel"] is None else ["Viaje de negocios", "Viaje personal"].index(st.session_state["type_of_travel"]),
            horizontal=True,
        )
        st.session_state["flight_distance"] = st.number_input(
            "Distancia de vuelo",
            min_value=0,
            value=int(st.session_state["flight_distance"] or 1000),
            help="Distancia total del vuelo (unidad alineada con el dataset).",
        )
        st.session_state["arrival_delay"] = st.number_input(
            "Retraso Llegada (minutos)",
            min_value=0,
            value=int(st.session_state["arrival_delay"] or 0),
            help="Minutos de retraso en la llegada.",
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_service_ratings() -> None:
    avg, rated_count, total_items = compute_live_csat(st.session_state["ratings"])

    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>Calificación de Calidad de Servicio</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ap-card-subtitle'>Califique la calidad esperada del servicio. Estas calificaciones alimentan la predicción de satisfacción.</div>",
        unsafe_allow_html=True,
    )

    if avg is None:
        st.metric(
            "Puntaje promedio de servicio (en vivo)",
            "—",
            help="Calculado solo con ítems calificados (1–5). N/A está excluido.",
        )
    else:
        st.metric(
            "Puntaje promedio de servicio (en vivo)",
            f"{avg} / 5",
            help="Calculado solo con ítems calificados (1–5). N/A está excluido.",
        )

    st.caption(f"Ítems calificados: {rated_count} de {total_items}. Puede marcar cualquiera como N/A si no aplica.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Accordion pattern to reduce fatigue on long forms (works great with Streamlit expanders)
    for section_title, section in SERVICE_BLOCKS.items():
        with st.expander(f"{section_title} — {section['description']}", expanded=False):
            st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='ap-card-title'>{section_title}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='ap-card-subtitle'>{section['description']}</div>", unsafe_allow_html=True)

            for key, label, help_text in section["items"]:
                csat_input(key, label, help_text)
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


def render_model_info() -> None:
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>Información del Modelo</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ap-card-subtitle'>Placeholders para documentación del modelo. Actualizaré esto una vez que el modelo final esté integrado.</div>",
        unsafe_allow_html=True,
    )

    st.write("**Qué hace esta herramienta**")
    st.write("- Predice la satisfacción del pasajero antes de cerrar un registro de vuelo.")
    st.write("- Destaca qué atributos de servicio están impulsando la satisfacción hacia arriba o abajo (una vez conectada la explicabilidad).")

    st.write("**Manejo de CSAT**")
    st.write("- Las calificaciones usan una escala Likert de 1–5 (Muy insatisfecho → Muy satisfecho).")
    st.write("- **0 se trata como No aplicable / No usado** (no es un puntaje bajo).")

    st.write("**Notas operativas**")
    st.write("- Para mejores resultados, califique los atributos más relevantes para el viaje del pasajero.")
    st.write("- Use N/A cuando un atributo no aplique, en lugar de forzar un puntaje.")

    st.markdown("</div>", unsafe_allow_html=True)


def render_actions() -> None:
    """
    CTA hierarchy:
      - Run prediction = primary (high emphasis)
      - Reset = secondary link style (doesn’t compete)
    """
    st.divider()

    left, right = st.columns([1, 1])

    with left:
        run = st.button("Ejecutar predicción", type="primary", use_container_width=True)
        st.caption("Sus datos se utilizarán para predecir la satisfacción del pasajero.")

    with right:
        st.markdown("<div class='ap-link-btn'>", unsafe_allow_html=True)
        reset = st.button("Reiniciar formulario", use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Borra todos los campos e inicia un nuevo escenario.")

    if reset:
        reset_all()

    # When the user clicks “Run prediction”, we take them straight to the Result page.
    if run:
        st.session_state["nav"] = "Resultado de Predicción"
        st.rerun()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    # Resolve assets path relative to this script
    assets_dir = os.path.join(os.path.dirname(__file__), "assets", "brand")
    favicon_path = os.path.join(assets_dir, "favicon-32.png")
    logo_path = os.path.join(assets_dir, "logo-airline-predict-light.png")

    st.set_page_config(
        page_title=f"{APP_NAME} G4", 
        page_icon=favicon_path if os.path.exists(favicon_path) else "✈️", 
        layout="wide"
    )
    inject_css()
    init_state()

    # Logo in sidebar
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, use_container_width=True)
    
    render_sidebar()
    render_header()

    if st.session_state["nav"] == "Datos del Vuelo":
        render_flight_inputs()
        render_actions()

    elif st.session_state["nav"] == "Evaluación de Servicio":
        render_service_ratings()
        render_actions()

    elif st.session_state["nav"] == "Resultado de Predicción":
        render_prediction_result()
        render_actions()

    elif st.session_state["nav"] == "Info del Modelo":
        render_model_info()


if __name__ == "__main__":
    main()
