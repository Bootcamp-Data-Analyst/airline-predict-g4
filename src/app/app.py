"""
Streamlit Web Application for Airline Satisfaction Prediction.
Provides an interactive interface for model inference.
"""

import streamlit as st
import pandas as pd
import sys
import os

DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"

# Adjust path to find scripts module (repo root / src layout)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from scripts.predict import predict_with_probability, validate_input
    from scripts.logging_utils import log_prediction_result, get_prediction_stats
    from scripts.model_utils import check_model_exists, load_metrics

except ModuleNotFoundError:
    if not DEMO_MODE:
        raise

    # --- DEMO STUBS (just to see UI without pipeline) ---
    def check_model_exists():
        return True

    def load_metrics():
        return {
            "test": {"accuracy": 0.92, "precision": 0.91, "recall": 0.90, "f1": 0.905},
            "overfitting": {"is_overfitting": False},
        }

    def validate_input(input_data):
        return True, []

    def predict_with_probability(input_data, model_type="rf"):
        rating_keys = [
            "Inflight wifi service",
            "Departure/Arrival time convenient",
            "Ease of Online booking",
            "Gate location",
            "Food and drink",
            "Online boarding",
            "Seat comfort",
            "Inflight entertainment",
            "On-board service",
            "Leg room service",
            "Baggage handling",
            "Checkin service",
            "Inflight service",
            "Cleanliness",
        ]
        vals = [input_data.get(k) for k in rating_keys if isinstance(input_data.get(k), (int, float))]
        avg = sum(vals) / len(vals) if vals else 3.0
        prob_sat = min(max((avg - 1) / 4, 0.05), 0.95)

        pred = 1 if prob_sat >= 0.5 else 0
        return {
            "prediction": pred,
            "prediction_label": "satisfied" if pred == 1 else "neutral/dissatisfied",
            "confidence": prob_sat if pred == 1 else (1 - prob_sat),
            "probability_satisfied": prob_sat,
            "probability_dissatisfied": 1 - prob_sat,
        }

    def log_prediction_result(result, input_data):
        return None

    def get_prediction_stats():
        return {"total_predictions": 12, "avg_confidence": 0.78}

# =============================================================================
# App constants
# =============================================================================

APP_NAME = "Airline Predict"
MODEL_BADGE = "Model v1 — Satisfaction"

# Brand assets (already in repo)
ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src", "app", "assets"))
LOGO_LIGHT = os.path.join(ASSETS_DIR, "logo-airline-predict-light.png")
LOGO_DARK = os.path.join(ASSETS_DIR, "logo-airline-predict-dark.png")
LOGO_ICON = os.path.join(ASSETS_DIR, "logo-airline-predict-icon.png")
FAVICON_32 = os.path.join(ASSETS_DIR, "favicon-32.png")


def file_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


def set_page() -> None:
    """
    I keep this in one place so it’s easy to tweak later.
    - If favicon exists, I use it.
    - Otherwise I fall back to an emoji to avoid breaking the app.
    """
    page_icon = FAVICON_32 if file_exists(FAVICON_32) else "✈️"
    st.set_page_config(
        page_title=f"{APP_NAME} — Predictor",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


# =============================================================================
# Styling (from my Streamlit UX version)
# =============================================================================

def inject_css() -> None:
    """
    I’m using a small set of design tokens so we keep consistency across:
    - cards
    - buttons
    - focus states
    - dark mode

    I’m keeping Streamlit defaults as much as possible so we don’t fight the framework.
    """
    st.markdown(
        """
        <style>
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
            --ap-font: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, "Apple Color Emoji", "Segoe UI Emoji";
          }

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

          html, body, [class*="stApp"]{
            font-family: var(--ap-font);
            color: var(--ap-text);
            text-rendering: optimizeLegibility;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
          }

          .block-container{
            padding-top: 1.15rem;
            padding-bottom: 2.25rem;
            max-width: 1180px;
          }

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

          .ap-microcopy{
            color: var(--ap-text-muted);
            font-size: 12px;
            margin-top: -6px;
            margin-bottom: 8px;
          }

          hr{
            border: none !important;
            height: 1px !important;
            background: var(--ap-border) !important;
            margin: .85rem 0 !important;
          }

          [data-testid="stCaptionContainer"]{
            color: var(--ap-text-muted);
          }

          .stTextInput input,
          .stNumberInput input,
          .stSelectbox div[data-baseweb="select"] > div,
          .stTextArea textarea{
            border-radius: var(--ap-radius-sm) !important;
            border-color: var(--ap-border-strong) !important;
          }

          :is(button, input, textarea, [role="combobox"], [role="radio"], a):focus{
            outline: none !important;
          }
          :is(button, input, textarea, [role="combobox"], a):focus-visible{
            box-shadow: 0 0 0 4px var(--ap-focus) !important;
            border-radius: var(--ap-radius-sm);
          }
          [data-testid="stRadio"] :focus-visible{
            box-shadow: 0 0 0 4px var(--ap-focus) !important;
            border-radius: 999px;
          }

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

          [data-testid="stSidebar"]{
            border-right: 1px solid var(--ap-border) !important;
          }

          /* Brand header */
          .ap-header{
            display:flex;
            align-items:center;
            gap:12px;
            margin-bottom: .25rem;
          }
          .ap-logo{
            height: 34px;
            width: auto;
            display:block;
          }
          .ap-title{
            font-size: 1.45rem;
            font-weight: 750;
            letter-spacing: .2px;
            margin: 0;
            line-height: 1.1;
          }
          .ap-subtitle{
            color: var(--ap-text-muted);
            margin-top: .25rem;
          }

          /* Prediction result cards */
          .ap-result{
            border-radius: var(--ap-radius);
            border: 1px solid var(--ap-border);
            padding: 16px;
            background: var(--ap-surface);
            box-shadow: var(--ap-shadow-soft);
          }
          .ap-result h3{
            margin: 0 0 8px 0;
          }

          @media (max-width: 640px){
            .block-container{ padding-top: .85rem; }
            .ap-logo{ height: 30px; }
            .ap-title{ font-size: 1.25rem; }
          }

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
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header() -> None:
    """
    I’m showing a light/dark logo based on user theme.
    Streamlit doesn’t expose theme directly, so I load a safe default and keep layout stable.
    """
    left, right = st.columns([3, 1])

    with left:
        # Prefer SVG if present, otherwise PNG.
        logo_path = None
        if file_exists(LOGO_LIGHT):
            logo_path = LOGO_LIGHT
        elif file_exists(LOGO_ICON):
            logo_path = LOGO_ICON

        if logo_path:
            st.markdown("<div class='ap-header'>", unsafe_allow_html=True)
            st.image(logo_path, width=160)
            st.markdown(
                f"<div><div class='ap-title'>{APP_NAME} <span class='ap-badge'>{MODEL_BADGE}</span></div>"
                f"<div class='ap-subtitle'>Passenger Satisfaction Predictor — CX Tool</div></div>",
                unsafe_allow_html=True,
            )
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.markdown(
                f"## ✈️ {APP_NAME} <span class='ap-badge'>{MODEL_BADGE}</span>",
                unsafe_allow_html=True,
            )
            st.caption("Passenger Satisfaction Predictor — CX Tool")

        st.info(
            "Fill this form to simulate a passenger experience and predict satisfaction level. "
            "Service ratings are the strongest signal for satisfaction forecasting."
        )

    with right:
        # I keep this simple: completion is “how much of the form is filled”.
        # For now I’m tracking only required passenger fields + service sliders.
        # If we want, we can refine this later into a proper progress model.
        st.metric("Session", "Ready")


# =============================================================================
# Main app (Mariana’s model logic stays intact)
# =============================================================================

def main():
    """Main Application Function."""

    render_brand_header()

    # Check Model Availability (unchanged)
    if not check_model_exists():
        st.error(
            """
            ⚠️ **Model Not Found**

            Please train the model first by running:
            ```bash
            python scripts/train_model.py
            ```
            """
        )
        st.stop()

    # Sidebar (same content, new layout style)
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.caption("Choose which model to use for inference.")

        model_options = ["Random Forest"]
       # Check if NN model exists (optional)
if not DEMO_MODE:
    from scripts.paths import MODEL_NN_PATH
    if os.path.exists(MODEL_NN_PATH):
        model_options.append("Neural Network (Keras)")
else:
    # Demo: keep only RF option to avoid missing pipeline modules
    pass

        model_choice = st.selectbox("Select Model", model_options)
        model_type = "rf" if model_choice == "Random Forest" else "nn"

        st.divider()
        st.markdown("### ℹ️ Information")

        metrics = load_metrics()
        if metrics:
            st.markdown("#### 📊 Model Metrics")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Accuracy", f"{metrics['test']['accuracy']:.2%}")
                st.metric("Precision", f"{metrics['test']['precision']:.2%}")
            with col2:
                st.metric("Recall", f"{metrics['test']['recall']:.2%}")
                st.metric("F1 Score", f"{metrics['test']['f1']:.2%}")

            if metrics.get("overfitting", {}).get("is_overfitting"):
                st.warning("⚠️ Potential Overfitting")
            else:
                st.success("✅ Robust Model")

        st.divider()

        stats = get_prediction_stats()
        if stats:
            st.markdown("#### 📈 Usage Stats")
            st.metric("Total Predictions", stats["total_predictions"])
            if "avg_confidence" in stats:
                st.metric("Avg Confidence", f"{stats['avg_confidence']:.2%}")

        st.divider()
        st.markdown("#### 🕵️ MLOps Monitoring")

        @st.cache_data(ttl=600)
        def get_drift_status_cached():
            from scripts.monitor import check_data_drift
            return check_data_drift()

        drift_report = get_drift_status_cached()

        st.write(f"Status: **{drift_report.get('status', 'Unknown')}**")
        st.caption(f"Based on recent {drift_report.get('production_samples', 0)} samples vs Training Data")

        if drift_report.get("drift_detected"):
            st.error("⚠️ DATA DRIFT DETECTED")
            st.write("Variables affected:")
            for feature in drift_report.get("drifting_features", []):
                st.write(f"- {feature}")
        else:
            st.success("✅ Distribution Stable")

        st.divider()
        st.caption("Developed by Airline Predict G4")
        st.caption("Dataset: Airlines Customer Satisfaction")

    # Main Form
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>📝 Passenger & Flight Details</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ap-card-subtitle'>Complete the passenger profile and rate service attributes to run a prediction.</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["👤 Personal Info", "✈️ Flight Info", "⭐ Service Ratings"])

    with tab1:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>Personal Info</div>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-subtitle'>Passenger profile used by the model.</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        gender = col1.selectbox("Gender", ["Male", "Female"])
        customer_type = col2.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
        age = col3.number_input("Age", 1, 120, 35)

        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>Flight Info</div>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-subtitle'>Trip context and operational delays.</div>", unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        travel_type = col1.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
        travel_class = col2.selectbox("Class", ["Business", "Eco Plus", "Eco"])
        flight_distance = col3.number_input("Flight Distance", 0, 10000, 1500)

        col4, col5 = st.columns(2)
        departure_delay = col4.number_input("Departure Delay (min)", 0, 2000, 0)
        arrival_delay = col5.number_input("Arrival Delay (min)", 0.0, 2000.0, 0.0)

        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>Service Ratings</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='ap-card-subtitle'>Rate from 0 (poor) to 5 (excellent). Use 0 if the service was not used.</div>",
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)

        with col1:
            wifi = st.slider("Inflight Wifi", 0, 5, 3)
            time_convenient = st.slider("Time Convenient", 0, 5, 3)
            online_booking = st.slider("Online Booking", 0, 5, 3)
            gate_location = st.slider("Gate Location", 0, 5, 3)
            food_drink = st.slider("Food and Drink", 0, 5, 3)
            online_boarding = st.slider("Online Boarding", 0, 5, 3)
            seat_comfort = st.slider("Seat Comfort", 0, 5, 4)

        with col2:
            entertainment = st.slider("Inflight Entertainment", 0, 5, 4)
            onboard_service = st.slider("On-board Service", 0, 5, 4)
            leg_room = st.slider("Leg Room Service", 0, 5, 3)
            baggage = st.slider("Baggage Handling", 0, 5, 4)
            checkin = st.slider("Checkin Service", 0, 5, 4)
            inflight_service = st.slider("Inflight Service", 0, 5, 4)
            cleanliness = st.slider("Cleanliness", 0, 5, 4)

        st.markdown("</div>", unsafe_allow_html=True)

    # Input Dictionary (unchanged)
    input_data = {
        "Gender": gender,
        "Customer Type": customer_type,
        "Age": age,
        "Type of Travel": travel_type,
        "Class": travel_class,
        "Flight Distance": flight_distance,
        "Inflight wifi service": wifi,
        "Departure/Arrival time convenient": time_convenient,
        "Ease of Online booking": online_booking,
        "Gate location": gate_location,
        "Food and drink": food_drink,
        "Online boarding": online_boarding,
        "Seat comfort": seat_comfort,
        "Inflight entertainment": entertainment,
        "On-board service": onboard_service,
        "Leg room service": leg_room,
        "Baggage handling": baggage,
        "Checkin service": checkin,
        "Inflight service": inflight_service,
        "Cleanliness": cleanliness,
        "Departure Delay in Minutes": departure_delay,
        "Arrival Delay in Minutes": arrival_delay,
    }

    st.divider()

    # Predict Button (unchanged logic, improved hierarchy)
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    if col_btn2.button("🔮 Predict Satisfaction", type="primary", use_container_width=True):
        is_valid, errors = validate_input(input_data)

        if not is_valid:
            st.error("❌ Validation Error:")
            for error in errors:
                st.write(f"- {error}")
        else:
            with st.spinner(f"Analyzing with {model_choice}..."):
                result = predict_with_probability(input_data, model_type=model_type)

                if result.get("error"):
                    st.error(f"❌ Prediction Error: {result['error']}")
                else:
                    log_prediction_result(result, input_data)

                    st.divider()
                    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
                    st.markdown("<div class='ap-card-title'>🎯 Prediction Result</div>", unsafe_allow_html=True)
                    st.markdown(
                        "<div class='ap-card-subtitle'>This is the predicted satisfaction label and confidence.</div>",
                        unsafe_allow_html=True,
                    )

                    conf = result["confidence"]

                    if result["prediction"] == 1:
                        st.markdown(
                            f"""
                            <div class="ap-result">
                              <h3>😊 SATISFIED CUSTOMER</h3>
                              <div class="ap-microcopy">Confidence: {conf:.1%}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
                        st.balloons()
                    else:
                        st.markdown(
                            f"""
                            <div class="ap-result">
                              <h3>😞 NEUTRAL OR DISSATISFIED</h3>
                              <div class="ap-microcopy">Confidence: {conf:.1%}</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )

                    st.divider()
                    st.subheader("📊 Probability Breakdown")
                    col1, col2 = st.columns(2)
                    col1.metric("Satisfied Probability", f"{result['probability_satisfied']:.1%}")
                    col2.metric("Dissatisfied Probability", f"{result['probability_dissatisfied']:.1%}")

                    st.progress(result["probability_satisfied"])

                    with st.expander("📋 View Input Data"):
                        st.json(input_data)

                    st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.caption("Airline Predict G4 — Satisfaction Classification Model")
    st.caption("Powered by Streamlit, Scikit-learn & Optuna")


if __name__ == "__main__":
    set_page()
    inject_css()
    main()
