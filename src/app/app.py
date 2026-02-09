"""
Streamlit Web Application for Airline Satisfaction Prediction.
Provides an interactive interface for model inference.

NOTES (demo/debug):
- I kept hitting ModuleNotFoundError because I still had "scripts" imports firing inside main().
  Streamlit re-runs the script a lot, so the failing import line looked "random".
- I fixed it by making demo mode a hard boundary: when DEMO_MODE=1, I do not import anything
  from the pipeline layer anywhere (not at top-level, not inside main, not inside cached functions).
- This lets me review UI/UX even if src/scripts/ doesn't exist on my machine yet.
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Tuple

import streamlit as st
import pandas as pd  # kept because the app uses it in places / future expansions

# -----------------------------------------------------------------------------
# Demo mode (UI preview without pipeline)
# -----------------------------------------------------------------------------
DEMO_MODE = os.getenv("DEMO_MODE", "0") == "1"

# Project layout: repo_root/src/app/app.py
# Real pipeline modules would live under repo_root/src/scripts/*
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SRC_DIR = os.path.join(REPO_ROOT, "src")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

# -----------------------------------------------------------------------------
# Assets (these DO exist in the repo)
# src/app/assets/*
# -----------------------------------------------------------------------------
ASSETS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "assets"))
LOGO_LIGHT = os.path.join(ASSETS_DIR, "logo-airline-predict-light.png")
LOGO_DARK = os.path.join(ASSETS_DIR, "logo-airline-predict-dark.png")
LOGO_ICON = os.path.join(ASSETS_DIR, "logo-airline-predict-icon.png")
FAVICON_32 = os.path.join(ASSETS_DIR, "favicon-32.png")

APP_NAME = "Airline Predict"
MODEL_BADGE = "Model v1 — Satisfaction"


def file_exists(path: str) -> bool:
    try:
        return os.path.exists(path)
    except Exception:
        return False


# -----------------------------------------------------------------------------
# Pipeline hooks (real vs demo)
# -----------------------------------------------------------------------------
if DEMO_MODE:
    # --- DEMO STUBS ---
    def check_model_exists() -> bool:
        return True

    def load_metrics() -> Dict[str, Any]:
        return {
            "test": {"accuracy": 0.92, "precision": 0.91, "recall": 0.90, "f1": 0.905},
            "overfitting": {"is_overfitting": False},
        }

    def validate_input(input_data: Dict[str, Any]) -> Tuple[bool, List[str]]:
        return True, []

    def predict_with_probability(input_data: Dict[str, Any], model_type: str = "rf") -> Dict[str, Any]:
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

    def log_prediction_result(result: Dict[str, Any], input_data: Dict[str, Any]) -> None:
        return None

    def get_prediction_stats() -> Dict[str, Any]:
        return {"total_predictions": 12, "avg_confidence": 0.78}

    def get_drift_report() -> Dict[str, Any]:
        return {
            "status": "Demo",
            "production_samples": 0,
            "drift_detected": False,
            "drifting_features": [],
        }

else:
    # --- REAL PIPELINE IMPORTS ---
    # If these don't exist, I want it to fail loudly because that's a real-mode issue.
    from scripts.predict import predict_with_probability, validate_input
    from scripts.logging_utils import log_prediction_result, get_prediction_stats
    from scripts.model_utils import check_model_exists, load_metrics
    from scripts.paths import MODEL_NN_PATH  # used only in real mode
    from scripts.monitor import check_data_drift

    @st.cache_data(ttl=600)
    def get_drift_report() -> Dict[str, Any]:
        return check_data_drift()


# -----------------------------------------------------------------------------
# Page + CSS
# -----------------------------------------------------------------------------
def set_page() -> None:
    page_icon = FAVICON_32 if file_exists(FAVICON_32) else "✈️"
    st.set_page_config(
        page_title=f"{APP_NAME} — Predictor",
        page_icon=page_icon,
        layout="wide",
        initial_sidebar_state="expanded",
    )


def inject_css() -> None:
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
              --ap-shadow-soft: 0 6px 18px rgba(0,0,0,.28);
              --ap-focus: rgba(120,155,255,.30);
              --ap-accent: #8FB2FF;
              --ap-accent-soft: rgba(143,178,255,.12);
            }
          }
          html, body, [class*="stApp"]{ font-family: var(--ap-font); color: var(--ap-text); }
          .block-container{ padding-top: 1.15rem; padding-bottom: 2.25rem; max-width: 1180px; }
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
            display:inline-flex; align-items:center; gap:.35rem;
            padding: 0.28rem 0.62rem; border-radius: 999px; font-size: 12px;
            border: 1px solid color-mix(in srgb, var(--ap-accent) 22%, transparent);
            background: var(--ap-accent-soft); color: var(--ap-accent);
            margin-left: 0.5rem; line-height: 1; letter-spacing: .2px;
          }
          .ap-card{
            border: 1px solid var(--ap-border); border-radius: var(--ap-radius);
            padding: 16px 16px 14px 16px; background: var(--ap-surface);
            box-shadow: var(--ap-shadow-soft); backdrop-filter: saturate(120%) blur(6px);
          }
          .ap-card-title{ font-size: 16px; font-weight: 650; margin-bottom: 2px; letter-spacing: .2px; }
          .ap-card-subtitle{ color: var(--ap-text-muted); font-size: 13px; margin-bottom: 10px; }
          .ap-microcopy{ color: var(--ap-text-muted); font-size: 12px; margin-top: -6px; margin-bottom: 8px; }
          hr{ border: none !important; height: 1px !important; background: var(--ap-border) !important; margin: .85rem 0 !important; }
          .stTextInput input, .stNumberInput input, .stTextArea textarea{
            border-radius: var(--ap-radius-sm) !important; border-color: var(--ap-border-strong) !important;
          }
          :is(button, input, textarea, [role="combobox"], [role="radio"], a):focus{ outline: none !important; }
          :is(button, input, textarea, [role="combobox"], a):focus-visible{
            box-shadow: 0 0 0 4px var(--ap-focus) !important; border-radius: var(--ap-radius-sm);
          }
          div.stButton > button[kind="primary"]{
            border-radius: 14px !important; padding: 0.70rem 1.05rem !important; font-weight: 650 !important;
            border: 1px solid color-mix(in srgb, var(--ap-accent) 35%, transparent) !important;
            background: linear-gradient(180deg,
              color-mix(in srgb, var(--ap-accent) 92%, #ffffff 8%),
              color-mix(in srgb, var(--ap-accent) 78%, #000000 22%)
            ) !important;
          }
          .ap-header{ display:flex; align-items:center; gap:12px; margin-bottom: .25rem; }
          .ap-title{ font-size: 1.45rem; font-weight: 750; letter-spacing: .2px; margin: 0; line-height: 1.1; }
          .ap-subtitle{ color: var(--ap-text-muted); margin-top: .25rem; }
          .ap-result{
            border-radius: var(--ap-radius); border: 1px solid var(--ap-border);
            padding: 16px; background: var(--ap-surface); box-shadow: var(--ap-shadow-soft);
          }
          .ap-result h3{ margin: 0 0 8px 0; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_brand_header() -> None:
    left, right = st.columns([3, 1])

    with left:
        logo_path = LOGO_LIGHT if file_exists(LOGO_LIGHT) else (LOGO_ICON if file_exists(LOGO_ICON) else None)

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
            st.markdown(f"## ✈️ {APP_NAME} <span class='ap-badge'>{MODEL_BADGE}</span>", unsafe_allow_html=True)
            st.caption("Passenger Satisfaction Predictor — CX Tool")

        st.info(
            "Fill this form to simulate a passenger experience and predict satisfaction level. "
            "Service ratings are the strongest signal for satisfaction forecasting."
        )

    with right:
        st.metric("Session", "Ready")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    render_brand_header()

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

    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        st.caption("Choose which model to use for inference.")

        model_options = ["Random Forest"]

        # Only in real mode do I even consider NN assets/paths
        if not DEMO_MODE:
            try:
                if os.path.exists(MODEL_NN_PATH):
                    model_options.append("Neural Network (Keras)")
            except Exception:
                pass

        model_choice = st.selectbox("Select Model", model_options)
        model_type = "rf" if model_choice == "Random Forest" else "nn"

        st.divider()
        st.markdown("### ℹ️ Information")

        metrics = load_metrics()
        if metrics:
            st.markdown("#### 📊 Model Metrics")
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Accuracy", f"{metrics['test']['accuracy']:.2%}")
                st.metric("Precision", f"{metrics['test']['precision']:.2%}")
            with c2:
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
            st.metric("Total Predictions", stats.get("total_predictions", 0))
            if "avg_confidence" in stats:
                st.metric("Avg Confidence", f"{stats['avg_confidence']:.2%}")

        st.divider()
        st.markdown("#### 🕵️ MLOps Monitoring")

        drift_report = get_drift_report()
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

    # Main content
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

        c1, c2, c3 = st.columns(3)
        gender = c1.selectbox("Gender", ["Male", "Female"])
        customer_type = c2.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"])
        age = c3.number_input("Age", 1, 120, 35)

        st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>Flight Info</div>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-subtitle'>Trip context and operational delays.</div>", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        travel_type = c1.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
        travel_class = c2.selectbox("Class", ["Business", "Eco Plus", "Eco"])
        flight_distance = c3.number_input("Flight Distance", 0, 10000, 1500)

        c4, c5 = st.columns(2)
        departure_delay = c4.number_input("Departure Delay (min)", 0, 2000, 0)
        arrival_delay = c5.number_input("Arrival Delay (min)", 0.0, 2000.0, 0.0)

        st.markdown("</div>", unsafe_allow_html=True)

    with tab3:
        st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
        st.markdown("<div class='ap-card-title'>Service Ratings</div>", unsafe_allow_html=True)
        st.markdown(
            "<div class='ap-card-subtitle'>Rate from 0 (poor) to 5 (excellent). Use 0 if the service was not used.</div>",
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        with c1:
            wifi = st.slider("Inflight Wifi", 0, 5, 3)
            time_convenient = st.slider("Time Convenient", 0, 5, 3)
            online_booking = st.slider("Online Booking", 0, 5, 3)
            gate_location = st.slider("Gate Location", 0, 5, 3)
            food_drink = st.slider("Food and Drink", 0, 5, 3)
            online_boarding = st.slider("Online Boarding", 0, 5, 3)
            seat_comfort = st.slider("Seat Comfort", 0, 5, 4)

        with c2:
            entertainment = st.slider("Inflight Entertainment", 0, 5, 4)
            onboard_service = st.slider("On-board Service", 0, 5, 4)
            leg_room = st.slider("Leg Room Service", 0, 5, 3)
            baggage = st.slider("Baggage Handling", 0, 5, 4)
            checkin = st.slider("Checkin Service", 0, 5, 4)
            inflight_service = st.slider("Inflight Service", 0, 5, 4)
            cleanliness = st.slider("Cleanliness", 0, 5, 4)

        st.markdown("</div>", unsafe_allow_html=True)

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

    _, col_btn, _ = st.columns([1, 2, 1])
    if col_btn.button("🔮 Predict Satisfaction", type="primary", use_container_width=True):
        is_valid, errors = validate_input(input_data)

        if not is_valid:
            st.error("❌ Validation Error:")
            for err in errors:
                st.write(f"- {err}")
        else:
            with st.spinner(f"Analyzing with {model_choice}..."):
                result = predict_with_probability(input_data, model_type=model_type)

            if result.get("error"):
                st.error(f"❌ Prediction Error: {result['error']}")
                return

            log_prediction_result(result, input_data)

            st.divider()
            st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
            st.markdown("<div class='ap-card-title'>🎯 Prediction Result</div>", unsafe_allow_html=True)
            st.markdown("<div class='ap-card-subtitle'>Predicted label and confidence.</div>", unsafe_allow_html=True)

            conf = float(result.get("confidence", 0.0))

            if result.get("prediction") == 1:
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
            c1, c2 = st.columns(2)
            c1.metric("Satisfied Probability", f"{float(result.get('probability_satisfied', 0.0)):.1%}")
            c2.metric("Dissatisfied Probability", f"{float(result.get('probability_dissatisfied', 0.0)):.1%}")

            st.progress(float(result.get("probability_satisfied", 0.0)))

            with st.expander("📋 View Input Data"):
                st.json(input_data)

            st.markdown("</div>", unsafe_allow_html=True)

    st.divider()
    st.caption("Airline Predict G4 — Satisfaction Classification Model")
    st.caption("Powered by Streamlit & Scikit-learn")


if __name__ == "__main__":
    set_page()
    inject_css()
    main()
