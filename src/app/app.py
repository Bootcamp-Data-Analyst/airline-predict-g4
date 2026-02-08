import streamlit as st
import pandas as pd
import sys
import os
from typing import Dict, Any, Optional, Tuple

# NOTE: `predict` is provided by the modeling/pipeline workstream.
# Right now I’m wiring the UI to the expected entry point, and we’ll adapt
# the input schema/return format once the final pipeline contract is confirmed.
from src.pipeline.predict import predict

# =============================================================================
# UX / Copy constants (kept here to keep the UI consistent and easy to tweak)
# =============================================================================

APP_NAME = "Airline Predict"
MODEL_BADGE = "Model v1 — Classification"

# CSAT semantics (1–5). We also support 0 = Not applicable.
# IMPORTANT: 0 is NOT a “bad score”. In airline survey datasets it typically means:
# “Not used / Not rated / Not applicable”.
CSAT_LABELS = {
    1: "Very dissatisfied",
    2: "Dissatisfied",
    3: "Neutral",
    4: "Satisfied",
    5: "Very satisfied",
    0: "Not applicable",
}

# I grouped the service ratings by passenger journey moments to reduce cognitive load.
# This matches how CX teams think about the experience (digital → airport → onboard).
SERVICE_BLOCKS = {
    "Digital & Time Convenience": {
        "description": "Rate digital and scheduling experience.",
        "items": [
            ("inflight_wifi_service", "Inflight Wi-Fi service", "Rate the quality and availability of onboard Wi-Fi."),
            ("ease_of_online_booking", "Ease of online booking", "Rate how easy it was to book online."),
            ("online_boarding", "Online boarding", "Rate the digital boarding pass and boarding process."),
            ("departure_arrival_time_convenient", "Departure/arrival time convenience", "Rate schedule convenience for this trip."),
        ],
    },
    "Airport Process": {
        "description": "Rate airport touchpoints.",
        "items": [
            ("gate_location", "Gate location", "Rate how convenient the gate location was."),
            ("checkin_service", "Check-in service", "Rate the check-in experience and service quality."),
            ("baggage_handling", "Baggage handling", "Rate baggage handling efficiency and reliability."),
        ],
    },
    "Onboard Comfort": {
        "description": "Rate onboard physical comfort.",
        "items": [
            ("seat_comfort", "Seat comfort", "Rate perceived seat comfort during the flight."),
            ("leg_room_service", "Leg room", "Rate available leg room and overall space."),
            ("cleanliness", "Cleanliness", "Rate cabin cleanliness and hygiene."),
        ],
    },
    "Service & Experience": {
        "description": "Rate service quality and entertainment.",
        "items": [
            ("food_and_drink", "Food and drink", "Rate quality of food and beverages."),
            ("inflight_entertainment", "Inflight entertainment", "Rate entertainment system and content."),
            ("onboard_service", "On-board service", "Rate cabin crew service and support."),
            ("inflight_service", "Inflight service", "Rate overall inflight service quality."),
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
          /* Layout */
          .block-container { padding-top: 1.25rem; padding-bottom: 2.5rem; max-width: 1180px; }

          /* Header badge */
          .ap-badge {
            display:inline-block;
            padding: 0.25rem 0.55rem;
            border-radius: 999px;
            font-size: 12px;
            border: 1px solid rgba(31,60,136,0.20);
            background: rgba(31,60,136,0.06);
            color: #1F3C88;
            vertical-align: middle;
            margin-left: 0.5rem;
          }

          /* Card container */
          .ap-card {
            border: 1px solid rgba(0,0,0,0.08);
            border-radius: 14px;
            padding: 16px 16px 14px 16px;
            background: white;
          }
          .ap-card-title {
            font-size: 16px;
            font-weight: 650;
            margin-bottom: 2px;
          }
          .ap-card-subtitle {
            color: rgba(0,0,0,0.60);
            font-size: 13px;
            margin-bottom: 10px;
          }

          /* Microcopy */
          .ap-microcopy {
            color: rgba(0,0,0,0.55);
            font-size: 12px;
            margin-top: -6px;
            margin-bottom: 8px;
          }

          /* Primary CTA */
          div.stButton > button[kind="primary"] {
            border-radius: 12px;
            padding: 0.65rem 1.0rem;
            font-weight: 650;
          }

          /* Link-style secondary action */
          .ap-link-btn button {
            background: transparent !important;
            border: none !important;
            color: #1F3C88 !important;
            padding: 0.25rem 0.25rem !important;
            font-weight: 600 !important;
            text-decoration: underline;
          }

          /*
            NOTE: Streamlit doesn’t expose per-option styling for radio inputs in a robust way.
            To avoid using “error red” for low scores, I’m relying on:
              - clear labels + microcopy
              - live CSAT feedback
              - neutral UI colors
            If we later move to a custom component, we can add a soft semantic gradient.
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
        "nav": "Service Ratings",
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
        None: "Select…",
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
        st.caption("Rate from 1 (very dissatisfied) to 5 (very satisfied). Use N/A if the service was not used.")
        st.session_state["ratings"][key] = None
        return None

    st.caption(f"Selected: **{option_labels[selected]}** — {CSAT_LABELS.get(selected, '')}")
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
            "Please complete the passenger profile first: "
            + ", ".join(missing)
            + ". This helps the model give a more reliable prediction.",
        )

    # Soft requirement: we don’t need every single attribute rated, but we do need some signal.
    avg, rated_count, _ = compute_live_csat(st.session_state["ratings"])
    if rated_count == 0:
        return (
            False,
            "To run a prediction, please rate at least one service attribute (or set it to N/A). "
            "Service ratings are the strongest signal for satisfaction forecasting.",
        )

    # Simple numeric sanity checks
    age = st.session_state.get("age")
    if age is not None and (age < 0 or age > 120):
        return (False, "Age looks out of range. Please enter a valid passenger age.")

    dist = st.session_state.get("flight_distance")
    if dist is not None and dist < 0:
        return (False, "Flight distance can’t be negative. Please check the value and try again.")

    return (True, "")


def build_input_dataframe() -> pd.DataFrame:
    """
    Creates a single-row DataFrame aligned to the dataset-style column names.

    NOTE: column names may need to be adjusted once the final pipeline expects
    snake_case vs original dataset labels. I’m keeping a clear mapping here so it’s easy to swap.
    """
    row: Dict[str, Any] = {
        "Gender": st.session_state["gender"],
        "Customer Type": st.session_state["customer_type"],
        "Age": st.session_state["age"],
        "Type of Travel": st.session_state["type_of_travel"],
        "Class": st.session_state["class"],
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
        st.caption("Passenger Satisfaction Predictor — CX Tool")
        st.info(
            "Fill this form to simulate a passenger experience and predict satisfaction level. "
            "Your ratings are used to forecast satisfaction and highlight improvement opportunities."
        )
    with right:
        st.metric(label="Form completion", value=f"{compute_progress()}%")
        st.progress(compute_progress() / 100.0)


def render_sidebar() -> None:
    st.sidebar.markdown(f"### {APP_NAME}")
    st.sidebar.caption("Operational CX forecasting for airline teams")

    nav = st.sidebar.radio(
        "Navigation",
        options=["Flight Inputs", "Service Ratings", "Prediction Result", "Model Info"],
        index=["Flight Inputs", "Service Ratings", "Prediction Result", "Model Info"].index(st.session_state["nav"]),
    )
    st.session_state["nav"] = nav

    st.sidebar.divider()
    st.sidebar.caption("Tip: Use **N/A** when the service was not used. It won’t lower the CSAT average.")


def render_flight_inputs() -> None:
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>Passenger Profile</div>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-subtitle'>Passenger and flight context data.</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.session_state["gender"] = st.radio(
            "Gender",
            options=["Female", "Male"],
            index=0 if st.session_state["gender"] is None else ["Female", "Male"].index(st.session_state["gender"]),
            horizontal=True,
        )
        st.session_state["age"] = st.number_input(
            "Age",
            min_value=0,
            max_value=120,
            value=st.session_state["age"] if st.session_state["age"] is not None else 30,
            help="Passenger age in years.",
        )
        st.session_state["class"] = st.selectbox(
            "Class",
            options=["Business", "Eco", "Eco Plus"],
            index=0 if st.session_state["class"] is None else ["Business", "Eco", "Eco Plus"].index(st.session_state["class"]),
            help="Travel class for this trip.",
        )
        st.session_state["departure_delay"] = st.number_input(
            "Departure Delay (minutes)",
            min_value=0,
            value=int(st.session_state["departure_delay"] or 0),
            help="Minutes delayed at departure.",
        )

    with c2:
        st.session_state["customer_type"] = st.selectbox(
            "Customer type",
            options=["Loyal Customer", "Disloyal Customer"],
            index=0 if st.session_state["customer_type"] is None else ["Loyal Customer", "Disloyal Customer"].index(st.session_state["customer_type"]),
            help="Customer relationship status.",
        )
        st.session_state["type_of_travel"] = st.radio(
            "Type of travel",
            options=["Business travel", "Personal travel"],
            index=0 if st.session_state["type_of_travel"] is None else ["Business travel", "Personal travel"].index(st.session_state["type_of_travel"]),
            horizontal=True,
        )
        st.session_state["flight_distance"] = st.number_input(
            "Flight distance",
            min_value=0,
            value=int(st.session_state["flight_distance"] or 1000),
            help="Total flight distance (unit aligned with the dataset used in modeling).",
        )
        st.session_state["arrival_delay"] = st.number_input(
            "Arrival Delay (minutes)",
            min_value=0,
            value=int(st.session_state["arrival_delay"] or 0),
            help="Minutes delayed at arrival.",
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_service_ratings() -> None:
    avg, rated_count, total_items = compute_live_csat(st.session_state["ratings"])

    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>Service Quality Ratings</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ap-card-subtitle'>Rate expected service quality. These ratings feed the satisfaction prediction.</div>",
        unsafe_allow_html=True,
    )

    if avg is None:
        st.metric(
            "Average service score (live)",
            "—",
            help="Calculated from rated items only (1–5). N/A is excluded.",
        )
    else:
        st.metric(
            "Average service score (live)",
            f"{avg} / 5",
            help="Calculated from rated items only (1–5). N/A is excluded.",
        )

    st.caption(f"Rated items: {rated_count} of {total_items}. You can set any attribute to N/A if it does not apply.")
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
    st.markdown("<div class='ap-card-title'>Prediction Result</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ap-card-subtitle'>Model output and key drivers. Use this to prioritize CX improvements.</div>",
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
            st.metric("Prediction", pred)
        with c2:
            if conf is None:
                st.metric("Confidence", "—")
            else:
                st.metric("Confidence", f"{int(round(conf * 100))}%")
                st.progress(min(max(conf, 0.0), 1.0))
        with c3:
            if csat_avg is None:
                st.metric("Average CSAT", "—")
            else:
                st.metric("Average CSAT", f"{csat_avg} / 5")

        st.divider()
        st.subheader("Top drivers (explainability)")
        if drivers:
            for d in drivers[:5]:
                st.write(f"• {d}")
        else:
            st.caption(
                "Driver insights will show up here once the pipeline exposes feature importance / SHAP outputs. "
                "UI is already prepared for it."
            )

        st.success("Prediction generated. You can tweak ratings to test different scenarios.")

    except Exception:
        # NOTE: Keeping the error message user-friendly.
        # The technical traceback can be added later to logs if we want.
        st.error(
            "We couldn’t generate a prediction with the current configuration. "
            "Please try again, or check that the model pipeline is connected and running."
        )

    st.markdown("</div>", unsafe_allow_html=True)


def render_model_info() -> None:
    st.markdown("<div class='ap-card'>", unsafe_allow_html=True)
    st.markdown("<div class='ap-card-title'>Model Info</div>", unsafe_allow_html=True)
    st.markdown(
        "<div class='ap-card-subtitle'>Placeholders for model documentation. I’ll update this once the final model is integrated.</div>",
        unsafe_allow_html=True,
    )

    st.write("**What this tool does**")
    st.write("- Predicts passenger satisfaction before closing a flight record.")
    st.write("- Highlights which service attributes are pushing satisfaction up or down (once explainability is wired).")

    st.write("**CSAT handling**")
    st.write("- Ratings use a 1–5 Likert scale (Very dissatisfied → Very satisfied).")
    st.write("- **0 is treated as Not applicable / Not used** (not a low score).")

    st.write("**Operational notes**")
    st.write("- For best results, rate the attributes most relevant to the passenger journey.")
    st.write("- Use N/A when an attribute does not apply, rather than forcing a score.")

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
        run = st.button("Run prediction", type="primary", use_container_width=True)
        st.caption("Your inputs will be used to predict passenger satisfaction.")

    with right:
        st.markdown("<div class='ap-link-btn'>", unsafe_allow_html=True)
        reset = st.button("Reset inputs", use_container_width=False)
        st.markdown("</div>", unsafe_allow_html=True)
        st.caption("Clears all fields and starts a new scenario.")

    if reset:
        reset_all()

    # When the user clicks “Run prediction”, we take them straight to the Result page.
    if run:
        st.session_state["nav"] = "Prediction Result"
        st.rerun()


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    st.set_page_config(page_title=f"{APP_NAME} G4", page_icon="✈️", layout="wide")
    inject_css()
    init_state()

    render_sidebar()
    render_header()

    if st.session_state["nav"] == "Flight Inputs":
        render_flight_inputs()
        render_actions()

    elif st.session_state["nav"] == "Service Ratings":
        render_service_ratings()
        render_actions()

    elif st.session_state["nav"] == "Prediction Result":
        render_prediction_result()
        render_actions()

    elif st.session_state["nav"] == "Model Info":
        render_model_info()


if __name__ == "__main__":
    main()
