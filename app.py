"""
Diabetes Risk Predictor — Streamlit App
----------------------------------------
Loads the trained Linear Regression model (and its selected feature list)
saved from the notebook, collects patient info through a form, and predicts
a diabetes risk score.

Required files in the same folder as this script:
    - diabetes_linear_regression_model.pkl
    - diabetes_model_selected_features.pkl

Run with:
    streamlit run app.py
"""

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Page config (must be the first Streamlit command)
# ------------------------------------------------------------------
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🩺",
    layout="centered",
)

# ------------------------------------------------------------------
# Bootstrap CSS (via CDN) + custom CSS to restyle Streamlit's native
# widgets so they visually match Bootstrap's look (rounded cards,
# Bootstrap's primary blue, spacing, etc.)
#
# NOTE: Streamlit renders each widget as its own React component, so raw
# Bootstrap <form>/<input> HTML tags can't submit data back to Python.
# Because of that, the actual inputs below stay as Streamlit widgets
# (st.number_input, st.selectbox, st.button) — but we use Bootstrap's
# CDN stylesheet for layout classes (cards, badges, alerts, grid) and
# override Streamlit's internal CSS classes so everything reads as one
# consistent Bootstrap-flavored design.
# ------------------------------------------------------------------
st.markdown(
    """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css"
          rel="stylesheet">

    <style>
        /* Page background */
        .stApp {
            background-color: #f4f6f9;
        }

        /* Hero header — Bootstrap "jumbotron" style card */
        .hero-card {
            background: linear-gradient(135deg, #0d6efd, #0a58ca);
            color: white;
            padding: 2rem 2rem;
            border-radius: 1rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 0.5rem 1rem rgba(0,0,0,0.15);
        }
        .hero-card h1 {
            font-weight: 700;
            margin-bottom: 0.25rem;
        }
        .hero-card p {
            opacity: 0.9;
            margin-bottom: 0;
        }

        /* Section card wrapper around the form */
        .form-card {
            background: white;
            padding: 1.75rem 1.75rem 0.5rem 1.75rem;
            border-radius: 1rem;
            box-shadow: 0 0.25rem 0.75rem rgba(0,0,0,0.06);
            margin-bottom: 1.5rem;
            border: 1px solid #e9ecef;
        }
        .form-card h4 {
            font-weight: 600;
            color: #0d6efd;
            margin-bottom: 1rem;
        }

        /* Restyle Streamlit's primary button to match Bootstrap's btn-primary */
        .stButton>button {
            background-color: #0d6efd;
            color: white;
            border: none;
            border-radius: 0.5rem;
            padding: 0.6rem 1.5rem;
            font-weight: 600;
            width: 100%;
            transition: background-color 0.15s ease-in-out;
        }
        .stButton>button:hover {
            background-color: #0a58ca;
            color: white;
        }

        /* Result alert boxes */
        .result-alert {
            padding: 1.25rem 1.5rem;
            border-radius: 0.75rem;
            font-size: 1.05rem;
            margin-top: 1rem;
        }
        .footer-note {
            text-align: center;
            color: #6c757d;
            font-size: 0.85rem;
            margin-top: 2rem;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <h1>🩺 Diabetes Risk Predictor</h1>
        <p>Enter patient details below to estimate a diabetes risk score
        using a trained Linear Regression model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Load model + selected features (cached so it only loads once)
# ------------------------------------------------------------------
@st.cache_resource
def load_model_and_features():
    model = joblib.load("diabetes_linear_regression_model.pkl")
    features = joblib.load("diabetes_model_selected_features.pkl")
    return model, features


try:
    model, best_features = load_model_and_features()
except FileNotFoundError:
    st.error(
        "Could not find 'diabetes_linear_regression_model.pkl' or "
        "'diabetes_model_selected_features.pkl'. Make sure both files "
        "are in the same folder as this app (export them from the "
        "notebook's 'Save the Models' step)."
    )
    st.stop()

# ------------------------------------------------------------------
# Config describing how to render each possible feature.
# (label, widget type, min, max, default, step)
# Only the features actually in `best_features` will be shown —
# this dict just knows sensible ranges for every possible column.
# ------------------------------------------------------------------
FEATURE_CONFIG = {
    "age":               ("Age (years)", "number", 18, 100, 40, 1),
    "gender":            ("Gender", "select", ["Female", "Male"], None, None, None),
    "bmi":               ("BMI (kg/m²)", "number", 10.0, 60.0, 24.5, 0.1),
    "glucose":           ("Glucose Level (mg/dL)", "number", 50, 300, 100, 1),
    "blood_pressure":    ("Blood Pressure (mmHg)", "number", 60, 200, 120, 1),
    "cholesterol":       ("Cholesterol (mg/dL)", "number", 100, 400, 180, 1),
    "heart_rate":        ("Heart Rate (bpm)", "number", 40, 180, 75, 1),
    "sleep_hours":       ("Sleep Hours / Night", "number", 0.0, 12.0, 7.0, 0.1),
    "physical_activity": ("Physical Activity Level", "select_activity", None, None, None, None),
    "smoking":           ("Smoker?", "select_yesno", None, None, None, None),
    "alcohol_intake":    ("Drinks Alcohol?", "select_yesno", None, None, None, None),
    "family_history":    ("Family History of Diabetes?", "select_yesno", None, None, None, None),
    "stress_level":      ("Stress Level (1-10)", "number", 1, 10, 5, 1),
    "diet_score":        ("Diet Quality Score (1-10)", "number", 1, 10, 5, 1),
    "steps_per_day":     ("Steps per Day", "number", 0, 30000, 6000, 100),
    "work_hours":        ("Work Hours per Day", "number", 0, 16, 8, 1),
    "water_intake_ltr":  ("Water Intake (liters/day)", "number", 0.0, 8.0, 2.5, 0.1),
    "insulin":           ("Insulin Level", "number", 0, 350, 100, 1),
}

# ------------------------------------------------------------------
# Build the form (Bootstrap "card" wrapper around Streamlit widgets)
# ------------------------------------------------------------------
st.markdown('<div class="form-card"><h4>Patient Information</h4>', unsafe_allow_html=True)

user_values = {}

with st.form("prediction_form"):
    # Lay widgets out in a responsive 2-column grid (Bootstrap-style row/col)
    cols = st.columns(2)

    for i, feat in enumerate(best_features):
        col = cols[i % 2]
        config = FEATURE_CONFIG.get(feat)

        with col:
            if config is None:
                # Fallback for any feature not in our config dict
                user_values[feat] = st.number_input(feat, value=0.0)
                continue

            label, kind, a, b, default, step = config

            if kind == "number":
                user_values[feat] = st.number_input(
                    label, min_value=a, max_value=b, value=default, step=step
                )
            elif kind == "select":
                choice = st.selectbox(label, a)
                # Encode: Male = 1, Female = 0
                user_values[feat] = 1 if choice == "Male" else 0
            elif kind == "select_activity":
                choice = st.selectbox(label, ["Low", "Medium", "High"])
                # Encode: Low = 0, Medium = 1, High = 2
                user_values[feat] = {"Low": 0, "Medium": 1, "High": 2}[choice]
            elif kind == "select_yesno":
                choice = st.selectbox(label, ["No", "Yes"])
                user_values[feat] = 1 if choice == "Yes" else 0

    submitted = st.form_submit_button("Predict Diabetes Risk")

st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------------------------------------------
# Prediction + Bootstrap-styled result alert
# ------------------------------------------------------------------
if submitted:
    # Build the input row in the exact column order the model expects
    input_row = pd.DataFrame([[user_values[feat] for feat in best_features]], columns=best_features)

    raw_score = float(model.predict(input_row)[0])
    # Clip for a cleaner display, since Linear Regression can output <0 or >1
    display_score = float(np.clip(raw_score, 0, 1))
    predicted_class = "High Risk" if raw_score >= 0.5 else "Low Risk"

    if raw_score >= 0.5:
        alert_class = "alert alert-danger result-alert"
        icon = "⚠️"
    else:
        alert_class = "alert alert-success result-alert"
        icon = "✅"

    st.markdown(
        f"""
        <div class="{alert_class}">
            {icon} <strong>Predicted: {predicted_class}</strong><br>
            Raw model score: {raw_score:.4f} &nbsp;|&nbsp; Clipped (0-1): {display_score:.4f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.progress(display_score)

    st.caption(
        "Note: this model was trained with plain Linear Regression on a "
        "binary (0/1) target as a learning exercise — treat the score as "
        "a rough relative indicator, not a calibrated probability or a "
        "medical diagnosis."
    )

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown(
    """
    <div class="footer-note">
        Built with Streamlit • Styled with Bootstrap 5 • For educational purposes only
    </div>
    """,
    unsafe_allow_html=True,
)
