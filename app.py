import os
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = "house_price_linear_regression.joblib"
# Feature order the model was trained on. Must match the notebook exactly.
FEATURES = ["area_sqft", "bedrooms"]

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# --------------------------------------------------------------------------
# Bootstrap CDN + custom CSS
# Streamlit widgets don't render actual Bootstrap markup, so this CSS
# re-skins Streamlit's own elements (inputs, sliders, buttons, alerts) to
# match Bootstrap 5's look — colors, spacing, radii, shadows, typography.
# --------------------------------------------------------------------------
st.markdown(
    """
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        html, body, [class*="css"] {
            font-family: -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        }

        .stApp {
            background-color: #f8f9fa;
        }

        .bs-hero {
            text-align: center;
            padding: 1.5rem 1rem 0.5rem 1rem;
        }
        .bs-hero h1 {
            font-weight: 700;
            color: #212529;
        }
        .bs-hero p {
            color: #6c757d;
            font-size: 1.05rem;
        }

        /* Card wrapper around the form */
        div[data-testid="stForm"] {
            background: #ffffff;
            border: 1px solid #dee2e6;
            border-radius: 0.75rem;
            padding: 1.75rem;
            box-shadow: 0 0.125rem 0.5rem rgba(0, 0, 0, 0.08);
        }

        /* Section subheader */
        h3 {
            font-weight: 600;
            color: #212529;
            border-bottom: 1px solid #e9ecef;
            padding-bottom: 0.5rem;
            margin-bottom: 1.25rem !important;
        }

        /* Number input / text input -> Bootstrap form-control look */
        div[data-testid="stNumberInput"] input {
            border-radius: 0.375rem;
            border: 1px solid #ced4da;
        }
        div[data-testid="stNumberInput"] input:focus {
            border-color: #86b7fe;
            box-shadow: 0 0 0 0.25rem rgba(13, 110, 253, 0.25);
        }

        /* Slider -> Bootstrap primary blue */
        div[data-testid="stSlider"] div[role="slider"] {
            background-color: #0d6efd !important;
            border-color: #0d6efd !important;
        }
        div[data-testid="stSlider"] .st-emotion-cache-1dj0hjr,
        div[data-testid="stSlider"] div[data-baseweb="slider"] > div > div {
            background: #0d6efd !important;
        }

        /* Primary button -> Bootstrap btn-primary */
        button[kind="primaryFormSubmit"], .stButton>button, button[kind="primary"] {
            background-color: #0d6efd;
            border: 1px solid #0d6efd;
            border-radius: 0.375rem;
            font-weight: 500;
            padding: 0.5rem 1.25rem;
            transition: background-color 0.15s ease-in-out;
        }
        button[kind="primaryFormSubmit"]:hover, .stButton>button:hover, button[kind="primary"]:hover {
            background-color: #0b5ed7;
            border-color: #0a58ca;
        }

        /* Success / error alerts -> Bootstrap alert look */
        div[data-testid="stAlert"] {
            border-radius: 0.5rem;
            border: 1px solid transparent;
        }

        label {
            font-weight: 500 !important;
            color: #343a40 !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Cached loader
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained model. Cached so it only loads once per session."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def predict(model, row: dict) -> float:
    """Build a single-row DataFrame in the right column order and predict."""
    df = pd.DataFrame([row])[FEATURES]
    pred = model.predict(df)[0]
    return float(pred)


# --------------------------------------------------------------------------
# App layout
# --------------------------------------------------------------------------
st.markdown(
    """
    <div class="bs-hero">
        <h1>🏠 House Price Predictor</h1>
        <p>Predict your House Price.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

model = load_model()

if model is None:
    st.error(
        f"Model file not found. Expected `{MODEL_PATH}` in the same folder "
        "as this script.\n\n"
        "Run the training notebook first (it saves the model with "
        "`joblib.dump(model, \"house_price_linear_regression.joblib\")`), "
        "then place that file next to `main.py`."
    )
    st.stop()

st.subheader("Enter house details")

with st.form("single_prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        area_sqft = st.number_input(
            "Area (sqft)", min_value=100.0, max_value=20000.0,
            value=1800.0, step=50.0,
        )
        bathrooms = st.slider(
            "Bathrooms", min_value=1, max_value=10, value=3, step=1,
        )
    with col2:
        bedrooms = st.slider(
            "Bedrooms", min_value=0, max_value=10, value=3, step=1,
        )

    submitted = st.form_submit_button("Predict Price", type="primary")

if submitted:
    try:
        input_values = {"area_sqft": area_sqft, "bedrooms": bedrooms, "bathrooms": bathrooms}
        price = predict(model, input_values)
        st.success(f"### Estimated Price: ${price:,.2f}")
    except Exception as e:
        st.error(f"Couldn't generate a prediction: {e}")
