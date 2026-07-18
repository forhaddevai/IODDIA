import os
import joblib
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_PATH = "house_price_linear_regression.joblib"

# Feature order the model was trained on. Must match the notebook exactly.

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
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
    df = pd.DataFrame([row])
    pred = model.predict(df)[0]
    return float(pred)


# --------------------------------------------------------------------------
# App layout
# --------------------------------------------------------------------------
st.title("House Price Predictor")
st.caption("Predict your House Price.")

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
        );
        bathrooms = st.slider(
            "Bathrooms", min_value=0, max_value=10, value=3, step=1,
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
