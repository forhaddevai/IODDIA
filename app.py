"""
House Price Prediction — Streamlit UI

Loads the Linear Regression model, StandardScaler, and feature list saved
by the training notebook (via joblib) and serves a simple web UI for:
  1. Single-house prediction using input widgets
  2. Batch prediction by uploading a CSV of houses

Run with:
    streamlit run main.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------
MODEL_DIR = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_linear_regression.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "house_price_scaler.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "house_price_features.pkl")

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# --------------------------------------------------------------------------
# Widget configuration per feature.
# Any feature not listed here automatically falls back to a plain
# number_input, so the app keeps working even if the feature set changes.
# --------------------------------------------------------------------------
FEATURE_WIDGETS = {
    "area_sqft": dict(label="Area (sqft)", kind="number", min_value=100.0,
                       max_value=20000.0, value=1800.0, step=50.0),
    "bedrooms": dict(label="Bedrooms", kind="slider", min_value=0,
                      max_value=10, value=3, step=1),
    "bathrooms": dict(label="Bathrooms", kind="slider", min_value=0,
                       max_value=10, value=2, step=1),
    "age_years": dict(label="Age of house (years)", kind="slider",
                       min_value=0, max_value=150, value=10, step=1),
    "garage_spaces": dict(label="Garage spaces", kind="slider",
                           min_value=0, max_value=5, value=1, step=1),
    "distance_to_city_km": dict(label="Distance to city center (km)",
                                 kind="number", min_value=0.0,
                                 max_value=200.0, value=8.0, step=0.5),
    "location_score": dict(label="Location / neighborhood score (1-10)",
                            kind="slider", min_value=1.0, max_value=10.0,
                            value=6.0, step=0.1),
}


# --------------------------------------------------------------------------
# Cached loaders
# --------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    """Load model, scaler, and feature list. Cached so it only loads once."""
    if not (os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH)
            and os.path.exists(FEATURES_PATH)):
        return None, None, None
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    features = joblib.load(FEATURES_PATH)
    return model, scaler, features


def humanize(feature_name: str) -> str:
    """Fallback label for a feature not in FEATURE_WIDGETS."""
    return feature_name.replace("_", " ").title()


def render_input(feature: str):
    """Render the appropriate Streamlit widget for a given feature name."""
    cfg = FEATURE_WIDGETS.get(feature)

    if cfg is None:
        # Unknown feature -> generic numeric input
        return st.number_input(humanize(feature), value=0.0, step=1.0)

    label = cfg["label"]
    if cfg["kind"] == "slider":
        return st.slider(
            label,
            min_value=cfg["min_value"],
            max_value=cfg["max_value"],
            value=cfg["value"],
            step=cfg["step"],
        )
    else:  # "number"
        return st.number_input(
            label,
            min_value=cfg["min_value"],
            max_value=cfg["max_value"],
            value=cfg["value"],
            step=cfg["step"],
        )


def predict(model, scaler, features, row: dict) -> float:
    """Build a single-row DataFrame in the right column order, scale, predict."""
    df = pd.DataFrame([row])[features]
    scaled = scaler.transform(df)
    pred = model.predict(scaled)[0]
    return float(pred)


# --------------------------------------------------------------------------
# App layout
# --------------------------------------------------------------------------
st.title("🏠 House Price Predictor")
st.caption("Linear Regression model trained in the companion notebook.")

model, scaler, features = load_artifacts()

if model is None:
    st.error(
        "Model files not found. Expected the following inside a `model/` "
        "folder next to this script:\n\n"
        f"- `{MODEL_PATH}`\n- `{SCALER_PATH}`\n- `{FEATURES_PATH}`\n\n"
        "Run the training notebook first (it saves these with `joblib.dump`), "
        "then copy the `model/` folder next to `main.py`."
    )
    st.stop()

tab_single, tab_batch, tab_about = st.tabs(
    ["🔮 Predict a Single House", "📄 Batch Predict (CSV)", "ℹ️ About"]
)

# --------------------------------------------------------------------------
# Tab 1: Single prediction
# --------------------------------------------------------------------------
with tab_single:
    st.subheader("Enter house details")

    with st.form("single_prediction_form"):
        cols = st.columns(2)
        input_values = {}
        for i, feature in enumerate(features):
            with cols[i % 2]:
                input_values[feature] = render_input(feature)

        submitted = st.form_submit_button("Predict Price", type="primary")

    if submitted:
        try:
            price = predict(model, scaler, features, input_values)
            st.success(f"### Estimated Price: ${price:,.2f}")

            with st.expander("Show input used for this prediction"):
                st.dataframe(
                    pd.DataFrame([input_values])[features],
                    use_container_width=True,
                )
        except Exception as e:
            st.error(f"Couldn't generate a prediction: {e}")

# --------------------------------------------------------------------------
# Tab 2: Batch prediction via CSV upload
# --------------------------------------------------------------------------
with tab_batch:
    st.subheader("Upload a CSV of houses")
    st.write(
        "The CSV must contain the following column(s), in any order:  \n"
        f"`{', '.join(features)}`"
    )

    template_df = pd.DataFrame([{f: FEATURE_WIDGETS.get(f, {}).get("value", 0) for f in features}])
    st.download_button(
        "Download CSV template",
        data=template_df.to_csv(index=False),
        file_name="house_price_template.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            missing = [f for f in features if f not in batch_df.columns]
            if missing:
                st.error(f"Missing required column(s): {', '.join(missing)}")
            else:
                X = batch_df[features]
                scaled = scaler.transform(X)
                batch_df["predicted_price"] = model.predict(scaled)
                st.success(f"Predicted prices for {len(batch_df)} houses.")
                st.dataframe(batch_df, use_container_width=True)

                st.download_button(
                    "Download predictions as CSV",
                    data=batch_df.to_csv(index=False),
                    file_name="house_price_predictions.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Couldn't process the file: {e}")

# --------------------------------------------------------------------------
# Tab 3: About / model info
# --------------------------------------------------------------------------
with tab_about:
    st.subheader("Model Info")
    st.write(f"**Algorithm:** Linear Regression (scikit-learn)")
    st.write(f"**Number of features:** {len(features)}")
    st.write(f"**Features used:** {', '.join(features)}")

    if hasattr(model, "coef_"):
        coef_df = pd.DataFrame({
            "feature": features,
            "coefficient": model.coef_,
        }).sort_values("coefficient", key=np.abs, ascending=False)
        st.write("**Coefficients** (on scaled features — magnitude reflects "
                 "relative importance):")
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

    if hasattr(model, "intercept_"):
        st.write(f"**Intercept:** {model.intercept_:,.2f}")

    st.info(
        "This model was trained on synthetic/sample data in the companion "
        "notebook (`house_price_prediction.ipynb`). Retrain on real data "
        "and re-save the artifacts to update this app."
    )
