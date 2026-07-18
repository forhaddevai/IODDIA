"""
House Price Prediction — Streamlit UI

Loads the Linear Regression model saved by the training notebook
(a single `house_price_linear_regression.joblib` file, trained on raw,
unscaled features) and serves a simple web UI for:
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
MODEL_PATH = "house_price_linear_regression.joblib"

# Feature order the model was trained on. Must match the notebook exactly.
FEATURES = ["area_sqft", "bedrooms"]

st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered",
)

# --------------------------------------------------------------------------
# Widget configuration per feature.
# --------------------------------------------------------------------------
FEATURE_WIDGETS = {
    "area_sqft": dict(label="Area (sqft)", kind="number", min_value=100.0,
                       max_value=20000.0, value=1800.0, step=50.0),
    "bedrooms": dict(label="Bedrooms", kind="slider", min_value=0,
                      max_value=10, value=3, step=1),
}


# --------------------------------------------------------------------------
# Cached loader
# --------------------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load the trained model. Cached so it only loads once per session."""
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


def humanize(feature_name: str) -> str:
    """Fallback label for a feature not in FEATURE_WIDGETS."""
    return feature_name.replace("_", " ").title()


def render_input(feature: str):
    """Render the appropriate Streamlit widget for a given feature name."""
    cfg = FEATURE_WIDGETS.get(feature)

    if cfg is None:
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


def predict(model, row: dict) -> float:
    """Build a single-row DataFrame in the right column order and predict."""
    df = pd.DataFrame([row])[FEATURES]
    pred = model.predict(df)[0]
    return float(pred)


# --------------------------------------------------------------------------
# App layout
# --------------------------------------------------------------------------
st.title("🏠 House Price Predictor")
st.caption("Linear Regression model trained in the companion notebook.")

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
        for i, feature in enumerate(FEATURES):
            with cols[i % 2]:
                input_values[feature] = render_input(feature)

        submitted = st.form_submit_button("Predict Price", type="primary")

    if submitted:
        try:
            price = predict(model, input_values)
            st.success(f"### Estimated Price: ${price:,.2f}")

            with st.expander("Show input used for this prediction"):
                st.dataframe(
                    pd.DataFrame([input_values])[FEATURES],
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
        f"`{', '.join(FEATURES)}`"
    )

    template_df = pd.DataFrame([{f: FEATURE_WIDGETS.get(f, {}).get("value", 0) for f in FEATURES}])
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
            missing = [f for f in FEATURES if f not in batch_df.columns]
            if missing:
                st.error(f"Missing required column(s): {', '.join(missing)}")
            else:
                X = batch_df[FEATURES]
                batch_df["predicted_price"] = model.predict(X)
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
    st.write("**Algorithm:** Linear Regression (scikit-learn)")
    st.write(f"**Number of features:** {len(FEATURES)}")
    st.write(f"**Features used:** {', '.join(FEATURES)}")

    if hasattr(model, "coef_"):
        coef_df = pd.DataFrame({
            "feature": FEATURES,
            "coefficient": model.coef_,
        }).sort_values("coefficient", key=np.abs, ascending=False)
        st.write("**Coefficients:**")
        st.dataframe(coef_df, use_container_width=True, hide_index=True)

    if hasattr(model, "intercept_"):
        st.write(f"**Intercept:** {model.intercept_:,.2f}")

    st.info(
        "This model was trained on synthetic/sample data in the companion "
        "notebook (`house_price_prediction.ipynb`). Retrain on real data "
        "and re-save `house_price_linear_regression.joblib` to update this app."
    )
