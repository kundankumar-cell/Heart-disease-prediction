import os
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide"
)

# ---------------------------------------------------------
# 1. ARTIFACT PATHS & SAFE LOADING
# ---------------------------------------------------------
MODEL_FILE = "Naive Bayes_heart.pkl"
SCALER_FILE = "scaler.pkl"
COLUMNS_FILE = "columns.pkl"

@st.cache_resource(show_spinner="Loading Machine Learning Artifacts...")
def load_ml_artifacts():
    """Safely loads model, scaler, and expected feature columns."""
    errors = []
    model, scaler, expected_columns = None, None, None

    # Verify model file
    if not os.path.exists(MODEL_FILE):
        errors.append(f"Model file '{MODEL_FILE}' not found.")
    else:
        try:
            model = joblib.load(MODEL_FILE)
        except Exception as e:
            errors.append(f"Failed to load '{MODEL_FILE}': {e}")

    # Verify scaler file
    if not os.path.exists(SCALER_FILE):
        errors.append(f"Scaler file '{SCALER_FILE}' not found.")
    else:
        try:
            scaler = joblib.load(SCALER_FILE)
        except Exception as e:
            errors.append(f"Failed to load '{SCALER_FILE}': {e}")

    # Verify columns file
    if not os.path.exists(COLUMNS_FILE):
        errors.append(f"Feature columns file '{COLUMNS_FILE}' not found.")
    else:
        try:
            expected_columns = joblib.load(COLUMNS_FILE)
        except Exception as e:
            errors.append(f"Failed to load '{COLUMNS_FILE}': {e}")

    return model, scaler, expected_columns, errors

# Load artifacts
model, scaler, expected_columns, load_errors = load_ml_artifacts()

# Display error banner if critical files are missing
if load_errors:
    for err in load_errors:
        st.error(f"⚠️ {err}")
    st.stop()

# ---------------------------------------------------------
# 2. APPLICATION HEADER
# ---------------------------------------------------------
st.title("❤️ Heart Disease Prediction")
st.markdown(
    "Provide clinical patient details to predict heart disease risk using "
    "the trained **Gaussian Naive Bayes** classifier."
)

# ---------------------------------------------------------
# 3. PATIENT DATA INPUTS
# ---------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics & Vitals")
    age = st.slider("Age", 18, 100, 40)
    sex = st.selectbox("Sex", ["M", "F"])
    resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
    cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])

with col2:
    st.subheader("Cardiovascular Examination")
    chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
    resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
    max_hr = st.slider("Max Heart Rate", 60, 220, 150)
    exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
    oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
    st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# ---------------------------------------------------------
# 4. PREDICTION ENGINE
# ---------------------------------------------------------
st.markdown("---")
if st.button("Predict", type="primary", use_container_width=True):
    try:
        # Create raw input dictionary matching training one-hot encoding
        raw_input = {
            'Age': age,
            'RestingBP': resting_bp,
            'Cholesterol': cholesterol,
            'FastingBS': fasting_bs,
            'MaxHR': max_hr,
            'Oldpeak': oldpeak,
            'Sex_' + sex: 1,
            'ChestPainType_' + chest_pain: 1,
            'RestingECG_' + resting_ecg: 1,
            'ExerciseAngina_' + exercise_angina: 1,
            'ST_Slope_' + st_slope: 1
        }

        # Create input dataframe
        input_df = pd.DataFrame([raw_input])

        # Fill in missing columns with 0s
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0

        # Maintain exact expected feature ordering
        input_df = input_df[expected_columns]

        # Scale the input features
        scaled_input = scaler.transform(input_df)

        # Make prediction
        prediction = model.predict(scaled_input)[0]

        # Calculate prediction probabilities if supported
        has_proba = hasattr(model, "predict_proba")
        if has_proba:
            probabilities = model.predict_proba(scaled_input)[0]
            disease_prob = probabilities[1] * 100
            normal_prob = probabilities[0] * 100
        else:
            disease_prob, normal_prob = None, None

        # Display Prediction Result Card
        st.subheader("Prediction Result")
        st.caption("Selected Model: **Gaussian Naive Bayes** (`Naive Bayes_heart.pkl`)")

        if prediction == 1:
            st.error("### ⚠️ Result: Heart Disease Detected (High Risk)")
            if has_proba:
                st.metric(
                    label="Estimated Disease Risk Probability",
                    value=f"{disease_prob:.1f}%",
                    help="Calculated by the Gaussian Naive Bayes model on scaled clinical features."
                )
                st.progress(disease_prob / 100.0)
            st.markdown(
                "**Clinical Summary:** The model detected risk patterns associated with heart disease. "
                "Consultation with a cardiologist is recommended."
            )
        else:
            st.success("### ✅ Result: Heart Disease Not Detected (Low Risk)")
            if has_proba:
                st.metric(
                    label="Estimated Normal (Low Risk) Probability",
                    value=f"{normal_prob:.1f}%",
                    help="Calculated by the Gaussian Naive Bayes model on scaled clinical features."
                )
                st.progress(normal_prob / 100.0)
            st.markdown(
                "**Clinical Summary:** The clinical parameters fall within low-risk thresholds for heart disease."
            )

        # Clinical Disclaimer
        st.info(
            "ℹ️ **Clinical Disclaimer:** This prediction is generated by a statistical Machine Learning model "
            "for informational and screening purposes only. It does not replace professional medical diagnosis, "
            "laboratory testing, or clinical consultation."
        )

    except Exception as ex:
        st.error(f"Prediction failed due to an unexpected error: {ex}")
