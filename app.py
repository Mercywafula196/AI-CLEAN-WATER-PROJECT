# app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ===============================
# Load the trained model
# ===============================
MODEL_PATH = "water_quality_model.pkl"

@st.cache_resource
def load_model(path=MODEL_PATH):
    bundle = joblib.load(path)
    return bundle["pipeline"], bundle.get("feature_names")

model, feature_names = load_model()

# ===============================
# App Layout
# ===============================
st.set_page_config(page_title="AI for Clean Water (SDG 6)", layout="centered")
st.title("💧 AI-Powered Water Quality Predictor")
st.write("Use this app to predict whether water is **safe to drink** based on chemical properties.")

# ===============================
# Input Options
# ===============================
st.sidebar.header("📥 Input Options")
upload = st.sidebar.file_uploader("Upload CSV file with water samples", type=["csv"])

if upload is not None:
    input_df = pd.read_csv(upload)
    st.sidebar.success("✅ File uploaded successfully!")
    st.dataframe(input_df.head())
else:
    st.sidebar.write("Or enter values manually below:")
    pH = st.sidebar.number_input("pH", 0.0, 14.0, 7.0)
    Hardness = st.sidebar.number_input("Hardness", 0.0, 1000.0, 200.0)
    Solids = st.sidebar.number_input("Solids (ppm)", 0.0, 50000.0, 15000.0)
    Chloramines = st.sidebar.number_input("Chloramines", 0.0, 20.0, 7.0)
    Sulfate = st.sidebar.number_input("Sulfate", 0.0, 500.0, 300.0)
    Conductivity = st.sidebar.number_input("Conductivity", 0.0, 2000.0, 400.0)
    Organic_carbon = st.sidebar.number_input("Organic Carbon", 0.0, 30.0, 10.0)
    Trihalomethanes = st.sidebar.number_input("Trihalomethanes", 0.0, 150.0, 80.0)
    Turbidity = st.sidebar.number_input("Turbidity", 0.0, 10.0, 4.0)

    input_df = pd.DataFrame([{
        "pH": pH,
        "Hardness": Hardness,
        "Solids": Solids,
        "Chloramines": Chloramines,
        "Sulfate": Sulfate,
        "Conductivity": Conductivity,
        "Organic_carbon": Organic_carbon,
        "Trihalomethanes": Trihalomethanes,
        "Turbidity": Turbidity
    }])

st.subheader("📊 Input Data Preview")
st.dataframe(input_df)

# ===============================
# Prediction
# ===============================
if st.button("🚀 Predict Water Quality"):
    try:
        preds = model.predict(input_df)
        probs = model.predict_proba(input_df)
        result = preds[0]

        if result == 1:
            st.success(f"✅ The water is likely **Safe (Potable)**. Confidence: {probs[0][1]:.2f}")
        else:
            st.warning(f"⚠️ The water is likely **Unsafe (Not Potable)**. Confidence: {probs[0][0]:.2f}")

        # Show probabilities
        st.write("Prediction Probabilities:")
        st.json({
            "Safe (Potable)": round(probs[0][1], 2),
            "Unsafe (Not Potable)": round(probs[0][0], 2)
        })

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ===============================
# Footer
# ===============================
st.markdown("---")
st.write("🧠 *AI for Clean Water — SDG 6 Project*")
st.caption("Disclaimer: This tool is for educational use only. Always verify results with certified lab tests.")
