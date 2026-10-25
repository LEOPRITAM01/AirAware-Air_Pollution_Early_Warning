import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="AirAware",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 AirAware")
st.subheader("Air Pollution Early Warning System")
st.write("Experimental next-day AQI forecasting using deep learning.")

MODEL_DIR = Path(__file__).parent / "airaware_model"

@st.cache_resource
def load_assets():
    with open(MODEL_DIR / "metadata.json") as f:
        metadata = json.load(f)

    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    models = {
        "LSTM": load_model(MODEL_DIR / "lstm_model.keras"),
        "GRU": load_model(MODEL_DIR / "gru_model.keras")
    }
    return metadata, scaler, models

try:
    metadata, scaler, models = load_assets()
except Exception as e:
    st.error(f"Unable to load model files: {e}")
    st.stop()

st.warning(
    "Research prototype only. This is not an official air-quality "
    "monitoring service or a substitute for health guidance."
)

st.header(f"Forecast for {metadata['city']}")

c1, c2, c3 = st.columns(3)
c1.metric("Baseline MAE", f"{metadata['baseline_mae']:.2f}")
c2.metric("LSTM MAE", f"{metadata['lstm_mae']:.2f}")
c3.metric("GRU MAE", f"{metadata['gru_mae']:.2f}")

model_name = st.selectbox("Forecast model", ["LSTM", "GRU"])

uploaded_file = st.file_uploader(
    "Upload a CSV with recent daily AQI readings",
    type=["csv"]
)

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    if "AQI" not in data.columns:
        st.error("Your CSV must contain a column named AQI.")
        st.stop()

    aqi = pd.to_numeric(data["AQI"], errors="coerce").dropna()
    lookback = int(metadata["lookback"])

    if len(aqi) < lookback:
        st.error(f"Upload at least {lookback} valid AQI readings.")
        st.stop()

    recent = aqi.iloc[-lookback:].to_numpy().reshape(-1, 1)
    scaled = scaler.transform(recent).reshape(1, lookback, 1)

    prediction_scaled = models[model_name].predict(
        scaled, verbose=0
    )
    prediction = float(
        scaler.inverse_transform(prediction_scaled)[0, 0]
    )

    st.metric("Predicted next-day AQI", f"{prediction:.1f}")

    if prediction <= 50:
        category = "Good"
    elif prediction <= 100:
        category = "Satisfactory"
    elif prediction <= 200:
        category = "Moderate"
    elif prediction <= 300:
        category = "Poor"
    elif prediction <= 400:
        category = "Very Poor"
    else:
        category = "Severe"

    st.write(f"Experimental AQI category: **{category}**")
    st.line_chart(aqi.tail(lookback).reset_index(drop=True))
