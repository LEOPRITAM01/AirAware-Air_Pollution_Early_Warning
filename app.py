import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from tensorflow.keras.models import load_model

st.set_page_config(
    page_title="AirAware | AQI Insights",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_DIR = Path(__file__).parent / "airaware_model"
LOOKBACK_FALLBACK = 30

# ---------- Styling ----------
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2.5rem;}
      .hero {
        padding: 1.25rem 1.5rem; border-radius: 16px;
        background: linear-gradient(120deg, #102a43 0%, #176b70 100%);
        color: white; margin-bottom: 1rem;
      }
      .hero h1 {margin: 0 0 .25rem 0; color: white;}
      .hero p {margin: 0; color: #e5f4f4;}
      .small-note {font-size: .88rem; color: #777;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_resource
def load_assets():
    with open(MODEL_DIR / "metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    models = {
        "LSTM": load_model(MODEL_DIR / "lstm_model.keras"),
        "GRU": load_model(MODEL_DIR / "gru_model.keras"),
    }
    return metadata, scaler, models

def aqi_category(value):
    """Indicative Indian AQI category bands; forecasts remain experimental."""
    if value <= 50:
        return "Good", "Air quality is generally satisfactory."
    if value <= 100:
        return "Satisfactory", "Sensitive people may wish to monitor symptoms."
    if value <= 200:
        return "Moderate", "Some people may experience discomfort, especially sensitive groups."
    if value <= 300:
        return "Poor", "Prolonged exposure may affect breathing for some people."
    if value <= 400:
        return "Very Poor", "Respiratory effects may occur with prolonged exposure."
    return "Severe", "Potential health effects may occur even in otherwise healthy people."

def metric_value(metadata, key):
    value = metadata.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def make_forecast_report(city, model_name, predicted, baseline_pred, category, last_value, delta):
    return pd.DataFrame([{
        "City": city,
        "Selected_Model": model_name,
        "Predicted_Next_Day_AQI": round(float(predicted), 2),
        "Experimental_Category": category,
        "Last_Observed_AQI": round(float(last_value), 2),
        "Change_vs_Last_Reading": round(float(delta), 2),
        "Persistence_Baseline_Prediction": round(float(baseline_pred), 2),
        "Disclaimer": "Experimental research forecast; not an official measurement or health advice.",
    }])

try:
    metadata, scaler, models = load_assets()
except Exception as exc:
    st.error("The model assets could not be loaded.")
    st.exception(exc)
    st.info("Check that airaware_model contains metadata.json, scaler.pkl, lstm_model.keras, and gru_model.keras.")
    st.stop()

city = str(metadata.get("city", "Selected city"))
lookback = int(metadata.get("lookback", LOOKBACK_FALLBACK))

st.markdown(
    f"""
    <div class="hero">
      <h1>🌍 AirAware</h1>
      <p>Air Pollution Early Warning System · Experimental next-day AQI forecasting</p>
      <p><b>Model city:</b> {city} &nbsp; · &nbsp; <b>Input window:</b> {lookback} daily readings</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.warning(
    "Research prototype only. Predictions are not official AQI readings, "
    "are not medically validated, and must not replace CPCB/official sources or health guidance."
)

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("⚙️ Forecast settings")
    model_name = st.selectbox("Prediction model", ["GRU", "LSTM"], index=0)
    st.caption(
        "Choose a model to forecast the next day. The persistence baseline "
        "assumes tomorrow's AQI equals the latest observed AQI."
    )
    st.divider()
    st.header("📂 Input data")
    uploaded_file = st.file_uploader(
        "Upload daily AQI CSV",
        type=["csv"],
        help=f"Needs an AQI column and at least {lookback} valid readings. An optional Date column is supported.",
    )
    st.caption("Expected columns: AQI, and optionally Date.")

# ---------- Model performance ----------
st.subheader("Model performance on the held-out test period")
mae_data = {
    "Persistence baseline": metric_value(metadata, "baseline_mae"),
    "LSTM": metric_value(metadata, "lstm_mae"),
    "GRU": metric_value(metadata, "gru_mae"),
}
rmse_data = {
    "Persistence baseline": metric_value(metadata, "baseline_rmse"),
    "LSTM": metric_value(metadata, "lstm_rmse"),
    "GRU": metric_value(metadata, "gru_rmse"),
}
performance_cols = st.columns(3)
for col, name in zip(performance_cols, ["Persistence baseline", "LSTM", "GRU"]):
    mae = mae_data[name]
    rmse = rmse_data[name]
    with col:
        st.metric(f"{name} · MAE", f"{mae:.2f}" if mae is not None else "N/A")
        st.metric(f"{name} · RMSE", f"{rmse:.2f}" if rmse is not None else "N/A")

valid_mae = {k: v for k, v in mae_data.items() if v is not None}
if valid_mae:
    best_mae_name = min(valid_mae, key=valid_mae.get)
    st.info(
        f"Evaluation insight: **{best_mae_name}** has the lowest recorded MAE "
        f"({valid_mae[best_mae_name]:.2f}). Lower MAE means a smaller average absolute prediction error "
        "on the held-out test set. This does not guarantee the same result for future data."
    )

with st.expander("How should I read these metrics?"):
    st.markdown(
        "- **MAE (Mean Absolute Error):** average absolute difference between predicted and observed AQI; lower is better.\n"
        "- **RMSE (Root Mean Squared Error):** penalizes larger errors more strongly; lower is better.\n"
        "- **Persistence baseline:** predicts the next AQI as the latest observed AQI. It is an important simple benchmark.\n"
        "- These metrics describe the historical test period, not the accuracy of the forecast for a particular uploaded file."
    )

st.divider()
st.subheader("📈 Explore recent AQI and forecast")
st.write(
    f"Upload a CSV with at least **{lookback} valid AQI values**. If a `Date` column is included, "
    "AirAware will sort the data by date before selecting the latest readings."
)

if uploaded_file is None:
    st.info("Upload a CSV from the sidebar to unlock the trend explorer, data checks, and next-day forecast.")
    with st.expander("CSV format example"):
        st.code("Date,AQI\n2026-09-25,142\n2026-09-26,151\n2026-09-27,138\n...,...", language="csv")
    st.stop()

try:
    raw = pd.read_csv(uploaded_file)
except Exception as exc:
    st.error(f"Could not read this CSV: {exc}")
    st.stop()

if "AQI" not in raw.columns:
    st.error("The uploaded CSV must contain a column named exactly `AQI`.")
    st.write("Columns found:", ", ".join(map(str, raw.columns)))
    st.stop()

data = raw.copy()
data["AQI"] = pd.to_numeric(data["AQI"], errors="coerce")
initial_rows = len(data)
data = data.dropna(subset=["AQI"]).copy()
invalid_count = initial_rows - len(data)

if "Date" in data.columns:
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    bad_dates = int(data["Date"].isna().sum())
    if bad_dates:
        st.warning(f"{bad_dates} row(s) have an invalid Date and will be omitted from the dated trend chart.")
    data = data.sort_values("Date", na_position="last")
else:
    bad_dates = None

if (data["AQI"] < 0).any():
    st.warning("Negative AQI values were found. Please verify the source data; AQI values are expected to be non-negative.")
if data.empty:
    st.error("No valid numeric AQI values were found.")
    st.stop()
if len(data) < lookback:
    st.error(f"Only {len(data)} valid AQI readings were found. Please upload at least {lookback}.")
    st.stop()

if data["AQI"].max() > 1000:
    st.warning("Some AQI values exceed 1,000. Verify the units and source before interpreting the forecast.")

recent_data = data.tail(lookback).copy()
recent_aqi = recent_data["AQI"].to_numpy(dtype=float)
last_aqi = float(recent_aqi[-1])
first_aqi = float(recent_aqi[0])
mean_aqi = float(np.mean(recent_aqi))
median_aqi = float(np.median(recent_aqi))
minimum_aqi = float(np.min(recent_aqi))
maximum_aqi = float(np.max(recent_aqi))
delta = last_aqi - float(recent_aqi[-2]) if len(recent_aqi) >= 2 else 0.0
trend_delta = last_aqi - first_aqi
category_now, category_description = aqi_category(last_aqi)

# Data quality panel
q1, q2, q3, q4 = st.columns(4)
q1.metric("Valid readings", f"{len(data)}", delta=f"{invalid_count} invalid removed" if invalid_count else None)
q2.metric("Latest AQI in file", f"{last_aqi:.1f}", delta=f"{delta:+.1f} vs previous row")
q3.metric(f"{lookback}-reading average", f"{mean_aqi:.1f}")
q4.metric("Recent range", f"{minimum_aqi:.1f}–{maximum_aqi:.1f}")

with st.expander("Data quality and preparation"):
    st.write(f"- Rows read: **{initial_rows}**")
    st.write(f"- Rows with missing/non-numeric AQI removed: **{invalid_count}**")
    st.write(f"- Valid AQI readings remaining: **{len(data)}**")
    if "Date" in data.columns:
        valid_dates = data["Date"].dropna()
        if not valid_dates.empty:
            st.write(f"- Dated range: **{valid_dates.min().date()} to {valid_dates.max().date()}**")
        st.write("- Rows are sorted by Date when selecting the latest readings.")
    else:
        st.write("- No Date column detected; input row order is assumed to be chronological.")
    st.write(f"- Model input uses the most recent **{lookback}** valid AQI values.")
    st.write("- Missing AQI rows are dropped; this dashboard does not infer missing measurements.")

# Charts: full history and recent model window
chart_tab, stats_tab, forecast_tab = st.tabs(["📊 Trend explorer", "🔎 Recent statistics", "🔮 Next-day forecast"])

with chart_tab:
    chart_data = data.copy()
    if "Date" in chart_data.columns and chart_data["Date"].notna().sum() > 1:
        chart_data = chart_data.dropna(subset=["Date"]).set_index("Date")
        chart_data = chart_data[["AQI"]]
        st.caption("Historical AQI from the uploaded file. The highlighted model window uses the latest valid readings.")
        st.line_chart(chart_data, y="AQI")
        st.line_chart(chart_data.tail(lookback), y="AQI")
    else:
        chart_data = chart_data[["AQI"]].reset_index(drop=True)
        chart_data.index.name = "Reading order"
        st.caption("No usable Date column detected, so the horizontal axis represents row order.")
        st.line_chart(chart_data, y="AQI")
        st.line_chart(chart_data.tail(lookback).reset_index(drop=True), y="AQI")

    st.markdown("**Recent-window interpretation**")
    if trend_delta > 10:
        st.write(f"AQI rose by {trend_delta:.1f} points from the first to the latest value in the model window.")
    elif trend_delta < -10:
        st.write(f"AQI fell by {abs(trend_delta):.1f} points from the first to the latest value in the model window.")
    else:
        st.write(f"AQI changed by {trend_delta:+.1f} points from the first to the latest value in the model window, suggesting a relatively small net change.")
    st.caption("This is a descriptive comparison, not a formal trend test.")

with stats_tab:
    stats = pd.DataFrame({
        "Statistic": ["Count", "Mean", "Median", "Minimum", "Maximum", "Standard deviation", "First in window", "Latest in window"],
        "AQI": [
            len(recent_aqi), mean_aqi, median_aqi, minimum_aqi, maximum_aqi,
            float(np.std(recent_aqi, ddof=1)) if len(recent_aqi) > 1 else 0.0,
            first_aqi, last_aqi
        ]
    })
    st.dataframe(stats, use_container_width=True, hide_index=True)
    st.caption("Statistics are calculated over the most recent model input window, not necessarily the full uploaded file.")

with forecast_tab:
    recent = recent_aqi.reshape(-1, 1)
    try:
        scaled = scaler.transform(recent).reshape(1, lookback, 1)
        prediction_scaled = models[model_name].predict(scaled, verbose=0)
        predicted = float(scaler.inverse_transform(np.asarray(prediction_scaled).reshape(-1, 1))[0, 0])
    except Exception as exc:
        st.error(f"Forecast could not be calculated: {exc}")
        st.stop()

    # AQI is non-negative by definition; retain raw model result for transparency.
    predicted_display = max(0.0, predicted)
    forecast_category, forecast_description = aqi_category(predicted_display)
    baseline_prediction = last_aqi
    change = predicted_display - last_aqi
    baseline_error = predicted_display - baseline_prediction

    st.subheader(f"Tomorrow's experimental forecast · {model_name}")
    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("Predicted next-day AQI", f"{predicted_display:.1f}", delta=f"{change:+.1f} vs latest AQI")
    fc2.metric("Indicative category", forecast_category)
    fc3.metric("Persistence forecast", f"{baseline_prediction:.1f}", help="Simple benchmark: tomorrow equals the latest uploaded AQI.")

    if predicted < 0:
        st.warning(f"The raw model output was {predicted:.1f}; the displayed value is floored at 0 because AQI cannot be negative.")
    st.write(f"**Category context:** {forecast_description}")
    if change > 25:
        st.warning("The model forecasts a notable increase compared with the latest uploaded reading. Treat this as an uncertain model signal, not an official warning.")
    elif change < -25:
        st.success("The model forecasts a notable decrease compared with the latest uploaded reading. Treat this as an uncertain model signal, not a guarantee.")
    else:
        st.info("The forecast is within 25 AQI points of the latest reading. Actual air quality may differ.")

    comparison = pd.DataFrame({
        "Series": ["Latest observed", "Persistence baseline", f"{model_name} forecast"],
        "AQI": [last_aqi, baseline_prediction, predicted_display],
    }).set_index("Series")
    st.bar_chart(comparison, y="AQI")

    st.markdown("**Forecast summary**")
    summary = (
        f"Using the latest {lookback} valid AQI readings, the selected {model_name} model predicts "
        f"a next-day AQI of {predicted_display:.1f} for the {city} model context. "
        f"The latest uploaded reading is {last_aqi:.1f}; the model prediction is {change:+.1f} points relative to it. "
        f"The simple persistence baseline predicts {baseline_prediction:.1f}."
    )
    st.write(summary)

    report = make_forecast_report(
        city, model_name, predicted_display, baseline_prediction,
        forecast_category, last_aqi, change
    )
    st.download_button(
        "⬇️ Download forecast summary (CSV)",
        data=report.to_csv(index=False).encode("utf-8"),
        file_name="airaware_forecast_summary.csv",
        mime="text/csv",
    )

st.divider()
st.subheader("About this prototype")
st.markdown(
    """
    - **Purpose:** demonstrate next-day AQI forecasting with recurrent neural networks.
    - **Models:** LSTM and GRU, compared against a persistence baseline.
    - **Evaluation:** historical held-out MAE and RMSE are displayed above.
    - **Important limitation:** this app forecasts from user-uploaded historical AQI values. It does not fetch live sensor data or official current AQI.
    - **Use responsibly:** check official air-quality sources for real-world decisions. The category labels are indicative and the model output is experimental.
    """
)
st.caption("AirAware · Academic research prototype")
