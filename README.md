# 🌍 AirAware — Air Pollution Early Warning System

## Overview

AirAware is a deep learning project that explores next-day Air Quality Index (AQI) forecasting using historical air-quality data from India. It combines time-series forecasting models with an interactive Streamlit dashboard to visualize AQI trends, compare model performance, and generate experimental next-day predictions.

## Features

- Historical AQI trend visualization and recent-window analysis
- Data-quality checks for uploaded AQI readings
- Interactive selection between LSTM and GRU forecasting models
- Comparison against a persistence baseline
- Model evaluation using MAE and RMSE
- Next-day AQI prediction with an indicative AQI category
- Downloadable forecast summary in CSV format

## Models

AirAware evaluates three approaches:

1. **Persistence Baseline:** Uses the latest observed AQI as the next-day prediction.
2. **Long Short-Term Memory (LSTM):** A recurrent neural network designed to learn patterns in sequential data.
3. **Gated Recurrent Unit (GRU):** A recurrent neural network architecture for sequence modelling.

## Dataset

**India Air Quality Data — Kaggle**

Dataset: https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

The initial experiment uses `city_day.csv` and focuses on daily AQI readings for Delhi. The forecasting workflow uses historical AQI values to construct time-series input sequences.

## Model Evaluation

Models are evaluated on a chronological held-out test set using:

- **Mean Absolute Error (MAE):** Measures the average absolute prediction error.
- **Root Mean Squared Error (RMSE):** Penalizes larger prediction errors more strongly.

### Experimental Results

| Model | MAE | RMSE |
|---|---:|---:|
| Persistence Baseline | 31.46 | 44.97 |
| LSTM | 33.98 | 45.92 |
| GRU | 32.23 | 44.44 |

**Result interpretation:** The persistence baseline achieved the lowest MAE, while GRU achieved the lowest RMSE. In this experiment, the deep learning models did not outperform the baseline on every metric.

## Interactive Dashboard

The Streamlit dashboard provides:

- Model performance metrics and comparisons
- Historical AQI trend charts
- Recent-window descriptive statistics
- Input data-quality checks
- LSTM/GRU next-day forecasting
- Comparison with the persistence forecast
- Downloadable forecast summary

The dashboard accepts a CSV file containing an `AQI` column with at least 30 valid readings. An optional `Date` column allows readings to be sorted chronologically.

**Important:** The dashboard uses uploaded historical AQI data. It does not fetch live pollution measurements or official current AQI readings.

## Repository Structure

```text
AirAware-Air_Pollution_Early_Warning/
├── app.py
├── AirAware.ipynb
├── requirements.txt
├── README.md
└── airaware_model/
    ├── lstm_model.keras
    ├── gru_model.keras
    ├── scaler.pkl
    └── metadata.json
```

## Installation and Usage

### 1. Clone the repository

```bash
git clone https://github.com/LEOPRITAM01/AirAware-Air_Pollution_Early_Warning.git
cd AirAware-Air_Pollution_Early_Warning
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Launch the dashboard

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal, then upload a suitable AQI CSV file to explore the dashboard and generate a forecast.

## Project Notebook

`AirAware.ipynb` documents the experimental workflow, including data preparation, exploratory analysis, time-series sequence creation, model training, and evaluation.

## Limitations

- This is an educational research prototype, not an official AQI monitoring service.
- Forecasts are experimental and have not been medically validated.
- Predictions depend on the uploaded historical data and may differ from actual future AQI.
- Model performance is based on a historical test period and does not guarantee future accuracy.
- AQI categories shown for predictions are indicative and should not be treated as official measurements.
- The project does not replace official air-quality reports or health guidance.

## Technologies Used

- Python
- Pandas and NumPy
- TensorFlow / Keras
- Scikit-learn
- Streamlit
- Jupyter Notebook

## Disclaimer

AirAware is intended for educational and research purposes only. For real-world air-quality information, consult official monitoring sources such as the Central Pollution Control Board (CPCB).
