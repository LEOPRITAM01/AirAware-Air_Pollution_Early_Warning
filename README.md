# AirAware — Air Pollution Early Warning System

## Overview
AirAware is a deep learning project exploring next-day Air Quality Index (AQI) forecasting using historical air-quality data from India.

## Models
- Persistence baseline
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)

## Dataset
India Air Quality Data:
https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india

The initial experiment uses `city_day.csv`.

## Evaluation
Models are evaluated using Mean Absolute Error (MAE) and Root Mean Squared Error (RMSE). See the completed experiment for actual results.

## Dashboard
The Streamlit dashboard accepts a CSV containing recent daily AQI readings and produces an experimental next-day forecast.

## Run locally
Install dependencies:

`pip install -r requirements.txt`

Launch the dashboard:

`streamlit run app.py`

## Limitations
This is an educational research prototype, not an official AQI monitoring service. Predictions are not medically validated and should not replace official air-quality reports or health guidance.
