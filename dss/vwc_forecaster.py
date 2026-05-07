import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/vwc_model.joblib")

_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_vwc_7days(
    initial_vwc: float,
    weather_forecast: list[dict],
):
    """
    weather_forecast = list of 7 dicts:
    [
      {"precip": .., "temp": .., "wind": .., "humidity": .., "solar": .., "pe": ..},
      ...
    ]
    """

    model = load_model()
    predictions = []

    current_vwc = initial_vwc

    for day in weather_forecast:
        X = pd.DataFrame([{
            "VWC_lag1": current_vwc,
            "Precipitation": day["precip"],
            "Air Temperature": day["temp"],
            "WindSpeed": day["wind"],
            "Relative Humidity": day["humidity"],
            "Solar Radiation": day["solar"],
            "PE": day["pe"],
        }])

        next_vwc = float(model.predict(X)[0])
        predictions.append(round(next_vwc, 2))

        current_vwc = next_vwc

    return predictions