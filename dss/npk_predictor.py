import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/npk_model.joblib")

_model = None

def load_model():
    global _model
    if _model is None:
        _model = joblib.load(MODEL_PATH)
    return _model


def predict_npk(
    temperature: float,
    humidity: float,
    ph: float,
    rainfall: float,
    soil_type: str,
    variety: str,
):
    model = load_model()

    X = pd.DataFrame([{
        "Temperature": temperature,
        "Humidity": humidity,
        "pH_Value": ph,
        "Rainfall": rainfall,
        "Soil_Type": soil_type,
        "Variety": variety,
    }])

    pred = model.predict(X)[0]

    return {
        "Nitrogen": round(float(pred[0]), 1),
        "Phosphorus": round(float(pred[1]), 1),
        "Potassium": round(float(pred[2]), 1),
    }