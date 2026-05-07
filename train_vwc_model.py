import pandas as pd
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = Path("data/soil_timeseries.csv")
MODEL_PATH = Path("models/vwc_model.joblib")

df = pd.read_csv(DATA_PATH, parse_dates=["DATE_TIME"])
df = df.sort_values("DATE_TIME")

# Create lag feature
df["VWC_lag1"] = df["VWC"].shift(1)
df = df.dropna()

features = [
    "VWC_lag1",
    "Precipitation",
    "Air Temperature",
    "WindSpeed",
    "Relative Humidity",
    "Solar Radiation",
    "PE"
]

X = df[features]
y = df["VWC"]

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X, y)

MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)

print("✅ VWC model trained.")