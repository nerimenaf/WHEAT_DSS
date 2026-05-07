import pandas as pd
import joblib
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

DATA_PATH = Path("data/wheat_only.csv")
MODEL_PATH = Path("models/npk_model.joblib")

# Load dataset
df = pd.read_csv(DATA_PATH)
df = df[df["Crop"] == "Wheat"]

# Features and targets
X = df[[
    "Temperature",
    "Humidity",
    "pH_Value",
    "Rainfall",
    "Soil_Type",
    "Variety",
]]

y = df[["Nitrogen", "Phosphorus", "Potassium"]]

# Preprocessing
categorical = ["Soil_Type", "Variety"]
numeric = ["Temperature", "Humidity", "pH_Value", "Rainfall"]

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
        ("num", "passthrough", numeric),
    ]
)

model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# Train
pipeline.fit(X, y)

# Save
MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(pipeline, MODEL_PATH)

print("✅ Model trained and saved.")