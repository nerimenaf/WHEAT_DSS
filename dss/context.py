from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import date, datetime, timezone

from .analytics import compute_analytics
from .weather import get_weather_summary
from .quantities import compute_estimates

from .npk_predictor import predict_npk
from .vwc_forecaster import predict_vwc_7days
from datetime import date, timedelta

from dss import weather

DEFAULT_CONTEXT_PATH = Path("shared_context.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_actions_section(ctx: dict[str, Any]) -> dict[str, Any]:
    ctx.setdefault("actions", {})
    ctx["actions"].setdefault("history", [])
    ctx["actions"].setdefault("latest", {})
    return ctx


def build_shared_context(
    *,
    field_name: str,
    soil_moisture_pct: float,
    soil_ph: float,
    n_ppm: float,
    p_ppm: float,
    k_ppm: float,
    sowing_date: date,
    lat: float,
    lon: float,
    soil_type: str,
    field_area_ha: float = 0.0,
) -> dict[str, Any]:

    # ----------------------------------
    # 1️⃣ Build initial field
    # ----------------------------------
    field = {
        "field_name": field_name,
        "crop_type": "wheat",
        "soil_moisture_pct": float(soil_moisture_pct),
        "soil_ph": float(soil_ph),
        "n_ppm": float(n_ppm),
        "p_ppm": float(p_ppm),
        "k_ppm": float(k_ppm),
        "sowing_date": sowing_date,
        "location": {"lat": float(lat), "lon": float(lon)},
        "soil_type": soil_type,
        "field_area_ha": float(field_area_ha),
    }

    # ----------------------------------
    # 2️⃣ Get weather first
    # ----------------------------------
    weather = get_weather_summary(lat, lon)

    # ----------------------------------
    # 3️⃣ ML Prediction (override NPK)
    # ----------------------------------
    try:
        predicted_npk = predict_npk(
            temperature=weather.get("avg_temp_next_24h_c", 20.0),
            humidity=weather.get("avg_humidity_next_24h_pct", 60.0),
            ph=soil_ph,
            rainfall=weather.get("rain_next_24h_mm", 0.0),
            soil_type=soil_type,
            variety="Hard Red",
        )

        field["n_ppm"] = predicted_npk["Nitrogen"]
        field["p_ppm"] = predicted_npk["Phosphorus"]
        field["k_ppm"] = predicted_npk["Potassium"]

        ml_layer = {
            "npk_overridden": True,
            "predicted_npk": predicted_npk,
            "note": "NPK overridden by ML prediction model."
        }

    except Exception as e:
        ml_layer = {
            "npk_overridden": False,
            "error": str(e)
        }
    
    # ----------------------------------
    # 3️⃣ ML Prediction (VWC forecast)
    # ----------------------------------


    weather_forecast = []

    daily_rain = weather.get("daily_rain_mm") or {}

    for i in range(7):
        d = (date.today() + timedelta(days=i)).isoformat()

        weather_forecast.append({
            "precip": daily_rain.get(d, 0.0),
            "temp": weather.get("avg_temp_next_24h_c", 20.0),
            "wind": weather.get("max_wind_next_24h_ms", 2.0),
            "humidity": weather.get("avg_humidity_next_24h_pct", 70.0),
            "solar": 100,
            "pe": 3,
        })

    vwc_7days = predict_vwc_7days(
        initial_vwc=field["soil_moisture_pct"],
        weather_forecast=weather_forecast,
    )


    # ----------------------------------
    # 4️⃣ Now compute analytics AFTER override
    # ----------------------------------
    analytics = compute_analytics(field)

    # ----------------------------------
    # 5️⃣ Prepare JSON-safe field
    # ----------------------------------
    field_jsonable = dict(field)
    field_jsonable["sowing_date"] = sowing_date.isoformat()

    # ----------------------------------
    # 6️⃣ Build final context
    # ----------------------------------
    ctx = {
        "field": field_jsonable,
        "analytics": analytics,
        "weather": weather,
        "ml_layer": ml_layer,
        "vwc_forecast": vwc_7days,
    }

    ctx = ensure_actions_section(ctx)

    # Estimates depend on analytics + overridden NPK
    ctx["estimates"] = compute_estimates(ctx)

    return ctx

def save_shared_context(ctx: dict[str, Any], path: Path = DEFAULT_CONTEXT_PATH) -> None:
    path.write_text(json.dumps(ctx, indent=2), encoding="utf-8")


def load_shared_context(path: Path = DEFAULT_CONTEXT_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def recompute_analytics_in_context(ctx: dict[str, Any]) -> dict[str, Any]:
    field = dict(ctx["field"])
    field["sowing_date"] = date.fromisoformat(field["sowing_date"])
    ctx["analytics"] = compute_analytics(field)
    ctx["estimates"] = compute_estimates(ctx)
    return ctx


def record_farmer_action(
    ctx: dict[str, Any],
    *,
    action_type: str,
    status: str,
    notes: str = "",
) -> dict[str, Any]:
    ensure_actions_section(ctx)
    event = {
        "time_utc": _utc_now_iso(),
        "action_type": action_type,
        "status": status,
        "notes": notes,
    }
    ctx["actions"]["history"].append(event)
    ctx["actions"]["latest"][action_type] = event
    return ctx


def apply_irrigation_effect(ctx: dict[str, Any], increase_pct: float) -> dict[str, Any]:
    old = float(ctx["field"]["soil_moisture_pct"])
    new = max(0.0, min(100.0, old + float(increase_pct)))
    ctx["field"]["soil_moisture_pct"] = round(new, 2)
    return recompute_analytics_in_context(ctx)


def apply_fertilization_effect(
    ctx: dict[str, Any],
    *,
    n_add_ppm: float = 0.0,
    p_add_ppm: float = 0.0,
    k_add_ppm: float = 0.0,
) -> dict[str, Any]:
    ctx["field"]["n_ppm"] = round(max(0.0, float(ctx["field"]["n_ppm"]) + float(n_add_ppm)), 2)
    ctx["field"]["p_ppm"] = round(max(0.0, float(ctx["field"]["p_ppm"]) + float(p_add_ppm)), 2)
    ctx["field"]["k_ppm"] = round(max(0.0, float(ctx["field"]["k_ppm"]) + float(k_add_ppm)), 2)
    return recompute_analytics_in_context(ctx)