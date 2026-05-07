from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import requests
from typing import Any

from .config import settings


@dataclass
class WeatherSummary:
    source: str
    fetched_at_utc: str
    lat: float
    lon: float
    rain_next_24h_mm: float
    avg_temp_next_24h_c: float
    avg_humidity_next_24h_pct: float
    max_wind_next_24h_ms: float
    notes: str
    # Optional daily summaries (next ~5 days based on 3h forecast)
    daily_rain_mm: dict[str, float]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_openweather_forecast(lat: float, lon: float) -> dict[str, Any]:
    if not settings.openweather_api_key:
        raise RuntimeError("OPENWEATHER_API_KEY not set.")

    params = {
        "lat": lat,
        "lon": lon,
        "appid": settings.openweather_api_key,
        "units": settings.units,
    }
    r = requests.get(settings.openweather_forecast_url, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def summarize_forecast(forecast_json: dict[str, Any], lat: float, lon: float) -> WeatherSummary:
    """
    Uses the OpenWeather 5-day/3h forecast list to compute a near-term (next 24h) summary
    and a simple daily rainfall summary (UTC day buckets).
    """
    lst = forecast_json.get("list", [])
    if not lst:
        return WeatherSummary(
            source="openweather",
            fetched_at_utc=_utc_now_iso(),
            lat=lat,
            lon=lon,
            rain_next_24h_mm=0.0,
            avg_temp_next_24h_c=0.0,
            avg_humidity_next_24h_pct=0.0,
            max_wind_next_24h_ms=0.0,
            notes="No forecast list data returned.",
            daily_rain_mm={},
        )

    now_utc = datetime.now(timezone.utc)
    horizon = now_utc + timedelta(hours=24)

    rain_24 = 0.0
    temps = []
    hums = []
    winds = []

    daily_rain: dict[str, float] = {}

    for item in lst:
        dt_utc = datetime.fromtimestamp(item["dt"], tz=timezone.utc)

        # Daily rain bucket
        day_key = dt_utc.date().isoformat()
        rain_3h = float(item.get("rain", {}).get("3h", 0.0) or 0.0)
        daily_rain[day_key] = daily_rain.get(day_key, 0.0) + rain_3h

        # Next 24h summary
        if now_utc <= dt_utc <= horizon:
            rain_24 += rain_3h
            main = item.get("main", {})
            wind = item.get("wind", {})
            if "temp" in main:
                temps.append(float(main["temp"]))
            if "humidity" in main:
                hums.append(float(main["humidity"]))
            if "speed" in wind:
                winds.append(float(wind["speed"]))

    def avg(xs: list[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    return WeatherSummary(
        source="openweather",
        fetched_at_utc=_utc_now_iso(),
        lat=lat,
        lon=lon,
        rain_next_24h_mm=round(rain_24, 2),
        avg_temp_next_24h_c=round(avg(temps), 2),
        avg_humidity_next_24h_pct=round(avg(hums), 2),
        max_wind_next_24h_ms=round(max(winds) if winds else 0.0, 2),
        notes="Near-term summary computed from 3-hourly forecast steps.",
        daily_rain_mm={k: round(v, 2) for k, v in daily_rain.items()},
    )


def get_weather_summary(lat: float, lon: float) -> dict[str, Any]:
    """
    Returns a JSON-serializable dict. If API key missing or call fails, returns a fallback stub.
    """
    try:
        forecast = fetch_openweather_forecast(lat, lon)
        summary = summarize_forecast(forecast, lat, lon)
        return summary.__dict__
    except Exception as e:
        return {
            "source": "fallback",
            "fetched_at_utc": _utc_now_iso(),
            "lat": lat,
            "lon": lon,
            "rain_next_24h_mm": 0.0,
            "avg_temp_next_24h_c": 0.0,
            "avg_humidity_next_24h_pct": 0.0,
            "max_wind_next_24h_ms": 0.0,
            "notes": f"Weather unavailable (using fallback). Error: {type(e).__name__}: {e}",
            "daily_rain_mm": {},
        }