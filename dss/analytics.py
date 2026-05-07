from __future__ import annotations

from datetime import date, datetime
from typing import Any


def soil_moisture_status(moisture_pct: float, soil_type: str) -> str:
    # Simple illustrative thresholds (adjust to your local calibration)
    if moisture_pct < 25:
        return "Low"
    if moisture_pct <= 60:
        return "Adequate"
    return "High"


def soil_ph_status(ph: float) -> str:
    if ph < 6.0:
        return "Acidic"
    if ph <= 7.5:
        return "Optimal"
    return "Alkaline"


def nutrient_status(value_ppm: float, nutrient: str) -> str:
    """
    Simple illustrative thresholds.
    N: <20 deficient, 20-50 adequate, >50 high
    P: <10 deficient, 10-25 adequate, >25 high
    K: <80 deficient, 80-150 adequate, >150 high
    """
    nutrient = nutrient.upper()
    if nutrient == "N":
        low, high = 20, 50
    elif nutrient == "P":
        low, high = 10, 25
    elif nutrient == "K":
        low, high = 80, 150
    else:
        low, high = 0, 0

    if value_ppm < low:
        return "Deficient"
    if value_ppm <= high:
        return "Adequate"
    return "High"


def days_after_sowing(sowing_date: date) -> int:
    today = date.today()
    delta = (today - sowing_date).days
    return max(0, delta)  # safeguard for future sowing dates


def phenological_stage(das: int) -> str:
    """
    Very simple rule-of-thumb staging by days after sowing (DAS).
    Adjust as needed for variety and temperature/region.
    """
    if das == 0:
        return "Not sown yet or just sown (DAS=0 safeguard)"
    if 1 <= das <= 14:
        return "Germination / Emergence"
    if 15 <= das <= 35:
        return "Tillering"
    if 36 <= das <= 65:
        return "Stem elongation"
    if 66 <= das <= 90:
        return "Heading / Flowering"
    if 91 <= das <= 120:
        return "Grain filling"
    return "Maturity / Late season"


def compute_analytics(field: dict[str, Any]) -> dict[str, Any]:
    das = days_after_sowing(field["sowing_date"])
    return {
        "soil_moisture_status": soil_moisture_status(field["soil_moisture_pct"], field["soil_type"]),
        "soil_ph_status": soil_ph_status(field["soil_ph"]),
        "n_status": nutrient_status(field["n_ppm"], "N"),
        "p_status": nutrient_status(field["p_ppm"], "P"),
        "k_status": nutrient_status(field["k_ppm"], "K"),
        "days_after_sowing": das,
        "phenological_stage": phenological_stage(das),
        "computed_at": datetime.utcnow().isoformat() + "Z",
        "soil_moisture_status": soil_moisture_status(field["soil_moisture_pct"], field["soil_type"]),
    }