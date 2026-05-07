from __future__ import annotations

from datetime import date, timedelta
from typing import Any

from .schemas import AgentRecommendation, OrchestratedOutput


def _is_positive_irrigation(decision: str) -> bool:
    if not decision:
        return False
    t = decision.lower().strip()

    if "do not" in t or "no irrigation" in t or "delay" in t:
        return False

    return "irrigate" in t or "apply water" in t

def _is_positive_fertilization(decision: str) -> bool:
    if not decision:
        return False
    t = decision.lower().strip()

    if "do not" in t or "delay" in t or "hold" in t:
        return False

    return "apply" in t or "fertiliz" in t or "topdress" in t



def orchestrate(
    ctx: dict[str, Any],
    irrigation: AgentRecommendation,
    fertilization: AgentRecommendation,
) -> OrchestratedOutput:

    analytics = ctx.get("analytics", {})
    weather = ctx.get("weather", {})

    soil_moisture_status = analytics.get("soil_moisture_status", "Unknown")
    soil_moisture = float(analytics.get("soil_moisture", 0.0) or 0.0)
    soil_max = float(analytics.get("soil_moisture_max", 90.0) or 90.0)

    stage = (analytics.get("phenological_stage") or "").lower()

    rain24 = float(weather.get("rain_next_24h_mm", 0.0) or 0.0)
    rain_today = float(weather.get("rain_today_mm", 0.0) or 0.0)

    irrigated_yesterday = analytics.get("irrigated_yesterday", False)
    fertilized_yesterday = analytics.get("fertilized_yesterday", False)

    fert_count = int(analytics.get("fertilization_count", 0) or 0)
    fert_max = int(ctx.get("max_fertilizations", 3))

    total_dose = float(analytics.get("total_fertilizer_applied", 0.0) or 0.0)
    dose_max = float(ctx.get("max_total_dose", 200.0))

    irrigate = _is_positive_irrigation(irrigation.decision)
    fertilize = _is_positive_fertilization(fertilization.decision)

    priority: list[str] = []
    notes = ""

    # =====================================================
    # RULE 0 – Heavy Rain Override (Hard Stop)
    # =====================================================
    if rain24 >= 10.0:
        notes = f"Heavy rain expected (~{rain24} mm). Delay irrigation and fertilization."
        weekly_plan = build_weekly_plan(ctx, irrigation, fertilization, [], heavy_rain=True)
        return OrchestratedOutput([], notes, weekly_plan)

    # =====================================================
    # RULE 1 – Rain Cancellation (Daily)
    # =====================================================
    rain_threshold = ctx.get("rain_cancel_threshold_mm", 8.0)

    if irrigate and rain_today > rain_threshold:
        irrigate = False
        notes += "Irrigation cancelled due to rainfall. "

    # =====================================================
    # RULE 2 – Soil Saturation Protection
    # =====================================================
    if irrigate and soil_moisture >= soil_max:
        irrigate = False
        notes += "Soil already saturated; irrigation cancelled. "

    # =====================================================
    # RULE 3 – Minimum Irrigation Interval
    # =====================================================
    if irrigate and irrigated_yesterday:
        irrigate = False
        notes += "Irrigation skipped to avoid consecutive events. "

    # =====================================================
    # RULE 4 – Late Season Nitrogen Restriction
    # =====================================================
    late_stage = any(k in stage for k in ["late", "ripening", "harvest"])

    if fertilize and late_stage:
        fertilize = False
        notes += "Late-season stage detected; nitrogen not prioritized. "

    # =====================================================
    # RULE 5 – Minimum Fertilization Interval
    # =====================================================
    if fertilize and fertilized_yesterday:
        fertilize = False
        notes += "Fertilization skipped to avoid consecutive application. "

    # =====================================================
    # RULE 6 – Max Fertilization Frequency
    # =====================================================
    if fertilize and fert_count >= fert_max:
        fertilize = False
        notes += "Maximum fertilization events reached for this season. "

    # =====================================================
    # RULE 7 – Maximum Cumulative Dosage
    # =====================================================
    proposed_dose = float(getattr(fertilization, "dose_kg", 0.0) or 0.0)

    if fertilize and (total_dose + proposed_dose > dose_max):
        fertilize = False
        notes += "Cumulative fertilizer limit exceeded. "

    # =====================================================
    # RULE 8 – Moisture Dependency for Nitrogen
    # =====================================================
    if fertilize and soil_moisture_status == "Low" and not irrigate:
        fertilize = False
        notes += "Low soil moisture; irrigation required before fertilization. "

    # =====================================================
    # RULE 9 – Fertilizer Requires Irrigation (Absorption Rule)
    # =====================================================
    if fertilize and not irrigate:
        irrigate = True
        notes += "Irrigation scheduled to support nutrient absorption. "

    # =====================================================
    # RULE 10 – Priority Ordering Logic
    # =====================================================
    if soil_moisture_status == "Low" and irrigate:
        priority.append("irrigation")

    if fertilize:
        if "irrigation" in priority:
            priority.append("fertilization")
        else:
            priority = ["fertilization"]

    if not irrigate and not fertilize and not notes:
        notes = "No immediate irrigation or fertilization action required."

    weekly_plan = build_weekly_plan(
        ctx,
        irrigation,
        fertilization,
        priority,
        heavy_rain=False,
    )

    return OrchestratedOutput(
        priority_order=priority,
        combined_notes=notes.strip(),
        weekly_plan=weekly_plan,
    )


def build_weekly_plan(
    ctx: dict[str, Any],
    irrigation: AgentRecommendation,
    fertilization: AgentRecommendation,
    priority: list[str],
    heavy_rain: bool,
) -> list[dict]:
    today = date.today()
    weather = ctx.get("weather", {})
    daily_rain = weather.get("daily_rain_mm", {}) or {}

    plan: list[dict] = []
    scheduled_irrigation = False
    scheduled_fertilization = False

    for i in range(7):
        d = today + timedelta(days=i)
        iso = d.isoformat()
        rain = float(daily_rain.get(iso, 0.0) or 0.0)

        tasks: list[str] = []
        tasks.append(f"Check field condition; forecast rain today ≈ {rain:.1f} mm.")

        if heavy_rain and i < 2:
            tasks.append("Avoid irrigation/fertilizer due to heavy rain risk; reassess after rain.")
        else:
            if "irrigation" in priority and not scheduled_irrigation:
                if rain < 5.0 and i in (0, 1):
                    tasks.append(f"Irrigation: {irrigation.decision}. Timing: {irrigation.timing}.")
                    scheduled_irrigation = True

            if "fertilization" in priority and not scheduled_fertilization:
                if rain < 10.0 and i in (1, 2):
                    tasks.append(f"Fertilization: {fertilization.decision}. Timing: {fertilization.timing}.")
                    scheduled_fertilization = True

        tasks.append("Scout for weeds/pests/disease; record observations.")
        plan.append({"date": iso, "tasks": tasks})

    return plan