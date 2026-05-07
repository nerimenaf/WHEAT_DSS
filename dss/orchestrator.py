from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone
from typing import Any

from .schemas import AgentRecommendation, OrchestratedOutput


def _first_number(text: str) -> float | None:
    if not text:
        return None
    m = re.search(r"([-+]?\d+(\.\d+)?)", text)
    return float(m.group(1)) if m else None


def _action_requested(rec: AgentRecommendation, kind: str) -> bool:
    """
    Robust detector:
    - If quantity contains a number: 0 => no action, >0 => action
    - else fallback to decision keywords
    """
    q = (getattr(rec, "quantity", "") or "").lower()
    d = (rec.decision or "").lower()

    num = _first_number(q)
    if num is not None:
        return num > 0.0

    negatives = ["do not", "don't", "avoid", "hold", "delay", "postpone", "no "]
    if any(n in d for n in negatives):
        return False

    if kind == "irrigation":
        return any(k in d for k in ["irrigate", "apply water", "irrigation"])
    if kind == "fertilization":
        return any(k in d for k in ["apply", "fertiliz", "topdress", "top-dress"])
    return False


def _parse_time_utc(s: str) -> datetime | None:
    if not s:
        return None
    try:
        # supports "...+00:00"
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _was_done_yesterday(ctx: dict[str, Any], action_type: str) -> bool:
    """
    Uses ctx['actions']['latest'][action_type]['time_utc'] if available.
    """
    latest = ((ctx.get("actions", {}) or {}).get("latest", {}) or {}).get(action_type)
    if not latest or latest.get("status") != "done":
        return False
    t = _parse_time_utc(latest.get("time_utc", ""))
    if not t:
        return False
    return (datetime.now(timezone.utc).date() - t.date()).days <= 1


def _count_done(ctx: dict[str, Any], action_type: str) -> int:
    hist = ((ctx.get("actions", {}) or {}).get("history", []) or [])
    return sum(1 for e in hist if e.get("action_type") == action_type and e.get("status") == "done")


def orchestrate(
    ctx: dict[str, Any],
    irrigation: AgentRecommendation,
    fertilization: AgentRecommendation,
) -> OrchestratedOutput:

    analytics = ctx.get("analytics", {}) or {}
    weather = ctx.get("weather", {}) or {}
    field = ctx.get("field", {}) or {}

    soil_moisture_status = analytics.get("soil_moisture_status", "Unknown")
    soil_moisture_pct = float(field.get("soil_moisture_pct", 0.0) or 0.0)

    stage = (analytics.get("phenological_stage") or "").lower()

    rain24 = float(weather.get("rain_next_24h_mm", 0.0) or 0.0)
    daily_rain = weather.get("daily_rain_mm", {}) or {}
    rain_today = float(daily_rain.get(date.today().isoformat(), 0.0) or 0.0)

    irrigated_yesterday = _was_done_yesterday(ctx, "irrigation")
    fertilized_yesterday = _was_done_yesterday(ctx, "fertilization")

    fert_count = _count_done(ctx, "fertilization")
    fert_max = int(ctx.get("max_fertilizations", 3) or 3)

    irrigate = _action_requested(irrigation, "irrigation")
    fertilize = _action_requested(fertilization, "fertilization")

    priority: list[str] = []
    notes_parts: list[str] = []

    # RULE 0 – Heavy Rain Override
    if rain24 >= 10.0:
        notes = f"Heavy rain expected (~{rain24} mm). Delay irrigation and fertilization."
        weekly_plan = build_weekly_plan(ctx, irrigation, fertilization, [], heavy_rain=True)
        return OrchestratedOutput(priority_order=[], combined_notes=notes, weekly_plan=weekly_plan)

    # RULE 1 – Rain cancellation (daily)
    rain_threshold = float(ctx.get("rain_cancel_threshold_mm", 8.0) or 8.0)
    if irrigate and rain_today > rain_threshold:
        irrigate = False
        notes_parts.append("Irrigation cancelled due to rainfall today.")

    # RULE 2 – Soil saturation protection (use moisture %)
    # (Prototype threshold; tune as needed)
    if irrigate and soil_moisture_pct >= 70.0:
        irrigate = False
        notes_parts.append("Soil already wet; irrigation cancelled.")

    # RULE 3 – Minimum irrigation interval
    if irrigate and irrigated_yesterday:
        irrigate = False
        notes_parts.append("Irrigation skipped to avoid consecutive events.")

    # RULE 4 – Late-season restriction (simple keyword rule)
    late_stage = any(k in stage for k in ["maturity", "late", "ripening", "harvest"])
    if fertilize and late_stage:
        fertilize = False
        notes_parts.append("Late-season stage detected; fertilization not prioritized.")

    # RULE 5 – Minimum fertilization interval
    if fertilize and fertilized_yesterday:
        fertilize = False
        notes_parts.append("Fertilization skipped to avoid consecutive application.")

    # RULE 6 – Max fertilization frequency
    if fertilize and fert_count >= fert_max:
        fertilize = False
        notes_parts.append("Maximum fertilization events reached for this season.")

    # RULE 8 – Moisture dependency for fertilization
    # If soil is low and fertilization is needed, ensure irrigation comes first
    if fertilize and soil_moisture_status == "Low":
        if not irrigate:
            irrigate = True
            notes_parts.append("Low soil moisture; schedule irrigation before fertilization.")
        else:
            notes_parts.append("Low soil moisture; irrigate first then fertilize.")

    # Priority ordering
    if irrigate:
        priority.append("irrigation")
    if fertilize:
        if "irrigation" in priority:
            priority.append("fertilization")
        else:
            priority = ["fertilization"]

    if not priority and not notes_parts:
        notes_parts.append("No immediate irrigation or fertilization action required.")

    weekly_plan = build_weekly_plan(ctx, irrigation, fertilization, priority, heavy_rain=False)

    return OrchestratedOutput(
        priority_order=priority,
        combined_notes=" ".join(notes_parts).strip(),
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
    daily_rain = weather.get("daily_rain_mm") or {}

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
            if "irrigation" in priority and not scheduled_irrigation and rain < 5.0 and i in (0, 1):
                tasks.append(
                    f"Irrigation: {irrigation.decision}. Qty: {getattr(irrigation,'quantity','')}. Timing: {irrigation.timing}."
                )
                scheduled_irrigation = True

            if "fertilization" in priority and not scheduled_fertilization and rain < 10.0 and i in (1, 2):
                tasks.append(
                    f"Fertilization: {fertilization.decision}. Qty: {getattr(fertilization,'quantity','')}. Timing: {fertilization.timing}."
                )
                scheduled_fertilization = True

        tasks.append("Scout for weeds/pests/disease; record observations.")
        plan.append({"date": iso, "tasks": tasks})

    return plan