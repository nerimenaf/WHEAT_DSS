from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EPISODES_PATH = Path("memory/shared_episodes.json")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def ensure_memory() -> None:
    EPISODES_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not EPISODES_PATH.exists():
        EPISODES_PATH.write_text("[]", encoding="utf-8")


def load_episodes() -> list[dict[str, Any]]:
    ensure_memory()
    return json.loads(EPISODES_PATH.read_text(encoding="utf-8"))


def save_episodes(rows: list[dict[str, Any]]) -> None:
    ensure_memory()
    EPISODES_PATH.write_text(json.dumps(rows, indent=2), encoding="utf-8")


# -----------------------------
# Constraints -> CVR
# -----------------------------
def _schedule_has(schedule: dict[str, Any], keyword: str) -> bool:
    kw = keyword.lower()
    for day in schedule.get("weekly_plan", []) or []:
        for t in (day.get("tasks", []) or []):
            if kw in (t or "").lower():
                return True
    return False


def _conflict_days(schedule: dict[str, Any]) -> int:
    conflicts = 0
    for day in schedule.get("weekly_plan", []) or []:
        tasks = " ".join(day.get("tasks", []) or []).lower()
        if ("irrigation:" in tasks or "irrigate" in tasks) and ("fertilization:" in tasks or "fertiliz" in tasks):
            conflicts += 1
    return conflicts


def compute_constraints(ctx: dict[str, Any], schedule: dict[str, Any]) -> dict[str, Any]:
    a = ctx.get("analytics", {}) or {}
    w = ctx.get("weather", {}) or {}
    soil_status = a.get("soil_moisture_status", "Unknown")
    rain24 = float(w.get("rain_next_24h_mm", 0.0) or 0.0)

    violations: list[str] = []

    # C1 heavy rain + operations scheduled
    if rain24 >= 10.0 and (_schedule_has(schedule, "irrigation") or _schedule_has(schedule, "fertil")):
        violations.append("Heavy rain expected (>=10mm) but operations scheduled.")

    # C2 soil high + irrigation scheduled
    if soil_status == "High" and _schedule_has(schedule, "irrig"):
        violations.append("Soil moisture high but irrigation scheduled.")

    # C3 conflict same day
    cdays = _conflict_days(schedule)
    if cdays > 0:
        violations.append(f"Irrigation and fertilization scheduled same day ({cdays}).")

    # C4 if soil low and both actions, irrigation should be first
    pr = schedule.get("priority_order", []) or []
    if soil_status == "Low" and ("irrigation" in pr) and ("fertilization" in pr) and pr[0] != "irrigation":
        violations.append("Soil dry but fertilization prioritized before irrigation.")

    total = 4
    cvr = len(violations) / total

    return {
        "violations": violations,
        "total_constraints": total,
        "cvr": round(cvr, 3),
        "conflict_days": cdays,
        "conflict_rate": round(cdays / 7.0, 3),
    }


def compute_feedback(constraints: dict[str, Any]) -> dict[str, Any]:
    cvr_raw = constraints.get("cvr", 1.0)
    if cvr_raw is None:
        cvr_raw = 1.0
    cvr = float(cvr_raw)

    score = max(0.0, min(1.0, 1.0 - cvr))
    return {"score": round(score, 3), "method": "1-CVR"}


# -----------------------------
# Metrics (WUE/FUE proxy + stability)
# -----------------------------
def _first_number(text: str) -> float | None:
    if not text:
        return None
    m = re.search(r"([-+]?\d+(\.\d+)?)", text)
    return float(m.group(1)) if m else None


def _sum_numbers(text: str) -> float:
    if not text:
        return 0.0
    nums = re.findall(r"([-+]?\d+(\.\d+)?)", text)
    return float(sum(float(n[0]) for n in nums)) if nums else 0.0


def _decision_flip(prev: str | None, curr: str | None) -> int:
    if not prev or not curr:
        return 0
    return 1 if prev.strip().lower() != curr.strip().lower() else 0


def compute_episode_metrics(
    *,
    episode: dict[str, Any],
    prev_episode: dict[str, Any] | None,
) -> dict[str, Any]:
    """
    Produces:
    - CVR/conflict_rate from constraints
    - WUE_proxy, FUE_proxy (prototype)
    - stability / flip_count relative to previous episode
    """
    constraints = episode.get("constraints_t", {}) or {}
    outcome = episode.get("outcome_t", {}) or {}
    decisions = episode.get("decisions_t", {}) or {}

    # amounts (best effort)
    irr_qty = ((decisions.get("irrigation", {}) or {}).get("quantity", "") or "")
    fert_qty = ((decisions.get("fertilization", {}) or {}).get("quantity", "") or "")

    irrigation_mm = _first_number(irr_qty) or 0.0
    fertilizer_amount_proxy = _sum_numbers(fert_qty)  # proxy: sums numbers in "Urea 100, DAP 50, ..."

    # outcomes (proxy)
    d_moist = float(outcome.get("delta_soil_moisture_pct", 0.0) or 0.0)
    d_n = float(outcome.get("delta_n_ppm", 0.0) or 0.0)
    d_p = float(outcome.get("delta_p_ppm", 0.0) or 0.0)
    d_k = float(outcome.get("delta_k_ppm", 0.0) or 0.0)

    wue = None
    if irrigation_mm > 0:
        wue = d_moist / irrigation_mm

    fue = None
    if fertilizer_amount_proxy > 0:
        fue = (d_n + d_p + d_k) / fertilizer_amount_proxy

    # stability vs previous episode (decision flips)
    flip_count = 0
    stability = 1.0
    if prev_episode:
        prev_dec = prev_episode.get("decisions_t", {}) or {}
        flip_count = (
            _decision_flip((prev_dec.get("irrigation", {}) or {}).get("decision"), (decisions.get("irrigation", {}) or {}).get("decision"))
            + _decision_flip((prev_dec.get("fertilization", {}) or {}).get("decision"), (decisions.get("fertilization", {}) or {}).get("decision"))
        )
        stability = 1.0 - (flip_count / 2.0)

    return {
        "cvr": constraints.get("cvr"),
        "conflict_rate": constraints.get("conflict_rate"),
        "irrigation_mm_proxy": round(irrigation_mm, 3),
        "fertilizer_amount_proxy": round(fertilizer_amount_proxy, 3),
        "WUE_proxy": None if wue is None else round(wue, 6),
        "FUE_proxy": None if fue is None else round(fue, 6),
        "flip_count": int(flip_count),
        "stability": round(float(stability), 3),
        "notes": "WUE/FUE are prototype proxies (no yield). Stability is based on decision flips vs previous episode.",
    }


# -----------------------------
# Episode lifecycle
# -----------------------------
def start_episode(
    *,
    ctx: dict[str, Any],
    irrigation_decision: dict[str, Any],
    fertilization_decision: dict[str, Any],
    schedule: dict[str, Any],
    system_config: dict[str, Any],
) -> str:
    rows = load_episodes()

    constraints = compute_constraints(ctx, schedule)
    feedback = compute_feedback(constraints)

    eid = str(uuid.uuid4())
    rows.append({
        "episode_id": eid,
        "time_utc": _utc_now_iso(),
        "system_config": system_config,  # stores toggles for experiments
        "context_t": ctx,
        "decisions_t": {
            "irrigation": irrigation_decision,
            "fertilization": fertilization_decision,
        },
        "schedule_t": schedule,
        "constraints_t": constraints,
        "feedback_t": feedback,
        "outcome_t": None,
        "context_t_plus_1": None,
        "metrics_t": None,  # computed after outcome is known
    })
    save_episodes(rows)
    return eid


def finalize_episode_outcome(*, episode_id: str, ctx_after: dict[str, Any]) -> dict[str, Any] | None:
    rows = load_episodes()
    idx = next((i for i, r in enumerate(rows) if r.get("episode_id") == episode_id), None)
    if idx is None:
        return None

    before = rows[idx].get("context_t", {}) or {}
    fb = before.get("field", {}) or {}
    fa = (ctx_after.get("field", {}) or {})

    def fnum(x, default=0.0):
        try:
            return float(x)
        except Exception:
            return float(default)

    outcome = {
        "delta_soil_moisture_pct": round(fnum(fa.get("soil_moisture_pct")) - fnum(fb.get("soil_moisture_pct")), 3),
        "delta_n_ppm": round(fnum(fa.get("n_ppm")) - fnum(fb.get("n_ppm")), 3),
        "delta_p_ppm": round(fnum(fa.get("p_ppm")) - fnum(fb.get("p_ppm")), 3),
        "delta_k_ppm": round(fnum(fa.get("k_ppm")) - fnum(fb.get("k_ppm")), 3),
        "latest_actions": (ctx_after.get("actions", {}) or {}).get("latest", {}),
    }

    rows[idx]["outcome_t"] = outcome
    rows[idx]["context_t_plus_1"] = ctx_after

    # compute metrics using previous episode (idx-1) as reference for stability
    prev_ep = rows[idx - 1] if idx > 0 else None
    rows[idx]["metrics_t"] = compute_episode_metrics(episode=rows[idx], prev_episode=prev_ep)

    save_episodes(rows)
    return rows[idx]