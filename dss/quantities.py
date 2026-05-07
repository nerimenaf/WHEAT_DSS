from __future__ import annotations

from typing import Any


def estimate_irrigation_mm(ctx: dict[str, Any]) -> dict[str, Any]:
    """
    Soil-type dependent irrigation estimator using root-zone water balance
    + optional VWC forecast adjustment.
    """

    a = ctx.get("analytics", {}) or {}
    w = ctx.get("weather", {}) or {}
    f = ctx.get("field", {}) or {}

    soil_status = a.get("soil_moisture_status", "Unknown")
    stage = a.get("phenological_stage", "")
    rain24 = float(w.get("rain_next_24h_mm", 0.0) or 0.0)

    soil_type = f.get("soil_type", "Loamy")
    soil_moisture_pct = float(f.get("soil_moisture_pct", 0.0) or 0.0)

    # --------------------------------------------------
    # 1️⃣ Soil water holding capacity (mm per meter)
    # --------------------------------------------------
    SOIL_WHC_MM_PER_M = {
        "Sandy": 80,
        "Loamy": 150,
        "Clay": 200,
        "Silty": 170,
        "Peaty": 140,
    }

    whc_per_m = SOIL_WHC_MM_PER_M.get(soil_type, 150)

    # --------------------------------------------------
    # 2️⃣ Root depth depends on stage
    # --------------------------------------------------
    if "Germination" in stage:
        root_depth_m = 0.3
    elif "Tillering" in stage:
        root_depth_m = 0.5
    elif "Stem elongation" in stage:
        root_depth_m = 0.7
    elif "Heading" in stage or "Flowering" in stage:
        root_depth_m = 0.9
    elif "Grain filling" in stage:
        root_depth_m = 1.0
    else:
        root_depth_m = 0.6  # default

    taw_mm = whc_per_m * root_depth_m

    # --------------------------------------------------
    # 3️⃣ Safeguards (rain + saturation)
    # --------------------------------------------------
    if soil_status == "High":
        gross = 0.0
        note = "Soil moisture high; avoid irrigation."

    elif rain24 >= 10.0:
        gross = 0.0
        note = "Heavy rain expected; delay irrigation."

    else:
        # --------------------------------------------------
        # 4️⃣ Target refill calculation
        # --------------------------------------------------
        target_pct = 70.0
        deficit_pct = max(0.0, target_pct - soil_moisture_pct)

        gross = (deficit_pct / 100.0) * taw_mm

        stage_critical = (
            "Heading" in stage or
            "Flowering" in stage or
            "Grain filling" in stage
        )

        if stage_critical:
            gross *= 1.15

        # --------------------------------------------------
        # ✅ 5️⃣ VWC Forecast Adjustment (SAFE POSITION)
        # --------------------------------------------------
        forecast = ctx.get("vwc_forecast", [])
        if forecast:
            min_future_vwc = min(forecast)
            if min_future_vwc < 25:
                gross *= 1.2  # increase irrigation 20%
                note = "Increased irrigation due to predicted low VWC."
            else:
                note = "Soil-type water balance model (WHC × root depth)."
        else:
            note = "Soil-type water balance model (WHC × root depth)."

        # Safety bounds
        gross = max(0.0, min(gross, taw_mm))

    gross = round(gross, 1)
    net = gross

    # --------------------------------------------------
    # 6️⃣ Convert to total volume if area known
    # --------------------------------------------------
    area_ha = float(f.get("field_area_ha", 0.0) or 0.0)
    total_m3 = None
    if area_ha > 0 and gross > 0:
        total_m3 = round(gross * area_ha * 10.0, 2)

    return {
        "soil_type": soil_type,
        "root_depth_m": round(root_depth_m, 2),
        "taw_mm": round(taw_mm, 1),
        "target_moisture_pct": 70.0,
        "net_mm": net,
        "gross_mm": gross,
        "total_m3_if_area_known": total_m3,
        "note": note,
    }


def estimate_fertilizer_rates(ctx: dict[str, Any]) -> dict[str, Any]:
    """
    Prototype fertilizer estimator (per hectare).
    Outputs nutrient targets and rough product conversions.
    """
    a = ctx.get("analytics", {}) or {}
    f = ctx.get("field", {}) or {}
    w = ctx.get("weather", {}) or {}

    das = int(a.get("days_after_sowing", 0) or 0)
    stage = a.get("phenological_stage", "")
    rain24 = float(w.get("rain_next_24h_mm", 0.0) or 0.0)

    n_status = a.get("n_status", "Unknown")
    p_status = a.get("p_status", "Unknown")
    k_status = a.get("k_status", "Unknown")

    # Very simple stage window preference: most topdressing early-mid season
    late_season = ("Maturity" in stage) or (das > 120)

    N = 0.0
    P2O5 = 0.0
    K2O = 0.0

    if not late_season:
        # Nitrogen
        if n_status == "Deficient":
            N = 50.0
        elif n_status == "Adequate":
            N = 25.0

        # Phosphorus (better earlier)
        if p_status == "Deficient" and das <= 45:
            P2O5 = 30.0

        # Potassium
        if k_status == "Deficient":
            K2O = 40.0

    note = "Prototype rates based on deficiency status + stage window; must be calibrated locally."

    if rain24 >= 10.0:
        note += " Heavy rain risk: consider delaying application to reduce loss."

    # Simple product conversions (not accounting for nutrient overlap like DAP provides N too)
    urea_kg_ha = round(N / 0.46, 1) if N > 0 else 0.0            # Urea ~46% N
    dap_kg_ha = round(P2O5 / 0.46, 1) if P2O5 > 0 else 0.0        # DAP ~46% P2O5
    mop_kg_ha = round(K2O / 0.60, 1) if K2O > 0 else 0.0          # MOP ~60% K2O

    area_ha = float(f.get("field_area_ha", 0.0) or 0.0)
    totals = None
    if area_ha > 0:
        totals = {
            "urea_kg_total": round(urea_kg_ha * area_ha, 1),
            "dap_kg_total": round(dap_kg_ha * area_ha, 1),
            "mop_kg_total": round(mop_kg_ha * area_ha, 1),
        }

    return {
        "N_kg_ha": round(N, 1),
        "P2O5_kg_ha": round(P2O5, 1),
        "K2O_kg_ha": round(K2O, 1),
        "urea_kg_ha": urea_kg_ha,
        "dap_kg_ha": dap_kg_ha,
        "mop_kg_ha": mop_kg_ha,
        "totals_if_area_known": totals,
        "note": note,
    }


def compute_estimates(ctx: dict[str, Any]) -> dict[str, Any]:
    return {
        "irrigation": estimate_irrigation_mm(ctx),
        "fertilization": estimate_fertilizer_rates(ctx),
    }