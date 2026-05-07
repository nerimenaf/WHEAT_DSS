from typing import Dict, Any, List


def build_time_series(
    moisture_start: float,
    moisture_trend: float,
    rain_pattern: List[float],
    temp_base: float,
) -> List[Dict[str, Any]]:

    series = []
    moisture = moisture_start

    for i in range(7):
        series.append({
            "day": i + 1,
            "soil_moisture": round(moisture, 1),
            "rain_mm": rain_pattern[i],
            "temperature": temp_base + (i % 3),
            "et0": round(4.5 + (temp_base - 25) * 0.1, 2),
        })
        moisture += moisture_trend

    return series


SCENARIOS: Dict[int, Dict[str, Any]] = {

    # =============================
    # FARM A – Semi-arid (Loamy-Sandy)
    # =============================

    1: {
        "name": "Tillering – Drought Onset",
        "farm": "A",
        "climate": "semi-arid",
        "soil_type": "Loamy",
        "stage": "Tillering",
        "description": "Progressive soil drying without rainfall.",
        "time_series": build_time_series(
            moisture_start=20,
            moisture_trend=-1,
            rain_pattern=[0,0,0,0,0,0,0],
            temp_base=28,
        ),
        "ground_truth": {
            "irrigation_weekly_range": (35, 49),
            "fertilization": {"N": 60, "P": 30, "K": 20},
        },
        "evaluation_focus": "Water stress + smooth scheduling"
    },

    2: {
        "name": "Stem Elongation – Heatwave",
        "farm": "A",
        "climate": "semi-arid",
        "soil_type": "Loamy",
        "stage": "Stem elongation",
        "description": "High ET demand with rapid moisture depletion.",
        "time_series": build_time_series(
            moisture_start=18,
            moisture_trend=-1.5,
            rain_pattern=[0,0,0,0,0,0,0],
            temp_base=34,
        ),
        "ground_truth": {
            "irrigation_weekly_range": (49, 70),
            "fertilization": {"N": 80, "P": 40, "K": 30},
        },
        "evaluation_focus": "High-demand irrigation + stability"
    },

    3: {
        "name": "Heading – Limited Water",
        "farm": "A",
        "climate": "semi-arid",
        "soil_type": "Loamy",
        "stage": "Heading / Flowering",
        "description": "Deficit irrigation constraint.",
        "time_series": build_time_series(
            moisture_start=22,
            moisture_trend=-0.5,
            rain_pattern=[0,0,0,0,0,0,0],
            temp_base=30,
        ),
        "ground_truth": {
            "irrigation_weekly_range": (21, 35),
            "fertilization": {"N": 40, "P": 20, "K": 20},
        },
        "evaluation_focus": "Water use efficiency"
    },

    4: {
        "name": "Grain Filling – Late Fertilization",
        "farm": "A",
        "climate": "semi-arid",
        "soil_type": "Loamy",
        "stage": "Grain filling",
        "description": "N deficiency during grain filling.",
        "time_series": build_time_series(
            moisture_start=19,
            moisture_trend=-0.8,
            rain_pattern=[0,0,0,0,0,0,0],
            temp_base=29,
        ),
        "ground_truth": {
            "irrigation_weekly_range": (28, 42),
            "fertilization": {"N": 50, "P": 0, "K": 0},
        },
        "evaluation_focus": "Late-stage nutrient management"
    },

    5: {
        "name": "Mid-week Rain Event",
        "farm": "A",
        "climate": "semi-arid",
        "soil_type": "Loamy",
        "stage": "Tillering",
        "description": "Rain at day 3–4 affecting irrigation schedule.",
        "time_series": build_time_series(
            moisture_start=18,
            moisture_trend=-0.5,
            rain_pattern=[0,0,12,15,0,0,0],
            temp_base=27,
        ),
        "ground_truth": {
            "irrigation_weekly_range": (16, 30),
            "fertilization": {"N": 60, "P": 30, "K": 20},
        },
        "evaluation_focus": "Weather-aware scheduling"
    },

    6: {
        "name": "Over-Irrigation Risk",
        "farm": "A",
        "climate": "semi-arid",
        "soil_type": "Loamy",
        "stage": "Tillering",
        "description": "Soil already near field capacity.",
        "time_series": build_time_series(
            moisture_start=25,
            moisture_trend=0,
            rain_pattern=[0,0,0,0,0,0,0],
            temp_base=26,
        ),
        "ground_truth": {
            "irrigation_weekly_range": (0, 14),
            "fertilization": {"N": 60, "P": 30, "K": 20},
        },
        "evaluation_focus": "Constraint enforcement"
    },

    # =============================
    # FARM B – Humid (Clay Soil)
    # =============================

    7: {
        "name": "Waterlogging Risk",
        "farm": "B",
        "climate": "humid",
        "soil_type": "Clay",
        "stage": "Tillering",
        "description": "Heavy rain and high moisture.",
        "time_series": build_time_series(
            moisture_start=32,
            moisture_trend=0.5,
            rain_pattern=[20,25,18,22,15,0,0],
            temp_base=22,
        ),
        "ground_truth": {
            "irrigation_weekly_range": (0, 0),
            "fertilization": {"N": 0, "P": 0, "K": 0},
        },
        "evaluation_focus": "Conflict avoidance"
    },

    8: {
        "name": "Nutrient Leaching",
        "farm": "B",
        "climate": "humid",
        "soil_type": "Clay",
        "stage": "Tillering",
        "description": "Nitrogen loss after heavy rainfall.",
        "time_series": build_time_series(
            moisture_start=30,
            moisture_trend=-0.5,
            rain_pattern=[30,0,0,0,0,0,0],
            temp_base=23,
        ),
        "ground_truth": {
            "irrigation_weekly_range": (0, 21),
            "fertilization": {"N": 70, "P": 30, "K": 20},
        },
        "evaluation_focus": "Adaptive fertilization"
    },

    9: {
        "name": "Clay Saturation Delay",
        "farm": "B",
        "climate": "humid",
        "soil_type": "Clay",
        "stage": "Stem elongation",
        "description": "High water retention.",
        "time_series": build_time_series(
            moisture_start=28,
            moisture_trend=-0.3,
            rain_pattern=[5,0,0,0,0,0,0],
            temp_base=24,
        ),
        "ground_truth": {
            "irrigation_weekly_range": (14, 28),
            "fertilization": {"N": 60, "P": 30, "K": 20},
        },
        "evaluation_focus": "Soil-specific reasoning"
    },

    10: {
        "name": "Alternating Rain & Sun",
        "farm": "B",
        "climate": "humid",
        "soil_type": "Clay",
        "stage": "Tillering",
        "description": "Unstable weather conditions.",
        "time_series": build_time_series(
            moisture_start=26,
            moisture_trend=-0.4,
            rain_pattern=[10,0,15,0,8,0,0],
            temp_base=25,
        ),
        "ground_truth": {
            "irrigation_weekly_range": (0, 35),
            "fertilization": {"N": 60, "P": 30, "K": 20},
        },
        "evaluation_focus": "Responsiveness + stability"
    },

    11: {
        "name": "Late-stage Potassium Boost",
        "farm": "B",
        "climate": "humid",
        "soil_type": "Clay",
        "stage": "Grain filling",
        "description": "Yield optimization phase.",
        "time_series": build_time_series(
            moisture_start=24,
            moisture_trend=-0.5,
            rain_pattern=[0,0,0,0,0,0,0],
            temp_base=27,
        ),
        "ground_truth": {
            "irrigation_weekly_range": (21, 35),
            "fertilization": {"N": 40, "P": 20, "K": 50},
        },
        "evaluation_focus": "FUE optimization"
    },

    12: {
        "name": "Combined Stress (Rain + Heat)",
        "farm": "B",
        "climate": "humid",
        "soil_type": "Clay",
        "stage": "Heading / Flowering",
        "description": "Rain followed by heat spike.",
        "time_series": build_time_series(
            moisture_start=27,
            moisture_trend=-1,
            rain_pattern=[15,0,0,0,0,0,0],
            temp_base=32,
        ),
        "ground_truth": {
            "irrigation_weekly_range": (14, 35),
            "fertilization": {"N": 50, "P": 20, "K": 20},
        },
        "evaluation_focus": "Complex coordination stress test"
    },
}