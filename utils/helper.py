"""Helper utilities for OptiCrop.

Contains the crop knowledge base (benefits, seasons, water needs, fertilizer
recommendations, etc.), suitability analysis for Scenario 2, and dashboard
analytics for the Research Dashboard (Scenario 3).
"""
import numpy as np
import pandas as pd

from . import preprocess as pp


# ---------------------------------------------------------------------------
# Crop knowledge base
# ---------------------------------------------------------------------------
CROP_INFO = {
    "rice": {
        "season": "Kharif (Monsoon season, June–November)",
        "water": "High (1200–2500 mm over the season). Needs standing water in paddies.",
        "fertilizer": "Apply 80–100 kg N/ha, 40 kg P/ha, 40 kg K/ha. Split N into 3 doses.",
        "productivity": "2.5–4.0 tonnes/ha (India average ~2.7 t/ha).",
        "benefits": "Staple food for billions; grows in waterlogged soils unsuitable for other crops; supports rural employment.",
        "practices": "Maintain 2–5 cm standing water; use SRI technique for higher yield; practice crop rotation with legumes.",
        "ideal": {"N": (80, 100), "P": (30, 50), "K": (35, 50), "temperature": (22, 26), "humidity": (75, 90), "ph": (5.5, 6.8), "rainfall": (180, 240)},
    },
    "maize": {
        "season": "Kharif (monsoon) and Rabi (winter) in some regions.",
        "water": "Moderate (500–800 mm). Sensitive to moisture stress at flowering.",
        "fertilizer": "120 kg N/ha, 60 kg P/ha, 40 kg K/ha. Apply N in 3 splits.",
        "productivity": "3.0–5.5 tonnes/ha.",
        "benefits": "Versatile cereal for food, feed, and biofuel; short duration crop; high yield potential.",
        "practices": "Sow at 60×20 cm spacing; ensure drainage; top-dress N at 30 DAS; control fall armyworm.",
        "ideal": {"N": (70, 90), "P": (40, 55), "K": (18, 28), "temperature": (22, 26), "humidity": (58, 72), "ph": (5.8, 6.8), "rainfall": (70, 100)},
    },
    "chickpea": {
        "season": "Rabi (October–March).",
        "water": "Low (250–350 mm). Drought-tolerant pulse.",
        "fertilizer": "20 kg N/ha (starter), 40–50 kg P/ha, 20 kg K/ha. Rhizobium inoculation recommended.",
        "productivity": "1.0–1.8 tonnes/ha.",
        "benefits": "Fixes atmospheric nitrogen; rich source of plant protein; improves soil fertility.",
        "practices": "Sow at 30×10 cm; treat seed with Rhizobium; irrigate once at pod-filling; avoid waterlogging.",
        "ideal": {"N": (30, 50), "P": (60, 80), "K": (70, 90), "temperature": (18, 24), "humidity": (40, 50), "ph": (6.5, 7.5), "rainfall": (65, 95)},
    },
    "kidneybeans": {
        "season": "Kharif and Rabi in mild climates.",
        "water": "Moderate (400–500 mm). Sensitive to waterlogging.",
        "fertilizer": "25 kg N/ha, 50 kg P/ha, 20 kg K/ha.",
        "productivity": "1.0–1.5 tonnes/ha.",
        "benefits": "High protein pulse; improves soil nitrogen; good market value.",
        "practices": "Well-drained loamy soils; inoculate with Rhizobium; irrigate at flowering and pod stage.",
        "ideal": {"N": (18, 32), "P": (55, 75), "K": (15, 28), "temperature": (18, 24), "humidity": (58, 72), "ph": (5.4, 6.4), "rainfall": (90, 120)},
    },
    "pigeonpeas": {
        "season": "Kharif (June–February). Long duration.",
        "water": "Low–moderate (300–600 mm). Deep-rooted and drought-tolerant.",
        "fertilizer": "20 kg N/ha, 50 kg P/ha, 20 kg K/ha.",
        "productivity": "1.0–1.5 tonnes/ha.",
        "benefits": "Protein-rich pulse; deep taproot improves soil structure; intercrop with cereals.",
        "practices": "Sow at 90×20 cm; intercrop with sorghum/maize; protect from pod borer.",
        "ideal": {"N": (22, 38), "P": (60, 80), "K": (15, 28), "temperature": (22, 28), "humidity": (48, 62), "ph": (5.6, 6.6), "rainfall": (130, 180)},
    },
    "mothbeans": {
        "season": "Kharif (arid regions).",
        "water": "Very low (200–300 mm). Highly drought-tolerant.",
        "fertilizer": "10–20 kg N/ha, 40 kg P/ha.",
        "productivity": "0.5–0.8 tonnes/ha.",
        "benefits": "Thrives in arid sandy soils; protein source; low input cost.",
        "practices": "Sow in light sandy loam; minimal irrigation; protect from pod borer.",
        "ideal": {"N": (18, 32), "P": (50, 70), "K": (15, 28), "temperature": (26, 32), "humidity": (48, 62), "ph": (6.0, 7.2), "rainfall": (48, 75)},
    },
    "mungbean": {
        "season": "Kharif and summer (short duration 60–70 days).",
        "water": "Low–moderate (300–500 mm).",
        "fertilizer": "15–20 kg N/ha, 40–50 kg P/ha.",
        "productivity": "0.8–1.2 tonnes/ha.",
        "benefits": "Short duration; fixes nitrogen; fits in multiple cropping systems; high protein.",
        "practices": "Treat seed with Rhizobium; sow at 30×10 cm; one irrigation at flowering.",
        "ideal": {"N": (18, 32), "P": (50, 70), "K": (15, 28), "temperature": (26, 32), "humidity": (54, 66), "ph": (6.0, 7.2), "rainfall": (78, 105)},
    },
    "blackgram": {
        "season": "Kharif and summer.",
        "water": "Moderate (400–600 mm).",
        "fertilizer": "15–20 kg N/ha, 40–50 kg P/ha.",
        "productivity": "0.8–1.2 tonnes/ha.",
        "benefits": "Nitrogen-fixing pulse; improves soil health; high protein and iron.",
        "practices": "Rhizobium seed treatment; avoid waterlogging; spray phosphorus if deficient.",
        "ideal": {"N": (18, 32), "P": (50, 70), "K": (15, 28), "temperature": (26, 32), "humidity": (58, 72), "ph": (6.0, 7.2), "rainfall": (85, 120)},
    },
    "lentil": {
        "season": "Rabi (November–March).",
        "water": "Low (250–350 mm).",
        "fertilizer": "20 kg N/ha, 40–50 kg P/ha.",
        "productivity": "0.9–1.5 tonnes/ha.",
        "benefits": "Cool-season protein pulse; fixes nitrogen; good for crop rotation.",
        "practices": "Sow at 22×10 cm; Rhizobium inoculation; one light irrigation at pod fill.",
        "ideal": {"N": (18, 32), "P": (55, 75), "K": (15, 28), "temperature": (22, 28), "humidity": (58, 72), "ph": (6.0, 7.2), "rainfall": (85, 120)},
    },
    "pomegranate": {
        "season": "Perennial; fruits in Kharif and Rabi.",
        "water": "Moderate (500–800 mm via irrigation).",
        "fertilizer": "625 g N, 250 g P, 250 g K per plant per year.",
        "productivity": "8–12 tonnes/ha.",
        "benefits": "High-value horticulture crop; rich in antioxidants; long shelf life.",
        "practices": "Drip irrigation; prune annually; manage fruit borer and bacterial blight.",
        "ideal": {"N": (15, 28), "P": (10, 22), "K": (10, 22), "temperature": (18, 26), "humidity": (58, 72), "ph": (6.0, 7.2), "rainfall": (105, 140)},
    },
    "banana": {
        "season": "Perennial; planted in Kharif/Rabi.",
        "water": "High (1500–2000 mm or equivalent irrigation).",
        "fertilizer": "200 g N, 60 g P, 300 g K per plant per year in 4 splits.",
        "productivity": "40–60 tonnes/ha.",
        "benefits": "High calorie fruit; year-round income; industrial uses (fiber, chips).",
        "practices": "Tissue culture plants; drip + fertigation; propping and de-suckering.",
        "ideal": {"N": (88, 112), "P": (72, 92), "K": (42, 58), "temperature": (26, 34), "humidity": (72, 88), "ph": (5.5, 6.8), "rainfall": (130, 180)},
    },
    "mango": {
        "season": "Perennial tree; fruits in summer (April–July).",
        "water": "Moderate (750–1250 mm).",
        "fertilizer": "1 kg N, 0.5 kg P, 1 kg K per mature tree per year.",
        "productivity": "8–15 tonnes/ha.",
        "benefits": "King of fruits; long-lived tree; high export value.",
        "practices": "Regular pruning; manage mango hoppers and anthracnose; irrigation during flowering.",
        "ideal": {"N": (18, 32), "P": (15, 28), "K": (22, 38), "temperature": (26, 34), "humidity": (52, 66), "ph": (5.4, 6.8), "rainfall": (105, 140)},
    },
    "grapes": {
        "season": "Perennial; pruned twice a year (Oct and April).",
        "water": "Moderate (700–1000 mm via drip).",
        "fertilizer": "500 g N, 250 g P, 750 g K per vine per year.",
        "productivity": "20–30 tonnes/ha.",
        "benefits": "High-value horticulture; export potential; value-added products (wine, raisins).",
        "practices": "Trellis system; drip irrigation; canopy management; disease control.",
        "ideal": {"N": (18, 32), "P": (120, 145), "K": (180, 220), "temperature": (22, 28), "humidity": (75, 90), "ph": (5.4, 6.8), "rainfall": (65, 95)},
    },
    "watermelon": {
        "season": "Summer (February–May).",
        "water": "Moderate (400–600 mm).",
        "fertilizer": "100 kg N, 50 kg P, 50 kg K/ha.",
        "productivity": "25–35 tonnes/ha.",
        "benefits": "Refreshing summer fruit; high water content; short duration profit crop.",
        "practices": "Sow on raised beds; drip irrigation; mulching; bee pollination.",
        "ideal": {"N": (105, 135), "P": (15, 28), "K": (15, 28), "temperature": (26, 34), "humidity": (58, 72), "ph": (6.0, 7.2), "rainfall": (65, 95)},
    },
    "muskmelon": {
        "season": "Summer (February–May).",
        "water": "Moderate (400–600 mm).",
        "fertilizer": "100 kg N, 50 kg P, 50 kg K/ha.",
        "productivity": "15–25 tonnes/ha.",
        "benefits": "Aromatic fruit; short duration; good market demand in summer.",
        "practices": "Raised beds; drip irrigation; mulch; harvest at full slip stage.",
        "ideal": {"N": (105, 135), "P": (15, 28), "K": (15, 28), "temperature": (26, 34), "humidity": (58, 72), "ph": (6.0, 7.2), "rainfall": (65, 95)},
    },
    "apple": {
        "season": "Perennial; temperate; harvest Aug–Oct.",
        "water": "Moderate (800–1200 mm).",
        "fertilizer": "700 g N, 350 g P, 700 g K per tree per year.",
        "productivity": "10–20 tonnes/ha.",
        "benefits": "Premium temperate fruit; long storage life; high market value.",
        "practices": "Prune in winter; manage scab and codling moth; thin fruits.",
        "ideal": {"N": (15, 28), "P": (120, 145), "K": (180, 220), "temperature": (18, 26), "humidity": (86, 98), "ph": (6.0, 7.2), "rainfall": (95, 130)},
    },
    "orange": {
        "season": "Perennial; fruits in winter (Nov–March).",
        "water": "Moderate (800–1200 mm).",
        "fertilizer": "600 g N, 200 g P, 400 g K per tree per year.",
        "productivity": "15–25 tonnes/ha.",
        "benefits": "Vitamin C rich; high demand; processing value (juice).",
        "practices": "Drip irrigation; manage citrus psylla and canker; balanced nutrition.",
        "ideal": {"N": (18, 32), "P": (15, 28), "K": (5, 18), "temperature": (22, 28), "humidity": (72, 88), "ph": (6.0, 7.2), "rainfall": (95, 130)},
    },
    "papaya": {
        "season": "Perennial; fruits year-round in tropics.",
        "water": "Moderate (1000–1500 mm).",
        "fertilizer": "200 g N, 100 g P, 200 g K per plant per year.",
        "productivity": "40–60 tonnes/ha.",
        "benefits": "Fast-growing; papain extraction; year-round fruiting.",
        "practices": "Plant at 1.8×1.8 m; manage papaya ring spot virus; regular fertigation.",
        "ideal": {"N": (42, 58), "P": (42, 58), "K": (42, 58), "temperature": (26, 34), "humidity": (72, 88), "ph": (6.0, 7.2), "rainfall": (130, 180)},
    },
    "coconut": {
        "season": "Perennial; fruits year-round.",
        "water": "Moderate–high (1200–2300 mm).",
        "fertilizer": "1 kg N, 0.5 kg P, 1.2 kg K per palm per year.",
        "productivity": "60–100 nuts/palm/year.",
        "benefits": "Tree of heaven; multiple uses (oil, water, coir); year-round income.",
        "practices": "Basin irrigation; organic matter; manage rhinoceros beetle.",
        "ideal": {"N": (18, 32), "P": (15, 28), "K": (15, 28), "temperature": (26, 34), "humidity": (72, 88), "ph": (5.4, 6.8), "rainfall": (175, 230)},
    },
    "cotton": {
        "season": "Kharif (April–September).",
        "water": "Moderate (600–1000 mm).",
        "fertilizer": "120 kg N, 60 kg P, 60 kg K/ha.",
        "productivity": "1.5–2.5 tonnes/ha (lint).",
        "benefits": "Major cash crop; raw material for textile industry; byproducts (oil, cake).",
        "practices": "Bt cotton; integrated pest management for bollworm; drip irrigation.",
        "ideal": {"N": (105, 135), "P": (32, 52), "K": (15, 28), "temperature": (22, 28), "humidity": (72, 88), "ph": (6.0, 7.2), "rainfall": (85, 120)},
    },
    "jute": {
        "season": "Kharif (March–June).",
        "water": "High (1500–2000 mm).",
        "fertilizer": "60 kg N, 30 kg P, 30 kg K/ha.",
        "productivity": "2.0–2.5 tonnes/ha (fiber).",
        "benefits": "Natural biodegradable fiber; eco-friendly packaging; supports rural industry.",
        "practices": "Retting in clean water; harvest at 50% flowering; rotate with rice.",
        "ideal": {"N": (68, 92), "P": (42, 58), "K": (32, 48), "temperature": (22, 28), "humidity": (72, 88), "ph": (6.0, 7.2), "rainfall": (160, 200)},
    },
    "coffee": {
        "season": "Perennial; flowering in March–April; harvest Nov–Feb.",
        "water": "Moderate (1250–2000 mm).",
        "fertilizer": "110 kg N, 40 kg P, 140 kg K/ha/year.",
        "productivity": "0.8–1.5 tonnes/ha (clean coffee).",
        "benefits": "High-value plantation crop; shade-grown supports biodiversity; export revenue.",
        "practices": "Shade management; mulching; manage coffee berry borer and leaf rust.",
        "ideal": {"N": (88, 112), "P": (22, 38), "K": (15, 28), "temperature": (22, 28), "humidity": (58, 72), "ph": (6.0, 7.2), "rainfall": (130, 180)},
    },
}


def get_crop_info(crop_name):
    """Return the knowledge-base entry for a crop, or a generic fallback.

    Args:
        crop_name (str): Lowercase crop name.

    Returns:
        dict: Crop information dictionary.
    """
    return CROP_INFO.get(
        crop_name.lower(),
        {
            "season": "Information not available for this crop.",
            "water": "Information not available.",
            "fertilizer": "Information not available.",
            "productivity": "Information not available.",
            "benefits": "Information not available.",
            "practices": "Information not available.",
            "ideal": {},
        },
    )


# ---------------------------------------------------------------------------
# Scenario 2 – suitability analysis
# ---------------------------------------------------------------------------
def suitability_score(values, crop_name):
    """Compute a 0–100 suitability score for a given crop and conditions.

    The score is the average percentage of features that fall inside the crop's
    ideal range, weighted equally.

    Args:
        values (dict): Mapping of feature name -> numeric value.
        crop_name (str): Crop to evaluate against.

    Returns:
        dict: ``score`` (float), ``advantages`` (list[str]),
            ``issues`` (list[str]), ``suggestions`` (list[str]).
    """
    info = get_crop_info(crop_name)
    ideal = info.get("ideal", {})
    advantages = []
    issues = []
    suggestions = []
    if not ideal:
        return {
            "score": 0.0,
            "advantages": ["No reference data available for this crop."],
            "issues": [],
            "suggestions": ["Consult a local agronomist for crop-specific guidance."],
        }

    scores = []
    for feat in pp.FEATURE_COLUMNS:
        val = values.get(feat)
        lo, hi = ideal.get(feat, (None, None))
        if lo is None or val is None:
            continue
        if lo <= val <= hi:
            scores.append(1.0)
            advantages.append(
                f"{feat.capitalize()} = {val} is within the ideal range ({lo}–{hi})."
            )
        else:
            # Partial credit based on distance from the ideal band.
            margin = max(abs(val - lo), abs(val - hi))
            band = hi - lo if hi != lo else 1
            feat_score = max(0.0, 1.0 - (margin / (band * 2)))
            scores.append(feat_score)
            issues.append(
                f"{feat.capitalize()} = {val} is outside the ideal range ({lo}–{hi})."
            )
            if val < lo:
                suggestions.append(
                    f"Increase {feat.capitalize()} (current {val}, ideal ≥ {lo})."
                )
            else:
                suggestions.append(
                    f"Reduce {feat.capitalize()} (current {val}, ideal ≤ {hi})."
                )
    score = round(float(np.mean(scores)) * 100, 1) if scores else 0.0
    return {
        "score": score,
        "advantages": advantages or ["No features within the ideal range."],
        "issues": issues,
        "suggestions": suggestions or ["All measured conditions are within the ideal range."],
    }


# ---------------------------------------------------------------------------
# Scenario 3 – research dashboard analytics
# ---------------------------------------------------------------------------
def dashboard_data():
    """Compute aggregated statistics for the Research Dashboard.

    Returns:
        dict: ``crop_counts`` (list[dict]), ``npk_stats`` (dict),
            ``distributions`` (dict), ``correlation`` (list[list]).
    """
    df = pp.load_dataset()
    df = pp.handle_missing_values(df)

    crop_counts = (
        df["label"].value_counts().rename_axis("crop").reset_index(name="count")
    ).to_dict(orient="records")

    npk_stats = {
        "N": {
            "mean": float(df["N"].mean()),
            "median": float(df["N"].median()),
            "min": float(df["N"].min()),
            "max": float(df["N"].max()),
        },
        "P": {
            "mean": float(df["P"].mean()),
            "median": float(df["P"].median()),
            "min": float(df["P"].min()),
            "max": float(df["P"].max()),
        },
        "K": {
            "mean": float(df["K"].mean()),
            "median": float(df["K"].median()),
            "min": float(df["K"].min()),
            "max": float(df["K"].max()),
        },
    }

    distributions = {
        "rainfall": df["rainfall"].tolist()[:200],
        "temperature": df["temperature"].tolist()[:200],
        "humidity": df["humidity"].tolist()[:200],
        "ph": df["ph"].tolist()[:200],
    }

    corr = df[pp.FEATURE_COLUMNS].corr().round(3)
    correlation = {
        "columns": corr.columns.tolist(),
        "values": corr.values.tolist(),
    }

    # Average NPK per crop (for grouped chart)
    npk_by_crop = (
        df.groupby("label")[["N", "P", "K"]].mean().round(1).reset_index()
    ).to_dict(orient="records")

    return {
        "crop_counts": crop_counts,
        "npk_stats": npk_stats,
        "distributions": distributions,
        "correlation": correlation,
        "npk_by_crop": npk_by_crop,
    }
