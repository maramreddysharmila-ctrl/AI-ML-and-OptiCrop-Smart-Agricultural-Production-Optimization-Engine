"""OptiCrop – Smart Agricultural Production Optimization Engine.

Flask application exposing the crop recommendation system, a crop suitability
checker (Scenario 2), and a research dashboard (Scenario 3).

Run: python app.py
"""
import os

import numpy as np
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify

from utils import predict as predict_mod
from utils import preprocess as pp
from utils import helper

app = Flask(__name__)
app.secret_key = "opticrop-smartbridge-secret-key"

# Load model artifacts once at startup.
MODEL, SCALER, ENCODER = predict_mod.load_artifacts()
CROP_CLASSES = list(ENCODER.classes_)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Home page with hero section, project description, and prediction form."""
    return render_template("index.html", crops=CROP_CLASSES)


@app.route("/predict", methods=["POST"])
def predict():
    """Scenario 1 – predict the best crop from environmental inputs.

    Validates the form, runs the model, and renders the result page.
    """
    raw = {feat: request.form.get(feat, "") for feat in pp.FEATURE_COLUMNS}
    is_valid, errors, parsed = pp.validate_input(raw)

    if not is_valid:
        for err in errors:
            flash(err, "danger")
        return render_template("index.html", crops=CROP_CLASSES, form=raw, errors=errors)

    try:
        result = predict_mod.predict_crop(parsed, MODEL, SCALER, ENCODER)
    except Exception as exc:  # pragma: no cover - defensive
        flash(f"Prediction failed: {exc}", "danger")
        return render_template("index.html", crops=CROP_CLASSES, form=raw)

    crop = result["crop"]
    info = helper.get_crop_info(crop)
    confidence_pct = round(result["confidence"] * 100, 2)

    # Top 3 alternative crops with probabilities.
    top_crops = list(result["probabilities"].items())[:3]

    return render_template(
        "result.html",
        crop=crop,
        confidence=confidence_pct,
        info=info,
        inputs=parsed,
        top_crops=top_crops,
    )


@app.route("/suitability", methods=["GET", "POST"])
def suitability():
    """Scenario 2 – check whether current conditions suit a selected crop."""
    if request.method == "POST":
        raw = {feat: request.form.get(feat, "") for feat in pp.FEATURE_COLUMNS}
        selected_crop = request.form.get("crop_select", "")
        is_valid, errors, parsed = pp.validate_input(raw)

        if not is_valid:
            for err in errors:
                flash(err, "danger")
            return render_template(
                "scenario2.html", crops=CROP_CLASSES, selected=selected_crop, form=raw
            )

        analysis = helper.suitability_score(parsed, selected_crop)
        info = helper.get_crop_info(selected_crop)
        return render_template(
            "scenario2.html",
            crops=CROP_CLASSES,
            selected=selected_crop,
            analysis=analysis,
            info=info,
            inputs=parsed,
            mode="result",
        )
    return render_template("scenario2.html", crops=CROP_CLASSES, mode="form")


@app.route("/dashboard")
def dashboard():
    """Scenario 3 – research dashboard with aggregated analytics."""
    data = helper.dashboard_data()
    return render_template("dashboard.html", data=data)


@app.route("/about")
def about():
    """About page describing the project and its objectives."""
    return render_template("about.html")


@app.route("/contact")
def contact():
    """Contact page."""
    return render_template("contact.html")


@app.errorhandler(404)
def not_found(error):
    """Custom 404 error page."""
    return render_template("error.html", code=404), 404


@app.errorhandler(500)
def server_error(error):
    """Custom 500 error page."""
    return render_template("error.html", code=500), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
