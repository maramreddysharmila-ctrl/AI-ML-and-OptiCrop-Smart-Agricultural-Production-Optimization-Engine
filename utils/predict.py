"""Inference helpers for OptiCrop.

Loads the persisted model, scaler, and label encoder, and exposes a clean
``predict_crop`` function used by the Flask application.
"""
import numpy as np

from . import preprocess as pp


def load_artifacts():
    """Load the trained model, scaler, and label encoder from disk.

    Returns:
        tuple: (model, scaler, encoder)
    """
    model = pp.load_object("crop_model.pkl")
    scaler = pp.load_object("scaler.pkl")
    encoder = pp.load_object("label_encoder.pkl")
    return model, scaler, encoder


def predict_crop(values, model=None, scaler=None, encoder=None):
    """Predict the best crop for a set of environmental features.

    Args:
        values (dict|list): Mapping of feature name -> value, or an ordered
            list matching :data:`preprocess.FEATURE_COLUMNS`.
        model: Fitted classifier (loaded if None).
        scaler: Fitted scaler (loaded if None).
        encoder: Fitted label encoder (loaded if None).

    Returns:
        dict: ``crop`` (str), ``confidence`` (float), ``probabilities`` (dict
            mapping crop -> probability, sorted descending).
    """
    if model is None or scaler is None or encoder is None:
        model, scaler, encoder = load_artifacts()

    # Build the feature row in the canonical order.
    if isinstance(values, dict):
        row = [values[feat] for feat in pp.FEATURE_COLUMNS]
    else:
        row = list(values)
    arr = np.array(row, dtype=float).reshape(1, -1)
    arr_scaled = scaler.transform(arr)

    pred_code = model.predict(arr_scaled)[0]
    crop = encoder.inverse_transform([pred_code])[0]

    # Confidence: use predict_proba when available, else fall back to 1.0.
    probabilities = {}
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(arr_scaled)[0]
        for code, prob in enumerate(probs):
            probabilities[encoder.inverse_transform([code])[0]] = float(prob)
        confidence = float(probs[pred_code])
        # Sort descending by probability.
        probabilities = dict(
            sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True)
        )
    else:
        confidence = 1.0

    return {
        "crop": crop,
        "confidence": confidence,
        "probabilities": probabilities,
    }


def predict_batch(rows, model=None, scaler=None, encoder=None):
    """Predict crops for multiple rows of features.

    Args:
        rows (list[dict]): Each dict maps feature name -> value.
        model, scaler, encoder: Optional pre-loaded artifacts.

    Returns:
        list[dict]: One result dict per row (see :func:`predict_crop`).
    """
    if model is None or scaler is None or encoder is None:
        model, scaler, encoder = load_artifacts()
    return [predict_crop(r, model, scaler, encoder) for r in rows]
