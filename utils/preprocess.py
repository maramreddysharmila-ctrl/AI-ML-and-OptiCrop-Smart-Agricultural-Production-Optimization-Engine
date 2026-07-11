"""Data preprocessing utilities for OptiCrop.

Handles dataset loading, missing-value treatment, feature scaling, and label
encoding so that the training and inference pipelines share a single,
consistent transformation path.
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Canonical feature order used everywhere in the project.
FEATURE_COLUMNS = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
LABEL_COLUMN = "label"

# Plausible ranges for input validation (used by the Flask app too).
FEATURE_RANGES = {
    "N": (0, 140),
    "P": (0, 145),
    "K": (0, 205),
    "temperature": (8, 45),
    "humidity": (10, 100),
    "ph": (3.5, 9.5),
    "rainfall": (10, 300),
}

DATASET_PATH = os.path.join(
    os.path.dirname(__file__), "..", "dataset", "Crop_recommendation.csv"
)
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "model")


def load_dataset(path=DATASET_PATH):
    """Load the crop recommendation dataset from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pandas.DataFrame: Loaded dataframe.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = pd.read_csv(path)
    return df


def handle_missing_values(df):
    """Return a copy of ``df`` with missing values handled.

    Numeric features are filled with their column median (robust to outliers);
    the label column, if missing, drops the row.

    Args:
        df (pandas.DataFrame): Raw dataframe.

    Returns:
        pandas.DataFrame: Cleaned dataframe.
    """
    df = df.copy()
    for col in FEATURE_COLUMNS:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    if LABEL_COLUMN in df.columns:
        df = df.dropna(subset=[LABEL_COLUMN])
    return df


def get_features_labels(df):
    """Split a dataframe into feature matrix ``X`` and label vector ``y``.

    Args:
        df (pandas.DataFrame): Source dataframe.

    Returns:
        tuple: (X, y) where X is a DataFrame of features and y is a Series of labels.
    """
    X = df[FEATURE_COLUMNS].copy()
    y = df[LABEL_COLUMN].copy() if LABEL_COLUMN in df.columns else None
    return X, y


def fit_scaler(X):
    """Fit a StandardScaler on the feature matrix.

    Args:
        X (pandas.DataFrame or numpy.ndarray): Feature matrix.

    Returns:
        StandardScaler: Fitted scaler instance.
    """
    scaler = StandardScaler()
    scaler.fit(X)
    return scaler


def transform_features(scaler, X):
    """Scale features using a fitted scaler.

    Args:
        scaler (StandardScaler): Fitted scaler.
        X (array-like): Feature matrix.

    Returns:
        numpy.ndarray: Scaled feature matrix.
    """
    return scaler.transform(X)


def fit_label_encoder(y):
    """Fit a LabelEncoder on the label vector.

    Args:
        y (array-like): Crop labels.

    Returns:
        LabelEncoder: Fitted encoder.
    """
    encoder = LabelEncoder()
    encoder.fit(y)
    return encoder


def encode_labels(encoder, y):
    """Transform string labels into integer codes.

    Args:
        encoder (LabelEncoder): Fitted encoder.
        y (array-like): Labels.

    Returns:
        numpy.ndarray: Encoded labels.
    """
    return encoder.transform(y)


def save_object(obj, filename):
    """Pickle an object into ``model/<filename>``.

    Args:
        obj: Any picklable object.
        filename (str): Name of the file inside the ``model/`` directory.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_object(filename):
    """Unpickle an object from ``model/<filename>``.

    Args:
        filename (str): Name of the file inside the ``model/`` directory.

    Returns:
        The unpickled object.
    """
    path = os.path.join(MODEL_DIR, filename)
    with open(path, "rb") as f:
        return pickle.load(f)


def preprocess_pipeline(df, fit=True, scaler=None, encoder=None):
    """Run the full preprocessing pipeline on a dataframe.

    Args:
        df (pandas.DataFrame): Raw dataframe.
        fit (bool): If True, fit new scaler/encoder; otherwise use provided ones.
        scaler (StandardScaler|None): Pre-fitted scaler (used when fit=False).
        encoder (LabelEncoder|None): Pre-fitted encoder (used when fit=False).

    Returns:
        dict: Dictionary with keys ``X_scaled``, ``y_encoded``, ``scaler``,
            ``encoder``, ``feature_names``.
    """
    df = handle_missing_values(df)
    X, y = get_features_labels(df)

    if fit:
        scaler = fit_scaler(X)
        encoder = fit_label_encoder(y)

    X_scaled = scaler.transform(X)
    y_encoded = encode_labels(encoder, y) if y is not None else None

    return {
        "X_scaled": X_scaled,
        "y_encoded": y_encoded,
        "scaler": scaler,
        "encoder": encoder,
        "feature_names": FEATURE_COLUMNS,
    }


def validate_input(values):
    """Validate a dictionary of user input values.

    Args:
        values (dict): Mapping of feature name -> string value from the form.

    Returns:
        tuple: (is_valid (bool), errors (list[str]), parsed (dict|None)).
    """
    errors = []
    parsed = {}
    for feat in FEATURE_COLUMNS:
        raw = values.get(feat, "").strip()
        if raw == "":
            errors.append(f"{feat.capitalize()} is required.")
            continue
        try:
            val = float(raw)
        except ValueError:
            errors.append(f"{feat.capitalize()} must be a numeric value.")
            continue
        if val < 0:
            errors.append(f"{feat.capitalize()} cannot be negative.")
            continue
        lo, hi = FEATURE_RANGES[feat]
        if val < lo or val > hi:
            errors.append(
                f"{feat.capitalize()} must be between {lo} and {hi}."
            )
            continue
        parsed[feat] = val
    return (len(errors) == 0, errors, parsed if len(errors) == 0 else None)
