"""Model evaluation utilities for OptiCrop.

Provides helper functions to compute metrics, build comparison tables, and
render confusion matrices / classification reports for every trained model.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Compute standard classification metrics for a single model.

    Args:
        model: Fitted classifier with a ``predict`` method.
        X_test (array-like): Test features.
        y_test (array-like): True test labels.
        model_name (str): Display name for the model.

    Returns:
        dict: Metrics dictionary with keys ``model``, ``accuracy``,
            ``precision``, ``recall``, ``f1``.
    """
    y_pred = model.predict(X_test)
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    return metrics


def evaluate_all(models, X_test, y_test):
    """Evaluate a dictionary of trained models and return a comparison frame.

    Args:
        models (dict): Mapping of model name -> fitted estimator.
        X_test (array-like): Test features.
        y_test (array-like): True test labels.

    Returns:
        pandas.DataFrame: Sorted comparison table (highest accuracy first).
    """
    rows = []
    for name, model in models.items():
        rows.append(evaluate_model(model, X_test, y_test, name))
    df = pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)
    return df


def get_confusion_matrix(model, X_test, y_test):
    """Return the confusion matrix for a fitted model.

    Args:
        model: Fitted classifier.
        X_test (array-like): Test features.
        y_test (array-like): True test labels.

    Returns:
        numpy.ndarray: Confusion matrix.
    """
    y_pred = model.predict(X_test)
    return confusion_matrix(y_test, y_pred)


def get_classification_report(model, X_test, y_test, target_names=None):
    """Return a text classification report for a fitted model.

    Args:
        model: Fitted classifier.
        X_test (array-like): Test features.
        y_test (array-like): True test labels.
        target_names (list[str]|None): Class display names.

    Returns:
        str: Classification report as a string.
    """
    y_pred = model.predict(X_test)
    return classification_report(
        y_test, y_pred, target_names=target_names, zero_division=0
    )


def best_model_name(comparison_df):
    """Return the name of the model with the highest accuracy.

    Args:
        comparison_df (pandas.DataFrame): Output of :func:`evaluate_all`.

    Returns:
        str: Name of the best model.
    """
    return comparison_df.iloc[0]["model"]
