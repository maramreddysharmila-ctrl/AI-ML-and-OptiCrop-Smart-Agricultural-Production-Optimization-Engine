"""Model training pipeline for OptiCrop.

Trains and compares Decision Tree, Random Forest, Logistic Regression, KNN, and
K-Means clustering models on the crop recommendation dataset, then persists the
best-performing supervised model together with its scaler and label encoder.

Run: python utils/train_model.py
"""
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from . import evaluation as ev
from . import preprocess as pp


# Random seed shared across the project for reproducibility.
RANDOM_SEED = 42


def build_models(random_state=RANDOM_SEED):
    """Instantiate all candidate models with sensible defaults.

    Args:
        random_state (int): Seed for reproducible models.

    Returns:
        dict: Mapping of model name -> unfitted estimator.
    """
    return {
        "Decision Tree": DecisionTreeClassifier(
            random_state=random_state, max_depth=12
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=random_state, n_jobs=-1
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=2000, random_state=random_state
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    }


def train_models(X_train, y_train, random_state=RANDOM_SEED):
    """Train every candidate model and return fitted estimators.

    Args:
        X_train (array-like): Scaled training features.
        y_train (array-like): Encoded training labels.
        random_state (int): Seed for reproducibility.

    Returns:
        dict: Mapping of model name -> fitted estimator.
    """
    models = build_models(random_state=random_state)
    fitted = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted[name] = model
        print(f"Trained {name}.")
    return fitted


def run_kmeans(X, n_clusters=22, random_state=RANDOM_SEED):
    """Fit a K-Means clustering model for exploratory analysis.

    Args:
        X (array-like): Scaled feature matrix.
        n_clusters (int): Number of clusters (defaults to number of crops).
        random_state (int): Seed for reproducibility.

    Returns:
        KMeans: Fitted K-Means estimator.
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X)
    return kmeans


def train_pipeline(test_size=0.2, random_state=RANDOM_SEED, save=True):
    """Run the full training pipeline and persist artifacts.

    Loads the dataset, preprocesses it, trains all models, evaluates them,
    selects the best by accuracy, and (optionally) saves the model, scaler, and
    label encoder to ``model/``.

    Args:
        test_size (float): Fraction of data held out for testing.
        random_state (int): Seed for reproducibility.
        save (bool): If True, persist artifacts to disk.

    Returns:
        dict: Results with keys ``comparison`` (DataFrame), ``best_model``,
            ``best_name``, ``scaler``, ``encoder``, ``kmeans``.
    """
    df = pp.load_dataset()
    df = pp.handle_missing_values(df)
    X, y = pp.get_features_labels(df)

    scaler = pp.fit_scaler(X)
    X_scaled = scaler.transform(X)

    encoder = pp.fit_label_encoder(y)
    y_encoded = encoder.transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=test_size, random_state=random_state,
        stratify=y_encoded
    )

    fitted = train_models(X_train, y_train, random_state=random_state)
    comparison = ev.evaluate_all(fitted, X_test, y_test)
    print("\nModel comparison:\n", comparison.to_string(index=False))

    best_name = comparison.iloc[0]["model"]
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name} (accuracy={comparison.iloc[0]['accuracy']:.4f})")

    # Cross-validate the best model for an honest score.
    cv_scores = cross_val_score(best_model, X_scaled, y_encoded, cv=5)
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # K-Means for exploratory clustering.
    kmeans = run_kmeans(X_scaled, n_clusters=len(encoder.classes_), random_state=random_state)

    if save:
        pp.save_object(best_model, "crop_model.pkl")
        pp.save_object(scaler, "scaler.pkl")
        pp.save_object(encoder, "label_encoder.pkl")
        pp.save_object(kmeans, "kmeans_model.pkl")
        pp.save_object(comparison, "model_comparison.pkl")
        print("\nArtifacts saved to model/")

    return {
        "comparison": comparison,
        "best_model": best_model,
        "best_name": best_name,
        "scaler": scaler,
        "encoder": encoder,
        "kmeans": kmeans,
    }


if __name__ == "__main__":
    train_pipeline()
