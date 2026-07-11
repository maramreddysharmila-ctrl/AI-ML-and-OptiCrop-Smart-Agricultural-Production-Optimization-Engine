"""Generate a realistic Crop Recommendation dataset.

This script creates a synthetic but realistic dataset for crop recommendation
based on N, P, K, temperature, humidity, pH, and rainfall values. Each crop has
characteristic ranges that mimic real agricultural conditions.

Run: python utils/generate_dataset.py
"""
import os
import numpy as np
import pandas as pd

# Crop profiles: (N, P, K, temperature, humidity, ph, rainfall)
# Each entry holds (mean, std) for the feature. Values are based on
# typical agronomic ranges found in the well-known Crop Recommendation dataset.
CROP_PROFILES = {
    "rice":        {"N": (90, 10),  "P": (40, 8),  "K": (42, 7),  "temp": (23, 2),  "humidity": (82, 5), "ph": (6.0, 0.5), "rainfall": (210, 30)},
    "maize":       {"N": (78, 12), "P": (48, 8),  "K": (22, 4),  "temp": (24, 2),  "humidity": (65, 6), "ph": (6.2, 0.5), "rainfall": (85, 15)},
    "chickpea":    {"N": (40, 8),  "P": (68, 10), "K": (80, 8),  "temp": (20, 2),  "humidity": (45, 5), "ph": (7.0, 0.4), "rainfall": (80, 12)},
    "kidneybeans": {"N": (25, 5),  "P": (65, 8),  "K": (20, 4),  "temp": (20, 2),  "humidity": (65, 5), "ph": (5.8, 0.4), "rainfall": (105, 15)},
    "pigeonpeas":  {"N": (30, 6),  "P": (68, 8),  "K": (20, 4),  "temp": (24, 2),  "humidity": (55, 5), "ph": (6.0, 0.4), "rainfall": (150, 20)},
    "mothbeans":   {"N": (25, 5),  "P": (60, 8),  "K": (20, 4),  "temp": (28, 2),  "humidity": (55, 5), "ph": (6.5, 0.4), "rainfall": (60, 10)},
    "mungbean":    {"N": (25, 5),  "P": (60, 8),  "K": (20, 4),  "temp": (28, 2),  "humidity": (60, 5), "ph": (6.5, 0.4), "rainfall": (90, 12)},
    "blackgram":   {"N": (25, 5),  "P": (60, 8),  "K": (20, 4),  "temp": (28, 2),  "humidity": (65, 5), "ph": (6.5, 0.4), "rainfall": (100, 15)},
    "lentil":      {"N": (25, 5),  "P": (65, 8),  "K": (20, 4),  "temp": (24, 2),  "humidity": (65, 5), "ph": (6.5, 0.4), "rainfall": (100, 15)},
    "pomegranate": {"N": (20, 5),  "P": (15, 4),  "K": (15, 4),  "temp": (22, 2),  "humidity": (65, 5), "ph": (6.5, 0.4), "rainfall": (120, 15)},
    "banana":      {"N": (100, 12),"P": (82, 8),  "K": (50, 6),  "temp": (30, 2),  "humidity": (80, 5), "ph": (6.0, 0.4), "rainfall": (150, 20)},
    "mango":       {"N": (25, 5),  "P": (20, 4),  "K": (30, 5),  "temp": (30, 2),  "humidity": (60, 5), "ph": (6.0, 0.4), "rainfall": (120, 15)},
    "grapes":      {"N": (25, 5),  "P": (130, 10),"K": (200, 12),"temp": (25, 2),  "humidity": (82, 5), "ph": (6.0, 0.4), "rainfall": (80, 12)},
    "watermelon":  {"N": (120, 12),"P": (20, 4),  "K": (20, 4),  "temp": (30, 2),  "humidity": (65, 5), "ph": (6.5, 0.4), "rainfall": (80, 12)},
    "muskmelon":   {"N": (120, 12),"P": (20, 4),  "K": (20, 4),  "temp": (30, 2),  "humidity": (65, 5), "ph": (6.5, 0.4), "rainfall": (80, 12)},
    "apple":       {"N": (20, 5),  "P": (135, 10),"K": (200, 12),"temp": (22, 2),  "humidity": (92, 4), "ph": (6.5, 0.4), "rainfall": (110, 15)},
    "orange":      {"N": (25, 5),  "P": (20, 4),  "K": (10, 3),  "temp": (25, 2),  "humidity": (80, 5), "ph": (6.5, 0.4), "rainfall": (110, 15)},
    "papaya":      {"N": (50, 8),  "P": (50, 8),  "K": (50, 6),  "temp": (30, 2),  "humidity": (80, 5), "ph": (6.5, 0.4), "rainfall": (150, 20)},
    "coconut":     {"N": (25, 5),  "P": (20, 4),  "K": (20, 4),  "temp": (30, 2),  "humidity": (80, 5), "ph": (6.0, 0.4), "rainfall": (200, 25)},
    "cotton":      {"N": (120, 12),"P": (40, 8),  "K": (20, 4),  "temp": (25, 2),  "humidity": (80, 5), "ph": (6.5, 0.4), "rainfall": (100, 15)},
    "jute":        {"N": (80, 10), "P": (50, 8),  "K": (40, 6),  "temp": (25, 2),  "humidity": (80, 5), "ph": (6.5, 0.4), "rainfall": (180, 20)},
    "coffee":      {"N": (100, 12),"P": (30, 6),  "K": (20, 4),  "temp": (25, 2),  "humidity": (65, 5), "ph": (6.5, 0.4), "rainfall": (150, 20)},
}

# Map canonical feature name -> key used inside CROP_PROFILES
FEATURE_KEYS = {"temperature": "temp"}
FEATURES = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
SAMPLES_PER_CROP = 100
RANDOM_SEED = 42


def generate_dataset(samples_per_crop=SAMPLES_PER_CROP, seed=RANDOM_SEED):
    """Generate a synthetic crop-recommendation DataFrame.

    Args:
        samples_per_crop (int): Number of samples to generate per crop.
        seed (int): Random seed for reproducibility.

    Returns:
        pandas.DataFrame: Dataset with 7 feature columns and a ``label`` column.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for crop, profile in CROP_PROFILES.items():
        for _ in range(samples_per_crop):
            row = {"label": crop}
            for feat in FEATURES:
                prof_key = FEATURE_KEYS.get(feat, feat)
                mean, std = profile[prof_key]
                value = rng.normal(mean, std)
                # Clamp to physically plausible ranges
                if feat in ("N", "P", "K", "rainfall"):
                    value = max(0, value)
                if feat == "ph":
                    value = min(max(value, 3.5), 9.5)
                if feat == "temperature":
                    value = min(max(value, 8), 45)
                if feat == "humidity":
                    value = min(max(value, 10), 100)
                row[feat] = round(value, 2)
            rows.append(row)
    return pd.DataFrame(rows)


def main():
    """Generate the dataset and write it to ``dataset/Crop_recommendation.csv``."""
    df = generate_dataset()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "dataset")
    out_path = os.path.join(out_dir, "Crop_recommendation.csv")
    df.to_csv(out_path, index=False)
    print(f"Dataset written to {out_path} with {len(df)} rows and {len(CROP_PROFILES)} crops.")


if __name__ == "__main__":
    main()
