import joblib
import pandas as pd
import numpy as np

# Neues Modell laden (auf 1–10 Skala trainiert)
position_model = joblib.load("position_model_scaled.pkl")

FEATURE_COLS = [
    "Sprint speed",
    "Stamina",
    "Strength",
    "Tactical Awareness",
    "Ball control",
    "Dribbling",
    "Finishing",
    "Long passing",
    "Short passing",
    "Shot power",
    "Tackling",
]


def _sharpen_proba(proba, gamma: float = 1.5):
    """
    Schärft Wahrscheinlichkeiten, ohne die Reihenfolge zu ändern.
    proba: array der Klassenwahrscheinlichkeiten (sum = 1)
    gamma > 1 -> Verteilung wird 'spitzer' (Top-Klassen dominanter).
    """
    p = np.asarray(proba, dtype=float)
    p = np.power(p, gamma)  # jede Wahrscheinlichkeit hoch gamma
    total = p.sum()
    if total > 0:
        p = p / total
    return p


def recommend_position_from_attributes(attrs: dict):
    """
    attrs: dict mit Keys wie in FEATURE_COLS,
    Werte aus deinem Formular (1–10).
    """

    row = {}
    for col in FEATURE_COLS:
        v = attrs.get(col, 0)
        try:
            v = float(v)
        except (TypeError, ValueError):
            v = 0.0

        # Werte auf 1–10 begrenzen (Fehleingaben abfangen)
        if v < 1:
            v = 1.0
        if v > 10:
            v = 10.0

        row[col] = [v]

    X = pd.DataFrame(row)

    # Roh-Wahrscheinlichkeiten vom Modell
    proba = position_model.predict_proba(X)[0]
    classes = position_model.classes_

    # Wahrscheinlichkeiten schärfen
    proba_sharp = _sharpen_proba(proba, gamma=1.5)

    # Sortiert nach geschärften Wahrscheinlichkeiten
    ranked = sorted(
        [(cls, float(p)) for cls, p in zip(classes, proba_sharp)],
        key=lambda x: x[1],
        reverse=True,
    )

    # Top 3 zurückgeben
    return ranked[:3]
