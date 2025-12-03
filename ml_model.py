import joblib
import pandas as pd
import numpy as np
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "position_model_scaled.pkl")

try:
    position_model = joblib.load(MODEL_PATH)
except FileNotFoundError:
    position_model = None
    print("⚠️ WARNUNG: ML-Modell nicht gefunden. KI-Vorschläge deaktiviert.")


# ⚠️ MUSS zu deinem TRAINING passen – inkl. Aggression, FK Accuracy, Long shots
FEATURE_COLS = [
    "Sprint speed",
    "Stamina",
    "Strength",
    "Aggression",
    "Ball control",
    "Dribbling",
    "FK Accuracy",
    "Finishing",
    "Long passing",
    "Long shots",
    "Short passing",
    "Shot power",
    "Tackling",
]

# Mapping: UI-Feldname → Modell-Feature
UI_TO_MODEL_FEATURES = {
    "speed":           "Sprint speed",
    "stamina":         "Stamina",
    "strength":        "Strength",
    "aggression":      "Aggression",
    "first_touch":     "Ball control",
    "dribbling":       "Dribbling",
    "finishing":       "Finishing",
    "long_passing":    "Long passing",
    "short_passing":   "Short passing",
    "shooting_power":  "Shot power",
    "tackling":        "Tackling",
    # Tactical Awareness, FK Accuracy, Long shots kommen unten als Defaults
}


def _clip_1_10(value, default=5.0) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        v = default
    if v < 1.0:
        v = 1.0
    if v > 10.0:
        v = 10.0
    return v


def _prepare_feature_vector(attrs: dict) -> pd.DataFrame:
    """Dict → 1xN DataFrame, strikt FEATURE_COLS, Werte in [1,10]."""
    row = {}
    for col in FEATURE_COLS:
        raw = attrs.get(col, 5.0)
        row[col] = [_clip_1_10(raw, default=5.0)]
    return pd.DataFrame(row)


def _sharpen_proba(proba, gamma: float = 1.5):
    """Verteilung spitzer machen (Top-Klasse klarer hervorheben)."""
    p = np.asarray(proba, dtype=float)
    p = np.power(p, gamma)
    s = p.sum()
    if s > 0:
        p = p / s
    return p


def recommend_position_from_attributes(attrs: dict):
    """
    attrs: Keys wie in FEATURE_COLS, Werte 1–10.
    Rückgabe: Liste [(position_code, prob), ...] Top3.
    """
    if position_model is None:
        print("⚠️ KI-Modell nicht geladen – Dummy-Vorschläge.")
        return [("CM", 0.34), ("ST", 0.33), ("CB", 0.33)]

    X = _prepare_feature_vector(attrs)
    proba = position_model.predict_proba(X)[0]
    classes = position_model.classes_
    proba_sharp = _sharpen_proba(proba, gamma=1.5)

    ranked = sorted(
        [(cls, float(p)) for cls, p in zip(classes, proba_sharp)],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:3]


def map_form_to_model_features(form):
    """
    request.form → Dict mit Keys wie in FEATURE_COLS.
    FK Accuracy & Long shots werden aktuell als neutrale 5.0 gesetzt.
    """
    attrs = {col: 5.0 for col in FEATURE_COLS}

    # Werte aus dem Formular mappen
    for ui_name, model_name in UI_TO_MODEL_FEATURES.items():
        raw_val = form.get(ui_name, 5)
        attrs[model_name] = _clip_1_10(raw_val, default=5.0)

    # FK Accuracy und Long shots aktuell NICHT im UI → neutrale 5
    attrs["FK Accuracy"] = 5.0
    attrs["Long shots"] = 5.0

    return attrs
