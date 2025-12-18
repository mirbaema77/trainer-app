import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

# MUST match your training columns
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


def _train_position_model_from_excel():
    """
    Train the model directly from 'formation_attributes_numeric.xlsx'.
    No .pkl file needed – built at app startup.
    """
    print("🔁 Training ML model from formation_attributes_numeric.xlsx ...")

    # 1. Load Excel
    df = pd.read_excel("formation_attributes_numeric.xlsx")
    df["Position"] = df["Position"].astype(str).str.strip().str.upper()

    # 2. Features & target
    X_raw = df[FEATURE_COLS].copy()
    y = df["Position"]

    # 3. Scale FIFA 0–99 → 1–10
    def scale_fifa_to_1_10(series: pd.Series) -> pd.Series:
        s = series.astype(float)
        min_val = s.min()
        max_val = s.max()
        if max_val == min_val:
            return pd.Series(5.0, index=s.index)
        scaled = 1.0 + (s - min_val) * 9.0 / (max_val - min_val)
        return scaled

    X_scaled = X_raw.copy()
    for col in FEATURE_COLS:
        X_scaled[col] = scale_fifa_to_1_10(X_scaled[col])

    # 4. Define RandomForest
    rf = RandomForestClassifier(
        n_estimators=75,
        max_depth=None,
        min_samples_leaf=2,
        n_jobs=-1,
        random_state=42,
        class_weight="balanced_subsample",
    )

    # 5. Train
    rf.fit(X_scaled, y)

    pipeline = Pipeline([
        ("rf", rf),
    ])

    print("✅ ML model trained. Classes:", list(pipeline.named_steps["rf"].classes_))
    return pipeline


# Train once when this module is imported
try:
    position_model = _train_position_model_from_excel()
except Exception as e:
    position_model = None
    print("⚠️ WARNING: Could not train ML model:", e)


# Mapping: UI field → model feature
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
    """Dict → 1xN DataFrame, strict FEATURE_COLS, values in [1,10]."""
    row = {}
    for col in FEATURE_COLS:
        raw = attrs.get(col, 5.0)
        row[col] = [_clip_1_10(raw, default=5.0)]
    return pd.DataFrame(row)


def _sharpen_proba(proba, gamma: float = 1.5):
    """Make distribution sharper (top class clearer)."""
    p = np.asarray(proba, dtype=float)
    p = np.power(p, gamma)
    s = p.sum()
    if s > 0:
        p = p / s
    return p


def recommend_position_from_attributes(attrs: dict):
    """
    attrs: keys like FEATURE_COLS, values 1–10.
    Returns: list [(position_code, prob), ...] Top3.
    """
    if position_model is None:
        print("⚠️ ML model not available – returning dummy suggestions.")
        return [("CM", 0.34), ("ST", 0.33), ("CB", 0.33)]

    X = _prepare_feature_vector(attrs)
    proba = position_model.predict_proba(X)[0]
    classes = position_model.classes_ if not hasattr(position_model, "named_steps") \
        else position_model.named_steps["rf"].classes_
    proba_sharp = _sharpen_proba(proba, gamma=1.5)

    ranked = sorted(
        [(cls, float(p)) for cls, p in zip(classes, proba_sharp)],
        key=lambda x: x[1],
        reverse=True,
    )
    return ranked[:3]


def map_form_to_model_features(form):
    """
    request.form → dict with keys like FEATURE_COLS.
    FK Accuracy & Long shots are currently set to neutral 5.0.
    """
    attrs = {col: 5.0 for col in FEATURE_COLS}

    # Map form values
    for ui_name, model_name in UI_TO_MODEL_FEATURES.items():
        raw_val = form.get(ui_name, 5)
        attrs[model_name] = _clip_1_10(raw_val, default=5.0)

    # Not in UI yet → neutral defaults
    attrs["FK Accuracy"] = 5.0
    attrs["Long shots"] = 5.0

    return attrs


def predict_position_proba_all(attrs: dict) -> dict:
    """
    Returns dict: {position_code: prob} for ALL classes.
    attrs: keys like FEATURE_COLS, values 1–10.
    """
    if position_model is None:
        return {}

    X = _prepare_feature_vector(attrs)
    proba = position_model.predict_proba(X)[0]

    classes = position_model.classes_ if not hasattr(position_model, "named_steps") \
        else position_model.named_steps["rf"].classes_

    proba_sharp = _sharpen_proba(proba, gamma=1.5)
    return {cls: float(p) for cls, p in zip(classes, proba_sharp)}

