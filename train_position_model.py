import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib

# -----------------------------------
# 1. Daten laden
# -----------------------------------
df = pd.read_excel("formation_attributes_numeric.xlsx")

# 2. Positions-Mapping
position_map = {
    "RB": "RB", "RWB": "RB",
    "LB": "LB", "LWB": "LB",
    "CB": "CB", "LCB": "CB", "RCB": "CB",
    "CDM": "CDM", "RDM": "CDM", "LDM": "CDM",
    "CM": "CM", "LCM": "CM", "RCM": "CM",
    "CAM": "CAM",
    "RM": "RM", "RW": "RW",
    "LM": "LM", "LW": "LM",  # LM & LW zusammenlegen
    "CF": "CF", "RF": "CF", "LF": "CF",
    "ST": "ST", "RS": "ST", "LS": "ST",
}

df["Position"] = df["Position"].astype(str).strip()
df = df[df["Position"].isin(position_map.keys())].copy()
df["Position_clean"] = df["Position"].map(position_map)

# -----------------------------------
# 3. Feature-Spalten (nur was du auch im UI hast)
# -----------------------------------
feature_cols = [
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

df_model = df.dropna(subset=feature_cols + ["Position_clean"]).copy()

X_raw = df_model[feature_cols]
y = df_model["Position_clean"]

# -----------------------------------
# 4. FIFA 0–99 → 1–10 skalieren
# -----------------------------------
def scale_fifa_to_1_10(series: pd.Series) -> pd.Series:
    # wir nutzen min/max aus dem echten Datensatz
    min_val = series.min()
    max_val = series.max()
    # lineare Skalierung auf [1, 10]
    scaled = 1 + (series - min_val) * 9.0 / (max_val - min_val)
    return scaled

X_scaled = X_raw.copy()
for col in feature_cols:
    X_scaled[col] = scale_fifa_to_1_10(X_scaled[col])

# -----------------------------------
# 5. Train/Test Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------------
# 6. RandomForest (kein extra Scaler nötig)
# -----------------------------------
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    random_state=42,
    n_jobs=-1
)

rf.fit(X_train, y_train)

# -----------------------------------
# 7. Evaluation
# -----------------------------------
y_pred = rf.predict(X_test)
print("CLASSIFICATION REPORT (1–10 Skala):")
print(classification_report(y_test, y_pred))

# -----------------------------------
# 8. Pipeline bauen & speichern
# -----------------------------------

# Hier machen wir eine einfache Pipeline, falls du später noch was ergänzen willst.
pipeline = Pipeline([
    ("rf", rf)
])

joblib.dump(pipeline, "position_model_scaled.pkl")
print("\nModel saved as: position_model_scaled.pkl")
