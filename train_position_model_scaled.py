import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
import joblib


# -----------------------------------
# 1. Daten laden
# -----------------------------------
# Stelle sicher, dass diese Datei im gleichen Ordner liegt
df = pd.read_excel("formation_attributes_numeric.xlsx")

print("Original rows:", len(df))

# -----------------------------------
# 2. Positions-Mapping
# -----------------------------------
position_map = {
    "RB": "RB", "RWB": "RB",
    "LB": "LB", "LWB": "LB",
    "CB": "CB", "LCB": "CB", "RCB": "CB",
    "CDM": "CDM", "RDM": "CDM", "LDM": "CDM",
    "CM": "CM", "LCM": "CM", "RCM": "CM",
    "CAM": "CAM",
    "RM": "RM", "RW": "RW",
    # LM & LW ähnlich, wir können sie zusammenlegen oder trennen.
    # Hier mappen wir beide auf "LM" (linke Seite).
    "LM": "LM", "LW": "LM",
    "CF": "CF", "RF": "CF", "LF": "CF",
    "ST": "ST", "RS": "ST", "LS": "ST",
}

df["Position"] = df["Position"].astype(str).str.strip()
df = df[df["Position"].isin(position_map.keys())].copy()
df["Position_clean"] = df["Position"].map(position_map)

print("Rows after position filter:", len(df))
print("Position distribution:")
print(df["Position_clean"].value_counts())


# -----------------------------------
# 3. Feature-Spalten (passend zu deinem UI)
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
print("Rows after dropping NaN:", len(df_model))

X_raw = df_model[feature_cols]
y = df_model["Position_clean"]


# -----------------------------------
# 4. FIFA 0–99 → 1–10 skalieren (pro Feature)
# -----------------------------------
def scale_fifa_to_1_10(series: pd.Series) -> pd.Series:
    min_val = series.min()
    max_val = series.max()
    # Schutz, falls min == max (sollte in echtem Datensatz nicht passieren)
    if max_val == min_val:
        return pd.Series(5.0, index=series.index)
    scaled = 1.0 + (series - min_val) * 9.0 / (max_val - min_val)
    return scaled


X_scaled = X_raw.copy()
for col in feature_cols:
    X_scaled[col] = scale_fifa_to_1_10(X_scaled[col])

print("Scaled feature sample:")
print(X_scaled.head())


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
# 6. RandomForest trainieren
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
print("CLASSIFICATION REPORT (Modell auf 1–10 Skala):")
print(classification_report(y_test, y_pred))

# -----------------------------------
# 8. Pipeline bauen & speichern
# -----------------------------------
pipeline = Pipeline([
    ("rf", rf)
])

model_path = "position_model_scaled.pkl"
joblib.dump(pipeline, model_path)
print(f"\nModel saved as: {model_path}")
