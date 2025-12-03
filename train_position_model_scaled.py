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
print("Rows loaded:", len(df))

# -----------------------------------
# 2. Feature-Spalten (müssen zu deiner Excel passen)
# -----------------------------------
feature_cols = [
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

# Position-Ziel
y = df["Position"].astype(str).str.strip().str.upper()

# Nur die Feature-Spalten
X_raw = df[feature_cols].copy()

# -----------------------------------
# 3. FIFA 0–99 → 1–10 skalieren (pro Feature)
# -----------------------------------
def scale_fifa_to_1_10(series: pd.Series) -> pd.Series:
    s = series.astype(float)
    min_val = s.min()
    max_val = s.max()
    if max_val == min_val:
        return pd.Series(5.0, index=s.index)
    scaled = 1.0 + (s - min_val) * 9.0 / (max_val - min_val)
    return scaled


X_scaled = X_raw.copy()
for col in feature_cols:
    X_scaled[col] = scale_fifa_to_1_10(X_scaled[col])

print("Scaled feature sample:")
print(X_scaled.head())

# -----------------------------------
# 4. Train/Test-Split
# -----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,   # jede Position proportional im Testset
)

print("Train size:", len(X_train))
print("Test size:", len(X_test))

# -----------------------------------
# 5. Modell definieren
# -----------------------------------
rf = RandomForestClassifier(
    n_estimators=400,
    max_depth=None,
    min_samples_leaf=2,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced_subsample",  # unbalancierte Positionen besser behandeln
)

# -----------------------------------
# 6. Trainieren
# -----------------------------------
rf.fit(X_train, y_train)

# -----------------------------------
# 7. Auswertung
# -----------------------------------
y_pred = rf.predict(X_test)
print("CLASSIFICATION REPORT (Modell auf 1–10 Skala):")
print(classification_report(y_test, y_pred))

# -----------------------------------
# 8. Pipeline bauen & speichern
# -----------------------------------
pipeline = Pipeline([
    ("rf", rf),
])

model_path = "position_model_scaled.pkl"
joblib.dump(pipeline, model_path)
print(f"\n✅ Model saved as: {model_path}")
