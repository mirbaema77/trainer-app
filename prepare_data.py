import pandas as pd

# 1. Datei laden (Pfad ggf. anpassen)
df = pd.read_excel("formation_attributes_cleaned.xlsx")

# 2. Attribut-Spalten definieren
attribute_cols = [
    "Sprint speed",
    "Stamina",
    "Strength",
    "Tactical Awareness",
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

# 3. Alle diese Spalten von "80\n" → 80.0 umwandeln
for col in attribute_cols:
    # sicherstellen, dass Spalte existiert
    if col in df.columns:
        df[col] = (
            df[col]
            .astype(str)                      # alles zu String
            .str.extract(r"(\d+)", expand=False)  # nur Zahl rausziehen
            .astype(float)                    # in float konvertieren
        )

# 4. Optional: Zeilen mit fehlenden Werten droppen
df = df.dropna(subset=attribute_cols)

# 5. Kontrollieren
print(df[attribute_cols].head())
print(df[attribute_cols].dtypes)

# 6. Neue, komplett numerische Datei speichern
df.to_excel("formation_attributes_numeric.xlsx", index=False)
