import pandas as pd
import numpy as np
import re

INPUT_FILE = "formation_attributes_cleaned.xlsx"
OUTPUT_FILE = "formation_attributes_numeric.xlsx"

# 1. Datei laden
df = pd.read_excel(INPUT_FILE)

# 2. Relevante Attribut-Spalten (GENAU wie in deiner Excel)
attribute_cols = [
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

# 3. Allgemeine String-Cleanups (Zeilenumbrüche, Spaces)
for col in df.columns:
    if df[col].dtype == object:
        df[col] = (
            df[col]
            .astype(str)
            .str.replace("\r", " ", regex=False)
            .str.replace("\n", " ", regex=False)
            .str.strip()
        )

# 4. Position & Also possible Positions aufbereiten
df["Position"] = df["Position"].astype(str).str.strip().str.upper()
df["Also possible Positions"] = df["Also possible Positions"].astype(str).str.strip().str.upper()

def derive_position(row):
    pos = row["Position"]
    also = row["Also possible Positions"]

    # Falls Position SUB/RES → aus Also possible Positions ableiten
    if pos in {"SUB", "RES"}:
        # Split nach Leerzeichen / Komma / Slash
        tokens = re.split(r"[,\s/]+", also)
        for t in tokens:
            t = t.strip().upper()
            if t and t not in {"SUB", "RES"}:
                return t
        # falls nichts Brauchbares → bleibt SUB/RES (wird später gefiltert)
        return pos
    else:
        return pos

df["Position"] = df.apply(derive_position, axis=1)

# 5. Positionen filtern, die wirklich unbrauchbar sind
invalid_positions = {"SUB", "RES", "", "NAN", "NONE"}
df = df[~df["Position"].isin(invalid_positions)]

# 5b. Feine Positionscodes auf grobe Rollen mappen
POSITION_NORMALIZATION = {
    # Innenverteidiger / Libero → CB
    "LCB": "CB",
    "RCB": "CB",
    "SW":  "CB",

    # Zentrales Mittelfeld
    "LCM": "CM",
    "RCM": "CM",

    # Defensives Mittelfeld
    "LDM": "CDM",
    "RDM": "CDM",

    # Offensives Mittelfeld
    "LAM": "CAM",
    "RAM": "CAM",

    # Flügel / Mittelfeld außen
    "LM":  "LW",
    "RM":  "RW",

    # Stürmer-Varianten → ST
    "LF":  "ST",
    "RF":  "ST",
    "LS":  "ST",
    "RS":  "ST",
    "CF":  "ST",

    # Wing-Backs → Außenverteidiger
    "LWB": "LB",
    "RWB": "RB",
}

df["Position"] = df["Position"].replace(POSITION_NORMALIZATION)

# 5c. GK komplett rauswerfen (kein Keeper-Modell nötig)
df = df[df["Position"] != "GK"]

# 6. Attribute in echte Zahlen umwandeln
#    z.B. "78 Ball control" → 78.0
for col in attribute_cols:
    df[col] = (
        df[col]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
    )

# 7. Fehlende Attribute mit Median auffüllen (statt Zeilen zu löschen)
for col in attribute_cols:
    if df[col].isna().any():
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)

# Sicherheit: Position darf nicht leer sein
df = df.dropna(subset=["Position"])

# 8. Clip auf sinnvollen Bereich 0–99
for col in attribute_cols:
    df[col] = df[col].clip(lower=0, upper=99)

# 9. Kontrolle
print("Rows after cleaning:", len(df))
print(df[["Position"] + attribute_cols].head())
print(df[attribute_cols].dtypes)

# 10. Bereinigte Datei speichern
df.to_excel(OUTPUT_FILE, index=False)
print(f"✅ Saved cleaned numeric data to: {OUTPUT_FILE}")
