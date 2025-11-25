from ml_model import recommend_position_from_attributes

attrs = {
    "Age": 25,
    "Weight (kg)": 75,
    "Body Size (cm)": 180,
    "Sprint speed": 80,
    "Stamina": 78,
    "Strength": 70,
    "Tactical Awareness": 65,
    "Ball control": 82,
    "Dribbling": 85,
    "FK Accuracy": 60,
    "Finishing": 78,
    "Long passing": 72,
    "Long shots": 74,
    "Short passing": 80,
    "Shot power": 76,
    "Tackling": 55,
}

print(recommend_position_from_attributes(attrs))
