from flask import Flask, render_template, request, redirect, url_for, session, send_file
import os
from email_utils import send_email

from datetime import datetime
import locale

import urllib.parse
import re


from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

import pandas as pd

from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired

from ml_model import recommend_position_from_attributes, map_form_to_model_features

#trainer-app-production-cf65.up.railway.app/_admin/download-db?token=<DEIN-DB_ADMIN_TOKEN>


NICE_POSITION_LABELS = {
    "RB": "Rechter Verteidiger",
    "LB": "Linker Verteidiger",
    "CB": "Innenverteidiger",
    "CDM": "Defensives Mittelfeld",
    "CM": "Zentrales Mittelfeld",
    "CAM": "Offensives Mittelfeld",
    "RM": "Rechtes Mittelfeld",
    "RW": "Rechter Flügel",
    "LM": "Linkes Mittelfeld",
    "LW": "Linker Flügel",
    "CF": "Hängende Spitze",
    "ST": "Mittelstürmer",
}


def get_position_label_for_formation(code: str, formation: str) -> str:
    base = NICE_POSITION_LABELS.get(code, code)

    if formation == "4-3-3":
        if code == "ST":
            return "Mittelstürmer"
        if code in ("LM", "LW"):
            return "Linker Flügelstürmer"
        if code in ("RM", "RW"):
            return "Rechter Flügelstürmer"
        if code == "CF":
            return "Zentrale Spitze"

    if formation == "4-2-3-1":
        if code in ("LM", "LW"):
            return "Linker Flügel"
        if code in ("RM", "RW"):
            return "Rechter Flügel"
        if code == "CAM":
            return "Zehner (ZOM)"
        if code == "CDM":
            return "Defensiver 6er"

    if formation == "4-4-2":
        if code == "ST":
            return "Stürmer"
        if code in ("LM", "LW"):
            return "Linkes Mittelfeld"
        if code in ("RM", "RW"):
            return "Rechtes Mittelfeld"

    return base


FORMATION_SLOTS = {
    "4-3-3": [
        {"id": "gk",  "code_key": "GK", "label": "TW", "css": "slot-433-gk"},
        {"id": "lb",  "code_key": "LB", "label": "LB", "css": "slot-433-lb"},
        {"id": "lcb", "code_key": "CB", "label": "CB", "css": "slot-433-lcb"},
        {"id": "rcb", "code_key": "CB", "label": "CB", "css": "slot-433-rcb"},
        {"id": "rb",  "code_key": "RB", "label": "RB", "css": "slot-433-rb"},
        {"id": "lcm", "code_key": "CM", "label": "CM", "css": "slot-433-lcm"},
        {"id": "cm",  "code_key": "CM", "label": "CM", "css": "slot-433-cm"},
        {"id": "rcm", "code_key": "CM", "label": "CM", "css": "slot-433-rcm"},
        {"id": "lw",  "code_key": "LW", "label": "LW", "css": "slot-433-lw"},
        {"id": "st",  "code_key": "ST", "label": "ST", "css": "slot-433-st"},
        {"id": "rw",  "code_key": "RW", "label": "RW", "css": "slot-433-rw"},
    ],
    "4-2-3-1": [
        {"id": "gk",  "code_key": "GK", "label": "TW", "css": "slot-4231-gk"},
        {"id": "lb",  "code_key": "LB", "label": "LB", "css": "slot-4231-lb"},
        {"id": "lcb", "code_key": "CB", "label": "CB", "css": "slot-4231-lcb"},
        {"id": "rcb", "code_key": "CB", "label": "CB", "css": "slot-4231-rcb"},
        {"id": "rb",  "code_key": "RB", "label": "RB", "css": "slot-4231-rb"},
        {"id": "ldm", "code_key": "CDM", "label": "DM", "css": "slot-4231-ldm"},
        {"id": "rdm", "code_key": "CDM", "label": "DM", "css": "slot-4231-rdm"},
        {"id": "lam", "code_key": "LM",  "label": "LW",  "css": "slot-4231-lam"},
        {"id": "cam", "code_key": "CAM", "label": "CAM", "css": "slot-4231-cam"},
        {"id": "ram", "code_key": "RM",  "label": "RW",  "css": "slot-4231-ram"},
        {"id": "st",  "code_key": "ST", "label": "ST", "css": "slot-4231-st"},
    ],
    "4-4-2": [
        {"id": "gk",  "code_key": "GK", "label": "TW", "css": "slot-442-gk"},
        {"id": "lb",  "code_key": "LB", "label": "LB", "css": "slot-442-lb"},
        {"id": "lcb", "code_key": "CB", "label": "CB", "css": "slot-442-lcb"},
        {"id": "rcb", "code_key": "CB", "label": "CB", "css": "slot-442-rcb"},
        {"id": "rb",  "code_key": "RB", "label": "RB", "css": "slot-442-rb"},
        {"id": "lm",  "code_key": "LM", "label": "LM", "css": "slot-442-lm"},
        {"id": "lcm", "code_key": "CM", "label": "CM", "css": "slot-442-lcm"},
        {"id": "rcm", "code_key": "CM", "label": "CM", "css": "slot-442-rcm"},
        {"id": "rm",  "code_key": "RM", "label": "RM", "css": "slot-442-rm"},
        {"id": "lst", "code_key": "ST", "label": "ST", "css": "slot-442-lst"},
        {"id": "rst", "code_key": "ST", "label": "ST", "css": "slot-442-rst"},
    ],
    "4-4-2-diamond": [
        {"id": "gk",  "code_key": "GK", "label": "TW", "css": "slot-442-gk"},
        {"id": "lb",  "code_key": "LB", "label": "LB", "css": "slot-442-lb"},
        {"id": "lcb", "code_key": "CB", "label": "CB", "css": "slot-442-lcb"},
        {"id": "rcb", "code_key": "CB", "label": "CB", "css": "slot-442-rcb"},
        {"id": "rb",  "code_key": "RB", "label": "RB", "css": "slot-442-rb"},
        {"id": "cdm", "code_key": "CDM", "label": "CDM", "css": "slot-442-lcm"},
        {"id": "lcm", "code_key": "CM",  "label": "CM",  "css": "slot-442-lm"},
        {"id": "rcm", "code_key": "CM",  "label": "CM",  "css": "slot-442-rm"},
        {"id": "cam", "code_key": "CAM", "label": "CAM", "css": "slot-442-rcm"},
        {"id": "lst", "code_key": "ST", "label": "ST", "css": "slot-442-lst"},
        {"id": "rst", "code_key": "ST", "label": "ST", "css": "slot-442-rst"},
    ],
    "4-1-4-1": [
        {"id": "gk",  "code_key": "GK", "label": "TW", "css": "slot-4231-gk"},
        {"id": "lb",  "code_key": "LB", "label": "LB", "css": "slot-4231-lb"},
        {"id": "lcb", "code_key": "CB", "label": "CB", "css": "slot-4231-lcb"},
        {"id": "rcb", "code_key": "CB", "label": "CB", "css": "slot-4231-rcb"},
        {"id": "rb",  "code_key": "RB", "label": "RB", "css": "slot-4231-rb"},
        {"id": "cdm", "code_key": "CDM", "label": "CDM", "css": "slot-4231-ldm"},
        {"id": "lam", "code_key": "LM",  "label": "LM",  "css": "slot-4231-lam"},
        {"id": "lcm", "code_key": "CM",  "label": "CM",  "css": "slot-4231-cam"},
        {"id": "rcm", "code_key": "CM",  "label": "CM",  "css": "slot-4231-ram"},
        {"id": "ram", "code_key": "RM",  "label": "RM",  "css": "slot-4231-ram"},
        {"id": "st",  "code_key": "ST",  "label": "ST",  "css": "slot-4231-st"},
    ],
    "4-3-1-2": [
        {"id": "gk",  "code_key": "GK", "label": "TW", "css": "slot-433-gk"},
        {"id": "lb",  "code_key": "LB", "label": "LB", "css": "slot-433-lb"},
        {"id": "lcb", "code_key": "CB", "label": "CB", "css": "slot-433-lcb"},
        {"id": "rcb", "code_key": "CB", "label": "CB", "css": "slot-433-rcb"},
        {"id": "rb",  "code_key": "RB", "label": "RB", "css": "slot-433-rb"},
        {"id": "lcm", "code_key": "CM",  "label": "CM",  "css": "slot-433-lcm"},
        {"id": "cm",  "code_key": "CM",  "label": "CM",  "css": "slot-433-cm"},
        {"id": "rcm", "code_key": "CM",  "label": "CM",  "css": "slot-433-rcm"},
        {"id": "cam", "code_key": "CAM", "label": "CAM", "css": "slot-4231-cam"},
        {"id": "lst", "code_key": "ST",  "label": "ST",  "css": "slot-442-lst"},
        {"id": "rst", "code_key": "ST",  "label": "ST",  "css": "slot-442-rst"},
    ],
    "3-5-2": [
        {"id": "gk",  "code_key": "GK", "label": "TW", "css": "slot-352-gk"},
        {"id": "lcb", "code_key": "CB", "label": "CB", "css": "slot-352-lcb"},
        {"id": "cb",  "code_key": "CB", "label": "CB", "css": "slot-352-cb"},
        {"id": "rcb", "code_key": "CB", "label": "CB", "css": "slot-352-rcb"},
        {"id": "lwb", "code_key": "LWB", "label": "LWB", "css": "slot-352-lwb"},
        {"id": "lcm", "code_key": "CM",  "label": "CM",  "css": "slot-352-lcm"},
        {"id": "cm",  "code_key": "CM",  "label": "CM",  "css": "slot-352-cm"},
        {"id": "rcm", "code_key": "CM",  "label": "CM",  "css": "slot-352-rcm"},
        {"id": "rwb", "code_key": "RWB", "label": "RWB", "css": "slot-352-rwb"},
        {"id": "lst", "code_key": "ST",  "label": "ST",  "css": "slot-352-lst"},
        {"id": "rst", "code_key": "ST",  "label": "ST",  "css": "slot-352-rst"},
    ],
    "3-4-3": [
        {"id": "gk",  "code_key": "GK", "label": "TW", "css": "slot-343-gk"},
        {"id": "lcb", "code_key": "CB", "label": "CB", "css": "slot-343-lcb"},
        {"id": "cb",  "code_key": "CB", "label": "CB", "css": "slot-343-cb"},
        {"id": "rcb", "code_key": "CB", "label": "CB", "css": "slot-343-rcb"},
        {"id": "lm",  "code_key": "LM",  "label": "LM",  "css": "slot-343-lm"},
        {"id": "lcm", "code_key": "CM",  "label": "CM",  "css": "slot-343-lcm"},
        {"id": "rcm", "code_key": "CM",  "label": "CM",  "css": "slot-343-rcm"},
        {"id": "rm",  "code_key": "RM",  "label": "RM",  "css": "slot-343-rm"},
        {"id": "lw",  "code_key": "LW",  "label": "LW",  "css": "slot-343-lw"},
        {"id": "st",  "code_key": "ST",  "label": "ST",  "css": "slot-343-st"},
        {"id": "rw",  "code_key": "RW",  "label": "RW",  "css": "slot-343-rw"},
    ],
}

FORMATION_ALLOWED = {
    "4-3-3": {"LB", "CB", "RB", "LWB", "RWB", "CDM", "CM", "CAM", "LM", "LW", "RM", "RW", "CF", "ST", "LF", "RF", "LS", "RS"},
    "4-2-3-1": {"LB", "CB", "RB", "CDM", "CM", "CAM", "LM", "LW", "RM", "RW", "CF", "ST"},
    "4-4-2": {"LB", "CB", "RB", "LM", "RM", "CM", "CDM", "CAM", "CF", "ST", "LS", "RS"},
    "4-4-2-diamond": {"LB", "CB", "RB", "CDM", "CM", "CAM", "CF", "ST", "LS", "RS"},
    "4-1-4-1": {"LB", "CB", "RB", "CDM", "CM", "CAM", "LM", "RM", "LW", "RW", "CF", "ST"},
    "4-3-1-2": {"LB", "CB", "RB", "CM", "CAM", "CF", "ST", "LS", "RS"},
    "3-5-2": {"CB", "LWB", "RWB", "CM", "CDM", "CAM", "LM", "RM", "CF", "ST", "LS", "RS"},
    "3-4-3": {"CB", "LWB", "RWB", "CM", "LM", "RM", "LW", "RW", "CF", "ST", "LF", "RF"},
}

POSITION_SIMILAR = {
    "LB":  ["LB", "LWB", "CB"],
    "RB":  ["RB", "RWB", "CB"],
    "CM":  ["CM", "CDM", "CAM"],
    "CDM": ["CDM", "CM", "CB"],
    "CAM": ["CAM", "CF", "CM"],
    "LM":  ["LM", "LW"],
    "RM":  ["RM", "RW"],
    "LW":  ["LW", "LM"],
    "RW":  ["RW", "RM"],
    "CF":  ["CF", "ST", "CAM"],
    "ST":  ["ST", "CF", "LS", "RS"],
    "LS":  ["LS", "ST", "CF"],
    "RS":  ["RS", "ST", "CF"],
    "LF":  ["LF", "LW", "CF"],
    "RF":  ["RF", "RW", "CF"],
}


def get_highlight_code_for_formation(best_code: str, formation: str) -> str:
    formation_def = FORMATION_SLOTS.get(formation, FORMATION_SLOTS["4-3-3"])
    slot_codes = {slot["code_key"] for slot in formation_def}

    if best_code in slot_codes:
        return best_code

    for candidate in POSITION_SIMILAR.get(best_code, []):
        if candidate in slot_codes:
            return candidate

    striker_aliases = {"CF", "LF", "RF", "LS", "RS"}
    if best_code in striker_aliases and "ST" in slot_codes:
        return "ST"

    return next(iter(slot_codes)) if slot_codes else best_code


def normalize_for_formation(code: str) -> str:
    if code in ("LW",):
        return "LM"
    if code in ("RW",):
        return "RM"
    if code in ("CF",):
        return "ST"
    if code in ("CDM", "CAM"):
        return code
    return code


app = Flask(__name__)
app.secret_key = "change-me-later"

app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///trainer.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


def generate_reset_token(coach_id: int) -> str:
    s = get_serializer()
    return s.dumps({"coach_id": coach_id})


def verify_reset_token(token: str, max_age_seconds: int = 3600 * 24) -> int | None:
    s = get_serializer()
    try:
        data = s.loads(token, max_age=max_age_seconds)
        return int(data.get("coach_id"))
    except (BadSignature, SignatureExpired, ValueError, TypeError):
        return None


class Player(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    coach_id = db.Column(db.Integer, db.ForeignKey("coach.id"), nullable=False)

    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120))
    phone = db.Column(db.String(50))

    height_cm = db.Column(db.Integer)
    weight_kg = db.Column(db.Integer)
    preferred_foot = db.Column(db.String(5))

    position = db.Column(db.String(50))

    speed = db.Column(db.Integer, default=5)
    stamina = db.Column(db.Integer, default=5)
    strength = db.Column(db.Integer, default=5)
    aggression = db.Column(db.Integer, default=5)
    tackling = db.Column(db.Integer, default=5)

    acceleration = db.Column(db.Integer, default=5)
    top_speed = db.Column(db.Integer, default=5)
    coordination = db.Column(db.Integer, default=5)
    agility = db.Column(db.Integer, default=5)

    dribbling = db.Column(db.Integer, default=5)
    first_touch = db.Column(db.Integer, default=5)
    short_passing = db.Column(db.Integer, default=5)
    long_passing = db.Column(db.Integer, default=5)
    finishing = db.Column(db.Integer, default=5)
    shooting_power = db.Column(db.Integer, default=5)

    decision_making = db.Column(db.Integer, default=5)
    marking = db.Column(db.Integer, default=5)

    vision = db.Column(db.Integer, default=5)
    creativity = db.Column(db.Integer, default=5)
    composure = db.Column(db.Integer, default=5)
    work_rate_attack = db.Column(db.Integer, default=5)
    work_rate_defense = db.Column(db.Integer, default=5)

    weak_foot = db.Column(db.Integer, default=3)

    @property
    def full_name(self):
        return f"{self.first_name} {self.last_name}"


class Coach(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

    birthdate = db.Column(db.String(20))
    gender = db.Column(db.String(10))
    teamname = db.Column(db.String(100))

    players = db.relationship("Player", backref="coach", lazy=True)
    trainings = db.relationship("Training", backref="coach", lazy=True)

    def set_password(self, password: str):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


class Training(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    coach_id = db.Column(db.Integer, db.ForeignKey("coach.id"), nullable=False)
    name = db.Column(db.String(200), nullable=False)

    age_group = db.Column(db.String(50))
    focus = db.Column(db.String(100))
    duration = db.Column(db.Integer)
    players = db.Column(db.Integer)
    physical = db.Column(db.String(50))

    created_at = db.Column(db.DateTime, default=datetime.utcnow)

if os.getenv("RESET_DB") == "1":
    if os.path.exists("database.db"):
        os.remove("database.db")


with app.app_context():
    db.create_all()


@app.route("/init-db")
def init_db():
    db.create_all()
    return "DB initialized"


def find_training_from_excel(age_group, focus, duration, players_count, physical):
    try:
        df = pd.read_excel("trainings.xlsx")
    except FileNotFoundError:
        return []

    df["Players Age"] = df["Players Age"].astype(str).str.strip()
    df["Training Focus"] = df["Training Focus"].astype(str).str.strip()

    if "Physically Challenge" in df.columns:
        df["Physically Challenge"] = (
            df["Physically Challenge"].astype(str).str.strip().str.lower()
        )

    if "Part" in df.columns:
        df["Part_num"] = pd.to_numeric(df["Part"], errors="coerce")
    else:
        df["Part_num"] = 0

    if "Trainingstime" in df.columns:
        minutes = df["Trainingstime"].astype(str).str.extract(r"(\d+)")[0]
        df["Trainingstime_min"] = pd.to_numeric(minutes, errors="coerce")
    else:
        df["Trainingstime_min"] = None

    if "Number of Players" in df.columns:
        df["NumberPlayers_num"] = pd.to_numeric(
            df["Number of Players"], errors="coerce"
        )
    else:
        df["NumberPlayers_num"] = None

    age_group_str = str(age_group).strip() if age_group else None
    focus_str = str(focus).strip() if focus else None
    physical_str = str(physical).strip().lower() if physical else None

    try:
        duration_int = int(duration) if duration is not None else None
    except (TypeError, ValueError):
        duration_int = None

    try:
        players_int = int(players_count) if players_count is not None else None
    except (TypeError, ValueError):
        players_int = None

    filt = pd.Series(True, index=df.index)

    if age_group_str:
        filt &= df["Players Age"] == age_group_str

    if focus_str:
        filt &= df["Training Focus"] == focus_str

    if physical_str and "Physically Challenge" in df.columns:
        filt &= df["Physically Challenge"] == physical_str

    if duration_int is not None:
        filt &= df["Trainingstime_min"] == duration_int

    if players_int is not None:
        filt &= df["NumberPlayers_num"] == players_int

    matches = df[filt]

    if matches.empty:
        return []

    matches = matches.sort_values("Part_num")

    trainings = []
    for _, row in matches.iterrows():
        trainings.append(
            {
                "part": row.get("Part", ""),
                "physically_challenge": row.get("Physically Challenge", ""),
                "equipment": row.get("Equipment", ""),
                "setup": row.get("Setup", ""),
                "instructions": row.get("Instructions", ""),
                "variations": row.get("Variations", ""),
                "coaching_points": row.get("Coaching Points", ""),
            }
        )

    return trainings

import re

VIDEO_EXCEL_PATH = "trainings_videos.xlsx"


def _parse_players_range(value):
    """
    Supports:
    - int / float (e.g. 12)
    - strings like "8 to 18", "12  to 18", "8-18"
    Returns (min_players, max_players) or (None, None) if unknown.
    """
    if value is None:
        return (None, None)

    # numeric cell
    try:
        if isinstance(value, (int, float)) and not pd.isna(value):
            n = int(value)
            return (n, n)
    except Exception:
        pass

    s = str(value).strip().lower()
    if not s or s == "nan":
        return (None, None)

    # normalize separators
    s = s.replace("–", "-").replace("—", "-")
    s = re.sub(r"\s+", " ", s)

    # "8 to 18" / "8 - 18"
    m = re.search(r"(\d+)\s*(to|-)\s*(\d+)", s)
    if m:
        return (int(m.group(1)), int(m.group(3)))

    # single number in string
    m2 = re.search(r"(\d+)", s)
    if m2:
        n = int(m2.group(1))
        return (n, n)

    return (None, None)


def _players_match(row_players_value, selected_players_count: int) -> bool:
    lo, hi = _parse_players_range(row_players_value)
    if lo is None or hi is None:
        return False
    return lo <= selected_players_count <= hi


def _drive_file_id(drive_link: str) -> str | None:
    if not drive_link:
        return None

    link = str(drive_link).strip()

    # covers:
    # /file/d/<ID>/view
    # /file/d/<ID>/preview
    m = re.search(r"/file/d/([^/]+)/", link)
    if m:
        return m.group(1)

    # covers:
    # .../d/<ID>/... (some shared formats)
    m = re.search(r"/d/([^/]+)", link)
    if m:
        return m.group(1)

    # covers:
    # open?id=<ID>
    # uc?id=<ID>
    try:
        parsed = urllib.parse.urlparse(link)
        qs = urllib.parse.parse_qs(parsed.query)
        if "id" in qs and qs["id"]:
            return qs["id"][0]
    except Exception:
        pass

    return None

def find_training_videos_from_excel(age_group, focus, intensity, players_count):
    try:
        df = pd.read_excel(VIDEO_EXCEL_PATH)
    except FileNotFoundError:
        return {"Aufwärmen": [], "Hauptteil 1": [], "Hauptteil 2": []}

    drive_col = "Google Drive Links "
    if drive_col not in df.columns and "Google Drive Links" in df.columns:
        drive_col = "Google Drive Links"

    def clean(x):
        s = "" if x is None else str(x).strip()
        return "" if s.lower() == "nan" else s

    for c in ["Spieleralter", "Trainingsschwerpunkt", "Intensität", "Trainingsphase", "Spieleranzahl"]:
        df[c] = df[c].apply(clean)
    df[drive_col] = df[drive_col].apply(clean)

    try:
        players_count = int(players_count)
    except Exception:
        return {"Aufwärmen": [], "Hauptteil 1": [], "Hauptteil 2": []}

    # strict filter
    df = df[
        (df["Spieleralter"] == clean(age_group)) &
        (df["Trainingsschwerpunkt"] == clean(focus)) &
        (df["Intensität"] == clean(intensity)) &
        (df[drive_col] != "")
    ]

    out = {"Aufwärmen": [], "Hauptteil 1": [], "Hauptteil 2": []}

    # 🔥 KEY FIX: group by phase
    for phase in out.keys():
        phase_df = df[df["Trainingsphase"] == phase]

        # match players range
        phase_df = phase_df[
            phase_df["Spieleranzahl"].apply(
                lambda v: _players_match(v, players_count)
            )
        ]

        if phase_df.empty:
            continue

        # ✅ deterministic: always first row in Excel for that phase
        row = phase_df.iloc[0]

        drive_link = row[drive_col]

        # extract file id
        m = re.search(r"/file/d/([^/]+)/", drive_link)
        file_id = m.group(1) if m else None

        out[phase] = [{
            "embed_url": drive_to_embed_url(drive_link),
            "thumbnail_url": (
                f"https://drive.google.com/thumbnail?id={file_id}&sz=w1000"
                if file_id else ""
            )
        }]

    return out




def drive_to_embed_url(drive_link: str) -> str:
    """
    Convert various Google Drive link formats into an embeddable preview URL.
    Works for:
      - https://drive.google.com/file/d/<ID>/view?...
      - https://drive.google.com/open?id=<ID>
      - https://drive.google.com/uc?id=<ID>&export=download
    Returns: https://drive.google.com/file/d/<ID>/preview
    """
    if not drive_link:
        return ""

    link = str(drive_link).strip()

    # file/d/<id>/...
    m = re.search(r"/file/d/([^/]+)/", link)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/file/d/{file_id}/preview"

    # ?id=<id>
    try:
        parsed = urllib.parse.urlparse(link)
        qs = urllib.parse.parse_qs(parsed.query)
        if "id" in qs and qs["id"]:
            file_id = qs["id"][0]
            return f"https://drive.google.com/file/d/{file_id}/preview"
    except Exception:
        pass

    # fallback: return original (might still work)
    return link


def compute_phase_minutes(age_group: str, intensity: str, duration_minutes: int) -> dict:
    """
    Returns minutes for 6 phases:
    Einlaufen, Aufwärmspiel, Hauptteil 1, Hauptteil 2, Abschlussspiel, Auslaufen

    Simple rule-set:
    - Base minutes depend on age group
    - Intensity modifies warm-up vs main load
    - Remaining time goes into Hauptteil 1/2 split
    """
    age_group = (age_group or "").strip()
    intensity = (intensity or "").strip()

    # base minutes by age
    if age_group == "E-Junioren (8–10)":
        base = {"Einlaufen": 6, "Aufwärmspiel": 12, "Abschlussspiel": 12, "Auslaufen": 4}
    elif age_group == "D/C-Junioren (11–15)":
        base = {"Einlaufen": 10, "Aufwärmspiel": 16, "Abschlussspiel": 16, "Auslaufen": 6}
    else:  # "16+"
        base = {"Einlaufen": 12, "Aufwärmspiel": 18, "Abschlussspiel": 18, "Auslaufen": 8}

    # intensity modifiers (Leicht / Mittel / Hoch)
    if intensity == "Leicht":
        base["Aufwärmspiel"] = max(8, base["Aufwärmspiel"] - 2)
    elif intensity == "Hoch":
        base["Aufwärmspiel"] = base["Aufwärmspiel"] + 4

    fixed = base["Einlaufen"] + base["Aufwärmspiel"] + base["Abschlussspiel"] + base["Auslaufen"]
    remaining = max(10, duration_minutes - fixed)

    # Split main part
    ht1 = int(round(remaining * 0.55))
    ht2 = remaining - ht1

    # If intensity high: shift a bit from main part into safer prep
    if intensity == "Hoch":
        shift = 4  # we added +4 warm-up, take it from main
        take1 = min(2, ht1 - 5)
        take2 = min(2, ht2 - 5)
        ht1 -= take1
        ht2 -= take2

    return {
        "Einlaufen": base["Einlaufen"],
        "Aufwärmspiel": base["Aufwärmspiel"],
        "Hauptteil 1": ht1,
        "Hauptteil 2": ht2,
        "Abschlussspiel": base["Abschlussspiel"],
        "Auslaufen": base["Auslaufen"],
    }


def compute_phase_text(age_group: str, focus: str, intensity: str) -> dict:
    """
    Auto text for phases without video.
    Keep it short and usable (MVP).
    """
    return {
        "Einlaufen": "Lockeres Einlaufen + Mobilität (Hüfte/Sprunggelenk) + 2–3 koordinative Aufgaben.",
        "Abschlussspiel": f"Abschlussspiel mit Schwerpunkt „{focus}“: klare Regeln, viele Aktionen, kurze Coaching-Impulse.",
        "Auslaufen": "3–5 Minuten locker auslaufen + kurze Mobilität/Dehnen, Puls runterfahren.",
    }




@app.route("/_debug/videos")
def debug_videos():
    # quick test with hard-coded values you know exist in Excel
    videos = find_training_videos_from_excel(
        age_group="E-Junioren (8–10)",
        focus="Abschluss",
        intensity="Mittel",
        players_count=10
    )
    return {"videos": videos}

@app.route("/_debug/video-values")
def debug_video_values():
    df = pd.read_excel(VIDEO_EXCEL_PATH)

    # detect drive column name (space / no space)
    drive_col = "Google Drive Links "
    if "Google Drive Links" in df.columns and drive_col not in df.columns:
        drive_col = "Google Drive Links"

    def uniq(col):
        if col not in df.columns:
            return []
        return sorted(
            {str(x).strip() for x in df[col].dropna().tolist() if str(x).strip() != ""}
        )

    return {
        "columns": list(df.columns),
        "Spieleralter": uniq("Spieleralter"),
        "Trainingsschwerpunkt": uniq("Trainingsschwerpunkt"),
        "Intensität": uniq("Intensität"),
        "Trainingsphase": uniq("Trainingsphase"),
        "spieleranzahl_examples": sorted({str(x).strip() for x in df["Spieleranzahl"].dropna().tolist()})[:25],
        "drive_col_used": drive_col
    }

@app.route("/_debug/thumbs")
def debug_thumbs():
    ids = [
        "1PvWDuHbkKfdfGkjpgT7Q06LLQuyA1fsT",
        "1EgWHuJZeyUw3iOj5vesMGqU60RMpKsc2",
    ]
    return {
        "thumbs": [
            f"https://drive.google.com/thumbnail?id={i}&sz=w1000" for i in ids
        ]
    }




@app.route("/")
def home():
    return render_template("splash.html")


@app.route("/start")
def start():
    return render_template("home.html")

@app.route("/training-type")
def training_type():
    return render_template("training_type.html")

@app.route("/special-training")
def special_training():
    return render_template("special_training.html")

@app.route("/pro-only")
def pro_only():
    return render_template("pro_only.html")



@app.route("/training", methods=["GET", "POST"])
def choose_age():
    if request.method == "POST":
        age_group = request.form.get("age_group")
        session["age_group"] = age_group
        return redirect(url_for("choose_players"))
    return render_template("age.html")


@app.route("/focus", methods=["GET", "POST"])
def choose_focus():
    if request.method == "POST":
        focus = request.form.get("focus")
        session["focus"] = focus
        return redirect(url_for("choose_physical"))
    return render_template("focus.html")


@app.route("/duration", methods=["GET", "POST"])
def choose_duration():
    if request.method == "POST":
        session["duration"] = request.form.get("duration")
        return redirect(url_for("ai_wishes_teaser"))

    return render_template("duration.html")



@app.route("/physical", methods=["GET", "POST"])
def choose_physical():
    if request.method == "POST":
        physical = request.form.get("physical")
        session["physical"] = physical
        return redirect(url_for("choose_duration"))

    physical = session.get("physical", "medium")
    return render_template("physical.html", physical=physical)


@app.route("/players-count", methods=["GET", "POST"])
def choose_players():
    if request.method == "POST":
        players_count = request.form.get("players")
        session["players"] = players_count
        return redirect(url_for("choose_focus"))

    players_count = session.get("players", 16)
    return render_template("players.html", players=players_count)


@app.route("/ai-wishes", methods=["GET", "POST"])
def ai_wishes_teaser():
    if request.method == "POST":
        if not session.get("coach_id"):
            session["next_url"] = url_for("summary")
            return redirect(url_for("auth_choice"))
        return redirect(url_for("summary"))

    return render_template("ai_wishes_teaser.html")


@app.route("/install")
def install_pwa():
    return render_template("install.html")

@app.route("/summary", methods=["GET", "POST"])
def summary():
    age_group = session.get("age_group")
    focus = session.get("focus")
    duration = session.get("duration")
    players_count = session.get("players")
    intensity = session.get("physical")  # we store intensity in "physical"

    # duration int
    try:
        duration_int = int(duration) if duration else 75
    except Exception:
        duration_int = 75

    # Load videos (from new excel)
    # Load videos (already includes embed_url + thumbnail_url)
    videos = find_training_videos_from_excel(
        age_group=age_group,
        focus=focus,
        intensity=intensity,
        players_count=players_count,
    )


    # Phase minutes + text
    phase_minutes = compute_phase_minutes(age_group, intensity, duration_int)
    phase_text = compute_phase_text(age_group, focus, intensity)

    try:
        locale.setlocale(locale.LC_TIME, "de_DE.UTF-8")
    except Exception:
        pass

    day_name = datetime.now().strftime("%A")

    return render_template(
        "summary_videos.html",
        age_group=age_group,
        focus=focus,
        duration=duration_int,
        players=players_count,
        physical=intensity,
        videos=videos,
        day_name=day_name,
        phase_minutes=phase_minutes,
        phase_text=phase_text,
    )


@app.route("/auth", methods=["GET"])
def auth_choice():
    return render_template("auth_choice.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    message = None

    if request.method == "POST":
        name = (request.form.get("name") or "").strip()
        email = (request.form.get("email") or "").strip()
        password = (request.form.get("password") or "").strip()
        birthdate = (request.form.get("birthdate") or "").strip()
        gender = (request.form.get("gender") or "").strip()
        teamname = (request.form.get("teamname") or "").strip()

        if not name or not email or not password:
            message = "Bitte Name, E-Mail und Passwort eingeben."
        else:
            existing = Coach.query.filter_by(email=email).first()
            if existing:
                message = "Für diese E-Mail existiert bereits ein Konto. Bitte einloggen."
            else:
                coach = Coach(
                    name=name,
                    email=email,
                    birthdate=birthdate,
                    gender=gender,
                    teamname=teamname,
                )
                coach.set_password(password)

                db.session.add(coach)
                db.session.commit()

                session["coach_id"] = coach.id
                session["coach_name"] = coach.name

                subject = "Willkommen bei Noqe"
                text_body = (
                    f"Hallo {name},\n\n"
                    "super, dass du jetzt Teil von Noqe bist! Dein Account wurde erfolgreich erstellt.\n\n"
                    f"E-Mail: {email}\n"
                    f"Team: {teamname or '-'}\n\n"
                    "Noqe stärkt dich als Trainer*in in jedem Moment deines Alltags – von der Trainingsvorbereitung über"
                    "die optimale Formationswahl bis hin zu deinem persönlichen Assistenten an deiner Seite."
                    "Wir freuen uns, dich auf diesem Weg zu begleiten.\n\n"
                    "Viel Erfolg und Spass mit Noqe!\n"
                    "Dein Noqe-Team"
                )

                html_body = f"""
                <p>Hallo {name},</p>

                <p>super, dass du jetzt Teil von <strong>Noqe</strong> bist! Dein Konto wurde erfolgreich erstellt.</p>

                <p>
                  <strong>E-Mail:</strong> {email}<br>
                  <strong>Team:</strong> {teamname or '-'}
                </p>

                <p>
                    Noqe stärkt dich als Trainer*in in jedem Moment deines Alltags – von der Trainingsvorbereitung über 
                    die optimale Formationswahl bis hin zu deinem persönlichen Assistenten an deiner Seite. 
                    Wir freuen uns darauf, dich auf deinem Weg zu begleiten!
                </p>

                <p>
                Viel Erfolg und Spass mit Noqe!<br>
                Dein <strong>Noqe-Team</strong>
                </p>
                """
                send_email(email, subject, text_body, html_body)

                return redirect(url_for("summary"))

    return render_template("register.html", message=message)


@app.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    info = None
    if request.method == "POST":
        email = (request.form.get("email") or "").strip()
        if email:
            coach = Coach.query.filter_by(email=email).first()
            if coach:
                token = generate_reset_token(coach.id)
                reset_url = url_for("reset_password", token=token, _external=True)
                subject = "Trainer App – Passwort zurücksetzen"
                text_body = (
                    f"Hallo {coach.name},\n\n"
                    "du hast eine Zurücksetzung deines Passworts angefordert.\n"
                    f"Klicke auf den folgenden Link, um ein neues Passwort zu vergeben:\n\n"
                    f"{reset_url}\n\n"
                    "Wenn du diese Anfrage nicht gestellt hast, kannst du diese E-Mail ignorieren.\n"
                )
                html_body = f"""
                <p>Hallo {coach.name},</p>
                <p>du hast eine Zurücksetzung deines Passworts angefordert.</p>
                <p>
                  Klicke auf den folgenden Link, um ein neues Passwort zu vergeben:<br>
                  <a href="{reset_url}">{reset_url}</a>
                </p>
                <p>Wenn du diese Anfrage nicht gestellt hast, kannst du diese E-Mail ignorieren.</p>
                """
                send_email(email, subject, text_body, html_body)

        info = "Wenn ein Konto mit dieser E-Mail existiert, wurde eine Nachricht versendet."
    return render_template("forgot_password.html", info=info)


@app.route("/reset-password/<token>", methods=["GET", "POST"])
def reset_password(token):
    coach_id = verify_reset_token(token)
    if not coach_id:
        return render_template("reset_password_invalid.html")

    coach = Coach.query.get_or_404(coach_id)

    message = None
    if request.method == "POST":
        password = (request.form.get("password") or "").strip()
        if not password:
            message = "Bitte ein neues Passwort eingeben."
        else:
            coach.set_password(password)
            db.session.commit()
            session["coach_id"] = coach.id
            session["coach_name"] = coach.name
            return redirect(url_for("summary"))

    return render_template("reset_password.html", message=message)


@app.route("/login", methods=["GET", "POST"])
def login():
    message = None

    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        coach = Coach.query.filter_by(email=email).first()
        if coach and coach.check_password(password):
            session["coach_id"] = coach.id
            session["coach_name"] = coach.name
            return redirect(url_for("summary"))
        else:
            message = "E-Mail oder Passwort falsch."

    return render_template("login.html", message=message)


@app.route("/logout")
def logout():
    session.pop("coach_id", None)
    session.pop("coach_name", None)
    return redirect(url_for("home"))


@app.route("/my-trainings")
def my_trainings():
    if not session.get("coach_id"):
        session["next_url"] = url_for("my_trainings")
        return redirect(url_for("auth_choice"))

    trainings = (
        Training.query.filter_by(coach_id=session["coach_id"])
        .order_by(Training.created_at.desc())
        .all()
    )
    return render_template("my_trainings.html", trainings=trainings)


@app.route("/training/<int:training_id>")
def training_detail(training_id):
    if not session.get("coach_id"):
        session["next_url"] = url_for("training_detail", training_id=training_id)
        return redirect(url_for("auth_choice"))

    training = Training.query.get_or_404(training_id)
    if training.coach_id != session["coach_id"]:
        return "Not found", 404

    trainings = find_training_from_excel(
        age_group=training.age_group,
        focus=training.focus,
        duration=training.duration,
        players_count=training.players,
        physical=training.physical,
    )

    return render_template(
        "training_detail.html",
        training=training,
        trainings=trainings,
    )


@app.route("/players")
def list_players():
    if not session.get("coach_id"):
        session["next_url"] = url_for("list_players")
        return redirect(url_for("auth_choice"))

    all_players = (
        Player.query
        .filter_by(coach_id=session["coach_id"])
        .order_by(Player.last_name, Player.first_name)
        .all()
    )
    return render_template("players_list.html", players=all_players)


@app.route("/players/new", methods=["GET", "POST"])
def new_player():
    if not session.get("coach_id"):
        session["next_url"] = url_for("new_player")
        return redirect(url_for("auth_choice"))

    if request.method == "POST":
        first_name = request.form.get("first_name")
        last_name = request.form.get("last_name")
        email = request.form.get("email")
        height_cm = request.form.get("height_cm") or None
        phone = request.form.get("phone")

        if height_cm is not None:
            height_cm = int(height_cm)

        player = Player(
            coach_id=session["coach_id"],
            first_name=first_name,
            last_name=last_name,
            email=email,
            height_cm=height_cm,
            phone=phone,
        )

        db.session.add(player)
        db.session.commit()

        return redirect(url_for("edit_player_attributes", player_id=player.id))

    return render_template("player_new.html")


def _load_owned_player(player_id):
    if not session.get("coach_id"):
        return None

    player = Player.query.get_or_404(player_id)
    if player.coach_id != session["coach_id"]:
        return None
    return player


@app.route("/players/<int:player_id>/suggest-position", methods=["POST"])
def suggest_position(player_id):
    if not session.get("coach_id"):
        session["next_url"] = url_for("select_formation", player_id=player_id)
        return redirect(url_for("auth_choice"))

    player = _load_owned_player(player_id)
    if player is None:
        return "Not found", 404

    formation = request.form.get("formation", "4-3-3")

    attrs_for_model = map_form_to_model_features({
        "speed": player.speed,
        "stamina": player.stamina,
        "strength": player.strength,
        "aggression": player.aggression,
        "tackling": player.tackling,
        "height_cm": player.height_cm or 180,
        "weight_kg": player.weight_kg or 70,
        "first_touch": player.first_touch,
        "dribbling": player.dribbling,
        "short_passing": player.short_passing,
        "long_passing": player.long_passing,
        "finishing": player.finishing,
        "shooting_power": player.shooting_power,
        "decision_making": player.decision_making,
    })

    top3 = recommend_position_from_attributes(attrs_for_model)

    top3_positions = []
    for code, prob in top3:
        top3_positions.append({
            "code": code,
            "label": get_position_label_for_formation(code, formation),
            "percent": f"{prob * 100:.1f}%"
        })

    best_code, best_prob = top3[0]

    formation_def = FORMATION_SLOTS.get(formation, FORMATION_SLOTS["4-3-3"])
    highlight_code = get_highlight_code_for_formation(best_code, formation)
    highlight_percent = f"{best_prob * 100:.1f}%"

    return render_template(
        "player_position_suggestion.html",
        player=player,
        formation=formation,
        slots=formation_def,
        highlight_code=highlight_code,
        highlight_percent=highlight_percent,
        top3_positions=top3_positions,
    )


@app.route("/players/<int:player_id>/formation", methods=["GET"])
def select_formation(player_id):
    player = _load_owned_player(player_id)
    if player is None:
        return redirect(url_for("auth_choice"))

    formations = [
        "4-3-3", "4-2-3-1", "4-4-2", "4-4-2-diamond",
        "4-1-4-1", "4-3-1-2", "3-5-2", "3-4-3"
    ]

    return render_template("player_formation_select.html", player=player, formations=formations)


@app.route("/players/<int:player_id>/attributes", methods=["GET", "POST"])
def edit_player_attributes(player_id):
    if not session.get("coach_id"):
        session["next_url"] = url_for("edit_player_attributes", player_id=player_id)
        return redirect(url_for("auth_choice"))

    player = _load_owned_player(player_id)
    if player is None:
        return "Not found", 404

    if request.method == "POST":
        player.speed = int(request.form.get("speed", player.speed))
        player.stamina = int(request.form.get("stamina", player.stamina))
        player.strength = int(request.form.get("strength", player.strength))
        player.aggression = int(request.form.get("aggression", player.aggression))
        player.tackling = int(request.form.get("tackling", player.tackling))
        player.height_cm = int(request.form.get("height_cm") or 0)
        player.weight_kg = int(request.form.get("weight_kg") or 0)

        player.first_touch = int(request.form.get("first_touch", player.first_touch))
        player.dribbling = int(request.form.get("dribbling", player.dribbling))
        player.short_passing = int(request.form.get("short_passing", player.short_passing))
        player.long_passing = int(request.form.get("long_passing", player.long_passing))
        player.finishing = int(request.form.get("finishing", player.finishing))
        player.shooting_power = int(request.form.get("shooting_power", player.shooting_power))
        player.decision_making = int(request.form.get("decision_making", player.decision_making))

        pf_val = request.form.get("preferred_foot_slider")
        if pf_val == "0":
            player.preferred_foot = "Left"
        elif pf_val == "1":
            player.preferred_foot = "Right"
        else:
            player.preferred_foot = None

        db.session.commit()

        return redirect(url_for("select_formation", player_id=player.id))

    return render_template("player_attributes.html", player=player)



@app.route("/_admin/reset-db")
def admin_reset_db():
    token = request.args.get("token")
    expected = os.environ.get("DB_ADMIN_TOKEN")

    if expected is None:
        return "DB_ADMIN_TOKEN not set", 500

    if token != expected:
        return "Forbidden", 403

    db_path = db.engine.url.database
    if not db_path or not os.path.exists(db_path):
        return "Database not found", 404

    os.remove(db_path)
    return "Database deleted. Restart app to recreate."




@app.route("/_admin/download-db")
def download_db():
    token = request.args.get("token")
    expected = os.environ.get("DB_ADMIN_TOKEN")

    if expected is None:
        return "DB_ADMIN_TOKEN not set on server", 500

    if token != expected:
        return "Forbidden", 403

    db_path = db.engine.url.database

    if not db_path:
        return "No sqlite database configured", 500

    if not os.path.exists(db_path):
        return f"DB file not found: {db_path}", 404

    return send_file(db_path, as_attachment=True)

@app.route("/test-email")
def test_email():
    """Einfache Test-Mail senden."""
    # Zieladresse aus URL ?to=... oder fallback auf SMTP_USERNAME
    to_address = request.args.get("to") or os.environ.get("SMTP_USERNAME")
    if not to_address:
        return "No target email (use ?to=...)", 400

    subject = "Trainer App – Testmail"
    text_body = "Das ist eine Testmail aus deiner Trainer App."
    html_body = "<p>Das ist eine <strong>Testmail</strong> aus deiner Trainer App.</p>"

    send_email(to_address, subject, text_body, html_body)
    return f"Testmail ausgelöst an: {to_address} (Details siehe Railway-Logs)"


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
