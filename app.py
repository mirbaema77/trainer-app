from flask import Flask, render_template, request, redirect, url_for, session, send_file
import os

from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from datetime import datetime

import pandas as pd

from ml_model import recommend_position_from_attributes, map_form_to_model_features


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


# ------------------------------------------------------------
# FORMATION CONFIG
# ------------------------------------------------------------

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
        {"id": "st",  "code_key": "ST", "label": "ST", "css": "slot-4231-st"},
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
        {"id": "lst", "code_key": "ST", "label": "ST", "css": "slot-442-lst"},
        {"id": "rst", "code_key": "ST", "label": "ST", "css": "slot-442-rst"},
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
        {"id": "lst", "code_key": "ST", "label": "ST", "css": "slot-352-lst"},
        {"id": "rst", "code_key": "ST", "label": "ST", "css": "slot-352-rst"},
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


class Player(db.Model):
    id = db.Column(db.Integer, primary_key=True)

    # ⭐ NEW: owner coach
    coach_id = db.Column(db.Integer, db.ForeignKey("coach.id"), nullable=False)

    # Basic info
    first_name = db.Column(db.String(50), nullable=False)
    last_name = db.Column(db.String(50), nullable=False)
    email = db.Column(db.String(120))
    phone = db.Column(db.String(50))

    # Physical data
    height_cm = db.Column(db.Integer)
    weight_kg = db.Column(db.Integer)
    preferred_foot = db.Column(db.String(5))

    position = db.Column(db.String(50))

    # Ratings ...
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

    # ⭐ optional: backref from Player
    players = db.relationship("Player", backref="coach", lazy=True)

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

    coach = db.relationship("Coach", backref="trainings")


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


@app.route("/")
def home():
    return render_template("splash.html")


@app.route("/start")
def start():
    return render_template("home.html")


@app.route("/training", methods=["GET", "POST"])
def choose_age():
    if request.method == "POST":
        age_group = request.form.get("age_group")
        session["age_group"] = age_group
        return redirect(url_for("choose_focus"))
    return render_template("age.html")


@app.route("/focus", methods=["GET", "POST"])
def choose_focus():
    if request.method == "POST":
        focus = request.form.get("focus")
        session["focus"] = focus
        return redirect(url_for("choose_duration"))
    return render_template("focus.html")


@app.route("/duration", methods=["GET", "POST"])
def choose_duration():
    if request.method == "POST":
        duration = request.form.get("duration")
        session["duration"] = duration
        return redirect(url_for("choose_physical"))
    return render_template("duration.html")


@app.route("/physical", methods=["GET", "POST"])
def choose_physical():
    if request.method == "POST":
        physical = request.form.get("physical")
        session["physical"] = physical
        return redirect(url_for("choose_players"))

    physical = session.get("physical", "medium")
    return render_template("physical.html", physical=physical)


@app.route("/players-count", methods=["GET", "POST"])
def choose_players():
    if request.method == "POST":
        players_count = request.form.get("players")
        session["players"] = players_count
        return redirect(url_for("ai_wishes_teaser"))

    players_count = session.get("players", 16)
    return render_template("players.html", players=players_count)


@app.route("/ai-wishes", methods=["GET", "POST"])
def ai_wishes_teaser():
    if request.method == "POST":
        return redirect(url_for("auth_choice"))
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
    physical = session.get("physical")

    save_message = None

    if request.method == "POST":
        pass

    trainings = find_training_from_excel(
        age_group=age_group,
        focus=focus,
        duration=duration,
        players_count=players_count,
        physical=physical,
    )

    return render_template(
        "summary.html",
        age_group=age_group,
        focus=focus,
        duration=duration,
        players=players_count,
        physical=physical,
        trainings=trainings,
        save_message=save_message,
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

                # 🔴 Ignore next_url for now to avoid 404
                return redirect(url_for("summary"))

    return render_template("register.html", message=message)

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

            # 🔴 Ignore next_url for now to avoid 404
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

    # ⭐ ensure training belongs to this coach
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
    # ⭐ require login + show only own players
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
    # ⭐ require login to create players
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
            coach_id=session["coach_id"],   # ⭐ owner
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
    """⭐ helper: load player and ensure it belongs to logged-in coach."""
    if not session.get("coach_id"):
        return None

    player = Player.query.get_or_404(player_id)
    if player.coach_id != session["coach_id"]:
        return None
    return player


@app.route("/players/<int:player_id>/suggest-position", methods=["POST"])
def suggest_position(player_id):
    # ⭐ require login
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
    # ⭐ protect + ownership
    if not session.get("coach_id"):
        session["next_url"] = url_for("select_formation", player_id=player_id)
        return redirect(url_for("auth_choice"))

    player = _load_owned_player(player_id)
    if player is None:
        return "Not found", 404

    formations = [
        "4-3-3", "4-2-3-1", "4-4-2", "4-4-2-diamond",
        "4-1-4-1", "4-3-1-2", "3-5-2", "3-4-3"
    ]
    return render_template("player_formation_select.html", player=player, formations=formations)


@app.route("/players/<int:player_id>/attributes", methods=["GET", "POST"])
def edit_player_attributes(player_id):
    # ⭐ protect + ownership
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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
