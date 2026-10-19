from __future__ import annotations

import json
import math
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from constants import CLASSES
from games.feedback import NeutralBaseline

MODES = ("recognition", "imitation")
ELIGIBLE = ("correct", "wrong", "timeout", "success")
DEFAULT_SETTINGS = {
    "adaptive": True,
    "recognition_level": 0,
    "imitation_level": 0,
    "threshold": 0.5,
    "hold_seconds": 1.2,
    "sensitivity": 1.0,
}


def progress_path() -> Path:
    if os.getenv("FER_GAME_DIR"):
        return Path(os.environ["FER_GAME_DIR"]) / "progress.sqlite3"
    root = (
        Path(os.getenv("LOCALAPPDATA", str(Path.home())))
        if os.name == "nt"
        else Path(os.getenv("XDG_DATA_HOME", str(Path.home() / ".local/share")))
    )
    return root / "fer-game" / "progress.sqlite3"


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_settings(settings: dict) -> dict:
    if set(settings) != set(DEFAULT_SETTINGS) or type(settings["adaptive"]) is not bool:
        raise ValueError("Invalid profile settings")
    for key in ("recognition_level", "imitation_level"):
        if type(settings[key]) is not int or settings[key] not in range(3):
            raise ValueError("Invalid difficulty")
    for key, lower, upper in (
        ("threshold", 0.35, 0.9),
        ("hold_seconds", 0.5, 5),
        ("sensitivity", 0.5, 2),
    ):
        if not isinstance(settings[key], int | float) or not lower <= settings[key] <= upper:
            raise ValueError(f"Invalid {key}")
    return settings.copy()


class ProgressStore:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or progress_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path, timeout=3)
        self.connection.row_factory = sqlite3.Row
        try:
            self.connection.execute("PRAGMA foreign_keys=ON")
            version = self.connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1):
                raise ValueError("Unsupported progress database version")
            self.connection.executescript("""
                CREATE TABLE IF NOT EXISTS profiles (
                    id INTEGER PRIMARY KEY, name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    settings TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY, profile_id INTEGER NOT NULL REFERENCES profiles(id),
                    mode TEXT NOT NULL, level INTEGER NOT NULL, model TEXT NOT NULL,
                    settings TEXT NOT NULL, started TEXT NOT NULL, ended TEXT
                );
                CREATE TABLE IF NOT EXISTS rounds (
                    session_id TEXT NOT NULL REFERENCES sessions(id), number INTEGER NOT NULL,
                    target TEXT NOT NULL, outcome TEXT NOT NULL, points INTEGER NOT NULL,
                    elapsed REAL NOT NULL, reference TEXT NOT NULL, details TEXT NOT NULL,
                    PRIMARY KEY(session_id, number)
                );
                CREATE TABLE IF NOT EXISTS calibrations (
                    profile_id INTEGER PRIMARY KEY REFERENCES profiles(id),
                    asset_hash TEXT NOT NULL, baseline TEXT NOT NULL, updated TEXT NOT NULL
                );
                PRAGMA user_version=1;
            """)
        except Exception:
            self.connection.close()
            raise

    def close(self) -> None:
        self.connection.close()

    def profiles(self) -> list[dict]:
        return [
            dict(row) for row in self.connection.execute("SELECT id,name FROM profiles ORDER BY id")
        ]

    def add_profile(self, name: str) -> int:
        name = name.strip()
        if not 1 <= len(name) <= 40:
            raise ValueError("Alias must contain 1-40 characters")
        with self.connection:
            cursor = self.connection.execute(
                "INSERT INTO profiles(name,settings) VALUES (?,?)",
                (name, json.dumps(DEFAULT_SETTINGS)),
            )
        return int(cursor.lastrowid)

    def settings(self, profile: int) -> dict:
        row = self.connection.execute(
            "SELECT settings FROM profiles WHERE id=?", (profile,)
        ).fetchone()
        if row is None:
            raise ValueError("Unknown profile")
        return validate_settings(json.loads(row[0]))

    def delete_profile(self, profile: int) -> None:
        with self.connection:
            self.connection.execute(
                "DELETE FROM rounds WHERE session_id IN "
                "(SELECT id FROM sessions WHERE profile_id=?)",
                (profile,),
            )
            self.connection.execute("DELETE FROM sessions WHERE profile_id=?", (profile,))
            self.connection.execute("DELETE FROM calibrations WHERE profile_id=?", (profile,))
            if self.connection.execute("DELETE FROM profiles WHERE id=?", (profile,)).rowcount != 1:
                raise ValueError("Unknown profile")

    def save_settings(self, profile: int, settings: dict) -> None:
        validated = validate_settings(settings)
        with self.connection:
            if (
                self.connection.execute(
                    "UPDATE profiles SET settings=? WHERE id=?", (json.dumps(validated), profile)
                ).rowcount
                != 1
            ):
                raise ValueError("Unknown profile")

    def start_session(self, profile: int, mode: str, level: int, model: str, settings: dict) -> str:
        if mode not in MODES or level not in range(3):
            raise ValueError("Invalid game session")
        identifier = uuid.uuid4().hex
        with self.connection:
            self.connection.execute(
                "INSERT INTO sessions VALUES (?,?,?,?,?,?,?,NULL)",
                (
                    identifier,
                    profile,
                    mode,
                    level,
                    model,
                    json.dumps(settings, allow_nan=False),
                    timestamp(),
                ),
            )
        return identifier

    def end_session(self, identifier: str) -> None:
        with self.connection:
            self.connection.execute(
                "UPDATE sessions SET ended=COALESCE(ended,?) WHERE id=?", (timestamp(), identifier)
            )

    def record_round(self, identifier: str, number: int, result: dict, reference: str) -> None:
        if (
            result["target"] not in CLASSES
            or result["outcome"] not in (*ELIGIBLE, "skipped", "cancelled", "unavailable")
            or type(result["points"]) is not int
            or result["points"] < 0
            or not math.isfinite(result["elapsed"])
            or result["elapsed"] < 0
            or number < 0
        ):
            raise ValueError("Invalid round result")
        values = (
            identifier,
            number,
            result["target"],
            result["outcome"],
            result["points"],
            result["elapsed"],
            reference,
            json.dumps(result, sort_keys=True, allow_nan=False),
        )
        previous = self.connection.execute(
            "SELECT * FROM rounds WHERE session_id=? AND number=?", (identifier, number)
        ).fetchone()
        if previous is not None:
            if tuple(previous) != values:
                raise ValueError("Conflicting result for an existing round")
            return
        with self.connection:
            self.connection.execute("INSERT INTO rounds VALUES (?,?,?,?,?,?,?,?)", values)

    def history(self, profile: int, mode: str | None = None) -> list[dict]:
        query = """SELECT s.mode,s.level,s.started,s.model,r.* FROM sessions s
                   JOIN rounds r ON s.id=r.session_id WHERE s.profile_id=?"""
        arguments: tuple = (profile,)
        if mode is not None:
            query += " AND s.mode=?"
            arguments += (mode,)
        query += " ORDER BY s.started,r.number"
        return [dict(row) for row in self.connection.execute(query, arguments)]

    def next_level(self, profile: int, mode: str, level: int) -> int:
        if mode not in MODES or level not in range(3):
            raise ValueError("Invalid adaptive difficulty input")
        rows = [
            r
            for r in self.history(profile, mode)
            if r["level"] == level and r["outcome"] in ELIGIBLE
        ][-8:]
        if len(rows) < 5:
            return level
        rate = sum(r["outcome"] in ("correct", "success") for r in rows) / len(rows)
        return min(2, level + 1) if rate >= 0.8 else max(0, level - 1) if rate <= 0.35 else level

    def save_calibration(self, profile: int, asset_hash: str, baseline: NeutralBaseline) -> None:
        data = NeutralBaseline.from_dict(baseline.to_dict()).to_dict()
        with self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO calibrations VALUES (?,?,?,?)",
                (profile, asset_hash, json.dumps(data, allow_nan=False), timestamp()),
            )

    def calibration(self, profile: int, asset_hash: str) -> NeutralBaseline | None:
        row = self.connection.execute(
            "SELECT baseline FROM calibrations WHERE profile_id=? AND asset_hash=?",
            (profile, asset_hash),
        ).fetchone()
        return NeutralBaseline.from_dict(json.loads(row[0])) if row else None
