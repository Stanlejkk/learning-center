"""SQLite storage. Everything lives in data/progress.db (local only)."""
from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DB_PATH = Path(os.environ.get("LC_DB", DATA / "progress.db"))

SCHEMA = """
CREATE TABLE IF NOT EXISTS mastery (
    skill_id       TEXT PRIMARY KEY,
    p              REAL NOT NULL,
    n_attempts     INTEGER NOT NULL DEFAULT 0,
    n_correct      INTEGER NOT NULL DEFAULT 0,
    last_seen      TEXT,
    certified      INTEGER NOT NULL DEFAULT 0,
    review_interval REAL NOT NULL DEFAULT 0,
    review_ease    REAL NOT NULL DEFAULT 2.5,
    review_due     TEXT,
    source         TEXT
);
CREATE TABLE IF NOT EXISTS sessions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    kind          TEXT NOT NULL,          -- placement | homework | checkpoint | remediation | review
    track         TEXT,
    title         TEXT,
    created_at    TEXT NOT NULL,
    started_at    TEXT,
    submitted_at  TEXT,
    graded_at     TEXT,
    time_limit_min INTEGER,
    status        TEXT NOT NULL,          -- created | started | submitted | graded | active | done
    data          TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS items (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER NOT NULL REFERENCES sessions(id),
    idx           INTEGER NOT NULL,
    skill_id      TEXT NOT NULL,
    level         INTEGER NOT NULL,
    gen_id        TEXT NOT NULL,
    seed          INTEGER NOT NULL,
    role          TEXT,                   -- frontier | review | stretch | remediation | placement | checkpoint
    problem       TEXT NOT NULL,          -- json of Problem
    user_answer   TEXT,
    correct       INTEGER,
    rubric_checked TEXT,                  -- json list of indices
    score         REAL,
    misconception TEXT,
    answered_at   TEXT
);
CREATE TABLE IF NOT EXISTS labs (
    lab_id        TEXT PRIMARY KEY,
    status        TEXT NOT NULL,
    score         REAL,
    passed        INTEGER,
    total         INTEGER,
    details       TEXT,
    submitted_at  TEXT
);
CREATE TABLE IF NOT EXISTS scans (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    INTEGER,
    path          TEXT NOT NULL,
    uploaded_at   TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    at            TEXT NOT NULL,
    kind          TEXT NOT NULL,
    skill_id      TEXT,
    detail        TEXT
);
CREATE INDEX IF NOT EXISTS idx_items_session ON items(session_id);
CREATE INDEX IF NOT EXISTS idx_items_skill ON items(skill_id);
"""

_db_path_override: Path | None = None


def set_db_path(path: Path | None) -> None:
    """Used by tests to point at a temporary database."""
    global _db_path_override
    _db_path_override = path


def db_path() -> Path:
    return _db_path_override or DB_PATH


def now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect():
    path = db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    try:
        yield con
        con.commit()
    finally:
        con.close()


def init_db() -> None:
    with connect() as con:
        con.executescript(SCHEMA)


def log_event(con, kind: str, skill_id: str | None = None, detail=None) -> None:
    con.execute("INSERT INTO events(at, kind, skill_id, detail) VALUES (?,?,?,?)",
                (now(), kind, skill_id, json.dumps(detail) if detail is not None else None))
