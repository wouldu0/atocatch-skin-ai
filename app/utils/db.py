import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import sqlite3
import json
from datetime import datetime

from config import DB_PATH

def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created  TEXT DEFAULT (datetime('now','localtime'))
        );
        CREATE TABLE IF NOT EXISTS records (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id         INTEGER NOT NULL,
            created         TEXT DEFAULT (datetime('now','localtime')),
            image_path      TEXT,
            predictions     TEXT,
            top_label       TEXT,
            top_prob        REAL,
            survey_score    INTEGER DEFAULT NULL,
            risk_level      TEXT DEFAULT NULL,
            survey_features TEXT DEFAULT NULL,
            FOREIGN KEY (user_id) REFERENCES users(id)
        );
    """)
    conn.commit()
    conn.close()

def get_user(username: str):
    conn = get_conn()
    row = conn.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
    conn.close()
    return dict(row) if row else None

def create_user(username: str, hashed_pw: str):
    conn = get_conn()
    try:
        conn.execute("INSERT INTO users (username, password) VALUES (?,?)",
                     (username, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def save_record(user_id, image_path, predictions, top_label, top_prob,
                survey_score=None, risk_level=None):
    conn = get_conn()
    conn.execute("""
        INSERT INTO records
          (user_id, image_path, predictions, top_label, top_prob, survey_score, risk_level)
        VALUES (?,?,?,?,?,?,?)
    """, (user_id, image_path, json.dumps(predictions, ensure_ascii=False),
          top_label, top_prob, survey_score, risk_level))
    conn.commit()
    record_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
    conn.close()
    return record_id

def update_record_survey(record_id, survey_score, risk_level, survey_features=None):
    conn = get_conn()
    try:
        conn.execute("ALTER TABLE records ADD COLUMN survey_features TEXT DEFAULT NULL")
        conn.commit()
    except Exception:
        pass
    conn.execute(
        "UPDATE records SET survey_score=?, risk_level=?, survey_features=? WHERE id=?",
        (survey_score, risk_level,
         json.dumps(survey_features, ensure_ascii=False) if survey_features else None,
         record_id)
    )
    conn.commit()
    conn.close()

def delete_record(record_id: int, user_id: int):
    conn = get_conn()
    row = conn.execute(
        "SELECT image_path FROM records WHERE id=? AND user_id=?",
        (record_id, user_id)
    ).fetchone()
    if row:
        conn.execute("DELETE FROM records WHERE id=? AND user_id=?",
                     (record_id, user_id))
        conn.commit()
        img = row["image_path"]
        if img and os.path.exists(img):
            try:
                os.remove(img)
            except OSError:
                pass
    conn.close()

def get_user_records(user_id):
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM records WHERE user_id=? ORDER BY created DESC", (user_id,)
    ).fetchall()
    conn.close()
    records = []
    for row in rows:
        r = dict(row)
        r["predictions"]     = json.loads(r["predictions"])     if r["predictions"]     else []
        r["survey_features"] = json.loads(r["survey_features"]) if r.get("survey_features") else None
        records.append(r)
    return records
