import json
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ALLOWED_ACTIONS = {"text.upper", "text.lower", "text.strip", "json.pretty", "csv.to_json"}


@dataclass(frozen=True)
class Skill:
    name: str
    version: int
    action: str
    config: dict[str, Any]
    description: str = ""


class SkillRegistry:
    """Versioned skill registry using SQLite locally or PostgreSQL when configured."""

    def __init__(self, db_path="kasr.db"):
        self.database_url = os.environ.get("KASR_DATABASE_URL") or os.environ.get("DATABASE_URL")
        self.backend = "postgres" if self.database_url and self.database_url.startswith(("postgres://", "postgresql://")) else "sqlite"
        self.db_path = str(db_path)
        if self.backend == "sqlite" and self.db_path != ":memory:":
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            self._execute(db, """CREATE TABLE IF NOT EXISTS skills (
                name TEXT NOT NULL,
                version INTEGER NOT NULL,
                action TEXT NOT NULL,
                config_json TEXT NOT NULL,
                description TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY(name, version)
            )""")

    def _connect(self):
        if self.backend == "postgres":
            try:
                import psycopg
                from psycopg.rows import dict_row
            except ImportError as exc:
                raise RuntimeError("PostgreSQL configured but psycopg is not installed") from exc
            return psycopg.connect(self.database_url, row_factory=dict_row, connect_timeout=8)
        db = sqlite3.connect(self.db_path)
        db.row_factory = sqlite3.Row
        return db

    def _execute(self, db, query, params=()):
        if self.backend == "postgres":
            query = query.replace("?", "%s")
        return db.execute(query, params)

    def add(self, name, action, config=None, description=""):
        name, action = name.strip(), action.strip()
        if not name or not action:
            raise ValueError("name and action are required")
        if action not in ALLOWED_ACTIONS:
            raise ValueError(f"action not allowlisted: {action}")
        if len(name) > 80:
            raise ValueError("name must be 80 characters or fewer")
        if len(description) > 500:
            raise ValueError("description must be 500 characters or fewer")
        if not isinstance(config or {}, dict):
            raise ValueError("config must be an object")
        config = config or {}
        with self._connect() as db:
            row = self._execute(db, "SELECT COALESCE(MAX(version),0) AS v FROM skills WHERE name=?", (name,)).fetchone()
            version = int(row["v"]) + 1
            self._execute(
                db,
                "INSERT INTO skills(name,version,action,config_json,description) VALUES(?,?,?,?,?)",
                (name, version, action, json.dumps(config, ensure_ascii=False), description),
            )
        return Skill(name, version, action, config, description)

    def get(self, name, version=None):
        with self._connect() as db:
            if version is None:
                row = self._execute(db, "SELECT * FROM skills WHERE name=? ORDER BY version DESC LIMIT 1", (name,)).fetchone()
            else:
                row = self._execute(db, "SELECT * FROM skills WHERE name=? AND version=?", (name, version)).fetchone()
        if row is None:
            return None
        return Skill(row["name"], row["version"], row["action"], json.loads(row["config_json"]), row["description"])

    def list(self):
        query = """SELECT s.* FROM skills s
                   JOIN (SELECT name,MAX(version) AS version FROM skills GROUP BY name) latest
                   ON s.name=latest.name AND s.version=latest.version
                   ORDER BY s.name"""
        with self._connect() as db:
            rows = self._execute(db, query).fetchall()
        return [Skill(r["name"], r["version"], r["action"], json.loads(r["config_json"]), r["description"]) for r in rows]
