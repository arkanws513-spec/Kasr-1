import json, sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALLOWED_ACTIONS = {"text.upper", "text.lower", "text.strip", "json.pretty", "csv.to_json"}

@dataclass(frozen=True)
class Skill:
    name: str
    version: int
    action: str
    config: dict[str, Any]
    description: str = ""

class SkillRegistry:
    def __init__(self, db_path="kasr.db"):
        self.db_path = str(db_path)
        if self.db_path != ":memory:": Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as db:
            db.execute("""CREATE TABLE IF NOT EXISTS skills (name TEXT NOT NULL, version INTEGER NOT NULL, action TEXT NOT NULL, config_json TEXT NOT NULL, description TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP, PRIMARY KEY(name, version))""")
    def _connect(self):
        db = sqlite3.connect(self.db_path); db.row_factory = sqlite3.Row; return db
    def add(self, name, action, config=None, description=""):
        name, action = name.strip(), action.strip()
        if not name or not action: raise ValueError("name and action are required")
        if action not in ALLOWED_ACTIONS: raise ValueError(f"action not allowlisted: {action}")
        if len(name) > 80:
            raise ValueError("name must be 80 characters or fewer")
        if len(description) > 500:
            raise ValueError("description must be 500 characters or fewer")
        if not isinstance(config or {}, dict):
            raise ValueError("config must be an object")
        config = config or {}
        with self._connect() as db:
            row = db.execute("SELECT COALESCE(MAX(version),0) v FROM skills WHERE name=?", (name,)).fetchone()
            version = int(row["v"]) + 1
            db.execute("INSERT INTO skills(name,version,action,config_json,description) VALUES(?,?,?,?,?)", (name,version,action,json.dumps(config,ensure_ascii=False),description))
        return Skill(name,version,action,config,description)
    def get(self, name, version=None):
        with self._connect() as db:
            if version is None: row = db.execute("SELECT * FROM skills WHERE name=? ORDER BY version DESC LIMIT 1",(name,)).fetchone()
            else: row = db.execute("SELECT * FROM skills WHERE name=? AND version=?",(name,version)).fetchone()
        if row is None: return None
        return Skill(row["name"],row["version"],row["action"],json.loads(row["config_json"]),row["description"])
    def list(self):
        with self._connect() as db:
            rows = db.execute("SELECT s.* FROM skills s JOIN (SELECT name,MAX(version) version FROM skills GROUP BY name) latest ON s.name=latest.name AND s.version=latest.version ORDER BY s.name").fetchall()
        return [Skill(r["name"],r["version"],r["action"],json.loads(r["config_json"]),r["description"]) for r in rows]
