import json, sqlite3
from pathlib import Path
from typing import Any

class ExperienceLog:
    def __init__(self, db_path="kasr.db"):
        self.db_path = str(db_path)
        if self.db_path != ":memory:": Path(self.db_path).parent.mkdir(parents=True,exist_ok=True)
        with sqlite3.connect(self.db_path) as db:
            db.execute("CREATE TABLE IF NOT EXISTS experiences (id INTEGER PRIMARY KEY, skill_name TEXT NOT NULL, skill_version INTEGER NOT NULL, input_json TEXT NOT NULL, result_json TEXT NOT NULL, verifier_status TEXT NOT NULL, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)")
    def record(self, skill_name: str, skill_version: int, input_data: Any, result: dict, verifier_status: str) -> None:
        with sqlite3.connect(self.db_path) as db:
            db.execute("INSERT INTO experiences(skill_name,skill_version,input_json,result_json,verifier_status) VALUES(?,?,?,?,?)",(skill_name,skill_version,json.dumps(input_data,ensure_ascii=False),json.dumps(result,ensure_ascii=False),verifier_status))
