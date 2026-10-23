import csv, io, json
from typing import Any
from kasr.skills.registry import Skill

class Runner:
    """Run only fixed, allowlisted transformations; never eval arbitrary code."""
    def run(self, skill: Skill, input_data: Any) -> dict[str, Any]:
        try:
            if skill.action == "text.upper": output = str(input_data).upper()
            elif skill.action == "text.lower": output = str(input_data).lower()
            elif skill.action == "text.strip": output = str(input_data).strip()
            elif skill.action == "json.pretty":
                parsed = json.loads(input_data) if isinstance(input_data,str) else input_data
                output = json.dumps(parsed,ensure_ascii=False,indent=2,sort_keys=True)
            elif skill.action == "csv.to_json":
                if not isinstance(input_data,str): raise TypeError("CSV input must be text")
                reader = csv.DictReader(io.StringIO(input_data))
                if not reader.fieldnames: raise ValueError("CSV must include a header row")
                output = [dict(row) for row in reader]
            else: return {"status":"rejected","error":"Action is not allowlisted"}
            return {"status":"passed","skill":skill.name,"version":skill.version,"output":output}
        except (ValueError,TypeError,csv.Error,json.JSONDecodeError) as exc:
            return {"status":"failed","skill":skill.name,"version":skill.version,"error":f"{type(exc).__name__}: {exc}"}
