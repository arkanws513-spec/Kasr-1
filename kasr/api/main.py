"""Local-first FastAPI interface. Keep this service private unless protected by auth."""
import os
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from kasr.engine.runner import Runner
from kasr.evaluation.verifier import verify_cases
from kasr.skills.registry import SkillRegistry

DB_PATH = os.environ.get("KASR_DB_PATH", "kasr.db")
ENABLE_WRITES = os.environ.get("KASR_ENABLE_WRITES", "false").lower() in {"1","true","yes"}
registry = SkillRegistry(DB_PATH)
runner = Runner()
app = FastAPI(title="KASR-1 API", version="0.1.0", description="Local-first, allowlisted skill runner")
origins = [x.strip() for x in os.environ.get("KASR_CORS_ORIGINS", "").split(",") if x.strip()]
if origins:
    app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=["GET","POST"], allow_headers=["Content-Type"])

class SkillCreate(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    action: str = Field(min_length=1, max_length=40)
    description: str = Field(default="", max_length=500)
class RunRequest(BaseModel):
    skill_name: str = Field(min_length=1, max_length=80)
    input: Any
class VerifyRequest(BaseModel):
    skill_name: str = Field(min_length=1, max_length=80)
    cases: list[dict[str, Any]] = Field(min_length=1, max_length=100)

@app.get("/")
def home():
    page = Path(__file__).resolve().parents[2] / "web" / "index.html"
    if not page.exists(): raise HTTPException(status_code=404, detail="Web UI file not found")
    return FileResponse(page)
@app.get("/api/health")
def health():
    return {"status":"ok", "service":"KASR-1", "writes_enabled":ENABLE_WRITES, "model_connected":False}
@app.get("/api/skills")
def list_skills():
    return {"skills":[{"name":s.name,"version":s.version,"action":s.action,"description":s.description} for s in registry.list()]}
@app.post("/api/skills")
def create_skill(body: SkillCreate):
    if not ENABLE_WRITES: raise HTTPException(status_code=403, detail="Skill creation is disabled. Set KASR_ENABLE_WRITES=true for a trusted local instance.")
    try: skill = registry.add(body.name, body.action, description=body.description)
    except ValueError as exc: raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"name":skill.name,"version":skill.version,"action":skill.action,"description":skill.description}
@app.post("/api/run")
def run_skill(body: RunRequest):
    skill = registry.get(body.skill_name)
    if skill is None: raise HTTPException(status_code=404, detail="Skill not found")
    return runner.run(skill, body.input)
@app.post("/api/verify")
def verify_skill(body: VerifyRequest):
    skill = registry.get(body.skill_name)
    if skill is None: raise HTTPException(status_code=404, detail="Skill not found")
    result = verify_cases(runner, skill, body.cases)
    return {"status":result.status,"passed":result.passed,"total":result.total,"details":list(result.details)}

if os.environ.get("KASR_SEED_DEMOS", "true").lower() in {"1","true","yes"}:
    for name, action, desc in [("uppercase","text.upper","Convert text to uppercase"),("trim","text.strip","Remove surrounding whitespace"),("csv-to-json","csv.to_json","Convert CSV text to JSON rows")]:
        if registry.get(name) is None:
            try: registry.add(name, action, description=desc)
            except ValueError: pass
