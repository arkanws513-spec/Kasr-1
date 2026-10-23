from typing import Any
from kasr.engine.runner import Runner
from kasr.evaluation.verifier import verify_cases
from kasr.skills.registry import Skill, SkillRegistry

class SkillOptimizer:
    def __init__(self, registry: SkillRegistry, runner=None): self.registry, self.runner = registry, runner or Runner()
    def evaluate_candidate(self, baseline: Skill, candidate_action: str, cases: list[dict[str,Any]], description=""):
        candidate = Skill(baseline.name,baseline.version+1,candidate_action,dict(baseline.config),description or baseline.description)
        before, after = verify_cases(self.runner,baseline,cases), verify_cases(self.runner,candidate,cases)
        # Promotion requires demonstrated improvement and never happens implicitly.
        accepted = after.status == "passed" and before.status != "passed"
        return {"baseline":before.__dict__,"candidate":after.__dict__,"accepted":accepted,"note":"No automatic persistence; explicit approval required."}
    def promote(self, name, action, cases, description=""):
        baseline = self.registry.get(name)
        if baseline is None: raise ValueError(f"Unknown skill: {name}")
        result = self.evaluate_candidate(baseline,action,cases,description)
        if not result["accepted"]: raise ValueError("Candidate did not demonstrate an improvement; not promoted")
        return self.registry.add(name,action,baseline.config,description or baseline.description)
