from dataclasses import dataclass
from typing import Any

@dataclass(frozen=True)
class Verification:
    status: str
    passed: int
    total: int
    details: tuple[str,...]

def verify_cases(runner, skill, cases: list[dict[str,Any]]) -> Verification:
    if not cases: return Verification("inconclusive",0,0,("No test cases supplied",))
    passed, details = 0, []
    for index, case in enumerate(cases,1):
        result = runner.run(skill,case.get("input"))
        ok = result.get("status") == "passed" and result.get("output") == case.get("expected")
        if ok: passed += 1
        else: details.append(f"case {index}: expected {case.get('expected')!r}; got {result.get('output',result.get('error'))!r}")
    return Verification("passed" if passed == len(cases) else "failed",passed,len(cases),tuple(details))
