import pytest
from kasr.skills.registry import SkillRegistry
from kasr.engine.runner import Runner
from kasr.evaluation.verifier import verify_cases
from kasr.learning.optimizer import SkillOptimizer

def registry(tmp_path): return SkillRegistry(tmp_path / "skills.db")
def test_versioned_skills(tmp_path):
    r=registry(tmp_path); a=r.add("x","text.strip"); b=r.add("x","text.upper")
    assert (a.version,b.version)==(1,2) and r.get("x").action=="text.upper" and r.get("x",1).action=="text.strip"
def test_allowlist(tmp_path):
    with pytest.raises(ValueError): registry(tmp_path).add("bad","python.exec")
def test_transforms(tmp_path):
    r,runner=registry(tmp_path),Runner(); assert runner.run(r.add("x","text.strip"),"  hi ")["output"]=="hi"
    assert runner.run(r.add("csv","csv.to_json"),"name,age\nMina,12")["output"]==[{"name":"Mina","age":"12"}]
def test_verifier(tmp_path):
    r,runner=registry(tmp_path),Runner(); s=r.add("up","text.upper")
    assert verify_cases(runner,s,[{"input":"a","expected":"A"}]).status=="passed"
    assert verify_cases(runner,s,[{"input":"a","expected":"a"}]).status=="failed"
    assert verify_cases(runner,s,[]).status=="inconclusive"
def test_optimizer_requires_improvement(tmp_path):
    r=registry(tmp_path); r.add("up","text.upper"); opt=SkillOptimizer(r)
    with pytest.raises(ValueError): opt.promote("up","text.lower",[{"input":"a","expected":"A"}])
    assert r.get("up").version==1
