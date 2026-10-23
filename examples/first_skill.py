from pathlib import Path
import tempfile
from kasr.skills.registry import SkillRegistry
from kasr.engine.runner import Runner

with tempfile.TemporaryDirectory() as directory:
    registry = SkillRegistry(Path(directory) / "demo.db")
    skill = registry.add("clean-text","text.strip",description="Remove leading and trailing whitespace")
    print(Runner().run(skill,"  KASR-1  "))
