# KASR-1

KASR-1 is a local-first prototype for versioned skills with evidence-based verification. This milestone uses deterministic allowlisted actions. It does not yet claim general intelligence, arbitrary-task learning, or self-modifying model weights.

## Included
- SQLite registry with immutable skill versions.
- Allowlisted text/data transformations; no eval, shell, or arbitrary Python execution.
- Independent expected-output verifier with passed/failed/inconclusive outcomes.
- Candidate evaluation and explicit-only promotion.
- No paid API keys or runtime dependencies for the core package.

## Quick start
Requires Python 3.10+.

~~~bash
python -m venv .venv
pip install -e '.[test]'
pytest
~~~

Example:

~~~python
from kasr.skills.registry import SkillRegistry
from kasr.engine.runner import Runner
registry = SkillRegistry('kasr.db')
skill = registry.add('uppercase', 'text.upper')
print(Runner().run(skill, 'Hello, Kasr!'))
~~~

Supported actions: text.upper, text.lower, text.strip, json.pretty, csv.to_json.

## Safety and limitations
No user-supplied code is executed, and this version contains no LLM. Saved skills are reusable procedures selected from a strict allowlist. Any future model adapter must remain optional and must not bypass validation or tool permissions. Do not expose a future execution service to untrusted network clients without authentication and isolation.

## Status
Milestone 0.1 core package and tests have been added. Test results must only be claimed after running tests in a real environment. See research/ for the design landscape and roadmap.
