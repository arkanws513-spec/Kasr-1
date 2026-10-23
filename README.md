# KASR-1

KASR-1 is a local-first prototype for versioned skills with evidence-based verification. This milestone uses deterministic allowlisted actions. It does not yet claim general intelligence, arbitrary-task learning, or self-modifying model weights.

## Included
- SQLite registry with version history.
- Allowlisted text/data transformations; no eval, shell, or arbitrary Python execution.
- Independent expected-output verifier with passed/failed/inconclusive outcomes.
- Candidate evaluation and explicit-only promotion.
- Arabic responsive web UI and FastAPI endpoints.
- No paid API keys; core skill engine has no runtime third-party dependencies.

## Run locally
Requires Python 3.10+.

~~~bash
python -m venv .venv
pip install -e '.[test]'
pip install -r requirements.txt
uvicorn kasr.api.main:app --reload
~~~

Open http://127.0.0.1:8000 for the web interface and http://127.0.0.1:8000/docs for the API docs. Run core tests with `pytest -q`.

Skill creation is disabled by default at the HTTP layer. For a trusted local instance, set `KASR_ENABLE_WRITES=true` before starting the server. The UI includes a few demo skills by default; set `KASR_SEED_DEMOS=false` to disable seeding. Use `KASR_DB_PATH` to choose the SQLite path.

## Python example

~~~python
from kasr.skills.registry import SkillRegistry
from kasr.engine.runner import Runner
registry = SkillRegistry('kasr.db')
skill = registry.add('uppercase', 'text.upper')
print(Runner().run(skill, 'Hello, Kasr!'))
~~~

Supported actions: `text.upper`, `text.lower`, `text.strip`, `json.pretty`, `csv.to_json`.

## Safety and limitations
No user-supplied code is executed, and this version contains no LLM. Saved skills are reusable procedures selected from a strict allowlist. The HTTP API is intended for local/trusted use; **do not deploy it publicly with writes enabled** before adding authentication, rate limits, and appropriate data isolation. Persistent storage on free hosting may be ephemeral. Any future model adapter must remain optional and must not bypass validation or tool permissions.

## Status
Milestone 0.2: core, HTTP API, and responsive UI are in the repository. GitHub Actions runs the core test suite on pushes and pull requests. Report test success only after the workflow finishes. See `research/` for the design landscape and roadmap.
