"""Bounded, persistent episodic memory."""
import json, time
from pathlib import Path
from typing import Any, Dict, List

class DynamicEpisodicMemory:
    def __init__(self, capacity: int = 1000, path: str = "data/memory.jsonl"):
        self.capacity = capacity
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.memory_store: List[Dict[str, Any]] = []
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines()[-capacity:]:
                try: self.memory_store.append(json.loads(line))
                except (ValueError, TypeError): pass
    def remember(self, user_id: str, pattern: str, feedback_score: float = 1.0):
        entry = {"user_id": str(user_id)[:120], "pattern": str(pattern)[:2000], "score": max(0.0, min(1.0, float(feedback_score))), "timestamp": time.time()}
        self.memory_store.append(entry)
        if len(self.memory_store) > self.capacity:
            self.memory_store = sorted(self.memory_store, key=lambda x: (x.get("score", 0), x.get("timestamp", 0)), reverse=True)[:self.capacity]
        self.path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in self.memory_store), encoding="utf-8")
    @staticmethod
    def _terms(text: str):
        return {w.casefold().strip(".,!?؟،:;()[]{}\"'") for w in text.split() if len(w) > 2}
    def retrieve_context(self, user_id: str, query: str) -> str:
        qterms = self._terms(query)
        candidates = []
        for item in self.memory_store:
            if item.get("user_id") != str(user_id) or float(item.get("score", 0)) < 0.7: continue
            overlap = len(qterms & self._terms(item.get("pattern", "")))
            candidates.append((overlap, float(item.get("timestamp", 0)), item["pattern"]))
        relevant = [x[2] for x in sorted(candidates, reverse=True)[:3]]
        if not relevant: return ""
        return "[ملاحظات سياقية من الذاكرة؛ تحقق من صحتها]:\n" + "\n".join(relevant)
