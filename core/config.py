from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict

@dataclass
class KaserConfig:
    vocab_size: int = 259
    n_layer: int = 4
    n_head: int = 8
    n_embd: int = 256
    block_size: int = 512
    dropout: float = 0.1
    bias: bool = False

    def __post_init__(self):
        if self.n_embd % self.n_head:
            raise ValueError("n_embd must be divisible by n_head")
        if self.block_size < 8:
            raise ValueError("block_size must be at least 8")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KaserConfig":
        allowed = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in data.items() if k in allowed})

    def save(self, path: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str) -> "KaserConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
