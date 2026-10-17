"""Atomic checkpoint save/load with config compatibility checks."""
import os, tempfile
from pathlib import Path
import torch
from core.config import KaserConfig
from model.architecture import KaserCoreEngine

def save_checkpoint(path, model, optimizer=None, step=0, best_loss=None):
    target = Path(path); target.parent.mkdir(parents=True, exist_ok=True)
    payload = {"model": model.state_dict(), "config": model.config.to_dict(), "step": int(step), "best_loss": best_loss}
    if optimizer is not None: payload["optimizer"] = optimizer.state_dict()
    fd, tmp = tempfile.mkstemp(prefix=target.name, dir=str(target.parent)); os.close(fd)
    try:
        torch.save(payload, tmp); os.replace(tmp, target)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)

def load_checkpoint(path, device="cpu", model=None, optimizer=None):
    payload = torch.load(path, map_location=device, weights_only=False)
    config = KaserConfig.from_dict(payload["config"])
    if model is None: model = KaserCoreEngine(config)
    elif model.config.to_dict() != config.to_dict(): raise ValueError("Checkpoint config does not match model config")
    model.load_state_dict(payload["model"]); model.to(device)
    if optimizer is not None and "optimizer" in payload: optimizer.load_state_dict(payload["optimizer"])
    return model, {"step": payload.get("step", 0), "best_loss": payload.get("best_loss")}
