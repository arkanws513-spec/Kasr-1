"""Vercel ASGI entry point for KASR-1."""
from pathlib import Path
import sys

# Ensure the repository root is importable in Vercel's isolated Python runtime.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kasr.api.main import app  # noqa: E402
