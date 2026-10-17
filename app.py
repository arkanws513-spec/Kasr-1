from __future__ import annotations

import hmac
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, field_validator

from inference import (
    CloudInferenceEngine,
    InferenceConfigurationError,
    InferenceProviderError,
)

load_dotenv()
BASE_DIR = Path(__file__).resolve().parent
MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "20000"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "30"))
engine = CloudInferenceEngine()


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def content_not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message content cannot be blank.")
        if len(value) > MAX_INPUT_CHARS:
            raise ValueError(f"Message exceeds the {MAX_INPUT_CHARS}-character limit.")
        return value


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=MAX_HISTORY_MESSAGES)
    system_prompt: str | None = Field(default=None, max_length=4000)


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield


app = FastAPI(
    title="Kasr-1 Cloud AI",
    version="1.0.0",
    description="A bilingual chat API backed by a hosted, pre-trained language model.",
    lifespan=lifespan,
)

origins = [
    item.strip()
    for item in os.getenv("CORS_ORIGINS", "").split(",")
    if item.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


def authorize_chat(authorization: str | None) -> None:
    """Fail closed unless a server-side access token is configured and supplied."""
    expected = os.getenv("KASR_ACCESS_TOKEN", "")
    if len(expected) < 32:
        raise HTTPException(
            status_code=503,
            detail="Chat access is not configured. Set a long KASR_ACCESS_TOKEN in deployment secrets.",
        )
    supplied = ""
    if authorization and authorization.startswith("Bearer "):
        supplied = authorization[7:].strip()
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise HTTPException(status_code=401, detail="Invalid or missing Kasr-1 access token.")


@app.get("/api/status")
def status() -> dict[str, object]:
    access_configured = len(os.getenv("KASR_ACCESS_TOKEN", "")) >= 32
    return {
        "app": "Kasr-1",
        "status": "ready" if engine.configured and access_configured else "needs_configuration",
        "provider": engine.base_url,
        "model": engine.model,
        "api_key_configured": engine.configured,
        "access_token_configured": access_configured,
        "note": (
            "Configuration is present; this does not prove the provider is reachable."
            if engine.configured and access_configured
            else "Set INFERENCE_API_KEY and a KASR_ACCESS_TOKEN of at least 32 characters in deployment secrets."
        ),
    }


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/chat")
def chat(
    request: ChatRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, str]:
    authorize_chat(authorization)
    history = [{"role": item.role, "content": item.content} for item in request.messages]
    if not any(item["role"] == "user" for item in history):
        raise HTTPException(status_code=422, detail="At least one user message is required.")

    try:
        answer = engine.generate_response(history, request.system_prompt)
        return {"response": answer, "model": engine.model}
    except InferenceConfigurationError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except InferenceProviderError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    index = BASE_DIR / "public" / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="Frontend file not found.")
    return FileResponse(index)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
