"""FastAPI service for locally trained Kaser-1 weights; no external model API."""
import os, secrets
from pathlib import Path
from typing import List, Literal, Optional
import torch
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from core.tokenizer import KaserTokenizer
from core.kernel import KaserAdaptiveKernel
from model.checkpoint import load_checkpoint
app = FastAPI(title="Kaser-1 Local Inference", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","), allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["Authorization", "Content-Type"])
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT = os.getenv("KASER_CHECKPOINT", "checkpoints/kaser1.pt")
ACCESS_TOKEN = os.getenv("KASER_ACCESS_TOKEN", "").strip()
model = None
kernel = None
load_error = None
def load_model():
    global model, kernel, load_error
    if not Path(CHECKPOINT).is_file():
        load_error = f"No trained checkpoint found at {CHECKPOINT}. Run training first."
        return
    try:
        model, _ = load_checkpoint(CHECKPOINT, device=DEVICE)
        model.eval(); kernel = KaserAdaptiveKernel(model, KaserTokenizer(), DEVICE); load_error = None
    except Exception as exc:
        model = kernel = None; load_error = f"Could not load checkpoint: {type(exc).__name__}: {exc}"
load_model()
class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)
class ChatRequest(BaseModel):
    messages: List[Message] = Field(min_length=1, max_length=30)
    user_id: str = Field(default="local-user", max_length=120)
    max_tokens: int = Field(default=160, ge=1, le=512)
class FeedbackRequest(BaseModel):
    user_id: str = Field(default="local-user", max_length=120)
    prompt: str = Field(min_length=1, max_length=12000)
    is_positive: bool = False
    corrected_text: Optional[str] = Field(default=None, max_length=12000)
def check_access(authorization):
    if ACCESS_TOKEN and not secrets.compare_digest((authorization or "").removeprefix("Bearer ").strip(), ACCESS_TOKEN):
        raise HTTPException(status_code=401, detail="Access token is invalid.")
@app.get("/api/status")
@app.get("/status")
def status():
    return {"service":"kaser-1","model_loaded":model is not None,"trained_weights_available":model is not None,"access_token_required":bool(ACCESS_TOKEN),"access_token_configured":bool(ACCESS_TOKEN),"api_key_configured":model is not None,"device":DEVICE,"checkpoint":CHECKPOINT,"detail":load_error}
@app.post("/api/chat")
@app.post("/chat")
def chat(body: ChatRequest, authorization: Optional[str] = Header(default=None)):
    check_access(authorization)
    if kernel is None: raise HTTPException(status_code=503, detail=load_error or "Trained Kaser-1 weights are not loaded.")
    recent=[m for m in body.messages if m.content.strip()][-12:]
    prompt="\n".join(("<|user|>" if m.role=="user" else "<|assistant|>")+m.content.strip() for m in recent)
    return kernel.generate_adaptive_response(body.user_id,prompt,body.max_tokens)
@app.post("/api/feedback")
@app.post("/feedback")
def feedback(body: FeedbackRequest, authorization: Optional[str] = Header(default=None)):
    check_access(authorization)
    if kernel is None: raise HTTPException(status_code=503,detail=load_error or "Model not loaded")
    kernel.learn_from_feedback(body.user_id,body.prompt,body.is_positive,body.corrected_text)
    return {"status":"saved","note":"Feedback is saved as memory/data; model weights change only during a training run."}
@app.post("/api/export-feedback")
def export_feedback(authorization: Optional[str] = Header(default=None)):
    check_access(authorization)
    if kernel is None: raise HTTPException(status_code=503,detail=load_error or "Model not loaded")
    return {"exported":kernel.export_buffer_for_offline_finetuning()}
