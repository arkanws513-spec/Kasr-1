# Kaser-1 — from-scratch language-model core

Kaser-1 is a small decoder-only Transformer built from random initialization. It uses a byte-level UTF-8 tokenizer, RMSNorm, RoPE, causal self-attention and SwiGLU. It does not download a pretrained model or call a third-party model API.

## Honest current status

- The model code and training/inference pipeline are implemented.
- A new checkout has no trained weights. Until you train on a substantial, lawful Arabic/English corpus, answers will not be useful and the API intentionally returns HTTP 503 instead of pretending a fixed string is model output.
- The included compact configuration is a development baseline, not GPT-3/Gemini-level capacity. Byte-level tokenization is simple and robust for Arabic/English but less compute-efficient than a trained subword tokenizer.
- Conversation memory stores contextual notes; it does not itself update model weights. Feedback must be exported and used in a later supervised training run.

## Run locally

Python 3.11+ is recommended. Install dependencies:

    python -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt

Place UTF-8 .txt, .md, or .jsonl training sources under data/raw/. Use public-domain, permissively licensed, or personally authored data; keep source/license notes. Convert them to one corpus:

    python -m training.dataset_prep --input data/raw --output data/corpus.txt

Train from random initialization (this can take a long time on CPU):

    python -m training.train --data data/corpus.txt --steps 3000 --batch-size 8 --block-size 256

The training script periodically saves checkpoints/kaser1.pt, including model configuration and optimizer state. Resume with --resume. More data and longer training are necessary for coherent dialogue; lower training loss alone does not prove conversational quality.

Start the inference service after a checkpoint exists:

    uvicorn inference.server:app --host 0.0.0.0 --port 8000

Check GET /api/status, then send POST /api/chat with JSON containing messages, e.g. {"messages":[{"role":"user","content":"اشرح الجاذبية ببساطة"}]}. If KASER_ACCESS_TOKEN is set, send Authorization: Bearer <token>; otherwise local development does not require an access token. Never commit production secrets.

## Model layout

- core/config.py: explicit model configuration
- core/tokenizer.py: dependency-free byte tokenizer (IDs 0–258)
- model/architecture.py: causal Transformer, RoPE and SwiGLU
- core/memory.py and core/kernel.py: persistent episodic memory and response orchestration
- training/: corpus preparation, next-token training and held-out loss evaluation
- inference/server.py: FastAPI service for local checkpoint inference

## Important deployment note

The static frontend on Vercel and the PyTorch model server are separate workloads. Vercel static hosting alone does not provide persistent GPU memory or a durable PyTorch model process. Run this API on a Python-capable host with enough RAM/CPU (or GPU), persist checkpoints/ and data/, then configure the frontend API base URL. Do not put model weights or private keys in the public public/ directory.
