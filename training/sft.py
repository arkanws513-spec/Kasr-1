"""Supervised fine-tuning for Kaser-1 on prompt/response conversation pairs."""
import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from core.tokenizer import KaserTokenizer
from model.checkpoint import load_checkpoint, save_checkpoint


def load_pairs(path: str) -> List[Tuple[str, str]]:
    """Load JSONL records with prompt/response or user/assistant messages."""
    pairs: List[Tuple[str, str]] = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON on line {line_no}: {exc.msg}") from exc
        if not isinstance(row, dict):
            continue
        prompt, response = row.get("prompt"), row.get("response")
        if isinstance(prompt, str) and isinstance(response, str) and prompt.strip() and response.strip():
            pairs.append((prompt.strip(), response.strip()))
            continue
        messages = row.get("messages")
        if isinstance(messages, list):
            pending_user = None
            for message in messages:
                if not isinstance(message, dict) or not isinstance(message.get("content"), str):
                    continue
                role, body = message.get("role"), message["content"].strip()
                if not body:
                    continue
                if role == "user":
                    pending_user = body
                elif role == "assistant" and pending_user is not None:
                    pairs.append((pending_user, body))
                    pending_user = None
    if len(pairs) < 2:
        raise ValueError(
            f"Need at least 2 prompt/response pairs in {path}; found {len(pairs)}. "
            "Use JSONL rows with prompt+response or messages arrays."
        )
    return pairs


def encode_pair(tokenizer: KaserTokenizer, prompt: str, response: str, block_size: int):
    prompt_ids = tokenizer.encode(prompt, add_bos=True)
    response_ids = tokenizer.encode(response, add_eos=True)
    # Keep at least one prompt token and one response target within the model context.
    response_ids = response_ids[:block_size]
    prompt_budget = block_size + 1 - len(response_ids)
    prompt_ids = prompt_ids[-max(1, prompt_budget):]
    if not response_ids or not prompt_ids:
        raise ValueError("Prompt and response must each contain at least one token")
    return prompt_ids, response_ids


@torch.no_grad()
def response_loss(model, tokenizer, pairs, device, limit=128):
    """Mean response-only token loss on held-out examples; does not update weights."""
    was_training = model.training
    model.eval()
    losses = []
    try:
        for prompt, response in pairs[:limit]:
            p, r = encode_pair(tokenizer, prompt, response, model.config.block_size)
            full = torch.tensor([p + r], dtype=torch.long, device=device)
            inputs, targets = full[:, :-1], full[:, 1:].clone()
            targets[:, :len(p) - 1] = -100
            if inputs.size(1) > model.config.block_size:
                inputs = inputs[:, -model.config.block_size:]
                targets = targets[:, -model.config.block_size:]
            _, loss = model(inputs, targets)
            if loss is not None and torch.isfinite(loss):
                losses.append(float(loss.cpu()))
        return sum(losses) / len(losses) if losses else float("nan")
    finally:
        model.train(was_training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="JSONL with prompt/response or messages")
    parser.add_argument("--checkpoint", default="checkpoints/kaser1.pt", help="Pretrained base checkpoint")
    parser.add_argument("--out", default="checkpoints/kaser1-sft.pt", help="Output fine-tuned checkpoint")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true", help="Resume from --out instead of --checkpoint")
    args = parser.parse_args()
    if args.steps < 1 or args.eval_every < 1 or args.lr <= 0:
        parser.error("--steps and --eval-every must be positive and --lr must be > 0")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    pairs = load_pairs(args.data)
    random.shuffle(pairs)
    split = max(1, int(len(pairs) * 0.9))
    train_pairs = pairs[:split]
    valid_pairs = pairs[split:] or pairs[-1:]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    source = args.out if args.resume else args.checkpoint
    if not Path(source).is_file():
        raise SystemExit(f"Checkpoint not found: {source}. Run base training first.")
    model, meta = load_checkpoint(source, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    if args.resume:
        model, meta = load_checkpoint(source, device=device, model=model, optimizer=optimizer)
    tokenizer = KaserTokenizer()
    start = int(meta.get("step") or 0) if args.resume else 0
    best_loss = meta.get("best_loss")
    best_loss = float(best_loss) if best_loss is not None else float("inf")

    print(f"Loaded {len(pairs)} pairs ({len(train_pairs)} train, {len(valid_pairs)} validation) on {device}")
    for step in range(start, args.steps):
        prompt, response = random.choice(train_pairs)
        prompt_ids, response_ids = encode_pair(tokenizer, prompt, response, model.config.block_size)
        prompt_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        response_tensor = torch.tensor([response_ids], dtype=torch.long, device=device)
        loss = model.step_learn(prompt_tensor, response_tensor, optimizer)

        current = step + 1
        if current % 20 == 0:
            print(f"sft_step={current}/{args.steps} train_response_loss={loss:.4f}")
        if current % args.eval_every == 0 or current == args.steps:
            val_loss = response_loss(model, tokenizer, valid_pairs, device)
            print(f"sft_step={current} validation_response_loss={val_loss:.4f}")
            # Always persist the latest state; report whether it improved validation.
            save_checkpoint(args.out, model, optimizer, current, val_loss)
            if val_loss < best_loss:
                best_loss = val_loss
                print("validation improved")
    print(f"SFT complete. Checkpoint: {args.out}")


if __name__ == "__main__":
    main()
