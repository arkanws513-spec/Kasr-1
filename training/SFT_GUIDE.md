# Kaser-1 supervised fine-tuning

This is stage 2 after base next-token training. It updates the same Kaser-1 architecture using the answer portion of examples as the loss target. Prompt tokens are masked, so the model is optimized to generate the response rather than reproduce the prompt.

## 1. Prepare data

Create a UTF-8 JSONL file, one object per line. Supported formats:

- A pair: `{"prompt":"ما عاصمة مصر؟","response":"عاصمة مصر هي القاهرة."}`
- A conversation: `{"messages":[{"role":"user","content":"Explain gravity simply."},{"role":"assistant","content":"Gravity attracts objects with mass."}]}`

Use at least two examples. For quality, prepare many varied, accurate Arabic and English examples and keep a separate held-out evaluation set. Use data you are permitted to train on.

## 2. Train the base model first

    python -m training.dataset_prep --input data/raw --output data/corpus.txt
    python -m training.train --data data/corpus.txt --steps 3000 --batch-size 8 --block-size 256

## 3. Fine-tune on responses

    python -m training.sft --data data/conversations.jsonl --checkpoint checkpoints/kaser1.pt --out checkpoints/kaser1-sft.pt --steps 1000

The script chooses CUDA when available, otherwise CPU; prints training and held-out response loss; and saves model configuration, weights, optimizer state and step. Continue an existing SFT run with:

    python -m training.sft --data data/conversations.jsonl --out checkpoints/kaser1-sft.pt --resume --steps 2000

With `--resume`, `--steps` is the target total step count, not the number of additional steps. The script currently saves the latest checkpoint at each evaluation interval; check held-out loss and generated responses before deploying it. Lower loss does not guarantee coherent conversation.
