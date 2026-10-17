"""Run Kaser-1 base training followed by supervised response fine-tuning."""
import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", default="data/corpus.txt")
    parser.add_argument("--conversations", required=True, help="JSONL prompt/response examples")
    parser.add_argument("--base-out", default="checkpoints/kaser1.pt")
    parser.add_argument("--sft-out", default="checkpoints/kaser1-sft.pt")
    parser.add_argument("--base-steps", type=int, default=3000)
    parser.add_argument("--sft-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--block-size", type=int, default=256)
    parser.add_argument("--base-lr", type=float, default=3e-4)
    parser.add_argument("--sft-lr", type=float, default=1e-4)
    parser.add_argument("--resume-base", action="store_true")
    parser.add_argument("--resume-sft", action="store_true")
    args = parser.parse_args()

    for file_path in (args.corpus, args.conversations):
        if not Path(file_path).is_file():
            parser.error(f"Input file not found: {file_path}")
    if min(args.base_steps, args.sft_steps, args.batch_size, args.block_size) < 1:
        parser.error("step counts, batch size and block size must be positive")

    base_cmd = [
        sys.executable, "-m", "training.train",
        "--data", args.corpus,
        "--out", args.base_out,
        "--steps", str(args.base_steps),
        "--batch-size", str(args.batch_size),
        "--block-size", str(args.block_size),
        "--lr", str(args.base_lr),
    ]
    if args.resume_base:
        base_cmd.append("--resume")

    print("Stage 1/2: base next-token training", flush=True)
    subprocess.run(base_cmd, check=True)

    sft_cmd = [
        sys.executable, "-m", "training.sft",
        "--data", args.conversations,
        "--checkpoint", args.base_out,
        "--out", args.sft_out,
        "--steps", str(args.sft_steps),
        "--lr", str(args.sft_lr),
    ]
    if args.resume_sft:
        sft_cmd.append("--resume")

    print("Stage 2/2: supervised response fine-tuning", flush=True)
    subprocess.run(sft_cmd, check=True)
    print(f"Pipeline complete. Deploy the fine-tuned checkpoint: {args.sft_out}")


if __name__ == "__main__":
    main()
