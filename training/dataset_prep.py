"""Prepare plain text / JSONL conversation data into a UTF-8 corpus."""
import argparse, json
from pathlib import Path

def extract_text(path: Path):
    if path.suffix.lower() not in {".jsonl", ".ndjson"}:
        return path.read_text(encoding="utf-8", errors="replace")
    rows = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try: item = json.loads(line)
        except ValueError: continue
        if isinstance(item, dict):
            if isinstance(item.get("text"), str): rows.append(item["text"])
            elif isinstance(item.get("messages"), list):
                for m in item["messages"]:
                    if isinstance(m, dict) and m.get("role") in {"user", "assistant"} and isinstance(m.get("content"), str): rows.append(f"<|{m['role']}|> {m['content']}")
            elif isinstance(item.get("prompt"), str) and isinstance(item.get("response"), str): rows.append(f"<|user|> {item['prompt']}\n<|assistant|> {item['response']}")
    return "\n\n".join(rows)

def main():
    p=argparse.ArgumentParser(); p.add_argument("--input", required=True); p.add_argument("--output", default="data/corpus.txt"); args=p.parse_args()
    source=Path(args.input); paths=[source] if source.is_file() else sorted(x for x in source.rglob("*") if x.is_file() and x.suffix.lower() in {".txt",".md",".jsonl",".ndjson"})
    text="\n\n".join(x for x in (extract_text(f).strip() for f in paths) if x)
    if len(text.encode("utf-8")) < 1000: raise SystemExit("Corpus is very small (<1 KB). Add licensed/public-domain text or conversation examples before training.")
    out=Path(args.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(text,encoding="utf-8")
    print(f"Prepared {len(text):,} characters from {len(paths)} file(s) -> {out}")
if __name__=="__main__": main()
