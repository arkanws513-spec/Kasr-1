"""Evaluate next-token loss on a text sample."""
import argparse
from pathlib import Path
import torch
from core.tokenizer import KaserTokenizer
from model.checkpoint import load_checkpoint
def main():
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",default="checkpoints/kaser1.pt"); p.add_argument("--data",required=True); a=p.parse_args()
    device="cuda" if torch.cuda.is_available() else "cpu"; model,meta=load_checkpoint(a.checkpoint,device); model.eval()
    ids=torch.tensor(KaserTokenizer().encode(Path(a.data).read_text(encoding="utf-8",errors="replace")),dtype=torch.long)
    n=model.config.block_size
    if len(ids)<n+1: raise SystemExit(f"Evaluation data must contain at least {n+1} byte tokens.")
    losses=[]
    with torch.inference_mode():
        for start in range(0,len(ids)-n-1,n):
            x=ids[start:start+n].unsqueeze(0).to(device); y=ids[start+1:start+n+1].unsqueeze(0).to(device); _,loss=model(x,y); losses.append(loss.item())
    mean=sum(losses)/len(losses); print(f"step={meta['step']} tokens={len(ids)} mean_loss={mean:.4f} perplexity={torch.exp(torch.tensor(mean)).item():.2f}")
if __name__=="__main__": main()
