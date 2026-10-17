import argparse
from pathlib import Path
import torch
from core.config import KaserConfig
from core.tokenizer import KaserTokenizer
from model.architecture import KaserCoreEngine
from model.checkpoint import save_checkpoint, load_checkpoint

def main():
 p=argparse.ArgumentParser()
 p.add_argument('--data',default='data/corpus.txt'); p.add_argument('--out',default='checkpoints/kaser1.pt')
 p.add_argument('--steps',type=int,default=3000); p.add_argument('--batch-size',type=int,default=8)
 p.add_argument('--block-size',type=int,default=256); p.add_argument('--width',type=int,default=256)
 p.add_argument('--layers',type=int,default=4); p.add_argument('--heads',type=int,default=8); p.add_argument('--lr',type=float,default=3e-4)
 p.add_argument('--resume',action='store_true'); a=p.parse_args()
 torch.manual_seed(42); text=Path(a.data).read_text(encoding='utf-8',errors='replace')
 tok=KaserTokenizer(); ids=torch.tensor(tok.encode(text),dtype=torch.long)
 if len(ids)<a.block_size+2: raise SystemExit('Corpus too small for configured block size.')
 cut=max(a.block_size+2,int(len(ids)*.95)); cut=min(cut,len(ids)-1); tr,va=ids[:cut],ids[cut:]
 if len(va)<a.block_size+2: va=tr[-max(a.block_size+2,len(tr)//10):]
 device='cuda' if torch.cuda.is_available() else 'cpu'
 cfg=KaserConfig(vocab_size=tok.vocab_size,n_layer=a.layers,n_head=a.heads,n_embd=a.width,block_size=a.block_size)
 model=KaserCoreEngine(cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=.1); start=0
 if a.resume and Path(a.out).exists(): model,meta=load_checkpoint(a.out,device,model,opt); start=meta['step']
 def batch(src):
  ix=torch.randint(0,len(src)-cfg.block_size-1,(a.batch_size,)); x=torch.stack([src[i:i+cfg.block_size] for i in ix]); y=torch.stack([src[i+1:i+cfg.block_size+1] for i in ix]); return x.to(device),y.to(device)
 for step in range(start,a.steps):
  model.train(); x,y=batch(tr); opt.zero_grad(set_to_none=True); _,loss=model(x,y); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); opt.step()
  if (step+1)%20==0: print(f'step {step+1}/{a.steps} loss={loss.item():.4f}')
  if (step+1)%200==0 or step+1==a.steps:
   model.eval()
   with torch.no_grad():
    x,y=batch(va); _,vloss=model(x,y)
   print(f'validation_loss={vloss.item():.4f}'); save_checkpoint(a.out,model,opt,step+1,vloss.item())
 print('Training complete. Saved checkpoint:',a.out)
if __name__=='__main__': main()
