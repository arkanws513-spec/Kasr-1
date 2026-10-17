import torch
from core.config import KaserConfig
from core.tokenizer import KaserTokenizer
from model.architecture import KaserCoreEngine

def test_byte_tokenizer_round_trip_arabic_and_english():
    tokenizer=KaserTokenizer(); text="أهلًا Kaser-1 🌟"
    assert tokenizer.decode(tokenizer.encode(text)) == text

def test_model_forward_and_backward():
    cfg=KaserConfig(vocab_size=259,n_layer=2,n_head=4,n_embd=64,block_size=32,dropout=0.0)
    model=KaserCoreEngine(cfg); x=torch.randint(3,259,(2,16)); y=torch.randint(3,259,(2,16))
    logits,loss=model(x,y)
    assert logits.shape == (2,16,259)
    assert torch.isfinite(loss)
    loss.backward()
