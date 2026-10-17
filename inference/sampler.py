"""Generation helpers."""
def generate_text(model, tokenizer, prompt, device="cpu", max_new_tokens=160, temperature=0.75, top_k=40, top_p=0.92):
    import torch
    ids = tokenizer.encode(prompt, add_bos=True)
    x = torch.tensor([ids[-model.config.block_size:]], dtype=torch.long, device=device)
    with torch.inference_mode(): y = model.generate(x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k, top_p=top_p)
    return tokenizer.decode(y[0, x.size(1):].tolist()).strip()
