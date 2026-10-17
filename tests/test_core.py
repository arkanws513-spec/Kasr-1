import torch

from core.config import KaserConfig
from core.tokenizer import KaserTokenizer
from model.architecture import KaserCoreEngine


def test_byte_tokenizer_round_trip_arabic_and_english():
    tokenizer = KaserTokenizer()
    text = "أهلًا Kaser-1 🌟"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def small_model(dropout=0.0):
    config = KaserConfig(
        vocab_size=259,
        n_layer=2,
        n_head=4,
        n_embd=64,
        block_size=32,
        dropout=dropout,
    )
    return KaserCoreEngine(config)


def test_model_forward_and_backward():
    model = small_model()
    x = torch.randint(3, 259, (2, 16))
    y = torch.randint(3, 259, (2, 16))
    logits, loss = model(x, y)
    assert logits.shape == (2, 16, 259)
    assert torch.isfinite(loss)
    loss.backward()


def test_cached_attention_matches_full_prefix_logits():
    torch.manual_seed(7)
    model = small_model().eval()
    ids = torch.randint(3, 259, (1, 9))
    with torch.no_grad():
        full_logits, _ = model(ids)
        prefix_logits, _, caches = model(ids[:, :-1], use_cache=True)
        last_logits, _, _ = model(
            ids[:, -1:],
            positions=torch.tensor([ids.size(1) - 1]),
            kv_caches=caches,
            use_cache=True,
        )
    torch.testing.assert_close(last_logits[:, -1], full_logits[:, -1], rtol=1e-4, atol=1e-5)


def test_generation_preserves_training_state_and_batch_shape():
    model = small_model()
    model.train()
    ids = torch.randint(3, 259, (2, 5))
    result = model.generate(ids, max_new_tokens=3, temperature=0)
    assert result.shape == (2, 8)
    assert model.training is True


def test_step_learn_updates_parameters_and_returns_finite_loss():
    model = small_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    prompt = torch.randint(3, 259, (2, 4))
    response = torch.randint(3, 259, (2, 3))
    before = model.token_embedding.weight.detach().clone()
    loss = model.step_learn(prompt, response, optimizer)
    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, model.token_embedding.weight.detach())
