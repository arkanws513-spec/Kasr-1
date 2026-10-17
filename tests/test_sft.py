import json

from training.sft import encode_pair, load_pairs
from core.tokenizer import KaserTokenizer


def test_load_pairs_supports_prompt_response_and_messages(tmp_path):
    path = tmp_path / "pairs.jsonl"
    path.write_text(
        json.dumps({"prompt": "ما عاصمة مصر؟", "response": "القاهرة."}, ensure_ascii=False)
        + "\n"
        + json.dumps({"messages": [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I am fine."},
        ]}, ensure_ascii=False),
        encoding="utf-8",
    )
    assert load_pairs(str(path)) == [
        ("ما عاصمة مصر؟", "القاهرة."),
        ("Hello", "Hi"),
        ("How are you?", "I am fine."),
    ]


def test_encode_pair_respects_context_and_keeps_response_targets():
    tokenizer = KaserTokenizer()
    prompt, response = encode_pair(tokenizer, "س" * 100, "جواب", block_size=32)
    assert len(prompt) >= 1
    assert len(response) >= 1
    assert len(prompt) + len(response) <= 33
    assert response[-1] == tokenizer.eos_id
