"""Dependency-free UTF-8 byte tokenizer. No external tokenizer or API keys."""
from typing import List
PAD_ID, BOS_ID, EOS_ID = 0, 1, 2
BYTE_OFFSET = 3
VOCAB_SIZE = 259
class KaserTokenizer:
    vocab_size = VOCAB_SIZE
    pad_id, bos_id, eos_id = PAD_ID, BOS_ID, EOS_ID
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids = [b + BYTE_OFFSET for b in text.encode("utf-8", errors="replace")]
        if add_bos: ids.insert(0, BOS_ID)
        if add_eos: ids.append(EOS_ID)
        return ids
    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        data = bytearray()
        for raw in ids:
            token = int(raw)
            if token < BYTE_OFFSET: continue
            if token < VOCAB_SIZE: data.append(token - BYTE_OFFSET)
        return data.decode("utf-8", errors="replace")
