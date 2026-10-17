"""Adaptive orchestration around the actual model; memory does not replace inference."""
import json, time
from pathlib import Path
from typing import Dict, Optional
import torch
from core.memory import DynamicEpisodicMemory
from core.tokenizer import KaserTokenizer

class KaserAdaptiveKernel:
    def __init__(self, model, tokenizer=None, device="cpu", memory_path="data/memory.jsonl", buffer_path="data/feedback_buffer.jsonl"):
        self.base_model = model
        self.tokenizer = tokenizer or KaserTokenizer()
        self.device = torch.device(device)
        self.memory = DynamicEpisodicMemory(path=memory_path)
        self.buffer_path = Path(buffer_path)
        self.training_buffer = []
    def generate_adaptive_response(self, user_id: str, prompt: str, max_tokens: int = 160) -> Dict:
        context = self.memory.retrieve_context(user_id, prompt)
        formatted = ("<|system|>أنت كاسر-1، مساعد واضح وصادق. أجب بلغة المستخدم، ولا تختلق معلومات.\n" + context + "\n<|user|>" + prompt + "\n<|assistant|>")
        ids = self.tokenizer.encode(formatted, add_bos=True)
        input_ids = torch.tensor([ids[-self.base_model.config.block_size:]], dtype=torch.long, device=self.device)
        with torch.inference_mode():
            output = self.base_model.generate(input_ids, max_new_tokens=max(1, min(int(max_tokens), 512)), temperature=0.75, top_k=40, top_p=0.92)
        new_ids = output[0, input_ids.size(1):].tolist()
        response = self.tokenizer.decode(new_ids).strip()
        if not response:
            response = "لم أتمكن من تكوين رد واضح بعد. تحتاج النواة إلى تدريب إضافي على بيانات محادثة جيدة."
        self.training_buffer.append({"user_id": str(user_id), "prompt": prompt, "context": context, "response": response, "timestamp": time.time()})
        return {"response": response, "applied_context": context, "status": "success"}
    def learn_from_feedback(self, user_id: str, prompt: str, is_positive: bool, corrected_text: Optional[str] = None):
        if corrected_text and corrected_text.strip():
            self.memory.remember(user_id, f"سؤال/موضوع: {prompt}\nالتصحيح الذي قدمه المستخدم: {corrected_text.strip()}", 1.0)
            self.training_buffer.append({"user_id": user_id, "prompt": prompt, "correction": corrected_text.strip(), "label": "correction", "timestamp": time.time()})
        elif is_positive:
            self.memory.remember(user_id, f"تفضيل أسلوب إجابة إيجابي في موضوع: {prompt[:180]}", 0.8)
        else:
            self.training_buffer.append({"user_id": user_id, "prompt": prompt, "label": "negative_feedback", "timestamp": time.time()})
    def export_buffer_for_offline_finetuning(self, filepath=None):
        target = Path(filepath) if filepath else self.buffer_path
        target.parent.mkdir(parents=True, exist_ok=True)
        if self.training_buffer:
            with target.open("a", encoding="utf-8") as f:
                for item in self.training_buffer: f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count = len(self.training_buffer)
            self.training_buffer.clear()
            return count
        return 0
