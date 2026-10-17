import json
import os
from http.server import BaseHTTPRequestHandler
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError

DEFAULT_MODEL = os.environ.get("INFERENCE_MODEL", "openai/gpt-oss-120b:fastest")
BASE_URL = os.environ.get("INFERENCE_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")

class handler(BaseHTTPRequestHandler):
    def _send(self, status, payload):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path not in ("/api/chat", "/api/chat.py"):
            return self._send(404, {"error": "Not found"})
        token = os.environ.get("INFERENCE_API_KEY") or os.environ.get("HF_TOKEN") or os.environ.get("GROQ_API_KEY")
        if not token:
            return self._send(503, {"error": "لم يتم ضبط INFERENCE_API_KEY في إعدادات الاستضافة بعد."})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length < 1 or length > 100_000:
                return self._send(400, {"error": "Invalid request size"})
            body = json.loads(self.rfile.read(length).decode("utf-8"))
            messages = body.get("messages", [])
            if not isinstance(messages, list) or not messages:
                return self._send(400, {"error": "أرسل رسالة أولًا."})
            safe_messages = []
            for item in messages[-24:]:
                if not isinstance(item, dict) or item.get("role") not in ("user", "assistant"):
                    continue
                content = item.get("content")
                if isinstance(content, str) and content.strip():
                    safe_messages.append({"role": item["role"], "content": content[:12000]})
            if not safe_messages or safe_messages[-1]["role"] != "user":
                return self._send(400, {"error": "Invalid messages"})
            language = "English" if body.get("language") == "en" else "Arabic"
            payload = {
                "model": DEFAULT_MODEL,
                "messages": [
                    {"role": "system", "content": "You are Kasr-1, a helpful, honest AI assistant. Reply in the user's language, especially natural Arabic when the user writes Arabic. Be clear and do not claim actions you did not perform."},
                    *safe_messages
                ],
                "temperature": 0.7,
                "max_tokens": 700
            }
            req = Request(
                BASE_URL + "/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Authorization": "Bearer " + token, "Content-Type": "application/json"},
                method="POST"
            )
            with urlopen(req, timeout=55) as response:
                result = json.loads(response.read().decode("utf-8"))
            choices = result.get("choices") or []
            reply = ((choices[0].get("message") or {}).get("content") if choices else None)
            if not isinstance(reply, str) or not reply.strip():
                return self._send(502, {"error": "لم يرجع النموذج ردًا نصيًا."})
            return self._send(200, {"reply": reply.strip(), "model": DEFAULT_MODEL})
        except HTTPError as exc:
            detail = exc.read(1200).decode("utf-8", "replace")
            if exc.code in (401, 403):
                return self._send(502, {"error": "رفض مزوّد النموذج المفتاح أو صلاحياته. تحقق من INFERENCE_API_KEY وصلاحية Inference Providers."})
            if exc.code == 429:
                return self._send(503, {"error": "وصلت إلى حد الاستخدام لدى مزوّد النموذج أو لا يوجد رصيد كافٍ."})
            return self._send(502, {"error": "فشل مزوّد النموذج (HTTP %s)." % exc.code})
        except (URLError, TimeoutError):
            return self._send(504, {"error": "انتهت مهلة الاتصال بمزوّد النموذج. حاول مرة أخرى."})
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError):
            return self._send(400, {"error": "صيغة الطلب غير صحيحة."})
        except Exception:
            return self._send(500, {"error": "حدث خطأ داخلي أثناء إنشاء الرد."})

    def do_GET(self):
        self._send(405, {"error": "Use POST /api/chat"})
