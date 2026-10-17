# Kasr-1 — Cloud AI Assistant

Kasr-1 is a lightweight, bilingual Arabic/English chat application backed by a **pre-trained model hosted by an inference provider**. It does not train a language model or download model weights.

## Architecture

- **Chat UI:** responsive Arabic/English interface in `public/index.html`.
- **API:** FastAPI routes for chat, health, and configuration status.
- **Inference adapter:** OpenAI-compatible API client; configured for Groq by default.
- **Hosting:** Docker image can be built and deployed through Ron Cloud's container runner or another Docker host.
- **Secrets:** provider keys and the private access token are supplied through environment variables; no real secrets belong in Git.

The first deployment uses one cloud inference provider. This keeps the app simple and lets us verify real model responses before adding optional providers or persistent server-side memory.

## Current default model

The default is `openai/gpt-oss-120b` on Groq, selected from the provider's published model catalog. Provider model IDs, access, quotas, and pricing can change. If it is unavailable to your account, set `INFERENCE_MODEL` to another currently supported model. Check the current catalog at <https://console.groq.com/docs/models>.

The default provider URL is `https://api.groq.com/openai/v1`. The adapter also accepts another OpenAI-compatible provider by changing `INFERENCE_BASE_URL`, `INFERENCE_API_KEY`, and `INFERENCE_MODEL`.

## Run locally

Requires Python 3.11+.

1. Copy `.env.example` to `.env`.
2. Add a valid provider key to `INFERENCE_API_KEY`. For compatibility, `GROQ_API_KEY` is also recognized.
3. Replace `KASR_ACCESS_TOKEN` with a random secret of at least 32 characters. The interface asks for this token and keeps it in session storage for the current browser session.
4. Install dependencies: `python -m pip install -r requirements.txt`.
5. Start the app: `python app.py`.
6. Open <http://localhost:8000>.

Without a provider API key, chat returns HTTP 503. Without a configured access token, chat fails closed with HTTP 503. This avoids leaving a public endpoint open to consume your provider quota.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `INFERENCE_API_KEY` | none | Secret key for the inference provider |
| `INFERENCE_BASE_URL` | Groq OpenAI-compatible URL | Provider API base URL |
| `INFERENCE_MODEL` | `openai/gpt-oss-120b` | Model ID from the provider's current catalog |
| `KASR_ACCESS_TOKEN` | none | Private access token; at least 32 characters |
| `MAX_INPUT_CHARS` | `20000` | Maximum characters per message |
| `MAX_HISTORY_MESSAGES` | `30` | Maximum conversation messages accepted |
| `MAX_OUTPUT_TOKENS` | `2048` | Response token limit |
| `REQUEST_TIMEOUT_SECONDS` | `60` | Provider request timeout |
| `CORS_ORIGINS` | `*` behavior when empty | Comma-separated browser origins; set explicitly for production |

For a public production deployment, set `CORS_ORIGINS` to the exact frontend origins. Keep both secrets in Ron Cloud's deployment secrets/environment settings, not in the repository.

## API

- `GET /api/health` — process health only.
- `GET /api/status` — reports provider and access-token configuration. It does **not** make a live model request.
- `POST /api/chat` — requires `Authorization: Bearer <KASR_ACCESS_TOKEN>`.

Request body:

```json
{
  "messages": [
    {"role": "user", "content": "ما عاصمة مصر؟"}
  ]
}
```

Successful response:

```json
{
  "response": "القاهرة.",
  "model": "openai/gpt-oss-120b"
}
```

## Docker / Ron Cloud

Build and run:

```sh
docker build -t kasr-1 .
docker run --rm -p 8000:8000 \
  -e INFERENCE_API_KEY="$INFERENCE_API_KEY" \
  -e KASR_ACCESS_TOKEN="$KASR_ACCESS_TOKEN" \
  -e INFERENCE_BASE_URL="https://api.groq.com/openai/v1" \
  -e INFERENCE_MODEL="openai/gpt-oss-120b" \
  kasr-1
```

In Ron Cloud, deploy this repository as a Docker application and set both secrets in the deployment configuration. Do not put secrets in a Dockerfile or commit them to Git. A successful container build is not the same as a successful live model call; verify both health and a real chat response after deployment.

## Tests

```sh
python -m pytest -q
```

Unit tests cover message normalization and access-token checks without needing a provider key. A live inference smoke test requires valid provider credentials and may consume account quota.

## Current limitations

- No training or local model weights.
- Conversation history is currently kept in the browser's local storage; it is not a cross-device account memory.
- The current release uses one configured provider per deployment, not automatic multi-provider failover.
- API status is configuration status, not proof that the provider is reachable.
- The shared access token is suitable for initial private testing; before a multi-user public launch, add user accounts, per-user quotas, and stronger abuse controls.
