"""Cloud inference adapter for Kasr-1.

Uses Hugging Face's OpenAI-compatible Inference Providers router by default.
The model is hosted by the provider; this project does not download or train model weights.
"""
from __future__ import annotations

import os

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI


class InferenceConfigurationError(RuntimeError):
    """Raised when required provider settings are missing."""


class InferenceProviderError(RuntimeError):
    """A safe, user-facing provider failure without secrets."""


def build_messages(
    history: list[dict[str, str]],
    system_prompt: str | None = None,
    max_history_messages: int = 30,
) -> list[dict[str, str]]:
    """Normalize conversation history and add the assistant's system prompt."""
    system = (
        system_prompt
        or os.getenv(
            "SYSTEM_PROMPT",
            "You are Kasr-1, a helpful AI assistant. Reply in the language used by "
            "the user (Arabic or English) unless asked otherwise. Be accurate, clear, "
            "and honest about uncertainty. Never claim to have performed actions you "
            "did not perform.",
        )
    ).strip()

    messages: list[dict[str, str]] = [{"role": "system", "content": system}]
    for item in history[-max_history_messages:]:
        role = item.get("role")
        content = item.get("content", "")
        if role not in ("user", "assistant") or not isinstance(content, str):
            continue
        content = content.strip()
        if content:
            messages.append({"role": role, "content": content})
    return messages


class CloudInferenceEngine:
    def __init__(self) -> None:
        self.api_key = (
            os.getenv("INFERENCE_API_KEY")
            or os.getenv("HF_TOKEN")
            or os.getenv("GROQ_API_KEY")
        )
        self.base_url = os.getenv(
            "INFERENCE_BASE_URL", "https://router.huggingface.co/v1"
        ).rstrip("/")
        self.model = os.getenv(
            "INFERENCE_MODEL", "openai/gpt-oss-120b:fastest"
        )
        self.timeout = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "60"))
        self.max_output_tokens = int(os.getenv("MAX_OUTPUT_TOKENS", "2048"))
        self.max_history_messages = int(os.getenv("MAX_HISTORY_MESSAGES", "30"))
        self.client: OpenAI | None = None
        if self.api_key:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=1,
            )

    @property
    def configured(self) -> bool:
        return self.client is not None

    def generate_response(
        self,
        history: list[dict[str, str]],
        system_prompt: str | None = None,
    ) -> str:
        if self.client is None:
            raise InferenceConfigurationError(
                "Inference API key is missing. Configure INFERENCE_API_KEY in deployment secrets."
            )

        messages = build_messages(history, system_prompt, self.max_history_messages)
        if not any(message["role"] == "user" for message in messages):
            raise ValueError("At least one non-empty user message is required.")

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.6,
                max_completion_tokens=self.max_output_tokens,
            )
            answer = completion.choices[0].message.content
            if not answer or not answer.strip():
                raise InferenceProviderError(
                    "The model returned an empty response. Please try again."
                )
            return answer.strip()
        except APITimeoutError as exc:
            raise InferenceProviderError(
                "The inference provider timed out. Please try again."
            ) from exc
        except APIConnectionError as exc:
            raise InferenceProviderError(
                "Could not connect to the inference provider."
            ) from exc
        except APIStatusError as exc:
            if exc.status_code in (401, 403):
                message = "The inference provider rejected the API token or model access."
            elif exc.status_code == 429:
                message = "The inference provider rate limit or credit allowance was reached. Please check your provider account."
            elif exc.status_code == 404:
                message = "The configured model was not found. Check INFERENCE_MODEL against the provider's current model list."
            else:
                message = f"The inference provider returned HTTP {exc.status_code}."
            raise InferenceProviderError(message) from exc
