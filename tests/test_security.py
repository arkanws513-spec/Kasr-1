import pytest
from fastapi import HTTPException

from app import authorize_chat


def test_chat_fails_closed_when_access_token_is_not_configured(monkeypatch):
    monkeypatch.delenv("KASR_ACCESS_TOKEN", raising=False)

    with pytest.raises(HTTPException) as error:
        authorize_chat(None)

    assert error.value.status_code == 503


def test_chat_rejects_missing_or_wrong_access_token(monkeypatch):
    monkeypatch.setenv("KASR_ACCESS_TOKEN", "a" * 40)

    with pytest.raises(HTTPException) as error:
        authorize_chat("Bearer " + "b" * 40)

    assert error.value.status_code == 401


def test_chat_accepts_correct_access_token(monkeypatch):
    token = "kasr-test-token-" + ("x" * 32)
    monkeypatch.setenv("KASR_ACCESS_TOKEN", token)

    authorize_chat("Bearer " + token)
