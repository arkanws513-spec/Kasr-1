from inference import build_messages


def test_build_messages_adds_system_prompt_and_keeps_recent_history():
    history = [
        {"role": "user", "content": "old"},
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
    ]

    messages = build_messages(history, "Test system", max_history_messages=2)

    assert messages[0] == {"role": "system", "content": "Test system"}
    assert messages[1:] == [
        {"role": "assistant", "content": "old answer"},
        {"role": "user", "content": "new question"},
    ]


def test_build_messages_ignores_invalid_roles_and_blank_content():
    history = [
        {"role": "system", "content": "not allowed from history"},
        {"role": "user", "content": "   "},
        {"role": "assistant", "content": "  useful answer  "},
    ]

    messages = build_messages(history, "Rules")

    assert messages == [
        {"role": "system", "content": "Rules"},
        {"role": "assistant", "content": "useful answer"},
    ]


def test_build_messages_uses_default_prompt_when_not_supplied(monkeypatch):
    monkeypatch.delenv("SYSTEM_PROMPT", raising=False)

    messages = build_messages([{"role": "user", "content": "Hello"}])

    assert messages[0]["role"] == "system"
    assert "Kasr-1" in messages[0]["content"]
