from unittest.mock import MagicMock

import pytest

import llamevol.llm as llm_mod
from llamevol.llm import LLMS, AnthropicClient, LLMmanager


class _DummyAnthropic:
    """Stand-in for anthropic.Anthropic that records its kwargs and requests."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.messages = MagicMock()


@pytest.fixture
def fake_anthropic(monkeypatch):
    """Swap anthropic.Anthropic so no API key or network access is needed."""
    monkeypatch.setattr(llm_mod.anthropic, "Anthropic", _DummyAnthropic)


def _anthropic_response(*texts, input_tokens=11, output_tokens=7):
    """Build a Messages API response with one text block per string."""
    return MagicMock(
        content=[MagicMock(type="text", text=t) for t in texts],
        usage=MagicMock(input_tokens=input_tokens, output_tokens=output_tokens),
    )


def test_llm_manager_creates_anthropic_client(fake_anthropic):
    llm = LLMmanager(model_name="claude-sonnet-5-5", api_key="sk-ant-test", client_str="anthropic")

    assert isinstance(llm.client, AnthropicClient)
    assert llm.model_name() == "claude-sonnet-5-5"
    assert llm.client.client.kwargs == {"api_key": "sk-ant-test", "base_url": None}


def test_llms_table_has_claude_models(fake_anthropic):
    for key in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5"):
        assert LLMS[key][4] == "anthropic"
        assert isinstance(LLMmanager(model_key=key).client, AnthropicClient)


def test_anthropic_chat_moves_system_prompt_and_counts_tokens(fake_anthropic):
    llm = LLMmanager(model_name="claude-opus-5-5", api_key="sk-ant-test", client_str="anthropic")
    thinking = MagicMock(type="thinking")
    response = _anthropic_response("Hello ", "world")
    response.content.insert(0, thinking)
    llm.client.client.messages.create.return_value = response

    res = llm.chat(
        [
            {"role": "system", "content": "You are an expert."},
            {"role": "user", "content": "Write code."},
        ],
        temperature=0.5,
        top_k=60,
    )

    assert res.error is None
    assert res.text == "Hello world"  # non-text blocks are skipped
    assert res.prompt_token_count == 11
    assert res.response_token_count == 7
    llm.client.client.messages.create.assert_called_once_with(
        model="claude-opus-5-5",
        messages=[{"role": "user", "content": "Write code."}],
        system="You are an expert.",
        max_tokens=AnthropicClient.DEFAULT_MAX_TOKENS,
        temperature=0.5,
        top_k=60,
    )


def test_anthropic_chat_keeps_caller_max_tokens(fake_anthropic):
    llm = LLMmanager(model_name="claude-haiku-5-5", api_key="sk-ant-test", client_str="anthropic")
    llm.client.client.messages.create.return_value = _anthropic_response("ok")

    llm.chat([{"role": "user", "content": "hi"}], max_tokens=123)

    kwargs = llm.client.client.messages.create.call_args.kwargs
    assert kwargs["max_tokens"] == 123
    assert "system" not in kwargs


def test_anthropic_chat_returns_error_instead_of_raising(fake_anthropic):
    llm = LLMmanager(model_name="claude-haiku-5-5", api_key="sk-ant-test", client_str="anthropic")
    llm.client.client.messages.create.side_effect = RuntimeError("overloaded")

    res = llm.chat([{"role": "user", "content": "hi"}])

    assert res.text is None
    assert isinstance(res.error, RuntimeError)
