"""LLM provider clients with the SDK transport replaced: no network, no real keys."""

from types import SimpleNamespace

import httpx2
import openai
import pytest

from rag_core.config import ENV_FILE, PROJECT_ROOT, Settings
from rag_core.generation import (
    AnthropicClient,
    LLMError,
    OpenAIClient,
    build_llm_client,
)

_REQUEST = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")


def _status_error(cls, status: int, code: str | None = None, message: str = "error"):
    body = {"code": code} if code else None
    return cls(message, response=httpx2.Response(status, request=_REQUEST), body=body)


def _reply(content: str | None, refusal: str | None = None, finish_reason: str = "stop"):
    message = SimpleNamespace(content=content, refusal=refusal)
    return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=finish_reason)])


class FakeCompletions:
    def __init__(self, result=None, error: Exception | None = None) -> None:
        self.result, self.error, self.kwargs = result, error, None

    def create(self, **kwargs):
        self.kwargs = kwargs
        if self.error:
            raise self.error
        return self.result


def _openai(
    model: str = "gpt-6-luna", base_url: str | None = None, **fake
) -> tuple[OpenAIClient, FakeCompletions]:
    client = OpenAIClient(
        api_key="sk-test", model=model, max_tokens=123, timeout_s=5, base_url=base_url
    )
    completions = FakeCompletions(**fake)
    client._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


def test_openai_success_sends_limits_and_messages():
    client, fake = _openai(result=_reply("Ответ [1]."))
    assert client.complete("system", "user") == "Ответ [1]."
    assert fake.kwargs["model"] == "gpt-6-luna"
    assert fake.kwargs["max_completion_tokens"] == 123
    assert fake.kwargs["reasoning_effort"] == "low"
    assert [m["role"] for m in fake.kwargs["messages"]] == ["system", "user"]


def test_openai_client_configures_timeout_and_retries():
    client = OpenAIClient(api_key="sk-test", timeout_s=7)
    assert client._client.max_retries == 2
    assert client._client.timeout == 7


def test_openai_legacy_model_skips_reasoning_effort():
    client, fake = _openai(model="gpt-4o-mini", result=_reply("ok"))
    client.complete("s", "u")
    assert "reasoning_effort" not in fake.kwargs


@pytest.mark.parametrize(
    "error, code",
    [
        (_status_error(openai.AuthenticationError, 401), "auth"),
        (_status_error(openai.RateLimitError, 429), "rate_limit"),
        (_status_error(openai.RateLimitError, 429, "insufficient_quota"), "quota"),
        (
            _status_error(
                openai.RateLimitError,
                429,
                message="Error code: 429 - You have no credits remaining. Add credits to continue.",
            ),
            "quota",
        ),
        (
            _status_error(
                openai.RateLimitError, 429, message="Error: insufficient_quota for this project"
            ),
            "quota",
        ),
        # Gemini's free tier words a per-minute limit as "quota": it clears by itself.
        (
            _status_error(
                openai.RateLimitError,
                429,
                message="You exceeded your current quota, please check your plan and billing.",
            ),
            "rate_limit",
        ),
        (
            _status_error(
                openai.RateLimitError, 429, message="Rate limit reached: tokens per minute (TPM)"
            ),
            "rate_limit",
        ),
        (_status_error(openai.NotFoundError, 404), "model"),
        (_status_error(openai.InternalServerError, 500), "api"),
        (openai.APIConnectionError(request=_REQUEST), "network"),
        (openai.APITimeoutError(request=_REQUEST), "network"),
    ],
)
def test_openai_errors_map_to_codes(error, code):
    client, _ = _openai(error=error)
    with pytest.raises(LLMError) as info:
        client.complete("s", "u")
    assert info.value.code == code
    assert "sk-test" not in str(info.value)


@pytest.mark.parametrize(
    "reply, code",
    [(_reply(None, refusal="no"), "refusal"), (_reply("", finish_reason="length"), "empty")],
)
def test_openai_refusal_and_empty_reply(reply, code):
    client, _ = _openai(result=reply)
    with pytest.raises(LLMError) as info:
        client.complete("s", "u")
    assert info.value.code == code


GROQ_URL = "https://api.groq.com/openai/v1"


def test_openai_client_defaults_to_api_openai_com():
    client = OpenAIClient(api_key="sk-test")
    assert client.base_url is None
    assert str(client._client.base_url).startswith("https://api.openai.com/")


def test_openai_compatible_host_gets_portable_parameters():
    real = OpenAIClient(api_key="gsk-test", model="m", base_url=GROQ_URL)
    assert str(real._client.base_url).rstrip("/") == GROQ_URL
    client, fake = _openai(model="llama-3.3-70b-versatile", base_url=GROQ_URL, result=_reply("ok"))
    assert client.complete("s", "u") == "ok"
    assert fake.kwargs["model"] == "llama-3.3-70b-versatile"
    assert fake.kwargs["max_tokens"] == 123
    assert "max_completion_tokens" not in fake.kwargs
    assert "reasoning_effort" not in fake.kwargs


def test_build_llm_client_passes_base_url_to_openai_only():
    common = {"model": "m", "effort": "low", "max_tokens": 10, "timeout_s": 1.0}
    client = build_llm_client("openai", "k", base_url=GROQ_URL, **common)
    assert isinstance(client, OpenAIClient) and client.base_url == GROQ_URL
    anthropic = build_llm_client("anthropic", "k", base_url=GROQ_URL, **common)
    assert isinstance(anthropic, AnthropicClient)


def test_build_llm_client_selects_provider():
    common = {"model": "m", "effort": "low", "max_tokens": 10, "timeout_s": 1.0}
    assert isinstance(build_llm_client("openai", "k", **common), OpenAIClient)
    assert isinstance(build_llm_client("anthropic", "k", **common), AnthropicClient)
    assert build_llm_client("none", "k", **common) is None
    assert build_llm_client("openai", None, **common) is None


def _settings(**values) -> Settings:
    return Settings(_env_file=None, **values)


def test_settings_pick_key_and_default_model_per_provider():
    s = _settings(llm_provider="openai", openai_api_key=" sk-o ", anthropic_api_key="sk-a")
    assert s.llm_api_key() == "sk-o" and s.resolved_llm_model == "gpt-6-luna"
    s = _settings(llm_provider="anthropic", anthropic_api_key="sk-a", llm_model="claude-x")
    assert s.llm_api_key() == "sk-a" and s.resolved_llm_model == "claude-x"
    assert _settings(llm_provider="openai", openai_api_key="  ").llm_api_key() is None
    assert _settings(llm_provider="none", openai_api_key="sk-o").llm_api_key() is None


def test_base_url_setting_requires_a_model():
    assert _settings().llm_base_url == ""
    s = _settings(llm_base_url=GROQ_URL, llm_model="llama-3.3-70b-versatile", openai_api_key="gsk")
    assert s.resolved_llm_model == "llama-3.3-70b-versatile" and s.llm_api_key() == "gsk"
    with pytest.raises(ValueError, match="LLM_MODEL is required"):
        _settings(llm_base_url=GROQ_URL)
    # Another provider ignores the base URL, so it needs no model.
    assert _settings(llm_provider="anthropic", llm_base_url=GROQ_URL).llm_base_url == GROQ_URL


def test_build_service_uses_the_base_url(tmp_path, monkeypatch):
    from rag_core import service

    monkeypatch.setattr(service, "SentenceTransformerEmbedder", lambda name: None)
    s = _settings(data_dir=tmp_path, llm_base_url=GROQ_URL, llm_model="m", openai_api_key="gsk")
    rag = service.build_service(s)
    assert rag.llm.base_url == GROQ_URL and rag.llm.model == "m"


def test_env_file_and_data_dir_resolve_from_project_root(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # a different working directory must not matter
    assert ENV_FILE == PROJECT_ROOT / ".env"
    assert (PROJECT_ROOT / "pyproject.toml").exists()
    assert _settings().data_dir == PROJECT_ROOT / "data"
    assert _settings(data_dir=tmp_path).data_dir == tmp_path


def test_env_file_values_are_loaded(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "LLM_PROVIDER=anthropic\nANTHROPIC_API_KEY=sk-from-file\nLLM_BASE_URL=http://x/v1\n",
        encoding="utf-8",
    )
    s = Settings(_env_file=env)
    assert s.llm_provider == "anthropic" and s.llm_api_key() == "sk-from-file"
    assert s.llm_base_url == "http://x/v1"
    assert "sk-from-file" not in repr(s)
