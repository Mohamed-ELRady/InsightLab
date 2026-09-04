"""Provider routing and the GUI-to-run settings boundary."""

from __future__ import annotations

from insightlab.core.config import Settings, with_model_access
from insightlab.core.reasoning import ReasoningEngine


def test_provider_native_ids_receive_the_right_litellm_prefix():
    cases = {
        "groq": ("openai/gpt-oss-120b", "groq/openai/gpt-oss-120b"),
        "google": ("gemini-3.7-flash", "gemini/gemini-3.7-flash"),
        "openrouter": ("openrouter/free", "openrouter/openrouter/free"),
        "cerebras": ("gpt-oss-120b", "cerebras/gpt-oss-120b"),
        "xai": ("grok-4.3", "xai/grok-4.3"),
    }

    for provider, (model, expected) in cases.items():
        settings = Settings(llm_provider=provider, llm_model=model)
        assert settings.model_identifier == expected


def test_custom_endpoint_needs_a_model_key_and_base_url():
    settings = Settings(
        llm_provider="custom",
        llm_model="local-model",
        api_key="secret",
        offline=False,
    )
    assert not settings.llm_available
    assert "base URL" in settings.configuration_issue

    configured = with_model_access(
        settings,
        provider="custom",
        model=" local-model ",
        api_key=" secret ",
        base_url=" http://localhost:11434/v1/ ",
    )
    assert configured.llm_available
    assert configured.model_identifier == "openai/local-model"
    assert configured.llm_base_url == "http://localhost:11434/v1"


def test_gui_settings_copy_does_not_mutate_environment_defaults():
    base = Settings(llm_provider="groq", llm_model="a", api_key=None)
    configured = with_model_access(
        base,
        provider="cerebras",
        model="gpt-oss-120b",
        api_key="temporary",
    )

    assert base.api_key is None
    assert configured.api_key == "temporary"
    assert configured.llm_provider == "cerebras"


def test_connection_error_never_echoes_the_api_key(monkeypatch):
    secret = "secret-that-must-not-leak"
    settings = Settings(
        llm_provider="groq",
        llm_model="openai/gpt-oss-120b",
        api_key=secret,
        offline=False,
    )
    engine = ReasoningEngine(settings)

    class BrokenModel:
        def call(self, messages):
            raise RuntimeError(f"Rejected credential {secret}")

    monkeypatch.setattr(engine, "_get_llm", lambda: BrokenModel())
    connected, message = engine.probe()

    assert not connected
    assert secret not in message
    assert "[redacted]" in message
