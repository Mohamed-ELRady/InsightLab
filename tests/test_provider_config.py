"""Provider routing and the GUI-to-run settings boundary."""

from __future__ import annotations

from collections import OrderedDict

from insightlab.core.config import (
    ProviderRoute, Settings, get_settings, reset_settings_cache, with_model_access,
)
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

    captured = {}

    class BrokenModel:
        def __init__(self, **kwargs):
            captured.update(kwargs)

        def call(self, messages):
            raise RuntimeError(f"Rejected credential {secret}")

    monkeypatch.setattr("insightlab.core.reasoning.LLM", BrokenModel)
    connected, message = engine.probe()

    assert not connected
    assert secret not in message
    assert "[redacted]" in message
    assert captured["max_tokens"] == 16
    assert captured["stream"] is False


def test_an_empty_model_reply_still_proves_the_connection(monkeypatch):
    """Reasoning tokens may consume the probe allowance before visible text."""
    settings = Settings(
        llm_provider="groq",
        llm_model="openai/gpt-oss-120b",
        api_key="working-key",
        offline=False,
    )
    engine = ReasoningEngine(settings)

    class EmptyButSuccessfulModel:
        def __init__(self, **kwargs):
            pass

        def call(self, messages):
            return ""

    monkeypatch.setattr("insightlab.core.reasoning.LLM", EmptyButSuccessfulModel)

    connected, message = engine.probe()

    assert connected
    assert message == "Connection works."


def test_identical_model_work_is_served_from_the_session_cache(monkeypatch):
    from insightlab.core.reasoning import AgentPersona

    settings = Settings(
        llm_provider="groq",
        llm_model="openai/gpt-oss-120b",
        api_key="temporary",
        offline=False,
    )
    engine = ReasoningEngine(settings)
    persona = AgentPersona(role="Analyst", goal="Explain", backstory="Careful")
    executions = []

    monkeypatch.setattr(engine, "agent", lambda *args: object())

    def run_once(agent, key, instruction, expected_output):
        executions.append((key, instruction, expected_output))
        return "A grounded answer."

    monkeypatch.setattr(engine, "_run_task", run_once)

    first = engine.ask("analyst", persona, "Same evidence", "One sentence")
    second = engine.ask("analyst", persona, "Same evidence", "One sentence")

    assert first == second == "A grounded answer."
    assert len(executions) == 1
    assert engine.cache_hits == 1
    assert engine.saved_tokens > 0
    assert engine.estimated_input_tokens > 0
    assert engine.estimated_output_tokens > 0


def test_runtime_tasks_apply_a_bounded_output_budget(monkeypatch):
    settings = Settings(api_key="temporary", offline=False, llm_max_output_tokens=900)
    engine = ReasoningEngine(settings)
    captured = []

    class BudgetedModel:
        def __init__(self, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr("insightlab.core.reasoning.LLM", BudgetedModel)
    assert engine._get_llm(max_output_tokens=350) is not None
    assert captured[0]["max_tokens"] == 350
    assert captured[0]["temperature"] == settings.llm_temperature


def test_cache_never_reuses_an_answer_for_different_evidence(monkeypatch):
    from insightlab.core.reasoning import AgentPersona

    settings = Settings(api_key="temporary", offline=False)
    engine = ReasoningEngine(settings)
    persona = AgentPersona(role="Analyst", goal="Explain", backstory="Careful")
    executions = []

    monkeypatch.setattr(engine, "agent", lambda *args: object())

    def echo(agent, key, instruction, expected_output):
        executions.append(instruction)
        return instruction

    monkeypatch.setattr(engine, "_run_task", echo)

    assert engine.ask("analyst", persona, "Evidence A", "One sentence") == "Evidence A"
    assert engine.ask("analyst", persona, "Evidence B", "One sentence") == "Evidence B"
    assert len(executions) == 2
    assert engine.cache_hits == 0


def test_session_cache_can_be_reused_by_a_new_analysis(monkeypatch):
    from insightlab.core.reasoning import AgentPersona

    settings = Settings(api_key="temporary", offline=False)
    persona = AgentPersona(role="Analyst", goal="Explain", backstory="Careful")
    shared_cache = OrderedDict()
    executions = []

    first_engine = ReasoningEngine(settings, response_cache=shared_cache)
    monkeypatch.setattr(first_engine, "agent", lambda *args: object())

    def run_once(agent, key, instruction, expected_output):
        executions.append(instruction)
        return "Stable answer"

    monkeypatch.setattr(first_engine, "_run_task", run_once)
    assert first_engine.ask("analyst", persona, "Same file", "Short") == "Stable answer"

    second_engine = ReasoningEngine(settings, response_cache=shared_cache)
    monkeypatch.setattr(second_engine, "agent", lambda *args: object())
    monkeypatch.setattr(second_engine, "_run_task", run_once)
    assert second_engine.ask("analyst", persona, "Same file", "Short") == "Stable answer"

    assert len(executions) == 1
    assert second_engine.cache_hits == 1


def test_session_cache_reuses_identical_work_across_rotating_credentials(monkeypatch):
    from insightlab.core.reasoning import AgentPersona

    shared_cache = OrderedDict()
    persona = AgentPersona(role="Analyst", goal="Explain", backstory="Careful")
    executions = []

    def execute(agent, key, instruction, expected_output):
        executions.append(instruction)
        return "Answer"

    for api_key in ("first-key", "second-key"):
        settings = Settings(api_key=api_key, offline=False)
        engine = ReasoningEngine(settings, response_cache=shared_cache)
        monkeypatch.setattr(engine, "agent", lambda *args: object())
        monkeypatch.setattr(engine, "_run_task", execute)
        assert engine.ask("analyst", persona, "Same file", "Short") == "Answer"

    assert len(executions) == 1


def test_invalid_json_is_not_cached_as_a_good_model_response(monkeypatch):
    from insightlab.core.reasoning import AgentPersona

    settings = Settings(api_key="temporary", offline=False)
    engine = ReasoningEngine(settings)
    persona = AgentPersona(role="Analyst", goal="Plan", backstory="Careful")
    replies = iter(("not json", '{"answerable": true}'))
    executions = []

    monkeypatch.setattr(engine, "agent", lambda *args: object())

    def respond(agent, key, instruction, expected_output):
        executions.append(instruction)
        return next(replies)

    monkeypatch.setattr(engine, "_run_task", respond)

    assert engine.ask_json("analyst", persona, "Plan this", "{}") is None
    assert engine.ask_json("analyst", persona, "Plan this", "{}") == {
        "answerable": True
    }
    assert engine.ask_json("analyst", persona, "Plan this", "{}") == {
        "answerable": True
    }
    assert len(executions) == 2
    assert engine.cache_hits == 1


def test_multiple_keys_are_expanded_before_the_next_provider():
    settings = with_model_access(
        Settings(offline=False), provider="groq", model="model-a", api_key="key-1",
        api_keys=("key-1", "key-2"),
        fallback_routes=(ProviderRoute("google", "model-b", ("key-3",)),),
    )

    candidates = settings.candidate_settings()
    assert [(item.llm_provider, item.api_key) for item in candidates] == [
        ("groq", "key-1"), ("groq", "key-2"), ("google", "key-3")
    ]


def test_environment_can_supply_additional_keys(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "primary")
    monkeypatch.setenv("LLM_API_KEYS", "secondary, third\nsecondary")
    monkeypatch.delenv("INSIGHTLAB_OFFLINE", raising=False)
    reset_settings_cache()
    try:
        settings = get_settings()
        assert settings.api_keys == ("primary", "secondary", "third")
        assert [item.api_key for item in settings.candidate_settings()] == [
            "primary", "secondary", "third"
        ]
    finally:
        reset_settings_cache()


def test_reasoning_fails_over_across_keys_then_providers(monkeypatch):
    from insightlab.core.reasoning import AgentPersona

    settings = with_model_access(
        Settings(offline=False), provider="groq", model="model-a", api_key="key-1",
        api_keys=("key-1", "key-2"),
        fallback_routes=(ProviderRoute("google", "model-b", ("key-3",)),),
    )
    engine = ReasoningEngine(settings)
    persona = AgentPersona(role="Analyst", goal="Explain", backstory="Careful")
    attempts = []
    monkeypatch.setattr(engine, "agent", lambda *args: object())

    def execute(agent, key, instruction, expected_output):
        attempts.append((engine.active_settings.llm_provider, engine.active_settings.api_key))
        if engine.active_settings.api_key in {"key-1", "key-2"}:
            engine.last_error = "quota exhausted"
            return None
        return "Recovered answer"

    monkeypatch.setattr(engine, "_run_task", execute)
    answer = engine.ask("analyst", persona, "Explain", "Short")

    assert answer == "Recovered answer"
    assert attempts == [("groq", "key-1"), ("groq", "key-2"), ("google", "key-3")]
    assert engine.active_settings.llm_provider == "google"
    assert len(engine.failover_log) == 2
    assert "key-1" not in " ".join(engine.failover_log)


def test_successful_secondary_key_stays_active_for_later_requests(monkeypatch):
    from insightlab.core.reasoning import AgentPersona

    settings = with_model_access(
        Settings(offline=False), provider="groq", model="model", api_key="bad",
        api_keys=("bad", "working"),
    )
    engine = ReasoningEngine(settings)
    persona = AgentPersona(role="Analyst", goal="Explain", backstory="Careful")
    attempts = []
    monkeypatch.setattr(engine, "agent", lambda *args: object())

    def execute(agent, key, instruction, expected_output):
        attempts.append(engine.active_settings.api_key)
        if engine.active_settings.api_key == "bad":
            return None
        return instruction

    monkeypatch.setattr(engine, "_run_task", execute)
    assert engine.ask("analyst", persona, "First", "Short") == "First"
    assert engine.ask("analyst", persona, "Second", "Short") == "Second"
    assert attempts == ["bad", "working", "working"]


def test_chatgpt_can_fail_over_to_an_explicit_api_provider(monkeypatch):
    from insightlab.core.chatgpt import ChatGPTError
    from insightlab.core.reasoning import AgentPersona

    settings = with_model_access(
        Settings(offline=False), provider="chatgpt", model="auto", api_key=None,
        chatgpt_connected=True,
        fallback_routes=(ProviderRoute("groq", "model", ("api-key",)),),
    )

    class ExhaustedChatGPT:
        connected = True
        cache_scope = "account"

        def complete(self, *args, **kwargs):
            raise ChatGPTError("usage limit reached")

    engine = ReasoningEngine(settings, chatgpt_client=ExhaustedChatGPT())
    persona = AgentPersona(role="Analyst", goal="Explain", backstory="Careful")
    monkeypatch.setattr(engine, "agent", lambda *args: object())
    monkeypatch.setattr(engine, "_run_task", lambda *args: "API fallback answer")

    assert engine.ask("analyst", persona, "Explain", "Short") == "API fallback answer"
    assert engine.active_settings.llm_provider == "groq"
    assert "ChatGPT" in engine.failover_log[0]


def test_probe_all_reports_each_key_and_redacts_failed_secret(monkeypatch):
    settings = with_model_access(
        Settings(offline=False), provider="groq", model="model", api_key="broken-secret",
        api_keys=("broken-secret", "working-secret"),
    )

    class ProbeModel:
        def __init__(self, **kwargs):
            self.api_key = kwargs["api_key"]

        def call(self, messages):
            if self.api_key == "broken-secret":
                raise RuntimeError("Rejected broken-secret")
            return "connected"

    monkeypatch.setattr("insightlab.core.reasoning.LLM", ProbeModel)
    checks = ReasoningEngine(settings).probe_all()

    assert [item["connected"] for item in checks] == [False, True]
    assert "broken-secret" not in checks[0]["message"]
    assert "[redacted]" in checks[0]["message"]


def test_live_reconfiguration_keeps_cache_and_updates_every_agent_route():
    shared_cache = OrderedDict()
    original = Settings(
        llm_provider="groq", llm_model="first-model", api_key="first-key",
        offline=False,
    )
    engine = ReasoningEngine(original, response_cache=shared_cache)
    engine.call_count = 4
    engine.saved_tokens = 120
    engine.failover_log.append("old route failed")
    engine._failed_routes.add(0)
    shared_cache["existing"] = "cached answer"

    updated = with_model_access(
        original,
        provider="google",
        model="gemini-3.7-flash",
        api_key="replacement-key",
    )

    assert engine.reconfigure(updated)
    assert engine.settings == updated
    assert engine.active_settings.llm_provider == "google"
    assert engine.active_route_label == "Google Gemini · key 1"
    assert engine.call_count == 4
    assert engine.saved_tokens == 120
    assert engine._response_cache["existing"] == "cached answer"
    assert not engine._failed_routes
    assert not engine.failover_log
    assert not engine.reconfigure(updated)
