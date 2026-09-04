"""Provider routing and the GUI-to-run settings boundary."""

from __future__ import annotations

from collections import OrderedDict

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
    assert captured["max_tokens"] == 32


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


def test_session_cache_is_scoped_to_the_api_credential(monkeypatch):
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

    assert len(executions) == 2


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
