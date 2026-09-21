"""No live login, tokens or model usage: exercise the real stdio client with a fake server."""

from __future__ import annotations

import json
import queue
import threading
from collections import OrderedDict
from pathlib import Path

import pytest

from insightlab.core import chatgpt
from insightlab.core.chatgpt import ChatGPTClient, ChatGPTError
from insightlab.core.config import Settings, with_model_access
from insightlab.core.reasoning import AgentPersona, ReasoningEngine


class Lines:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        while (line := self.queue.get()) is not None:
            yield line

    def emit(self, message):
        self.queue.put(json.dumps(message) + "\n")

    def close(self):
        pass


class FakeProcess:
    def __init__(self, command, **kwargs):
        self.command, self.kwargs = command, kwargs
        self.stdout = Lines()
        self.stdin = self
        self.returncode = None
        self.calls = []
        self.account = None
        self.hang = None
        self.fail_turn = False
        self.auth_url = "https://auth.openai.com/oauth/authorize?state=example"
        self.only_turn_items = False
        self.drop_turn_events = False
        self.reject_turn = False

    def write(self, line):
        message = json.loads(line)
        self.calls.append(message)
        method = message.get("method")
        if method is None or "id" not in message or method == self.hang:
            return
        result = {}
        if method == "account/read":
            result = {"account": self.account}
        elif method == "account/login/start":
            result = {"type": "chatgpt", "loginId": "login-1", "authUrl": self.auth_url}
        elif method == "account/logout":
            self.account = None
        elif method == "thread/start":
            result = {"thread": {"id": "thread-1"}}
        elif method == "turn/start":
            if self.reject_turn:
                self.stdout.emit({"id": message["id"], "error": {"message": "usage limit reached"}})
                return
            result = {"turn": {"id": "turn-1"}}
            if self.drop_turn_events:
                self.stdout.emit({"id": message["id"], "result": result})
                return
            # Events deliberately arrive before the RPC response.
            item = {"id": "answer-1", "type": "agentMessage", "phase": "final_answer", "text": '{"answer": "grounded"}'}
            if not self.only_turn_items:
                self.stdout.emit({"method": "item/completed", "params": {
                    "threadId": "thread-1", "turnId": "turn-1",
                    "item": {**item, "phase": "commentary", "text": "Thinking…"},
                }})
                self.stdout.emit({"method": "item/completed", "params": {"threadId": "thread-1", "turnId": "turn-1", "item": item}})
            self.stdout.emit({"method": "turn/completed", "params": {
                "threadId": "thread-1", "turn": {
                    "id": "turn-1", "status": "failed" if self.fail_turn else "completed",
                    "error": {"message": "usage limit reached: secret-token"} if self.fail_turn else None,
                    "items": [item],
                },
            }})
        self.stdout.emit({"id": message["id"], "result": result})

    def flush(self):
        pass

    def close(self):
        pass

    def poll(self):
        return self.returncode

    def terminate(self):
        self.returncode = 0
        self.stdout.queue.put(None)

    kill = terminate

    def wait(self, timeout=None):
        return self.returncode


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(chatgpt, "find_codex", lambda: "/fake/codex")
    monkeypatch.setattr(chatgpt.subprocess, "Popen", FakeProcess)
    instance = ChatGPTClient()
    yield instance
    instance.close()


def sign_in(client):
    client._process.account = {"type": "chatgpt", "email": "owner@example.com", "planType": "plus"}
    client.refresh_account()


def test_isolated_ephemeral_auth_does_not_inherit_api_keys(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "do-not-use")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://do-not-use.example")
    other = ChatGPTClient()
    try:
        env = other._process.kwargs["env"]
        assert "OPENAI_API_KEY" not in env
        assert "OPENAI_BASE_URL" not in env
        assert env["CODEX_HOME"] != client._process.kwargs["env"]["CODEX_HOME"]
        command = other._process.command
        for flag in ('cli_auth_credentials_store="ephemeral"', 'forced_login_method="chatgpt"',
                     'features.shell_tool=false', 'features.apps=false', 'web_search="disabled"'):
            assert flag in command
        directory = Path(env["CODEX_HOME"]).parent
    finally:
        other.close()
    assert not directory.exists()


def test_login_refresh_and_logout_only_affect_this_session(client):
    assert not client.connected
    login = client.start_login()
    assert client.start_login() == login
    assert len([call for call in client._process.calls if call.get("method") == "account/login/start"]) == 1
    scope = client.cache_scope
    sign_in(client)
    assert client.connected
    assert client.cache_scope != scope
    assert client.login is None
    client.logout()
    assert not client.connected
    assert client.account is None
    assert client._process.poll() is not None


def test_login_cancellation(client):
    client.start_login()
    client.cancel_login()
    assert client.login is None
    assert client._process.calls[-1]["method"] == "account/login/cancel"


def test_rejects_non_official_login_link(client):
    client._process.auth_url = "https://example.com/oauth"
    with pytest.raises(ChatGPTError, match="unexpected"):
        client.start_login()
    assert not client.connected
    assert client._process.poll() is not None


def test_rejects_api_key_account(client):
    client._process.account = {"type": "apiKey"}
    with pytest.raises(ChatGPTError, match="not API-key"):
        client.refresh_account()
    assert not client.connected


@pytest.mark.parametrize("only_turn_items", [False, True])
def test_final_answer_survives_interleaved_events_and_ignores_commentary(client, only_turn_items):
    sign_in(client)
    client._process.only_turn_items = only_turn_items
    assert client.complete("Evidence only") == '{"answer": "grounded"}'
    start = next(call["params"] for call in client._process.calls if call.get("method") == "thread/start")
    assert start["sandbox"] == "read-only"
    assert start["ephemeral"] is True
    assert "model" not in start
    assert client._process.calls[-1]["method"] == "thread/unsubscribe"


def test_usage_error_is_safe_and_does_not_fall_back_to_api(client):
    sign_in(client)
    client._process.fail_turn = True
    with pytest.raises(ChatGPTError, match="usage limit") as error:
        client.complete("Evidence")
    assert "secret-token" not in str(error.value)
    assert not any(call.get("params", {}).get("type") == "apiKey" for call in client._process.calls)


def test_timeout_closes_process_and_drops_auth(client):
    sign_in(client)
    client._process.hang = "account/read"
    with pytest.raises(ChatGPTError, match="too long"):
        client._rpc("account/read", {}, timeout=0.01)
    assert not client.connected
    assert client._process.poll() is not None


def test_timeout_during_generation_stops_the_process(client):
    sign_in(client)
    client._process.drop_turn_events = True
    with pytest.raises(ChatGPTError, match="too long"):
        client.complete("Evidence", timeout=0.02)
    assert client._process.poll() is not None
    assert not client.connected


def test_expired_deadline_never_sends_a_request(client):
    before = len(client._process.calls)
    with pytest.raises(ChatGPTError, match="too long"):
        client._rpc("turn/start", {}, timeout=-1)
    assert len(client._process.calls) == before


def test_rejected_turn_does_not_leave_an_active_thread(client):
    sign_in(client)
    client._process.reject_turn = True
    with pytest.raises(ChatGPTError, match="usage limit"):
        client.complete("Evidence")
    assert client._process.calls[-1]["method"] == "thread/unsubscribe"


def test_no_execution_without_login(client):
    with pytest.raises(ChatGPTError, match="Sign in"):
        client.complete("Evidence")
    assert not any(call.get("method") == "turn/start" for call in client._process.calls)


def test_reader_denies_server_requests():
    process = FakeProcess([])
    process.stdout.emit({"id": 200, "method": "item/commandExecution/requestApproval", "params": {}})
    process.stdout.queue.put(None)
    chatgpt._reader(process, {}, threading.Lock(), threading.Lock(), queue.Queue())
    assert process.calls == [{"id": 200, "error": {"code": -32601, "message": "InsightLab does not allow tool requests."}}]


def test_chatgpt_settings_need_login_not_key():
    base = Settings(offline=False)
    settings = with_model_access(base, provider="chatgpt", model="auto", api_key="ignored", base_url="https://ignored", chatgpt_connected=True)
    assert settings.llm_available
    assert settings.api_key is None and settings.llm_base_url is None
    assert not Settings(llm_provider="chatgpt", api_key="not-enough").llm_available
    assert not Settings(llm_provider="chatgpt", chatgpt_connected=True, offline=True).llm_available


def test_engine_routes_json_through_codex_with_scoped_cache(client, monkeypatch):
    sign_in(client)
    monkeypatch.setattr("insightlab.core.reasoning.LLM", lambda **kwargs: pytest.fail("Must not build a paid API client"))
    engine = ReasoningEngine(Settings(llm_provider="chatgpt", llm_model="auto", chatgpt_connected=True), chatgpt_client=client, response_cache=OrderedDict())
    persona = AgentPersona("Analyst", "Explain", "Use evidence")
    assert engine.probe()[0]
    assert engine.call_count == 0  # Account checks do not use model quota.
    assert engine.ask_json("a", persona, "Evidence", "{}") == {"answer": "grounded"}
    assert engine.ask_json("a", persona, "Evidence", "{}") == {"answer": "grounded"}
    assert engine.call_count == 1 and engine.cache_hits == 1
    client.cache_scope = "new-login"
    engine.ask_json("a", persona, "Evidence", "{}")
    assert engine.call_count == 2
    client.close()
    assert engine.ask_json("a", persona, "Evidence", "{}") is None


def test_engine_exposes_safe_failure_and_keeps_statistical_fallback(client):
    sign_in(client)
    client._process.fail_turn = True
    engine = ReasoningEngine(Settings(llm_provider="chatgpt", chatgpt_connected=True), chatgpt_client=client)
    assert engine.ask("a", AgentPersona("r", "g", "b"), "task", "answer") is None
    assert "usage limit" in engine.last_error
    assert engine.failure_count == 1
    assert not engine._response_cache
    assert engine.ask("b", AgentPersona("r", "g", "b"), "next stage", "answer") is None
    assert engine.failure_count == 1  # Do not hammer the service at every stage.
    assert engine.call_count == 1
    assert not engine.available


def test_missing_cli_has_actionable_error(monkeypatch):
    monkeypatch.setattr(chatgpt, "find_codex", lambda: None)
    with pytest.raises(ChatGPTError, match="Install the official Codex CLI"):
        ChatGPTClient()


@pytest.fixture
def gui(monkeypatch, tmp_path):
    from streamlit.testing.v1 import AppTest
    from insightlab.app import chatgpt_panel
    import streamlit as st

    monkeypatch.setenv("INSIGHTLAB_OFFLINE", "1")
    monkeypatch.setenv("INSIGHTLAB_WORKSPACE", str(tmp_path / "runs"))
    monkeypatch.setattr(chatgpt_panel, "find_codex", lambda: "/fake/codex")
    original = st.get_option
    monkeypatch.setattr(st, "get_option", lambda key: "127.0.0.1" if key == "server.address" else original(key))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "insightlab/app/main.py"))
    app.run()
    app.selectbox(key="llm_provider").set_value("chatgpt").run()
    return app


def test_gui_chatgpt_has_login_instead_of_api_key(gui):
    assert not gui.exception
    assert gui.button(key="chatgpt_sign_in")
    assert not [field for field in gui.text_input if field.label == "API key"]
    assert not [button for button in gui.button if button.key == "test_llm_connection"]


def test_gui_login_and_logout_use_the_same_session_client(gui, client, monkeypatch):
    from insightlab.app import chatgpt_panel
    monkeypatch.setattr(chatgpt_panel, "ChatGPTClient", lambda: client)
    gui.button(key="chatgpt_sign_in").click().run()
    assert not gui.exception
    assert gui.session_state["chatgpt_client"] is client
    assert client.login
    sign_in(client)
    gui.button(key="chatgpt_refresh").click().run()
    assert not gui.exception
    assert any(message.value == "Signed in with ChatGPT" for message in gui.success)
    gui.button(key="chatgpt_sign_out").click().run()
    assert not gui.exception
    assert not client.connected
    assert gui.button(key="chatgpt_sign_in")


def test_gui_does_not_expose_login_when_bound_publicly(gui, monkeypatch):
    import streamlit as st
    original = st.get_option
    monkeypatch.setattr(st, "get_option", lambda key: "0.0.0.0" if key == "server.address" else original(key))
    gui.run()
    assert not gui.exception
    assert any("non-local server" in warning.value for warning in gui.warning)
    assert not [button for button in gui.button if button.key == "chatgpt_sign_in"]


def test_gui_passes_chatgpt_client_to_pipeline_and_can_disconnect_during_run(gui, client):
    sign_in(client)
    gui.session_state["chatgpt_client"] = client
    gui.run()
    [button for button in gui.button if button.label == "Try it with sample data"][0].click().run()
    assert not gui.exception
    engine = gui.session_state["supervisor"].reasoning
    assert engine.chatgpt_client is client
    assert engine.settings.llm_provider == "chatgpt"
    assert engine.settings.api_key is None
    gui.button(key="chatgpt_run_sign_out").click().run()
    assert not gui.exception
    assert not engine.available


def test_arabic_panel_has_translated_controls_and_errors(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from insightlab.app import chatgpt_panel
    from insightlab.core.language import ARABIC
    import streamlit as st

    original = st.get_option
    monkeypatch.setattr(st, "get_option", lambda key: "127.0.0.1" if key == "server.address" else original(key))
    monkeypatch.setattr(chatgpt_panel, "find_codex", lambda: "/fake/codex")
    app = AppTest.from_string(
        "from insightlab.app.chatgpt_panel import render\n"
        "from insightlab.core.language import ARABIC\n"
        "render(ARABIC)\n"
    ).run()
    assert not app.exception
    assert app.button(key="chatgpt_sign_in").label == "سجّل الدخول بحساب ChatGPT"
    assert "حد استخدام" in chatgpt_panel.error_message(ARABIC, "ChatGPT usage limit reached.")
