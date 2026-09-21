"""The reasoning layer that sits between the agents and the language model.

Two things matter here.

First, every agent in the pipeline is a real CrewAI agent with its own role,
goal and backstory, so the model reasons in character rather than as a generic
assistant.

Second, the pipeline must still work when there is no API key, when the network
is down, or when the model returns something unusable. Nothing in this module
raises on a failed call: it returns ``None`` and the calling agent falls back to
the deterministic heuristic it already had. The consequence is that InsightLab
degrades to a competent statistical tool instead of breaking.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable

from crewai import Agent, Crew, LLM, Process, Task

from .config import PROVIDERS, Settings, get_settings
from .chatgpt import ChatGPTClient
from .language import DEFAULT as DEFAULT_LANGUAGE
from .language import Language

logger = logging.getLogger(__name__)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

#: Prepended to every agent backstory so the whole system speaks one language.
HOUSE_STYLE = (
    "You are speaking to a person who understands the real-world context of "
    "their data but may never have studied statistics. Never assume the data "
    "is commercial: follow the detected domain, row meaning, measures and the "
    "user's stated goal. Use plain language and never use "
    "technical jargon without explaining it in the same sentence. Explain why "
    "something matters before asking them to decide anything, and say what the "
    "decision will change. Ask as few questions as possible. They should feel "
    "they are talking to an experienced analyst in the dataset's own domain, "
    "not filling in a form."
)


@dataclass
class AgentPersona:
    """The character an agent reasons in."""

    role: str
    goal: str
    backstory: str


class ReasoningEngine:
    """Builds CrewAI agents and runs single-task crews against them."""

    def __init__(
        self,
        settings: Settings | None = None,
        language: Language | None = None,
        response_cache: OrderedDict[str, str] | None = None,
        chatgpt_client: ChatGPTClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.language = language or DEFAULT_LANGUAGE
        self._routes = self.settings.candidate_settings()
        self._active_index = 0
        self._llms: dict[tuple[int, int], LLM] = {}
        self._agents: dict[tuple[int, str, int], Agent] = {}
        self._failed_routes: set[int] = set()
        self._response_cache = (
            response_cache if response_cache is not None else OrderedDict()
        )
        self.call_count = 0
        self.failure_count = 0
        # Successes since the most recent connection configuration. This is
        # separate from total usage so the always-visible status indicator does
        # not claim a newly edited key works because an older one once did.
        self.route_success_count = 0
        self.cache_hits = 0
        self.estimated_input_tokens = 0
        self.estimated_output_tokens = 0
        self.saved_tokens = 0
        self.chatgpt_client = chatgpt_client
        self.last_error: str | None = None
        self.failover_log: list[str] = []

    def reconfigure(
        self,
        settings: Settings,
        *,
        chatgpt_client: ChatGPTClient | None = None,
    ) -> bool:
        """Apply sidebar connection edits to this live run immediately.

        Agents keep a reference to this engine, so rebuilding the route table
        here updates pending decisions, chat and later stages without restarting
        the data workflow. The successful-response cache and usage totals remain
        intact; provider clients and failures belong to the old configuration
        and are discarded.
        """
        client_changed = chatgpt_client is not self.chatgpt_client
        if settings == self.settings and not client_changed:
            return False
        self.settings = settings
        self.chatgpt_client = chatgpt_client
        self._routes = settings.candidate_settings()
        self._active_index = 0
        self._llms.clear()
        self._agents.clear()
        self._failed_routes.clear()
        self.last_error = None
        self.failover_log.clear()
        self.route_success_count = 0
        return True

    def set_language(self, language: Language) -> None:
        """Switch the language every agent writes in.

        Cached agents are dropped: the language is part of the backstory, so a
        cached agent would keep writing in the previous one.
        """
        if language.code != self.language.code:
            self.language = language
            self._agents.clear()

    # -- availability ------------------------------------------------------

    @property
    def available(self) -> bool:
        return any(
            index not in self._failed_routes and self._route_available(route)
            for index, route in enumerate(self._routes)
        )

    @property
    def active_settings(self) -> Settings:
        if self._routes:
            return self._routes[min(self._active_index, len(self._routes) - 1)]
        return self.settings

    @property
    def configured_route_count(self) -> int:
        return len(self._routes)

    @property
    def active_route_number(self) -> int:
        return self._active_index + 1 if self._routes else 0

    @property
    def active_route_label(self) -> str:
        if not self._routes:
            return ""
        return self._route_label(min(self._active_index, len(self._routes) - 1))

    def _route_available(self, route: Settings) -> bool:
        if route.llm_provider == "chatgpt":
            return route.llm_available and bool(
                self.chatgpt_client and self.chatgpt_client.connected
            )
        return route.llm_available

    def status(self) -> str:
        active = self.active_settings
        suffix = f" · route {self._active_index + 1}/{len(self._routes)}" if len(self._routes) > 1 else ""
        return active.describe_llm() + suffix

    # -- construction ------------------------------------------------------

    def _get_llm(self, route_index: int | None = None,
                 max_output_tokens: int | None = None) -> LLM | None:
        index = self._active_index if route_index is None else route_index
        if not 0 <= index < len(self._routes):
            return None
        route = self._routes[index]
        if route.llm_provider == "chatgpt":
            return None  # Never route subscription auth through a billed API.
        if index in self._failed_routes or not self._route_available(route):
            return None
        output_limit = max(64, min(
            int(max_output_tokens or route.llm_max_output_tokens),
            route.llm_max_output_tokens,
        ))
        cache_key = (index, output_limit)
        if cache_key not in self._llms:
            try:
                self._llms[cache_key] = LLM(
                    model=route.model_identifier,
                    api_key=route.api_key,
                    base_url=route.llm_base_url,
                    temperature=route.llm_temperature,
                    timeout=route.llm_timeout,
                    max_tokens=output_limit,
                )
            except Exception as error:  # pragma: no cover - provider specific
                logger.warning("Could not build the language model: %s", error)
                self.last_error = self._redact_error(error, route)
                return None
        return self._llms[cache_key]

    def probe(self) -> tuple[bool, str]:
        """Make one minimal call to verify GUI-supplied model credentials.

        The response content is deliberately irrelevant: a request completing
        without an exception proves that the provider, model ID, endpoint and
        key agree. Reasoning models can spend a small completion allowance on
        hidden reasoning and legitimately return an empty visible message, so
        treating empty text as an authentication failure creates a false
        negative even though the provider accepted and billed the request.
        Credential text is stripped from errors defensively before anything is
        shown in the interface.
        """
        results = self.probe_all()
        working = [item for item in results if item["connected"]]
        if working:
            if len(results) == 1:
                return True, str(working[0]["message"])
            return True, f"{len(working)}/{len(results)} configured connection(s) work."
        if results:
            return False, results[-1]["message"]
        issue = self.settings.configuration_issue or "Model access is unavailable."
        return False, issue

    def probe_all(self) -> list[dict[str, Any]]:
        """Test every key and provider without stopping after the first success."""
        if self.settings.offline:
            return []
        ordered: list[dict[str, Any] | None] = []
        jobs: list[tuple[int, Callable[[], dict[str, Any]]]] = []

        def add_job(job: Callable[[], dict[str, Any]]) -> None:
            jobs.append((len(ordered), job))
            ordered.append(None)

        for route in self.settings.routes:
            label = route.label
            issue = self.settings._route_issue(route)
            if issue is not None:
                ordered.append(
                    {"label": label, "connected": False, "message": issue}
                )
                continue
            if route.provider == "chatgpt":
                def check_chatgpt(label=label):
                    if not self.chatgpt_client:
                        return {
                            "label": label,
                            "connected": False,
                            "message": "Sign in with ChatGPT first.",
                        }
                    try:
                        account = self.chatgpt_client.refresh_account()
                        connected = bool(account)
                        message = (
                            "Signed in. Model access will be checked on the first analysis request."
                            if connected else "Sign in with ChatGPT first."
                        )
                    except Exception as error:
                        connected, message = False, str(error)
                    return {
                        "label": label,
                        "connected": connected,
                        "message": message,
                    }

                add_job(check_chatgpt)
                continue
            prefix = PROVIDERS[route.provider].litellm_prefix
            for key_index, api_key in enumerate(route.api_keys, start=1):
                key_label = f"{label} · key {key_index}"

                def check_api(
                    route=route, prefix=prefix, api_key=api_key, key_label=key_label
                ):
                    try:
                        llm = LLM(
                            model=f"{prefix}/{route.model}", api_key=api_key,
                            base_url=route.base_url, temperature=0, max_tokens=16,
                            timeout=min(self.settings.llm_timeout, 20), stream=False,
                        )
                        self.call_count += 1
                        llm.call(messages=[{"role": "user", "content": "Reply OK."}])
                        connected, message = True, "Connection works."
                    except Exception as error:  # pragma: no cover - requires a live provider
                        self.failure_count += 1
                        connected = False
                        raw = str(error).replace(api_key, "[redacted]")
                        message = (
                            "Connection failed: "
                            f"{' '.join(raw.split())[:320] or type(error).__name__}"
                        )
                    return {
                        "label": key_label,
                        "connected": connected,
                        "message": message,
                    }

                add_job(check_api)

        if jobs:
            with ThreadPoolExecutor(max_workers=min(8, len(jobs))) as pool:
                futures = {pool.submit(job): index for index, job in jobs}
                for future in as_completed(futures):
                    ordered[futures[future]] = future.result()
        return [item for item in ordered if item is not None]

    def agent(self, key: str, persona: AgentPersona,
              max_output_tokens: int | None = None) -> Agent | None:
        """Return the CrewAI agent for ``key``, building it on first use."""
        output_limit = max(64, min(
            int(max_output_tokens or self.active_settings.llm_max_output_tokens),
            self.active_settings.llm_max_output_tokens,
        ))
        llm = self._get_llm(max_output_tokens=output_limit)
        if llm is None:
            return None
        cache_key = (self._active_index, key, output_limit)
        if cache_key not in self._agents:
            backstory = f"{persona.backstory}\n\n{HOUSE_STYLE}"
            if self.language.instruction:
                backstory += f"\n\n{self.language.instruction}"
            try:
                self._agents[cache_key] = Agent(
                    role=persona.role,
                    goal=persona.goal,
                    backstory=backstory,
                    llm=llm,
                    verbose=False,
                    allow_delegation=False,
                    max_iter=3,
                )
            except Exception as error:  # pragma: no cover - provider specific
                self.last_error = self._redact_error(error, self.active_settings)
                logger.warning("Could not build agent %s: %s", key, self.last_error)
                return None
        return self._agents[cache_key]

    # -- execution ---------------------------------------------------------

    def ask(
        self,
        key: str,
        persona: AgentPersona,
        instruction: str,
        expected_output: str,
        *,
        cache_namespace: str = "text",
        cache_validator: Callable[[str], bool] | None = None,
        max_output_tokens: int | None = None,
    ) -> str | None:
        """Run one task, reusing an identical successful answer in this session.

        The cache key covers the model, language, full persona, instruction and
        expected output. A hit therefore changes neither the request nor its
        answer; it only avoids paying twice when a UI rerun or repeated question
        asks for the exact same work.
        """
        if not self.available:
            return None
        output_limit = max(64, min(
            int(max_output_tokens or 600), self.settings.llm_max_output_tokens
        ))
        cache_key = self._cache_key(
            key,
            persona,
            instruction,
            expected_output,
            cache_namespace,
            output_limit,
        )
        prompt_tokens = self._estimate_prompt_tokens(
            persona, instruction, expected_output
        )
        if cache_key in self._response_cache:
            self.cache_hits += 1
            self._response_cache.move_to_end(cache_key)
            cached = self._response_cache[cache_key]
            self.saved_tokens += prompt_tokens + self._estimate_tokens(cached)
            return cached

        text: str | None = None
        for index in range(self._active_index, len(self._routes)):
            route = self._routes[index]
            if index in self._failed_routes or not self._route_available(route):
                continue
            self._active_index = index
            self.last_error = None
            self.estimated_input_tokens += prompt_tokens
            if route.llm_provider == "chatgpt":
                text = self._run_chatgpt(
                    persona, instruction, expected_output, output_limit
                )
            else:
                agent = self.agent(key, persona, output_limit)
                if agent is not None:
                    text = self._run_task(agent, key, instruction, expected_output)
            if text:
                break
            self._disable_current_route()
        if text:
            self.route_success_count += 1
            self.estimated_output_tokens += self._estimate_tokens(text)
        if text and (cache_validator is None or cache_validator(text)):
            self._response_cache[cache_key] = text
            self._response_cache.move_to_end(cache_key)
            while len(self._response_cache) > 128:
                self._response_cache.popitem(last=False)
        return text

    def _run_chatgpt(self, persona: AgentPersona, instruction: str,
                     expected_output: str, max_output_tokens: int) -> str | None:
        prompt = (
            f"Role: {persona.role}\nGoal: {persona.goal}\n{persona.backstory}\n\n"
            f"{HOUSE_STYLE}\n{self.language.instruction or ''}\n\n"
            f"Task:\n{instruction}\n\nExpected output:\n{expected_output}\n"
            f"Keep the response within approximately {max_output_tokens} tokens."
        )
        self.call_count += 1
        self.last_error = None
        try:
            return self.chatgpt_client.complete(
                prompt, model=self.active_settings.llm_model,
                timeout=self.active_settings.llm_timeout,
            )
        except Exception as error:
            self.failure_count += 1
            self.last_error = str(error)
            logger.warning("ChatGPT analysis failed: %s", self.last_error)
            return None

    def _run_task(
        self,
        agent: Agent,
        key: str,
        instruction: str,
        expected_output: str,
    ) -> str | None:
        """Execute one uncached model task."""
        task = Task(
            description=instruction,
            expected_output=expected_output,
            agent=agent,
        )
        try:
            self.call_count += 1
            crew = Crew(
                agents=[agent],
                tasks=[task],
                process=Process.sequential,
                verbose=False,
            )
            result = crew.kickoff()
        except Exception as error:
            self.failure_count += 1
            self.last_error = self._redact_error(error, self.active_settings)
            logger.warning("Reasoning call for %s failed: %s", key, self.last_error)
            return None

        text = getattr(result, "raw", None) or str(result)
        return text.strip() or None

    def _cache_key(
        self,
        key: str,
        persona: AgentPersona,
        instruction: str,
        expected_output: str,
        cache_namespace: str,
        max_output_tokens: int,
    ) -> str:
        """Hash response-relevant inputs; credentials do not change the answer."""
        route_scope = []
        for route in self._routes:
            route_scope.extend((route.model_identifier, route.llm_base_url or ""))
        parts = (
            self.chatgpt_client.cache_scope if self.chatgpt_client else "",
            *route_scope,
            str(self.settings.llm_temperature),
            self.language.code,
            cache_namespace,
            str(max_output_tokens),
            key,
            persona.role,
            persona.goal,
            persona.backstory,
            instruction,
            expected_output,
        )
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Conservative local estimate that works for English and Arabic."""
        if not text:
            return 0
        return max(1, math.ceil(len(text.encode("utf-8")) / 4))

    def _estimate_prompt_tokens(self, persona: AgentPersona, instruction: str,
                                expected_output: str) -> int:
        return self._estimate_tokens(
            f"{persona.role}\n{persona.goal}\n{persona.backstory}\n{HOUSE_STYLE}\n"
            f"{self.language.instruction}\n{instruction}\n{expected_output}"
        )

    def _route_label(self, index: int) -> str:
        route = self._routes[index]
        same_provider_before = sum(
            earlier.llm_provider == route.llm_provider
            for earlier in self._routes[:index]
        )
        key_label = f" · key {same_provider_before + 1}" if route.llm_provider != "chatgpt" else ""
        return f"{route.provider.label}{key_label}"

    @staticmethod
    def _redact_error(error: Exception, route: Settings) -> str:
        message = str(error)
        if route.api_key:
            message = message.replace(route.api_key, "[redacted]")
        return " ".join(message.split())[:320] or type(error).__name__

    def _disable_current_route(self) -> None:
        index = self._active_index
        if index in self._failed_routes:
            return
        self._failed_routes.add(index)
        label = self._route_label(index)
        reason = self.last_error or "empty or unavailable response"
        next_index = next(
            (candidate for candidate in range(index + 1, len(self._routes))
             if candidate not in self._failed_routes and self._route_available(self._routes[candidate])),
            None,
        )
        if next_index is None:
            event = f"{label} failed ({reason}); no model route remains."
        else:
            event = f"{label} failed ({reason}); switched to {self._route_label(next_index)}."
            self._active_index = next_index
        self.failover_log.append(event)
        logger.warning(event)

    def ask_json(
        self,
        key: str,
        persona: AgentPersona,
        instruction: str,
        shape: str,
        *,
        max_output_tokens: int | None = None,
    ) -> Any | None:
        """Run one task expecting JSON back, and parse it defensively.

        ``shape`` is a description of the JSON structure that gets appended to
        the instruction. Returns the parsed object, or ``None`` when the call
        failed or the reply could not be parsed.
        """
        full_instruction = (
            f"{instruction}\n\n"
            f"Reply with JSON only, matching this shape:\n{shape}\n"
            "Do not wrap the JSON in code fences and do not add commentary."
        )
        text = self.ask(
            key,
            persona,
            full_instruction,
            expected_output="A single valid JSON value and nothing else.",
            cache_namespace="json",
            cache_validator=lambda value: parse_json(value) is not None,
            max_output_tokens=max_output_tokens or self.settings.llm_max_output_tokens,
        )
        if text is None:
            return None
        return parse_json(text)


def parse_json(text: str) -> Any | None:
    """Pull a JSON value out of a model reply.

    Models wrap JSON in code fences, prefix it with a sentence, or append a
    sign-off. All three are handled here rather than in each caller.
    """
    candidate = _FENCE.sub("", text).strip()
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        pass

    for opening, closing in (("{", "}"), ("[", "]")):
        start = candidate.find(opening)
        end = candidate.rfind(closing)
        if start != -1 and end > start:
            try:
                return json.loads(candidate[start : end + 1])
            except json.JSONDecodeError:
                continue
    logger.debug("Could not parse a JSON value from the reply: %s", text[:200])
    return None
