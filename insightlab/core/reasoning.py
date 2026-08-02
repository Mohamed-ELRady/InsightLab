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

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

from crewai import Agent, Crew, LLM, Process, Task

from .config import Settings, get_settings
from .language import DEFAULT as DEFAULT_LANGUAGE
from .language import Language

logger = logging.getLogger(__name__)

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.MULTILINE)

#: Prepended to every agent backstory so the whole system speaks one language.
HOUSE_STYLE = (
    "You are speaking to a business owner who is an expert in their own trade "
    "but has never studied statistics. Use plain language and never use "
    "technical jargon without explaining it in the same sentence. Explain why "
    "something matters before asking them to decide anything, and say what the "
    "decision will change. Ask as few questions as possible. They should feel "
    "they are talking to an experienced business consultant, not filling in a "
    "form."
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
        self, settings: Settings | None = None, language: Language | None = None
    ) -> None:
        self.settings = settings or get_settings()
        self.language = language or DEFAULT_LANGUAGE
        self._llm: LLM | None = None
        self._agents: dict[str, Agent] = {}
        self.call_count = 0
        self.failure_count = 0

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
        return self.settings.llm_available

    def status(self) -> str:
        return self.settings.describe_llm()

    # -- construction ------------------------------------------------------

    def _get_llm(self) -> LLM | None:
        if not self.available:
            return None
        if self._llm is None:
            try:
                self._llm = LLM(
                    model=self.settings.model_identifier,
                    api_key=self.settings.api_key,
                    temperature=self.settings.llm_temperature,
                    timeout=self.settings.llm_timeout,
                )
            except Exception as error:  # pragma: no cover - provider specific
                logger.warning("Could not build the language model: %s", error)
                return None
        return self._llm

    def agent(self, key: str, persona: AgentPersona) -> Agent | None:
        """Return the CrewAI agent for ``key``, building it on first use."""
        llm = self._get_llm()
        if llm is None:
            return None
        if key not in self._agents:
            backstory = f"{persona.backstory}\n\n{HOUSE_STYLE}"
            if self.language.instruction:
                backstory += f"\n\n{self.language.instruction}"
            self._agents[key] = Agent(
                role=persona.role,
                goal=persona.goal,
                backstory=backstory,
                llm=llm,
                verbose=False,
                allow_delegation=False,
                max_iter=3,
            )
        return self._agents[key]

    # -- execution ---------------------------------------------------------

    def ask(
        self,
        key: str,
        persona: AgentPersona,
        instruction: str,
        expected_output: str,
    ) -> str | None:
        """Run one task and return the raw text, or ``None`` on any failure."""
        agent = self.agent(key, persona)
        if agent is None:
            return None

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
            logger.warning("Reasoning call for %s failed: %s", key, error)
            return None

        text = getattr(result, "raw", None) or str(result)
        return text.strip() or None

    def ask_json(
        self,
        key: str,
        persona: AgentPersona,
        instruction: str,
        shape: str,
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
