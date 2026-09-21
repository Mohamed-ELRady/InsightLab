"""The base every pipeline agent inherits from.

An agent's ``run`` method is a generator. It does its work and, whenever it
reaches a judgement that belongs to the user rather than to us, it yields a
:class:`Decision` and receives an :class:`Answer` back:

    answer = yield self.decide(...)

The supervisor drives that generator. In interactive mode it stops and waits for
a human; in autonomous mode it answers with the suggestion immediately. Neither
the agent nor the user interface needs to know which of the two happened, which
is what keeps the two modes from drifting apart.
"""

from __future__ import annotations

from typing import Any, Generator, Iterable

from ..core.activity_log import EventKind
from ..core.decision import Answer, Decision, Option
from ..core.reasoning import AgentPersona, ReasoningEngine
from ..core.state import PipelineState

#: What a ``run`` generator yields and receives.
Flow = Generator[Decision, Answer, None]


class Agent:
    """One stage of the pipeline."""

    #: Key in ``PipelineState.stage_status``.
    stage: str = ""

    #: Short identifier used to cache the CrewAI agent.
    key: str = ""

    #: Title shown to the user while this agent is working.
    title: str = ""

    persona: AgentPersona = AgentPersona(
        role="Data analyst", goal="Help the user understand their data", backstory=""
    )

    def __init__(self, reasoning: ReasoningEngine) -> None:
        self.reasoning = reasoning

    # -- to implement ------------------------------------------------------

    def run(self, state: PipelineState) -> Flow:  # pragma: no cover - abstract
        raise NotImplementedError
        yield  # pragma: no cover - marks this as a generator for type checkers

    # -- reasoning helpers -------------------------------------------------

    def explain(self, instruction: str, expected: str = "A short paragraph.",
                *, max_output_tokens: int = 500) -> str | None:
        """Ask the model for prose. ``None`` means fall back to a heuristic."""
        return self.reasoning.ask(
            self.key, self.persona, instruction, expected,
            max_output_tokens=max_output_tokens,
        )

    def reason(self, instruction: str, shape: str,
               *, max_output_tokens: int | None = None) -> Any | None:
        """Ask the model for structured data. ``None`` means fall back."""
        return self.reasoning.ask_json(
            self.key, self.persona, instruction, shape,
            max_output_tokens=max_output_tokens,
        )

    def memory_block(self, state: PipelineState) -> str:
        """The project and domain facts so far, formatted for a prompt.

        Returns an empty string when nothing is known, so prompts can leave the
        whole section out rather than telling the model there are no facts.
        """
        block = state.memory.as_prompt_block()
        preferences = getattr(state, "project_preferences", {})
        directive = getattr(state, "analysis_directive", "").strip()
        if not block and not preferences and not directive:
            return ""
        memory_text = (
            "\n\nThe user has already established the following facts about this "
            f"project and dataset. Treat these as true and let them override anything the "
            f"numbers suggest:\n{block}"
            if block else ""
        )
        if preferences:
            preference_lines = "\n".join(
                f"- {key.replace('_', ' ')}: {value}"
                for key, value in sorted(preferences.items())
            )
            memory_text += (
                "\n\nThe user has chosen these presentation preferences. They may change "
                "wording, emphasis and question style, but never calculations or evidence:\n"
                f"{preference_lines}"
            )
        if directive:
            memory_text += (
                "\n\nFor this run, the user explicitly requested the following. "
                "Follow it wherever the available data supports it, and state "
                "clearly when it cannot be done:\n- " + directive
            )
        return memory_text

    # -- decision helpers --------------------------------------------------

    def decide(
        self,
        *,
        topic: str,
        question: str,
        context: str,
        suggestion: Option,
        alternatives: Iterable[Option] = (),
        custom_prompt: str = "",
        skip_effect: str = "",
        evidence: dict[str, Any] | None = None,
    ) -> Decision:
        """Build a decision for this stage."""
        decision = Decision(
            stage=self.stage,
            topic=topic,
            question=question,
            context=context,
            suggestion=suggestion,
            alternatives=list(alternatives),
            evidence=evidence or {},
        )
        if custom_prompt:
            decision.custom_prompt = custom_prompt
        if skip_effect:
            decision.skip_effect = skip_effect
        return decision

    def note(self, state: PipelineState, message: str, **details: Any) -> None:
        state.log.record(EventKind.NOTE, self.stage, message, **details)

    def warn(self, state: PipelineState, message: str, **details: Any) -> None:
        state.log.record(EventKind.WARNING, self.stage, message, **details)

    def apply_operation(self, state: PipelineState, operation) -> None:
        """Record a cleaning operation and make its result the working data."""
        state.set_frame(operation.frame, self.stage, operation.description)

    def capture_custom(
        self,
        state: PipelineState,
        decision: Decision,
        answer: Answer,
        *,
        category: str = "context",
    ) -> None:
        """Save a free-text answer as a project and domain fact.

        Anything the user types in their own words is context about the dataset
        or its domain, so it goes into shared memory where every later agent can
        see it - not just the one that asked.
        """
        if answer.is_custom and answer.text:
            state.remember(
                answer.text,
                category=category,
                stage=self.stage,
                topic=decision.topic,
            )
