"""The shared decision framework.

Every agent that needs a judgement call from the user raises the *same* four
options, so the experience is identical no matter which stage of the pipeline
raised it:

1. Use the suggestion the analyst proposed.
2. Choose one of the ready-made alternatives.
3. Write a custom instruction in free text.
4. Skip this step.

Agents never build these options by hand. They construct a :class:`Decision`
describing the situation and yield it; the supervisor decides whether a human
answers it (interactive mode) or the suggestion is taken automatically
(autonomous mode).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class Choice(str, Enum):
    """Which of the four standard options was taken."""

    SUGGESTION = "suggestion"
    ALTERNATIVE = "alternative"
    CUSTOM = "custom"
    SKIP = "skip"


@dataclass
class Option:
    """One concrete course of action an agent can take.

    ``payload`` is the machine-readable part the agent acts on. ``label`` and
    ``rationale`` are what the user reads, and must be written in plain
    business language.
    """

    label: str
    rationale: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"label": self.label, "rationale": self.rationale, "payload": self.payload}


@dataclass
class Decision:
    """A checkpoint the pipeline has reached and cannot pass unattended."""

    stage: str
    topic: str
    question: str
    context: str
    suggestion: Option
    alternatives: list[Option] = field(default_factory=list)
    custom_prompt: str = "Tell us how you would like this handled, in your own words."
    skip_effect: str = "This step is left untouched and the data moves on unchanged."
    evidence: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def option_at(self, index: int) -> Option | None:
        if 0 <= index < len(self.alternatives):
            return self.alternatives[index]
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "stage": self.stage,
            "topic": self.topic,
            "question": self.question,
            "context": self.context,
            "suggestion": self.suggestion.to_dict(),
            "alternatives": [option.to_dict() for option in self.alternatives],
            "custom_prompt": self.custom_prompt,
            "skip_effect": self.skip_effect,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Answer:
    """The resolution of a :class:`Decision`.

    Instances are produced either by the user interface or, in autonomous mode,
    by the supervisor. ``automatic`` records which of the two it was so the run
    log stays honest about what the user actually approved.
    """

    choice: Choice
    decision_id: str = ""
    alternative_index: int | None = None
    text: str = ""
    automatic: bool = False
    answered_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    _decision: Decision | None = field(default=None, repr=False, compare=False)

    @classmethod
    def accept(cls, decision: Decision, *, automatic: bool = False) -> "Answer":
        return cls(
            choice=Choice.SUGGESTION,
            decision_id=decision.id,
            automatic=automatic,
            _decision=decision,
        )

    @classmethod
    def alternative(cls, decision: Decision, index: int) -> "Answer":
        return cls(
            choice=Choice.ALTERNATIVE,
            decision_id=decision.id,
            alternative_index=index,
            _decision=decision,
        )

    @classmethod
    def custom(cls, decision: Decision, text: str) -> "Answer":
        return cls(
            choice=Choice.CUSTOM,
            decision_id=decision.id,
            text=text.strip(),
            _decision=decision,
        )

    @classmethod
    def skip(cls, decision: Decision) -> "Answer":
        return cls(choice=Choice.SKIP, decision_id=decision.id, _decision=decision)

    @property
    def selected(self) -> Option | None:
        """The option the answer points at, or ``None`` when skipped."""
        if self._decision is None:
            return None
        if self.choice is Choice.SUGGESTION:
            return self._decision.suggestion
        if self.choice is Choice.ALTERNATIVE and self.alternative_index is not None:
            return self._decision.option_at(self.alternative_index)
        return None

    @property
    def payload(self) -> dict[str, Any]:
        """Machine-readable instruction for the agent that raised the decision.

        A custom answer carries no payload of its own; the agent is expected to
        interpret ``text`` and may fall back on the suggestion's payload.
        """
        option = self.selected
        return dict(option.payload) if option else {}

    @property
    def is_skip(self) -> bool:
        return self.choice is Choice.SKIP

    @property
    def is_custom(self) -> bool:
        return self.choice is Choice.CUSTOM

    def describe(self) -> str:
        """One line summarising what was decided, for the run log."""
        if self.choice is Choice.SKIP:
            return "Skipped."
        if self.choice is Choice.CUSTOM:
            return f"Custom instruction: {self.text}"
        option = self.selected
        label = option.label if option else "unknown option"
        if self.choice is Choice.SUGGESTION:
            return f"Accepted the suggestion: {label}"
        return f"Chose the alternative: {label}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_id": self.decision_id,
            "choice": self.choice.value,
            "alternative_index": self.alternative_index,
            "text": self.text,
            "automatic": self.automatic,
            "answered_at": self.answered_at.isoformat(),
            "summary": self.describe(),
        }
