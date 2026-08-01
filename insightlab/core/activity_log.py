"""Chronological record of everything the pipeline did.

Two audiences read this log. The user reads it to understand what happened to
their data and why; an auditor reads it to reproduce the run. Both need every
decision, including the ones the system took on its own in autonomous mode, so
nothing is filtered out at write time.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Iterable


class EventKind(str, Enum):
    STAGE_STARTED = "stage_started"
    STAGE_FINISHED = "stage_finished"
    STAGE_SKIPPED = "stage_skipped"
    DECISION_RAISED = "decision_raised"
    DECISION_ANSWERED = "decision_answered"
    DATA_CHANGED = "data_changed"
    FACT_RECORDED = "fact_recorded"
    ARTEFACT_CREATED = "artefact_created"
    NOTE = "note"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class Event:
    kind: EventKind
    stage: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind.value,
            "stage": self.stage,
            "message": self.message,
            "details": self.details,
            "at": self.at.isoformat(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Event":
        return cls(
            kind=EventKind(raw["kind"]),
            stage=raw.get("stage", ""),
            message=raw.get("message", ""),
            details=raw.get("details", {}),
            at=datetime.fromisoformat(raw["at"]),
        )


class ActivityLog:
    """Append-only list of :class:`Event` records."""

    def __init__(self, events: Iterable[Event] | None = None) -> None:
        self._events: list[Event] = list(events or [])

    def __len__(self) -> int:
        return len(self._events)

    def __iter__(self):
        return iter(self._events)

    def record(
        self,
        kind: EventKind,
        stage: str,
        message: str,
        **details: Any,
    ) -> Event:
        event = Event(kind=kind, stage=stage, message=message, details=details)
        self._events.append(event)
        return event

    def events(self) -> list[Event]:
        return list(self._events)

    def for_stage(self, stage: str) -> list[Event]:
        return [event for event in self._events if event.stage == stage]

    def of_kind(self, *kinds: EventKind) -> list[Event]:
        wanted = set(kinds)
        return [event for event in self._events if event.kind in wanted]

    def data_operations(self) -> list[Event]:
        """Just the events that changed the data, for the cleaning summary."""
        return self.of_kind(EventKind.DATA_CHANGED)

    def to_list(self) -> list[dict[str, Any]]:
        return [event.to_dict() for event in self._events]

    @classmethod
    def from_list(cls, raw: list[dict[str, Any]]) -> "ActivityLog":
        return cls(Event.from_dict(item) for item in raw)
