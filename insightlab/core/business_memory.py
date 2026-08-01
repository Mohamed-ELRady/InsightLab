"""Shared store for business facts the user tells us.

Anything the user states about how their business works - a VIP threshold, a
peak season, a rule about which rows to ignore - is written here once and read
by every agent for the rest of the project. There is deliberately one store per
run rather than one per agent, so a fact stated during cleaning is still known
when the report is written.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Iterable


#: Broad buckets used to group facts when they are shown back to the user.
CATEGORIES = (
    "definition",  # "VIP customers spend more than 5,000"
    "seasonality",  # "Peak season starts in November"
    "exclusion",  # "Ignore cancelled invoices"
    "classification",  # "We group products into A/B/C tiers"
    "target",  # "We aim for a 30% margin"
    "context",  # anything else worth remembering
)


@dataclass
class Fact:
    """A single statement about how the user's business works."""

    statement: str
    category: str = "context"
    source_stage: str = "unknown"
    source_topic: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "category": self.category,
            "source_stage": self.source_stage,
            "source_topic": self.source_topic,
            "recorded_at": self.recorded_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Fact":
        recorded = raw.get("recorded_at")
        return cls(
            statement=raw["statement"],
            category=raw.get("category", "context"),
            source_stage=raw.get("source_stage", "unknown"),
            source_topic=raw.get("source_topic", ""),
            id=raw.get("id", uuid.uuid4().hex[:8]),
            recorded_at=(
                datetime.fromisoformat(recorded)
                if recorded
                else datetime.now(timezone.utc)
            ),
        )


class BusinessMemory:
    """Ordered collection of :class:`Fact` objects, unique by statement."""

    def __init__(self, facts: Iterable[Fact] | None = None) -> None:
        self._facts: list[Fact] = list(facts or [])

    def __len__(self) -> int:
        return len(self._facts)

    def __iter__(self):
        return iter(self._facts)

    def __bool__(self) -> bool:
        return bool(self._facts)

    def remember(
        self,
        statement: str,
        *,
        category: str = "context",
        stage: str = "unknown",
        topic: str = "",
    ) -> Fact | None:
        """Record a fact. Returns ``None`` if it is blank or already known."""
        cleaned = " ".join(statement.split())
        if not cleaned:
            return None
        if category not in CATEGORIES:
            category = "context"
        for existing in self._facts:
            if existing.statement.casefold() == cleaned.casefold():
                return None

        fact = Fact(
            statement=cleaned,
            category=category,
            source_stage=stage,
            source_topic=topic,
        )
        self._facts.append(fact)
        return fact

    def forget(self, fact_id: str) -> bool:
        """Drop a fact by id. Returns True when something was removed."""
        before = len(self._facts)
        self._facts = [fact for fact in self._facts if fact.id != fact_id]
        return len(self._facts) < before

    def facts(self) -> list[Fact]:
        return list(self._facts)

    def by_category(self) -> dict[str, list[Fact]]:
        grouped: dict[str, list[Fact]] = {}
        for fact in self._facts:
            grouped.setdefault(fact.category, []).append(fact)
        return grouped

    def as_prompt_block(self, *, limit: int = 40) -> str:
        """Render the memory for inclusion in an agent prompt.

        Returns an empty string when nothing has been recorded, so callers can
        drop the whole section rather than injecting a "none" placeholder.
        """
        if not self._facts:
            return ""
        lines = [
            f"- ({fact.category}) {fact.statement}" for fact in self._facts[-limit:]
        ]
        return "\n".join(lines)

    def to_list(self) -> list[dict[str, Any]]:
        return [fact.to_dict() for fact in self._facts]

    @classmethod
    def from_list(cls, raw: list[dict[str, Any]]) -> "BusinessMemory":
        return cls(Fact.from_dict(item) for item in raw)
