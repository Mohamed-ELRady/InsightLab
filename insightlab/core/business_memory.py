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
from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:  # pragma: no cover
    from .claims import Claim


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
    """A single statement about how the user's business works.

    ``claim`` is the machine-readable reading of the statement, when we managed
    to make one. The sentence is always what the user reads; the claim is what
    lets the fact be tested against new data and applied automatically. A fact
    with no claim behaves exactly as it always did.
    """

    statement: str
    category: str = "context"
    source_stage: str = "unknown"
    source_topic: str = ""
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    recorded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    claim: "Claim | None" = None
    #: Set when a later file contradicted this and the owner kept it anyway.
    disputed: bool = False

    @property
    def is_testable(self) -> bool:
        return self.claim is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "category": self.category,
            "source_stage": self.source_stage,
            "source_topic": self.source_topic,
            "recorded_at": self.recorded_at.isoformat(),
            "claim": self.claim.to_dict() if self.claim else None,
            "disputed": self.disputed,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Fact":
        from .claims import Claim

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
            claim=Claim.from_dict(raw["claim"]) if raw.get("claim") else None,
            disputed=bool(raw.get("disputed", False)),
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
        claim: Any = None,
    ) -> Fact | None:
        """Record a fact. Returns ``None`` if it is blank or already known."""
        cleaned = " ".join(statement.split())
        if not cleaned:
            return None
        if category not in CATEGORIES:
            category = "context"
        for existing in self._facts:
            if existing.statement.casefold() == cleaned.casefold():
                # Re-stating a known fact with structure this time is an
                # upgrade, not a duplicate.
                if claim is not None and existing.claim is None:
                    existing.claim = claim
                return None

        fact = Fact(
            statement=cleaned,
            category=category,
            source_stage=stage,
            source_topic=topic,
            claim=claim,
        )
        self._facts.append(fact)
        return fact

    def testable(self) -> list[Fact]:
        """Facts that can be checked against a dataset."""
        return [fact for fact in self._facts if fact.is_testable]

    def replace(self, fact_id: str, statement: str, claim: Any = None) -> bool:
        """Update a fact in place, keeping its id and its history."""
        for fact in self._facts:
            if fact.id == fact_id:
                fact.statement = " ".join(statement.split())
                fact.claim = claim
                fact.disputed = False
                return True
        return False

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
