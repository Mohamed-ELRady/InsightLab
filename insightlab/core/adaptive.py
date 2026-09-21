"""Transparent personalisation from explicit owner feedback.

The learner only changes presentation order. It cannot change calculations,
filter rows, edit memory, or invent a conclusion.
"""

from __future__ import annotations

from typing import Any, Iterable, TypeVar


T = TypeVar("T")


def preference_score(kind: str, value: str, profile: dict[str, Any],
                     policy: dict[str, float] | None = None) -> float:
    policy = policy or {}
    weight_key = {
        "axes": "axis_weight", "kpis": "kpi_weight",
        "chart_kinds": "chart_weight", "confidence": "confidence_weight",
    }.get(kind, "axis_weight")
    return float(profile.get(kind, {}).get(str(value).casefold(), 0.0)) * float(
        policy.get(weight_key, 1.0)
    )


def rank_values(values: Iterable[str], kind: str, profile: dict[str, Any],
                policy: dict[str, float] | None = None) -> list[str]:
    """Stable ranking: ties retain the product's deterministic default order."""
    return sorted(
        values,
        key=lambda value: -preference_score(kind, value, profile, policy),
    )


def rank_objects(items: Iterable[T], *, profile: dict[str, Any],
                 policy: dict[str, float] | None = None,
                 axis_attr: str = "axis", name_attr: str = "") -> list[T]:
    def score(item: T) -> float:
        total = preference_score("axes", getattr(item, axis_attr, ""), profile, policy)
        if name_attr:
            total += preference_score("kpis", getattr(item, name_attr, ""), profile, policy)
        kind = getattr(item, "kind", "")
        if kind:
            total += preference_score("chart_kinds", kind, profile, policy)
        confidence = getattr(item, "confidence", "")
        if confidence:
            total += preference_score("confidence", confidence, profile, policy)
        return total

    return sorted(items, key=lambda item: -score(item))
