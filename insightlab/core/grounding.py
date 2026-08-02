"""Checking that every figure in a generated sentence exists in the data.

The insight agent asks a model to write conclusions containing numbers. Nothing
else in the system checks that those numbers came from anywhere. One sentence
saying revenue grew 45% when it grew 4.5% ends the user's trust in every other
figure on the page, and they have no way to tell which one was wrong.

So before any generated text reaches a user, every number in it is matched
against an allow-list built from the data itself: every value in every chart
table, every KPI, every profile statistic, and the arithmetic a reader would
reasonably do with them. A sentence carrying a number that is not on the list
does not get shown.

There is no model in this module. A checker that could itself hallucinate would
be pointless.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import pandas as pd

#: Relative tolerance when matching a figure. Generous enough for honest
#: rounding ("2.8M" for 2,764,183), tight enough to catch a wrong number.
TOLERANCE = 0.02

#: Numbers at or below this are matched on absolute difference instead, since a
#: 2% relative tolerance on the number 3 is meaningless.
SMALL_NUMBER = 20.0

#: Absolute tolerance for those small numbers.
SMALL_TOLERANCE = 0.5

#: Figures this common carry no claim - a year, a count of items in a list, an
#: ordinal. Checking them produces false alarms, not safety.
ALWAYS_ALLOWED = frozenset(float(n) for n in range(0, 13)) | frozenset(
    float(year) for year in range(1990, 2101)
)

#: Suffix multipliers a model uses when writing business figures.
SUFFIXES = {
    "k": 1_000.0,
    "thousand": 1_000.0,
    "m": 1_000_000.0,
    "mn": 1_000_000.0,
    "million": 1_000_000.0,
    "b": 1_000_000_000.0,
    "bn": 1_000_000_000.0,
    "billion": 1_000_000_000.0,
}

_NUMBER = re.compile(
    r"""
    (?<![\w.])                      # not mid-identifier
    (?P<sign>[-+]?)
    (?P<digits>\d{1,3}(?:,\d{3})+|\d+(?:\.\d+)?)
    \s*
    (?P<suffix>k|m|mn|bn|b|thousand|million|billion)?
    \s*
    (?P<percent>%)?
    (?![\w])
    """,
    re.VERBOSE | re.IGNORECASE,
)

#: Sentence boundary that does not split on a decimal point or a suffix.
_SENTENCE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")


@dataclass
class Figure:
    """One number found in generated text."""

    text: str
    value: float
    is_percentage: bool
    start: int
    end: int


@dataclass
class GroundingResult:
    """What happened when a piece of text was checked."""

    text: str
    checked: int = 0
    verified: int = 0
    unverified: list[Figure] = field(default_factory=list)
    removed_sentences: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return not self.unverified

    @property
    def summary(self) -> str:
        if self.is_clean:
            return f"All {self.checked} figures matched the data."
        listed = ", ".join(figure.text for figure in self.unverified[:4])
        return (
            f"{len(self.unverified)} of {self.checked} figures did not match "
            f"anything in the data ({listed})."
        )


def extract_figures(text: str) -> list[Figure]:
    """Every number a reader would take as a factual claim."""
    figures: list[Figure] = []
    for match in _NUMBER.finditer(text):
        raw = match.group("digits").replace(",", "")
        try:
            value = float(raw)
        except ValueError:
            continue

        suffix = (match.group("suffix") or "").lower()
        if suffix:
            value *= SUFFIXES[suffix]
        if match.group("sign") == "-":
            value = -value

        figures.append(
            Figure(
                text=match.group(0).strip(),
                value=value,
                is_percentage=bool(match.group("percent")),
                start=match.start(),
                end=match.end(),
            )
        )
    return figures


class FactBase:
    """Every figure the data actually supports.

    Built once per check from the state. Holds the raw values and, separately,
    the percentages a reader would derive from them - a share of a total, a
    period-on-period change - because a model quoting "52% of revenue" is
    quoting a real fact that appears nowhere as a stored number.
    """

    def __init__(self) -> None:
        self._values: set[float] = set()
        self._percentages: set[float] = set()

    def __len__(self) -> int:
        return len(self._values) + len(self._percentages)

    # -- population --------------------------------------------------------

    def add(self, value: Any) -> None:
        number = _as_number(value)
        if number is None:
            return
        self._values.add(number)
        for rounded in _display_roundings(number):
            self._values.add(rounded)

    def add_percentage(self, value: Any) -> None:
        number = _as_number(value)
        if number is None:
            return
        self._percentages.add(number)
        for places in (0, 1):
            self._percentages.add(round(number, places))

    def add_series(self, series: Iterable[Any]) -> None:
        for item in series:
            self.add(item)

    def add_frame(self, frame: pd.DataFrame | None) -> None:
        if frame is None or frame.empty:
            return
        for column in frame.columns:
            values = pd.to_numeric(frame[column], errors="coerce").dropna()
            if values.empty:
                continue
            self.add_series(values.tolist())
            # Totals and shares of a column are the arithmetic a reader does
            # without thinking, so a model quoting them is quoting the data.
            total = float(values.sum())
            self.add(total)
            self.add(float(values.mean()))
            if total:
                for value in values:
                    self.add_percentage(float(value) / total * 100)

    def add_derived_shares(self, values: Iterable[float]) -> None:
        """Shares and pairwise changes between a set of related figures."""
        numbers = [number for number in (_as_number(v) for v in values) if number]
        total = sum(numbers)
        if total:
            for number in numbers:
                self.add_percentage(number / total * 100)
        for first in numbers:
            for second in numbers:
                if first and first != second:
                    self.add_percentage((second - first) / abs(first) * 100)

    # -- checking ----------------------------------------------------------

    def holds(self, figure: Figure) -> bool:
        if figure.value in ALWAYS_ALLOWED and not figure.is_percentage:
            return True

        pool = self._percentages if figure.is_percentage else self._values
        if _matches(figure.value, pool):
            return True

        # A percentage may have been stored as a fraction, or the other way
        # round, depending on which agent produced it.
        if figure.is_percentage and _matches(figure.value / 100, self._values):
            return True
        if not figure.is_percentage and _matches(figure.value, self._percentages):
            return True
        return False


def _display_roundings(number: float) -> set[float]:
    """Every way this figure legitimately appears once it is written down.

    The product itself renders 1,858,741 as "1.9M" - two significant figures.
    A tolerance wide enough to accept that on its own would be wide enough to
    hide a real error, so instead the roundings are stored as facts in their own
    right. Two to four significant figures covers every format the product uses
    and nothing looser: "2M" for 1.86M is still, correctly, unverified.
    """
    roundings = {round(number, places) for places in (0, 1, 2)}
    if number:
        magnitude = math.floor(math.log10(abs(number)))
        for digits in (2, 3, 4):
            places = digits - 1 - magnitude
            roundings.add(round(number, places))
    return roundings


def _as_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(number) or math.isinf(number):
        return None
    return number


def _matches(value: float, pool: set[float]) -> bool:
    if value in pool:
        return True
    magnitude = abs(value)
    tolerance = SMALL_TOLERANCE if magnitude <= SMALL_NUMBER else magnitude * TOLERANCE
    return any(abs(value - candidate) <= tolerance for candidate in pool)


def build_fact_base(state) -> FactBase:
    """Every figure the current run can legitimately support."""
    facts = FactBase()

    facts.add(len(state.frame) if state.frame is not None else 0)
    facts.add(state.profile.row_count)
    facts.add(state.profile.column_count)
    facts.add(state.profile.duplicate_rows)

    for column in state.profile.columns:
        facts.add(column.missing_count)
        facts.add(column.unique_count)
        facts.add_percentage(column.missing_rate * 100)
        for key, value in column.stats.items():
            if key == "top_values" and isinstance(value, dict):
                facts.add_series(value.values())
                facts.add_derived_shares([v for v in value.values()])
            elif key == "top_share":
                facts.add_percentage(_as_number(value) * 100 if _as_number(value) else None)
            else:
                facts.add(value)

    for kpi in state.kpis:
        facts.add(kpi.value)
        if kpi.unit == "%":
            facts.add_percentage(kpi.value)

    # Figures from the previous run are real facts too. Without them, every
    # finding about what has changed since last time cites a number that is not
    # in this file and gets stripped as invented.
    comparison = getattr(state, "comparison", None)
    if comparison is not None:
        for change in comparison.changes:
            facts.add(change.before)
            facts.add(change.now)
            facts.add(change.difference)
            facts.add_percentage(change.relative * 100)
            facts.add_percentage(abs(change.difference))

    for chart in state.charts:
        facts.add_frame(chart.table)

    # The working frame itself: a model may quote a single large order.
    if state.frame is not None:
        for name in state.frame.columns:
            values = pd.to_numeric(state.frame[name], errors="coerce").dropna()
            if values.empty:
                continue
            facts.add(float(values.sum()))
            facts.add(float(values.mean()))
            facts.add(float(values.median()))
            facts.add(float(values.min()))
            facts.add(float(values.max()))

    return facts


def check(text: str, facts: FactBase) -> GroundingResult:
    """Report which figures in ``text`` the data does not support."""
    result = GroundingResult(text=text)
    for figure in extract_figures(text):
        result.checked += 1
        if facts.holds(figure):
            result.verified += 1
        else:
            result.unverified.append(figure)
    return result


def strip_unverified(text: str, facts: FactBase) -> GroundingResult:
    """Remove whole sentences carrying a figure the data does not support.

    Sentences, not numbers: deleting "45%" out of the middle of a sentence
    leaves something that still reads as a claim but no longer says anything.
    Removing the claim entirely is the honest edit.
    """
    result = check(text, facts)
    if result.is_clean:
        return result

    kept: list[str] = []
    for sentence in _SENTENCE.split(text):
        bad = [
            figure
            for figure in extract_figures(sentence)
            if not facts.holds(figure)
        ]
        if bad:
            result.removed_sentences.append(sentence.strip())
        else:
            kept.append(sentence)

    result.text = " ".join(part.strip() for part in kept).strip()
    return result
