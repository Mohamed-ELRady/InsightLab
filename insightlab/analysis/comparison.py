"""Comparing this run with the last one on the same kind of file.

"Revenue is 11.8M" is a fact. "Revenue is 11.8M, down 8% on the period you
analysed in March" is the sentence an owner actually wants, and until now every
run was an island that could not produce it.

Two runs are comparable when their files have the same shape. That is decided by
a fingerprint over the column names and roles - not over the filename, which
changes every month, and not over the row count, which always changes.

The period alignment matters more than it looks. Comparing a twelve-month file
with a three-month one on totals would report a collapse that is really just a
shorter file, so totals are only compared when the periods are comparable, and
per-month averages are used when they are not.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.state import DatasetProfile, Kpi, PipelineState, Role
from .profiling import try_parse_datetime

#: A change smaller than this is not worth a sentence.
MATERIAL_CHANGE = 0.05

#: Periods must be within this much of each other in length before totals can be
#: compared directly rather than per-period.
LENGTH_TOLERANCE = 0.25


def fingerprint(frame: pd.DataFrame | None) -> str:
    """A stable identifier for the shape of a file.

    Deliberately computed from the column names of the file **as supplied**,
    before anything is cleaned or derived. Three things were tried and rejected:

    The filename, which changes every month. The row count, which always
    changes. And column roles, which are inferred from the data and therefore
    diverge between two exports of the same report - a column that happens to
    hold one value this month profiles as constant, and the fingerprints stop
    matching for a reason that has nothing to do with the file being different.

    Names are also why this must run at load time. Cleaning drops columns and
    feature engineering adds them, both depending on what the data looks like,
    so a fingerprint taken afterwards never matches twice.
    """
    if frame is None or frame.empty:
        return ""
    parts = sorted(str(name).strip().casefold() for name in frame.columns)
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]


@dataclass
class Period:
    """The stretch of time a run's data covers."""

    start: str = ""
    end: str = ""
    months: float = 0.0

    @property
    def known(self) -> bool:
        return bool(self.start and self.end and self.months > 0)

    def label(self) -> str:
        if not self.known:
            return "an unknown period"
        return f"{self.start} to {self.end}"

    def to_dict(self) -> dict[str, Any]:
        return {"start": self.start, "end": self.end, "months": self.months}

    @classmethod
    def from_dict(cls, raw: dict[str, Any] | None) -> "Period":
        if not isinstance(raw, dict):
            return cls()
        return cls(
            start=str(raw.get("start", "")),
            end=str(raw.get("end", "")),
            months=float(raw.get("months", 0) or 0),
        )


@dataclass
class Change:
    """How one measure moved between two runs."""

    name: str
    now: float
    before: float
    unit: str = ""
    per_period: bool = False

    @property
    def difference(self) -> float:
        return self.now - self.before

    @property
    def relative(self) -> float:
        return self.difference / abs(self.before) if self.before else 0.0

    @property
    def is_material(self) -> bool:
        return abs(self.relative) >= MATERIAL_CHANGE

    @property
    def direction(self) -> str:
        return "up" if self.difference > 0 else "down"

    def describe(self, previous_label: str) -> str:
        basis = " per month" if self.per_period else ""
        if self.unit == "%":
            return (
                f"{self.name} is {self.now:.1f}%, against {self.before:.1f}% "
                f"in {previous_label} - {abs(self.difference):.1f} points "
                f"{self.direction}."
            )
        return (
            f"{self.name}{basis} is {_format(self.now)}, against "
            f"{_format(self.before)} in {previous_label} - "
            f"{abs(self.relative):.0%} {self.direction}."
        )


@dataclass
class Comparison:
    """The result of putting two runs side by side."""

    previous_run_id: str
    previous_date: str
    previous_period: Period
    current_period: Period
    changes: list[Change]
    comparable_periods: bool

    @property
    def material(self) -> list[Change]:
        return [change for change in self.changes if change.is_material]

    def label(self) -> str:
        if self.previous_period.known:
            return self.previous_period.label()
        return f"your analysis of {self.previous_date[:10]}"

    def describe(self) -> str:
        if not self.material:
            return (
                f"Nothing has moved much since {self.label()}: every headline "
                "figure is within five per cent of where it was."
            )

        biggest = max(self.material, key=lambda change: abs(change.relative))
        lines = [f"Compared with {self.label()}: {biggest.describe(self.label())}"]
        for change in self.material:
            if change is biggest:
                continue
            lines.append(change.describe(self.label()))

        if not self.comparable_periods:
            lines.append(
                "The two files cover periods of different lengths, so totals are "
                "compared per month rather than outright."
            )
        return " ".join(lines)


# ---------------------------------------------------------------------------
# Finding the previous run
# ---------------------------------------------------------------------------


def period_of(frame: pd.DataFrame, profile: DatasetProfile) -> Period:
    """The stretch of time a dataset covers."""
    for column in profile.columns:
        if column.role is not Role.DATETIME or column.name not in frame.columns:
            continue
        series = frame[column.name]
        if not pd.api.types.is_datetime64_any_dtype(series):
            parsed = try_parse_datetime(series)
            if parsed is None:
                continue
            series = parsed
        clean = series.dropna()
        if clean.empty:
            continue
        span = (clean.max() - clean.min()).days
        return Period(
            start=str(clean.min().date()),
            end=str(clean.max().date()),
            months=round(max(span, 1) / 30.44, 2),
        )
    return Period()


def find_previous(
    state: PipelineState, workspace: Path | None = None
) -> dict[str, Any] | None:
    """The most recent earlier run over the same shape of file."""
    from ..core.storage import list_runs

    # Taken at load time from the file as supplied. Recomputing it here from
    # the working frame would compare post-cleaning shapes, which never match.
    current = state.fingerprint or fingerprint(state.raw_frame)
    if not current:
        return None

    for directory in list_runs(workspace):
        if directory.name == state.run_id:
            continue
        summary_path = directory / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if summary.get("fingerprint") != current:
            continue
        if not summary.get("kpis"):
            continue
        return summary
    return None


def compare(
    state: PipelineState, previous: dict[str, Any]
) -> Comparison | None:
    """Put the current run's headline figures against an earlier run's."""
    earlier = {
        item.get("name"): item
        for item in previous.get("kpis", [])
        if isinstance(item, dict) and item.get("value") is not None
    }
    if not earlier:
        return None

    previous_period = Period.from_dict(previous.get("period"))
    current_period = period_of(state.frame, state.profile) if state.frame is not None else Period()

    comparable = _periods_comparable(current_period, previous_period)

    changes: list[Change] = []
    for kpi in state.kpis:
        if kpi.value is None or not _is_numeric_measure(kpi):
            continue
        before = earlier.get(kpi.name)
        if before is None:
            continue
        try:
            previous_value = float(before["value"])
        except (TypeError, ValueError, KeyError):
            continue

        now = float(kpi.value)
        per_period = False

        # A total from a twelve-month file against one from a three-month file
        # would report a collapse that is only a shorter file. Rates and
        # averages are unaffected, so only totals are rebased.
        if not comparable and _is_total(kpi) and current_period.months and previous_period.months:
            now = now / current_period.months
            previous_value = previous_value / previous_period.months
            per_period = True

        changes.append(
            Change(
                name=kpi.name,
                now=now,
                before=previous_value,
                unit=kpi.unit,
                per_period=per_period,
            )
        )

    if not changes:
        return None

    return Comparison(
        previous_run_id=str(previous.get("run_id", "")),
        previous_date=str(previous.get("started_at", "")),
        previous_period=previous_period,
        current_period=current_period,
        changes=changes,
        comparable_periods=comparable,
    )


def _periods_comparable(current: Period, previous: Period) -> bool:
    if not (current.known and previous.known):
        return True  # No dates to go on; treat totals at face value.
    if not previous.months:
        return True
    ratio = abs(current.months - previous.months) / previous.months
    return ratio <= LENGTH_TOLERANCE


def _is_numeric_measure(kpi: Kpi) -> bool:
    """Whether this figure is a quantity that can meaningfully move.

    "Strongest month" carries the peak month's revenue as its value but shows a
    date as its label. Comparing the values across two runs is arithmetically
    valid and reads as nonsense - "Strongest month is 824,209, down 54%" - so
    figures whose displayed form is not the number are left out.
    """
    display = kpi.display_value.replace(",", "").replace("%", "").strip()
    for suffix in ("M", "K", "B"):
        display = display.removesuffix(suffix)
    try:
        float(display)
    except ValueError:
        return False
    return True


def _is_total(kpi: Kpi) -> bool:
    """Whether a figure grows simply because the file covers more time."""
    if kpi.unit == "%":
        return False
    lowered = kpi.name.casefold()
    return any(
        word in lowered
        for word in ("total", "number of", "units", "count", "customers")
    )


def _format(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if magnitude >= 1_000:
        return f"{value:,.0f}"
    if float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}"
