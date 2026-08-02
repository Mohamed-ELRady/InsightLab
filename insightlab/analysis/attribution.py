"""Why a number moved.

"November was your best month" is an observation. "November was 1.6M above
average, and 62% of that came from Electronics in Cairo" is something a person
can act on tomorrow. The second one is the first question every owner asks and
the one a chart cannot answer.

The decomposition is exact and requires no model: the contributions of the
parts sum to the movement of the whole, by construction. That matters, because
an attribution whose parts do not add up is worse than none - the reader will
check, and they will find the gap.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from ..core.state import DatasetProfile, PipelineState, Role
from .cleaning import UNUSUAL_SUFFIX
from .profiling import try_parse_datetime

#: Contributors below this share of the movement are folded into "everything
#: else". Listing a 0.4% contributor is noise dressed as detail.
MIN_CONTRIBUTION = 0.03

#: Contributors listed per dimension before the rest are grouped.
TOP_CONTRIBUTORS = 5

#: A dimension with more distinct values than this cannot be explained to
#: anyone, however well it decomposes.
MAX_DIMENSION_VALUES = 40

#: Dimensions listed in the written explanation. Twelve breakdowns of the same
#: movement is not an explanation, it is a data dump.
MAX_DIMENSIONS_DESCRIBED = 3

#: Columns that are functions of the date. When the movement being explained is
#: itself a period, these are circular - "November was high because of
#: November" - so they are excluded from a time-based attribution. They stay
#: available when explaining the gap between two groups, where seasonality is a
#: real and useful answer.
CALENDAR_DERIVED = frozenset(
    {
        "_period", "year", "month", "month_name", "quarter", "week",
        "day", "day_of_week", "day_of_month", "is_weekend", "weekday",
    }
)


@dataclass
class Contribution:
    """One group's share of a movement."""

    label: str
    change: float
    share: float
    before: float
    after: float

    @property
    def direction(self) -> str:
        return "added to" if self.change >= 0 else "took away from"


@dataclass
class DimensionBreakdown:
    """How one dimension explains a movement."""

    column: str
    contributions: list[Contribution]
    concentration: float

    @property
    def headline(self) -> Contribution | None:
        return self.contributions[0] if self.contributions else None

    def describe(self, measure: str) -> str:
        if not self.contributions:
            return ""
        readable = self.column.replace("_", " ")
        top = self.contributions[0]
        parts = [
            f"By {readable}: {top.label} accounts for {abs(top.share):.0%} of the "
            f"movement on its own"
        ]
        if len(self.contributions) > 1:
            second = self.contributions[1]
            parts.append(
                f", followed by {second.label} at {abs(second.share):.0%}"
            )
        return "".join(parts) + "."


@dataclass
class Attribution:
    """The full account of one movement."""

    measure: str
    period_label: str
    baseline_label: str
    period_value: float
    baseline_value: float
    dimensions: list[DimensionBreakdown] = field(default_factory=list)

    @property
    def change(self) -> float:
        return self.period_value - self.baseline_value

    @property
    def relative_change(self) -> float:
        return self.change / self.baseline_value if self.baseline_value else 0.0

    @property
    def best_dimension(self) -> DimensionBreakdown | None:
        """The dimension that concentrates the movement most tightly.

        A movement spread evenly across every region explains nothing; one
        where a single region carries most of it is the actual answer.
        """
        if not self.dimensions:
            return None
        return max(self.dimensions, key=lambda item: item.concentration)

    def describe(self) -> str:
        readable = self.measure.replace("_", " ")
        direction = "above" if self.change >= 0 else "below"
        opening = (
            f"{self.period_label} was {abs(self.change):,.0f} {direction} "
            f"{self.baseline_label} on {readable} "
            f"({abs(self.relative_change):.0%})."
        )

        best = self.best_dimension
        if best is None or not best.contributions:
            return (
                opening + " No single group accounts for much of it, so it is a "
                "broad movement rather than a specific cause."
            )

        lines = [opening]
        top = best.contributions[0]
        # A share above 100% is arithmetically correct - one group rose while
        # others fell - but reads as an error, so it is worded rather than
        # printed as a percentage.
        if abs(top.share) > 1:
            magnitude = (
                f"more than the whole movement on its own ({abs(top.change):,.0f}), "
                "with other groups pulling in the opposite direction"
            )
        else:
            magnitude = (
                f"{abs(top.change):,.0f} of it, or {abs(top.share):.0%} of the total"
            )
        lines.append(
            f"Most of it is {best.column.replace('_', ' ')}: {top.label} alone "
            f"{top.direction} it by {magnitude}."
        )

        for other in self.dimensions[:MAX_DIMENSIONS_DESCRIBED]:
            if other is best:
                continue
            text = other.describe(self.measure)
            if text:
                lines.append(text)
        return " ".join(lines)

    def to_dict(self) -> dict:
        return {
            "measure": self.measure,
            "period": self.period_label,
            "baseline": self.baseline_label,
            "period_value": self.period_value,
            "baseline_value": self.baseline_value,
            "change": self.change,
            "relative_change": self.relative_change,
            "dimensions": [
                {
                    "column": dimension.column,
                    "concentration": dimension.concentration,
                    "contributions": [
                        {
                            "label": item.label,
                            "change": item.change,
                            "share": item.share,
                        }
                        for item in dimension.contributions
                    ],
                }
                for dimension in self.dimensions
            ],
        }


# ---------------------------------------------------------------------------
# Building it
# ---------------------------------------------------------------------------


def _explains_nothing(name: str, measure: str, exclude_calendar: bool) -> bool:
    """Whether breaking down by this column would be circular.

    Three ways it can be. It is our own cleaning marker; it is a function of
    the measure being explained, so of course it tracks it; or, when the thing
    being explained is a month, it is a function of the date.
    """
    lowered = str(name).casefold()
    if lowered.endswith(UNUSUAL_SUFFIX):
        return True
    if exclude_calendar and lowered in CALENDAR_DERIVED:
        return True
    stem = measure.casefold()
    return lowered.startswith(f"{stem}_") or lowered.endswith(f"_{stem}")


def _dimensions(
    frame: pd.DataFrame,
    profile: DatasetProfile,
    measure: str,
    *,
    exclude_calendar: bool = False,
) -> list[str]:
    """Columns worth breaking a movement down by."""
    usable: list[str] = []
    for column in profile.columns:
        if column.name not in frame.columns or column.name == measure:
            continue
        if column.role not in (Role.CATEGORY, Role.BOOLEAN):
            continue
        if _explains_nothing(column.name, measure, exclude_calendar):
            continue
        if 2 <= column.unique_count <= MAX_DIMENSION_VALUES:
            usable.append(column.name)

    for name in frame.columns:
        if name in usable or name == measure:
            continue
        if _explains_nothing(name, measure, exclude_calendar):
            continue
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(
            series
        ):
            continue
        if 2 <= series.nunique() <= MAX_DIMENSION_VALUES:
            usable.append(str(name))
    return usable


def _breakdown(
    period_rows: pd.DataFrame,
    baseline_rows: pd.DataFrame,
    column: str,
    measure: str,
    total_change: float,
) -> DimensionBreakdown | None:
    """Split a movement across the levels of one dimension."""
    if column not in period_rows.columns or not total_change:
        return None

    def totals(frame: pd.DataFrame) -> pd.Series:
        values = pd.to_numeric(frame[measure], errors="coerce")
        return values.groupby(frame[column].astype(str)).sum()

    after = totals(period_rows)
    before = totals(baseline_rows)
    levels = sorted(set(after.index) | set(before.index))
    if len(levels) < 2:
        return None

    contributions: list[Contribution] = []
    for level in levels:
        end = float(after.get(level, 0.0))
        start = float(before.get(level, 0.0))
        change = end - start
        if not change:
            continue
        contributions.append(
            Contribution(
                label=str(level),
                change=change,
                share=change / total_change,
                before=start,
                after=end,
            )
        )
    if not contributions:
        return None

    # Largest absolute mover first: a group that took 400k away matters as much
    # as one that added 400k.
    contributions.sort(key=lambda item: abs(item.change), reverse=True)

    kept = [item for item in contributions if abs(item.share) >= MIN_CONTRIBUTION]
    kept = kept[:TOP_CONTRIBUTORS]
    remainder = [item for item in contributions if item not in kept]
    if remainder:
        # Always folded in, however small its total. The contributions summing
        # exactly to the movement is the property that makes this trustworthy:
        # a reader who adds up the column and finds a gap will not believe any
        # of the rest of it either.
        total = sum(item.change for item in remainder)
        kept.append(
            Contribution(
                label=f"everything else ({len(remainder)} groups)",
                change=total,
                share=total / total_change,
                before=sum(item.before for item in remainder),
                after=sum(item.after for item in remainder),
            )
        )

    concentration = abs(kept[0].share) if kept else 0.0
    return DimensionBreakdown(column=column, contributions=kept, concentration=concentration)


def _period_series(frame: pd.DataFrame, profile: DatasetProfile) -> pd.Series | None:
    for column in profile.columns:
        if column.role is Role.DATETIME and column.name in frame.columns:
            series = frame[column.name]
            if pd.api.types.is_datetime64_any_dtype(series):
                return series
            parsed = try_parse_datetime(series)
            if parsed is not None:
                return parsed
    return None


def explain(
    state: PipelineState,
    *,
    measure: str | None = None,
    period: str | None = None,
    grain: str = "M",
) -> Attribution | None:
    """Explain why one period differs from the rest.

    With no period given, the most extreme one is chosen - which is the one the
    user is looking at when they ask the question.
    """
    frame = state.frame
    if frame is None or frame.empty:
        return None

    from .exploration import resolve_columns

    columns = resolve_columns(frame, state.profile)
    measure = measure or columns.primary_measure
    if not measure or measure not in frame.columns:
        return None

    dates = _period_series(frame, state.profile)
    if dates is None:
        return None

    working = frame.assign(_period=dates.dt.to_period(grain).astype(str))
    working = working[working["_period"].notna()]
    if working.empty:
        return None

    totals = (
        pd.to_numeric(working[measure], errors="coerce")
        .groupby(working["_period"])
        .sum()
        .sort_index()
    )
    if len(totals) < 3:
        return None

    if period is None:
        # The period furthest from typical is the one being asked about.
        average = float(totals.mean())
        period = str((totals - average).abs().idxmax())
    if period not in totals.index:
        return None

    period_rows = working[working["_period"] == period]
    baseline_rows = working[working["_period"] != period]
    if baseline_rows.empty:
        return None

    period_value = float(totals.loc[period])
    other_periods = totals.drop(index=period)
    baseline_value = float(other_periods.mean())
    change = period_value - baseline_value
    if not change:
        return None

    # The baseline is an average period, so each dimension's baseline must be
    # scaled to one period's worth for the contributions to add up.
    scale = 1 / len(other_periods)
    scaled_baseline = baseline_rows.assign(
        **{measure: pd.to_numeric(baseline_rows[measure], errors="coerce") * scale}
    )

    attribution = Attribution(
        measure=measure,
        period_label=str(period),
        baseline_label="an average month" if grain == "M" else "an average period",
        period_value=period_value,
        baseline_value=baseline_value,
    )

    for column in _dimensions(
        working, state.profile, measure, exclude_calendar=True
    ):
        breakdown = _breakdown(period_rows, scaled_baseline, column, measure, change)
        if breakdown is not None:
            attribution.dimensions.append(breakdown)

    attribution.dimensions.sort(key=lambda item: item.concentration, reverse=True)
    return attribution


def explain_group_gap(
    state: PipelineState, group_column: str, measure: str, left: str, right: str
) -> Attribution | None:
    """Explain why one group differs from another, rather than one period.

    Same decomposition, different slice: useful for "why is Cairo ahead of
    Aswan" as opposed to "why was November high".
    """
    frame = state.frame
    if frame is None or group_column not in frame.columns or measure not in frame.columns:
        return None

    groups = frame[group_column].astype(str)
    left_rows = frame[groups == left]
    right_rows = frame[groups == right]
    if left_rows.empty or right_rows.empty:
        return None

    left_value = float(pd.to_numeric(left_rows[measure], errors="coerce").sum())
    right_value = float(pd.to_numeric(right_rows[measure], errors="coerce").sum())
    change = left_value - right_value
    if not change:
        return None

    attribution = Attribution(
        measure=measure,
        period_label=left,
        baseline_label=right,
        period_value=left_value,
        baseline_value=right_value,
    )
    for column in _dimensions(frame, state.profile, measure):
        if column == group_column:
            continue
        breakdown = _breakdown(left_rows, right_rows, column, measure, change)
        if breakdown is not None:
            attribution.dimensions.append(breakdown)

    attribution.dimensions.sort(key=lambda item: item.concentration, reverse=True)
    return attribution
