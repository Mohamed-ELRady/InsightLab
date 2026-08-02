"""A closed vocabulary for asking the data a question.

The obvious way to build "ask anything" is to let a model write pandas or SQL
and run it. That is also the wrong way: it puts arbitrary code execution in the
product, and it moves the calculation into the model, which is exactly what the
rest of this codebase refuses to do.

Instead a question becomes a :class:`QueryPlan` — a small, fixed set of fields
describing *what* to compute. The plan is validated against the real schema
before anything runs, and then executed by ordinary pandas. A plan that names a
column which does not exist fails as a plan, with a message the user can read,
rather than as a traceback.

The vocabulary is deliberately narrow. It covers the questions a business owner
actually asks — how much, of what, broken down by what, over what period, top
few — and nothing else.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

from ..core.state import DatasetProfile, Role
from .profiling import try_parse_datetime

Aggregation = Literal["sum", "mean", "median", "count", "min", "max", "nunique"]
Grain = Literal["day", "week", "month", "quarter", "year"]

AGGREGATIONS: tuple[str, ...] = (
    "sum", "mean", "median", "count", "min", "max", "nunique",
)

GRAINS: dict[str, str] = {
    "day": "D",
    "week": "W",
    "month": "M",
    "quarter": "Q",
    "year": "Y",
}

#: Comparison operators a filter may use.
OPERATORS: tuple[str, ...] = ("=", "!=", ">", ">=", "<", "<=", "in", "not in", "contains")

#: Rows returned unless the plan asks for fewer.
DEFAULT_LIMIT = 20

#: Hard ceiling. A chat answer is read, not exported.
MAX_LIMIT = 200


class PlanError(ValueError):
    """The plan cannot be run against this data, and why."""


@dataclass
class Filter:
    column: str
    operator: str
    value: Any

    def describe(self) -> str:
        readable = {"=": "is", "!=": "is not", "in": "is one of",
                    "not in": "is not one of", "contains": "contains"}
        word = readable.get(self.operator, self.operator)
        return f"{self.column.replace('_', ' ')} {word} {self.value}"


@dataclass
class QueryPlan:
    """What to compute. Never how."""

    measure: str = ""
    aggregation: Aggregation = "sum"
    group_by: list[str] = field(default_factory=list)
    time_grain: Grain | None = None
    filters: list[Filter] = field(default_factory=list)
    sort_descending: bool = True
    limit: int = DEFAULT_LIMIT
    question: str = ""

    def describe(self) -> str:
        """What this plan will work out, in the user's language.

        Shown alongside the answer so the user can see we understood the
        question, and correct us when we did not.
        """
        measure = self.measure.replace("_", " ") if self.measure else "records"
        word = {
            "sum": "Total", "mean": "Average", "median": "Typical",
            "count": "Number of", "min": "Lowest", "max": "Highest",
            "nunique": "Distinct count of",
        }.get(self.aggregation, self.aggregation.title())

        parts = [f"{word} {measure}"]
        if self.time_grain:
            parts.append(f"per {self.time_grain}")
        if self.group_by:
            parts.append("by " + " and ".join(name.replace("_", " ") for name in self.group_by))
        if self.filters:
            parts.append("where " + " and ".join(item.describe() for item in self.filters))
        return " ".join(parts) + "."

    def to_dict(self) -> dict[str, Any]:
        return {
            "measure": self.measure,
            "aggregation": self.aggregation,
            "group_by": self.group_by,
            "time_grain": self.time_grain,
            "filters": [
                {"column": item.column, "operator": item.operator, "value": item.value}
                for item in self.filters
            ],
            "sort_descending": self.sort_descending,
            "limit": self.limit,
            "question": self.question,
        }


@dataclass
class QueryResult:
    """What running a plan produced."""

    plan: QueryPlan
    table: pd.DataFrame
    headline: str
    row_count: int
    chart_kind: str = ""

    @property
    def is_single_value(self) -> bool:
        return self.table.shape == (1, 1)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _schema(frame: pd.DataFrame, profile: DatasetProfile) -> dict[str, str]:
    """Column name to role, for everything actually present in the frame."""
    roles = {
        column.name: column.role.value
        for column in profile.columns
        if column.name in frame.columns
    }
    # Feature-engineered columns may not be in the profile yet.
    for name in frame.columns:
        if name not in roles:
            series = frame[name]
            if pd.api.types.is_numeric_dtype(series):
                roles[str(name)] = Role.MEASURE.value
            elif pd.api.types.is_datetime64_any_dtype(series):
                roles[str(name)] = Role.DATETIME.value
            else:
                roles[str(name)] = Role.CATEGORY.value
    return roles


def _resolve(name: str, schema: dict[str, str]) -> str | None:
    """Match a column name a model produced against the real schema.

    Models drop underscores, change case and pluralise. Rejecting "Product
    Category" when the column is `product_category` would be pedantry, not
    safety, so the match is lenient about shape and strict about identity.
    """
    if name in schema:
        return name
    wanted = name.strip().casefold().replace(" ", "").replace("_", "")
    for candidate in schema:
        if candidate.casefold().replace(" ", "").replace("_", "") == wanted:
            return candidate
    return None


def validate(
    raw: dict[str, Any], frame: pd.DataFrame, profile: DatasetProfile
) -> QueryPlan:
    """Turn a proposed plan into one that is guaranteed to run.

    Raises :class:`PlanError` with a readable message when it cannot, which is
    shown to the user rather than logged and swallowed.
    """
    if not isinstance(raw, dict):
        raise PlanError("The question could not be turned into something to calculate.")

    schema = _schema(frame, profile)

    aggregation = str(raw.get("aggregation", "sum")).casefold()
    if aggregation not in AGGREGATIONS:
        aggregation = "sum"

    measure = str(raw.get("measure") or "").strip()
    resolved_measure = _resolve(measure, schema) if measure else None
    if measure and resolved_measure is None:
        raise PlanError(
            f'There is no column called "{measure}" in your data. The columns '
            f"available are: {', '.join(sorted(schema))}."
        )
    if resolved_measure and aggregation in ("sum", "mean", "median", "min", "max"):
        if schema[resolved_measure] not in (Role.MEASURE.value,):
            if not pd.api.types.is_numeric_dtype(frame[resolved_measure]):
                raise PlanError(
                    f"{resolved_measure.replace('_', ' ')} does not hold numbers, "
                    f"so it cannot be totalled or averaged. It can be counted."
                )

    group_by: list[str] = []
    for name in raw.get("group_by") or []:
        resolved = _resolve(str(name), schema)
        if resolved is None:
            raise PlanError(
                f'There is no column called "{name}" to group by. The columns '
                f"available are: {', '.join(sorted(schema))}."
            )
        if resolved not in group_by:
            group_by.append(resolved)
    # More than two breakdowns produces a table nobody can read.
    group_by = group_by[:2]

    grain = raw.get("time_grain")
    time_grain = str(grain).casefold() if grain else None
    if time_grain not in GRAINS:
        time_grain = None
    if time_grain and not any(
        role == Role.DATETIME.value for role in schema.values()
    ):
        time_grain = None

    filters: list[Filter] = []
    for item in raw.get("filters") or []:
        if not isinstance(item, dict):
            continue
        resolved = _resolve(str(item.get("column", "")), schema)
        if resolved is None:
            raise PlanError(
                f'The condition mentions "{item.get("column")}", which is not a '
                "column in your data."
            )
        operator = str(item.get("operator", "=")).casefold().strip()
        if operator not in OPERATORS:
            operator = "="
        filters.append(Filter(resolved, operator, item.get("value")))

    try:
        limit = int(raw.get("limit") or DEFAULT_LIMIT)
    except (TypeError, ValueError):
        limit = DEFAULT_LIMIT

    return QueryPlan(
        measure=resolved_measure or "",
        aggregation=aggregation,  # type: ignore[arg-type]
        group_by=group_by,
        time_grain=time_grain,  # type: ignore[arg-type]
        filters=filters,
        sort_descending=bool(raw.get("sort_descending", True)),
        limit=max(1, min(limit, MAX_LIMIT)),
        question=str(raw.get("question") or ""),
    )


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _apply_filters(frame: pd.DataFrame, filters: list[Filter]) -> pd.DataFrame:
    result = frame
    for item in filters:
        if item.column not in result.columns:
            continue
        series = result[item.column]

        if item.operator in (">", ">=", "<", "<="):
            numbers = pd.to_numeric(series, errors="coerce")
            try:
                threshold = float(item.value)
            except (TypeError, ValueError):
                continue
            mask = {
                ">": numbers > threshold,
                ">=": numbers >= threshold,
                "<": numbers < threshold,
                "<=": numbers <= threshold,
            }[item.operator]
        elif item.operator in ("in", "not in"):
            wanted = item.value if isinstance(item.value, list) else [item.value]
            wanted = {str(value).casefold() for value in wanted}
            inside = series.astype(str).str.casefold().isin(wanted)
            mask = inside if item.operator == "in" else ~inside
        elif item.operator == "contains":
            mask = series.astype(str).str.contains(str(item.value), case=False, na=False)
        else:
            equal = series.astype(str).str.casefold() == str(item.value).casefold()
            mask = equal if item.operator == "=" else ~equal

        result = result[mask.fillna(False)]
    return result


def _time_column(frame: pd.DataFrame, profile: DatasetProfile) -> str | None:
    for column in profile.columns:
        if column.role is Role.DATETIME and column.name in frame.columns:
            return column.name
    for name in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[name]):
            return str(name)
    return None


def run(
    plan: QueryPlan, frame: pd.DataFrame, profile: DatasetProfile
) -> QueryResult:
    """Execute a validated plan with pandas."""
    working = _apply_filters(frame, plan.filters)
    if working.empty:
        return QueryResult(
            plan=plan,
            table=pd.DataFrame(),
            headline="No records match those conditions.",
            row_count=0,
        )

    keys: list[str] = []
    if plan.time_grain:
        time_column = _time_column(working, profile)
        if time_column:
            series = working[time_column]
            if not pd.api.types.is_datetime64_any_dtype(series):
                parsed = try_parse_datetime(series)
                series = parsed if parsed is not None else pd.to_datetime(
                    series, errors="coerce"
                )
            working = working.assign(
                **{"period": series.dt.to_period(GRAINS[plan.time_grain]).astype(str)}
            )
            keys.append("period")
    keys.extend(plan.group_by)

    measure = plan.measure or (keys[0] if keys else working.columns[0])
    label = _value_label(plan, measure)

    if not keys:
        value = _aggregate_series(working, measure, plan.aggregation)
        table = pd.DataFrame({label: [value]})
        return QueryResult(
            plan=plan,
            table=table,
            headline=f"{label}: {_format(value)}",
            row_count=len(working),
        )

    if plan.aggregation == "count":
        grouped = working.groupby(keys, observed=True).size().reset_index(name=label)
    elif plan.aggregation == "nunique":
        grouped = (
            working.groupby(keys, observed=True)[measure].nunique().reset_index(name=label)
        )
    else:
        numeric = pd.to_numeric(working[measure], errors="coerce")
        grouped = (
            working.assign(**{"_value": numeric})
            .groupby(keys, observed=True)["_value"]
            .agg(plan.aggregation)
            .reset_index()
            .rename(columns={"_value": label})
        )

    # A time series reads in time order; everything else reads biggest first.
    if "period" in keys:
        grouped = grouped.sort_values("period")
    else:
        grouped = grouped.sort_values(label, ascending=not plan.sort_descending)

    total_rows = len(grouped)

    # The headline is computed from the whole result, then the table is cut for
    # display. Describing a trend from a truncated window would state a change
    # across "the period" that is not the change across the period.
    headline = _headline(grouped, keys, label, plan)

    if total_rows > plan.limit:
        # For a time series the recent end is the interesting one; for a
        # ranking it is the top.
        grouped = (
            grouped.tail(plan.limit) if "period" in keys else grouped.head(plan.limit)
        )
        headline += (
            f" Showing {plan.limit} of {total_rows:,} rows."
        )
    grouped = grouped.reset_index(drop=True)

    if not grouped.empty and label in grouped.columns:
        grouped[label] = pd.to_numeric(grouped[label], errors="coerce").round(2)
    return QueryResult(
        plan=plan,
        table=grouped,
        headline=headline,
        row_count=total_rows,
        chart_kind="line" if "period" in keys else "bar",
    )


def _aggregate_series(frame: pd.DataFrame, column: str, how: str) -> float:
    if how == "count":
        return float(len(frame))
    if column not in frame.columns:
        return float(len(frame))
    if how == "nunique":
        return float(frame[column].nunique())
    values = pd.to_numeric(frame[column], errors="coerce").dropna()
    if values.empty:
        return 0.0
    return float(getattr(values, how)())


def _value_label(plan: QueryPlan, measure: str) -> str:
    readable = measure.replace("_", " ")
    return {
        "sum": f"Total {readable}",
        "mean": f"Average {readable}",
        "median": f"Typical {readable}",
        "count": "Number of records",
        "min": f"Lowest {readable}",
        "max": f"Highest {readable}",
        "nunique": f"Distinct {readable}",
    }.get(plan.aggregation, readable)


def _format(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if magnitude >= 1_000:
        return f"{value:,.0f}"
    # A count of 315 is not "315.00". Only show decimals when there are any.
    if float(value).is_integer():
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _headline(
    table: pd.DataFrame, keys: list[str], label: str, plan: QueryPlan
) -> str:
    """One factual sentence about the result, computed rather than written."""
    if table.empty:
        return "Nothing matched."

    top = table.iloc[0]
    name = " / ".join(str(top[key]) for key in keys)
    value = top[label] if label in table.columns else None

    if "period" in keys and len(table) > 1:
        first, last = table.iloc[0], table.iloc[-1]
        start, end = float(first[label]), float(last[label])
        change = ((end - start) / start * 100) if start else 0.0
        direction = "up" if change > 2 else ("down" if change < -2 else "flat")
        return (
            f"{label} runs from {_format(start)} in {first[keys[0]]} to "
            f"{_format(end)} in {last[keys[0]]} — {direction} "
            f"{abs(change):.0f}% across the period."
        )

    total = pd.to_numeric(table[label], errors="coerce").sum() if label in table else 0
    share = (float(value) / float(total) * 100) if total and value is not None else 0.0
    tail = f", which is {share:.0f}% of the {len(table)} shown" if share else ""
    return f"{name} is highest at {_format(float(value))}{tail}."
