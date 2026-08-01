"""Deterministic cleaning operations.

Each function takes a dataframe and returns a new one plus a plain-language
description of what it did. The description is what ends up in the cleaning log
the user is handed at the end, so it is written for them, not for a developer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ..core.state import DatasetProfile, Role
from .profiling import try_parse_datetime

#: Rows beyond this many IQRs from the quartiles are flagged for review.
IQR_MULTIPLIER = 1.5

#: A column is only worth an outlier question if it has at least this many rows.
MIN_ROWS_FOR_OUTLIERS = 30

#: Above this share of flagged rows the "outliers" are really the distribution.
MAX_OUTLIER_SHARE = 0.15


@dataclass
class Operation:
    """The result of one cleaning operation."""

    frame: pd.DataFrame
    description: str
    rows_removed: int = 0
    columns_removed: int = 0
    cells_changed: int = 0
    details: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Duplicates
# ---------------------------------------------------------------------------


@dataclass
class DuplicateReport:
    total: int
    groups: int
    example: pd.DataFrame
    subset: list[str] | None = None

    @property
    def found(self) -> bool:
        return self.total > 0


def find_duplicates(
    frame: pd.DataFrame, subset: list[str] | None = None, *, examples: int = 12
) -> DuplicateReport:
    """Locate rows that repeat, either wholly or on a chosen set of columns."""
    marked = frame.duplicated(subset=subset, keep=False)
    total = int(frame.duplicated(subset=subset, keep="first").sum())
    duplicated_rows = frame[marked]
    groups = 0
    if not duplicated_rows.empty:
        key = subset or list(frame.columns)
        groups = int(duplicated_rows.groupby(key, dropna=False).ngroups)
    return DuplicateReport(
        total=total,
        groups=groups,
        example=duplicated_rows.head(examples).copy(),
        subset=subset,
    )


def drop_duplicates(frame: pd.DataFrame, subset: list[str] | None = None) -> Operation:
    result = frame.drop_duplicates(subset=subset, keep="first").reset_index(drop=True)
    removed = len(frame) - len(result)
    scope = "identical rows" if not subset else f"rows repeating on {', '.join(subset)}"
    return Operation(
        frame=result,
        description=f"Removed {removed:,} duplicate {scope}, keeping the first of each.",
        rows_removed=removed,
    )


def merge_duplicates(
    frame: pd.DataFrame, profile: DatasetProfile, subset: list[str] | None = None
) -> Operation:
    """Collapse repeated rows into one, adding up the quantities.

    Used when a repeat is a real second transaction that was exported twice
    under one key: the totals should survive the merge even though the row
    count drops.
    """
    keys = subset or [
        column.name
        for column in profile.columns
        if column.role is not Role.MEASURE and column.name in frame.columns
    ]
    keys = [key for key in keys if key in frame.columns]
    if not keys:
        return drop_duplicates(frame, subset)

    measures = [
        column.name
        for column in profile.columns
        if column.role is Role.MEASURE and column.name in frame.columns
    ]
    if not measures:
        return drop_duplicates(frame, subset)

    aggregation: dict[str, Any] = {name: "sum" for name in measures}
    for name in frame.columns:
        if name not in keys and name not in aggregation:
            aggregation[name] = "first"

    result = (
        frame.groupby(keys, dropna=False, as_index=False)
        .agg(aggregation)
        .reset_index(drop=True)
    )
    removed = len(frame) - len(result)
    return Operation(
        frame=result,
        description=(
            f"Merged {removed:,} repeated rows into their originals, adding up "
            f"{', '.join(measures)} so the totals stay correct."
        ),
        rows_removed=removed,
        details={"keys": keys, "summed": measures},
    )


# ---------------------------------------------------------------------------
# Outliers
# ---------------------------------------------------------------------------


@dataclass
class OutlierReport:
    column: str
    count: int
    share: float
    lower_bound: float
    upper_bound: float
    low_values: list[float]
    high_values: list[float]
    rows: pd.DataFrame

    @property
    def found(self) -> bool:
        return self.count > 0


def find_outliers(
    frame: pd.DataFrame, column: str, *, examples: int = 10
) -> OutlierReport | None:
    """Flag unusually small or large values in one numeric column."""
    if column not in frame.columns:
        return None

    values = pd.to_numeric(frame[column], errors="coerce")
    clean = values.dropna()
    if len(clean) < MIN_ROWS_FOR_OUTLIERS or clean.nunique() < 5:
        return None

    first_quartile = float(clean.quantile(0.25))
    third_quartile = float(clean.quantile(0.75))
    spread = third_quartile - first_quartile
    if spread <= 0:
        return None

    lower = first_quartile - IQR_MULTIPLIER * spread
    upper = third_quartile + IQR_MULTIPLIER * spread
    mask = (values < lower) | (values > upper)
    count = int(mask.sum())
    if count == 0:
        return None

    flagged = frame[mask.fillna(False)]
    flagged_values = values[mask.fillna(False)].dropna()
    return OutlierReport(
        column=column,
        count=count,
        share=count / len(clean),
        lower_bound=lower,
        upper_bound=upper,
        low_values=sorted(float(v) for v in flagged_values[flagged_values < lower])[:examples],
        high_values=sorted(
            (float(v) for v in flagged_values[flagged_values > upper]), reverse=True
        )[:examples],
        rows=flagged.head(examples).copy(),
    )


def outlier_candidates(frame: pd.DataFrame, profile: DatasetProfile) -> list[OutlierReport]:
    """Outlier reports worth asking the user about, most severe first.

    Columns where a sixth of the rows are flagged are skipped: that is a skewed
    distribution, not a data problem, and asking about it wastes the user's
    attention.
    """
    reports = []
    for column in profile.columns:
        if column.role is not Role.MEASURE:
            continue
        report = find_outliers(frame, column.name)
        if report is not None and report.share <= MAX_OUTLIER_SHARE:
            reports.append(report)
    return sorted(reports, key=lambda item: item.share, reverse=True)


def remove_outliers(frame: pd.DataFrame, report: OutlierReport) -> Operation:
    values = pd.to_numeric(frame[report.column], errors="coerce")
    keep = ~((values < report.lower_bound) | (values > report.upper_bound)).fillna(False)
    result = frame[keep].reset_index(drop=True)
    removed = len(frame) - len(result)
    return Operation(
        frame=result,
        description=(
            f"Removed {removed:,} rows where {report.column} fell outside the "
            f"normal range, treating them as data-entry errors."
        ),
        rows_removed=removed,
    )


def cap_outliers(frame: pd.DataFrame, report: OutlierReport) -> Operation:
    """Pull extreme values back to the edge of the normal range.

    Keeps the row - and so the customer, the region and the date - while
    stopping one number from dominating every average.
    """
    result = frame.copy()
    values = pd.to_numeric(result[report.column], errors="coerce")
    changed = int(
        ((values < report.lower_bound) | (values > report.upper_bound)).fillna(False).sum()
    )
    result[report.column] = values.clip(report.lower_bound, report.upper_bound)
    return Operation(
        frame=result,
        description=(
            f"Capped {changed:,} extreme values in {report.column} to the edge of "
            f"the normal range, keeping the rows but stopping them dominating averages."
        ),
        cells_changed=changed,
    )


def flag_outliers(frame: pd.DataFrame, report: OutlierReport) -> Operation:
    """Mark the rows without touching the numbers."""
    result = frame.copy()
    values = pd.to_numeric(result[report.column], errors="coerce")
    flag_name = f"{report.column}_is_unusual"
    result[flag_name] = (
        (values < report.lower_bound) | (values > report.upper_bound)
    ).fillna(False)
    return Operation(
        frame=result,
        description=(
            f"Left the values in {report.column} untouched and added a "
            f"{flag_name} marker so they can be looked at separately."
        ),
        details={"flag_column": flag_name},
    )


# ---------------------------------------------------------------------------
# Columns
# ---------------------------------------------------------------------------


def drop_column(frame: pd.DataFrame, column: str, reason: str = "") -> Operation:
    if column not in frame.columns:
        return Operation(frame=frame, description=f"{column} was already gone.")
    result = frame.drop(columns=[column])
    tail = f" {reason}" if reason else ""
    return Operation(
        frame=result,
        description=f"Dropped the column {column}.{tail}",
        columns_removed=1,
    )


def rename_column(frame: pd.DataFrame, column: str, new_name: str) -> Operation:
    if column not in frame.columns or not new_name or new_name == column:
        return Operation(frame=frame, description=f"Left {column} named as it was.")
    result = frame.rename(columns={column: new_name})
    return Operation(
        frame=result,
        description=f"Renamed {column} to {new_name}.",
        details={"from": column, "to": new_name},
    )


def convert_column(frame: pd.DataFrame, column: str, target: str) -> Operation:
    """Convert a column to ``number``, ``date``, ``text`` or ``category``."""
    if column not in frame.columns:
        return Operation(frame=frame, description=f"{column} is not in the data.")

    result = frame.copy()
    original = result[column]
    before_missing = int(original.isna().sum())

    if target == "number":
        converted = pd.to_numeric(
            original.astype(str).str.replace(r"[^\d.\-eE]", "", regex=True),
            errors="coerce",
        )
    elif target == "date":
        converted = try_parse_datetime(original)
        if converted is None:
            converted = pd.to_datetime(original, errors="coerce", format="mixed")
    elif target == "category":
        converted = original.astype("category")
    elif target == "text":
        converted = original.astype(str)
    else:
        return Operation(frame=frame, description=f"Unknown target type {target!r}.")

    result[column] = converted
    new_missing = int(converted.isna().sum())
    lost = max(0, new_missing - before_missing)
    note = (
        f" {lost:,} values could not be read as a {target} and are now empty."
        if lost
        else ""
    )
    return Operation(
        frame=result,
        description=f"Converted {column} to a {target}.{note}",
        cells_changed=int(len(result)) - before_missing,
        details={"target": target, "unreadable": lost},
    )


def fill_missing(frame: pd.DataFrame, column: str, strategy: str) -> Operation:
    """Fill blanks with ``median``, ``mean``, ``mode``, ``zero`` or a label."""
    if column not in frame.columns:
        return Operation(frame=frame, description=f"{column} is not in the data.")

    result = frame.copy()
    series = result[column]
    blanks = int(series.isna().sum())
    if blanks == 0:
        return Operation(frame=frame, description=f"{column} had no blanks to fill.")

    if strategy in {"median", "mean"}:
        numeric = pd.to_numeric(series, errors="coerce")
        value = float(numeric.median() if strategy == "median" else numeric.mean())
        result[column] = numeric.fillna(value)
        wording = f"the {strategy} ({value:,.2f})"
    elif strategy == "zero":
        result[column] = pd.to_numeric(series, errors="coerce").fillna(0)
        wording = "zero"
    elif strategy == "mode":
        modes = series.dropna().mode()
        if modes.empty:
            return Operation(frame=frame, description=f"{column} has no value to copy.")
        value = modes.iloc[0]
        result[column] = series.fillna(value)
        wording = f"the most common value ({value})"
    else:
        label = strategy if strategy else "Unknown"
        result[column] = series.astype(object).fillna(label)
        wording = f'the label "{label}"'

    return Operation(
        frame=result,
        description=f"Filled {blanks:,} blank values in {column} with {wording}.",
        cells_changed=blanks,
        details={"strategy": strategy, "filled": blanks},
    )


def drop_missing_rows(frame: pd.DataFrame, column: str) -> Operation:
    if column not in frame.columns:
        return Operation(frame=frame, description=f"{column} is not in the data.")
    result = frame[frame[column].notna()].reset_index(drop=True)
    removed = len(frame) - len(result)
    return Operation(
        frame=result,
        description=f"Removed {removed:,} rows that had no value for {column}.",
        rows_removed=removed,
    )


def filter_rows(frame: pd.DataFrame, column: str, exclude: list[Any]) -> Operation:
    """Drop rows whose value in ``column`` is one the user wants ignored."""
    if column not in frame.columns or not exclude:
        return Operation(frame=frame, description="Nothing was filtered out.")
    wanted = {str(value).casefold() for value in exclude}
    keep = ~frame[column].astype(str).str.casefold().isin(wanted)
    result = frame[keep].reset_index(drop=True)
    removed = len(frame) - len(result)
    listed = ", ".join(str(value) for value in exclude)
    return Operation(
        frame=result,
        description=f"Removed {removed:,} rows where {column} was {listed}.",
        rows_removed=removed,
        details={"column": column, "excluded": list(exclude)},
    )


def suggest_column_action(column, row_count: int) -> tuple[str, str, dict[str, Any]]:
    """Recommend what to do with one column.

    Returns ``(action, reason, payload)`` where action is one of ``keep``,
    ``drop``, ``fill``, ``convert``.
    """
    if column.role is Role.CONSTANT:
        return (
            "drop",
            "every row holds the same value, so it cannot explain any difference "
            "between rows",
            {"action": "drop"},
        )

    if column.missing_rate >= 0.6:
        return (
            "drop",
            f"{column.missing_rate:.0%} of the rows are empty, which is too few to "
            "draw a conclusion from",
            {"action": "drop"},
        )

    if column.role is Role.MEASURE and 0 < column.missing_rate < 0.6:
        return (
            "fill",
            f"{column.missing_rate:.0%} of rows are blank; filling them with the "
            "middle value keeps those rows usable in totals and averages",
            {"action": "fill", "strategy": "median"},
        )

    if column.role is Role.CATEGORY and 0 < column.missing_rate < 0.6:
        return (
            "fill",
            f"{column.missing_rate:.0%} of rows are blank; grouping them under one "
            'label keeps them visible instead of silently dropping them',
            {"action": "fill", "strategy": "Unknown"},
        )

    if column.role is Role.TEXT and column.unique_count > row_count * 0.8:
        return (
            "drop",
            "almost every row has different free text, which cannot be grouped or "
            "counted",
            {"action": "drop"},
        )

    return ("keep", "the values look consistent and usable as they are", {"action": "keep"})


def apply_column_action(
    frame: pd.DataFrame, column: str, payload: dict[str, Any]
) -> Operation:
    """Run whichever column action a payload describes."""
    action = payload.get("action", "keep")
    if action == "drop":
        return drop_column(frame, column)
    if action == "rename":
        return rename_column(frame, column, str(payload.get("new_name", "")))
    if action == "convert":
        return convert_column(frame, column, str(payload.get("target", "text")))
    if action == "fill":
        return fill_missing(frame, column, str(payload.get("strategy", "median")))
    if action == "drop_missing":
        return drop_missing_rows(frame, column)
    return Operation(frame=frame, description=f"Left {column} exactly as it is.")
