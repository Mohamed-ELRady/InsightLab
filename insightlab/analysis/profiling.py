"""Dataset profiling and column role detection.

Everything here is deterministic pandas work. The language model is never asked
what type a column is - it is only asked to explain, in business language, what
this module already established. That keeps the numbers trustworthy and keeps
the cost of a run low.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ..core.state import ColumnProfile, DatasetProfile, Role

#: Column names that usually mean "this is a key, not a quantity".
IDENTIFIER_PATTERN = re.compile(
    r"(^|[_\s])(id|ids|key|code|no|num|number|ref|reference|uuid|guid|sku|barcode)([_\s]|$)",
    re.IGNORECASE,
)

#: Column names that usually carry a date even when stored as text.
DATE_NAME_PATTERN = re.compile(
    r"(date|time|day|month|year|timestamp|created|updated|ordered|shipped|due|period)",
    re.IGNORECASE,
)

#: Column names that usually carry money.
MONEY_NAME_PATTERN = re.compile(
    r"(revenue|sales|amount|price|cost|profit|margin|total|value|spend|income|"
    r"expense|charge|fee|payment|discount|tax|salary|budget)",
    re.IGNORECASE,
)

#: Column names that usually carry a count.
QUANTITY_NAME_PATTERN = re.compile(
    r"(quantity|qty|units|count|volume|items|pieces|orders|visits|clicks)",
    re.IGNORECASE,
)

_BOOLEAN_TOKENS = {
    frozenset({"true", "false"}),
    frozenset({"yes", "no"}),
    frozenset({"y", "n"}),
    frozenset({"1", "0"}),
    frozenset({"t", "f"}),
}

#: Below this share of parseable values we do not treat a text column as dates.
DATE_PARSE_THRESHOLD = 0.8

#: Calendar labels. pandas parses "January" and "2024" as dates, but a column of
#: month names is a set of groups to compare, not a timeline to plot, and a year
#: column is a label you cannot meaningfully average.
CALENDAR_LABEL_NAMES = frozenset(
    {"year", "month", "quarter", "week", "day", "month_name", "day_of_week", "weekday"}
)

_MONTH_AND_DAY_WORDS = frozenset(
    {
        "january", "february", "march", "april", "may", "june", "july",
        "august", "september", "october", "november", "december",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept",
        "oct", "nov", "dec",
        "monday", "tuesday", "wednesday", "thursday", "friday", "saturday",
        "sunday", "mon", "tue", "tues", "wed", "thu", "thur", "thurs", "fri",
        "sat", "sun",
        "q1", "q2", "q3", "q4",
    }
)

#: A column with at least this share of distinct values looks like a key.
IDENTIFIER_UNIQUE_RATIO = 0.9

#: Numeric columns with at most this many distinct values may be codes.
LOW_CARDINALITY_NUMERIC = 12


def try_parse_datetime(series: pd.Series) -> pd.Series | None:
    """Parse a text column as dates, or return ``None`` if it clearly is not.

    pandas will happily coerce almost anything, so the result only counts when
    a large majority of the non-empty values actually parsed.
    """
    non_null = series.dropna()
    if non_null.empty:
        return None

    sample = non_null.head(2000).astype(str)
    # A column of bare integers parses as nanosecond timestamps, which is
    # almost never what the user meant.
    if sample.str.fullmatch(r"\d+(\.\d+)?").mean() > 0.5:
        return None

    # "January" and "Monday" parse as dates but are labels for grouping, not
    # points on a timeline.
    if sample.str.strip().str.casefold().isin(_MONTH_AND_DAY_WORDS).mean() > 0.5:
        return None

    parsed = pd.to_datetime(series, errors="coerce", format="mixed")
    success = parsed.notna().sum() / max(len(non_null), 1)
    return parsed if success >= DATE_PARSE_THRESHOLD else None


def _looks_boolean(series: pd.Series) -> bool:
    values = series.dropna().unique()
    if len(values) == 0 or len(values) > 2:
        return False
    tokens = frozenset(str(value).strip().casefold() for value in values)
    if tokens in _BOOLEAN_TOKENS:
        return True
    return any(tokens <= candidate for candidate in _BOOLEAN_TOKENS)


def detect_role(series: pd.Series, name: str) -> Role:
    """Work out what a column means in business terms."""
    non_null = series.dropna()
    if non_null.empty:
        return Role.UNKNOWN

    unique_count = int(non_null.nunique())
    if unique_count == 1:
        return Role.CONSTANT

    # A calendar part is a label whatever it is stored as. Averaging a month
    # number or totalling a year would both be meaningless.
    if name.casefold() in CALENDAR_LABEL_NAMES:
        return Role.CATEGORY

    if pd.api.types.is_bool_dtype(series) or _looks_boolean(series):
        return Role.BOOLEAN

    if pd.api.types.is_datetime64_any_dtype(series):
        return Role.DATETIME

    unique_ratio = unique_count / len(non_null)

    if pd.api.types.is_numeric_dtype(series):
        # An integer key column is numeric but is not a quantity: averaging an
        # invoice number is meaningless, so catch it by name and uniqueness.
        if IDENTIFIER_PATTERN.search(name) and unique_ratio >= IDENTIFIER_UNIQUE_RATIO:
            return Role.IDENTIFIER
        if (
            IDENTIFIER_PATTERN.search(name)
            and unique_count <= LOW_CARDINALITY_NUMERIC
            and not MONEY_NAME_PATTERN.search(name)
            and not QUANTITY_NAME_PATTERN.search(name)
        ):
            return Role.CATEGORY
        return Role.MEASURE

    # Text-like from here on.
    if DATE_NAME_PATTERN.search(name) and try_parse_datetime(series) is not None:
        return Role.DATETIME

    if IDENTIFIER_PATTERN.search(name) and unique_ratio >= IDENTIFIER_UNIQUE_RATIO:
        return Role.IDENTIFIER

    average_length = non_null.astype(str).str.len().mean()
    if unique_ratio > 0.5 and average_length > 30:
        return Role.TEXT
    if unique_ratio >= IDENTIFIER_UNIQUE_RATIO:
        return Role.IDENTIFIER
    if unique_count <= 60 or unique_ratio < 0.35:
        return Role.CATEGORY
    return Role.TEXT


def measure_flavour(name: str) -> str:
    """Whether a measure reads as money, a count, or something else."""
    if MONEY_NAME_PATTERN.search(name):
        return "money"
    if QUANTITY_NAME_PATTERN.search(name):
        return "quantity"
    return "value"


def _numeric_stats(series: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return {}
    described = clean.describe()
    stats = {
        "min": float(described["min"]),
        "max": float(described["max"]),
        "mean": float(described["mean"]),
        "median": float(clean.median()),
        "std": float(described["std"]) if not np.isnan(described["std"]) else 0.0,
        "sum": float(clean.sum()),
        "zeros": int((clean == 0).sum()),
        "negatives": int((clean < 0).sum()),
    }
    if len(clean) > 2:
        stats["skew"] = float(clean.skew())
    return stats


def _category_stats(series: pd.Series) -> dict[str, Any]:
    counts = series.dropna().astype(str).value_counts()
    if counts.empty:
        return {}
    total = int(counts.sum())
    top = counts.head(8)
    return {
        "top_values": {str(index): int(value) for index, value in top.items()},
        "top_share": round(float(counts.iloc[0]) / total, 4),
        "distinct": int(counts.size),
    }


def _datetime_stats(series: pd.Series) -> dict[str, Any]:
    parsed = series
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        maybe = try_parse_datetime(series)
        if maybe is None:
            return {}
        parsed = maybe
    clean = parsed.dropna()
    if clean.empty:
        return {}
    span = clean.max() - clean.min()
    return {
        "earliest": str(clean.min().date()),
        "latest": str(clean.max().date()),
        "span_days": int(span.days),
    }


def profile_column(series: pd.Series, name: str) -> ColumnProfile:
    """Build the profile for a single column."""
    total = len(series)
    missing = int(series.isna().sum())
    role = detect_role(series, name)

    profile = ColumnProfile(
        name=name,
        dtype=str(series.dtype),
        role=role,
        missing_count=missing,
        missing_rate=missing / total if total else 0.0,
        unique_count=int(series.dropna().nunique()),
        sample_values=list(series.dropna().unique()[:5]),
    )

    if role is Role.MEASURE:
        profile.stats = _numeric_stats(series)
        profile.stats["flavour"] = measure_flavour(name)
    elif role in (Role.CATEGORY, Role.BOOLEAN, Role.IDENTIFIER):
        profile.stats = _category_stats(series)
    elif role is Role.DATETIME:
        profile.stats = _datetime_stats(series)
    elif role is Role.CONSTANT:
        values = series.dropna().unique()
        profile.stats = {"value": str(values[0]) if len(values) else ""}

    profile.note = _column_note(profile)
    return profile


def _column_note(profile: ColumnProfile) -> str:
    """One plain sentence about anything unusual in this column."""
    notes: list[str] = []
    if profile.missing_rate >= 0.5:
        notes.append(
            f"over half the rows have no value ({profile.missing_rate:.0%} empty)"
        )
    elif profile.missing_rate >= 0.1:
        notes.append(f"{profile.missing_rate:.0%} of rows are empty")

    if profile.role is Role.CONSTANT:
        notes.append("every row holds the same value, so it cannot explain anything")

    if profile.role is Role.MEASURE:
        signed_measure = any(
            token in profile.name.casefold()
            for token in (
                "latitude", "longitude", "temperature", "temp", "change",
                "delta", "difference", "deviation", "balance", "profit",
                "loss", "elevation", "altitude", "coordinate",
            )
        )
        if profile.stats.get("negatives", 0) and not signed_measure:
            notes.append(f"{profile.stats['negatives']} rows hold a negative number")
        if abs(profile.stats.get("skew", 0.0)) > 2:
            notes.append("a few very large values pull the average up")

    if profile.role is Role.CATEGORY and profile.stats.get("top_share", 0) > 0.9:
        notes.append("almost every row falls into a single group")

    return "; ".join(notes)


def profile_dataset(frame: pd.DataFrame) -> DatasetProfile:
    """Build the full profile for a dataframe."""
    profile = DatasetProfile(
        row_count=int(len(frame)),
        column_count=int(frame.shape[1]),
        duplicate_rows=int(frame.duplicated().sum()),
        memory_mb=float(frame.memory_usage(deep=True).sum()) / (1024 * 1024),
        columns=[profile_column(frame[name], str(name)) for name in frame.columns],
    )
    profile.summary = describe_shape(profile)
    return profile


def ensure_profile(state) -> DatasetProfile:
    """Return the current state's profile, rebuilding only after data changed.

    Keeping this helper here avoids making :mod:`core.state` depend on the
    analysis package.  ``set_frame`` is the single invalidation point, so reuse
    never serves statistics from an older dataframe.
    """
    if state.profile_is_current:
        return state.profile
    return state.set_profile(profile_dataset(state.frame))


def describe_shape(profile: DatasetProfile) -> str:
    """A single factual sentence about the dataset, used as a prompt anchor."""
    role_counts: dict[str, int] = {}
    for column in profile.columns:
        role_counts[column.role.value] = role_counts.get(column.role.value, 0) + 1
    parts = [f"{count} {role}" for role, count in sorted(role_counts.items())]
    return (
        f"{profile.row_count:,} rows and {profile.column_count} columns "
        f"({', '.join(parts)})."
    )


def frame_digest(frame: pd.DataFrame, profile: DatasetProfile, *, rows: int = 5) -> str:
    """A compact text rendering of the dataset for use inside a prompt.

    Model context is finite and user data can be enormous, so this sends the
    shape, the per-column facts and a handful of example rows rather than the
    data itself.
    """
    lines = [describe_shape(profile), "", "Columns:"]
    for column in profile.columns:
        detail = f"- {column.name} ({column.role.value}, {column.dtype})"
        if column.missing_rate:
            detail += f", {column.missing_rate:.0%} empty"
        if column.role is Role.MEASURE and column.stats:
            detail += (
                f", ranges {column.stats.get('min', 0):,.2f}"
                f" to {column.stats.get('max', 0):,.2f}"
            )
        elif column.role is Role.CATEGORY and column.stats.get("top_values"):
            top = list(column.stats["top_values"])[:4]
            detail += f", e.g. {', '.join(top)}"
        elif column.role is Role.DATETIME and column.stats:
            detail += (
                f", {column.stats.get('earliest')} to {column.stats.get('latest')}"
            )
        lines.append(detail)

    lines.append("")
    lines.append(f"First {rows} rows:")
    lines.append(frame.head(rows).to_csv(index=False).strip())
    return "\n".join(lines)
