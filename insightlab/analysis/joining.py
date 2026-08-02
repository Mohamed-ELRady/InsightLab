"""Putting several files together.

Real businesses export sales, products and customers separately. One file at a
time is a hard ceiling: margin analysis is impossible when cost lives in a
second export, and customer analysis is impossible when the customer table does.

Join keys are proposed from **measured value overlap**, not from name
similarity. Two columns both called ``id`` that share no values are not a join;
``customer_id`` and ``client_ref`` that share 94% of their values are. Names
only break ties.

The dangerous part is not finding the join, it is fan-out. Joining on a column
that is not unique in the second table multiplies rows, and every total after
that is silently wrong by a factor nobody notices. So the row multiplication is
computed before anything is joined, and it is stated in the question.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from ..core.state import DatasetProfile, Role

#: Below this share of matching values, two columns are not the same thing.
MIN_OVERLAP = 0.5

#: Above this, the join is safe enough to recommend without hesitation.
STRONG_OVERLAP = 0.85

#: A column on the fact table with fewer distinct values than this is a
#: category, not a key: joining on it would match almost every row to almost
#: every row.
MIN_KEY_VALUES = 5

#: The lookup side has no such floor. A product table with three categories is
#: a perfectly ordinary lookup, and requiring five distinct values there would
#: make small reference files impossible to attach - which is most of them.
MIN_LOOKUP_VALUES = 2

#: Row growth beyond this is flagged as a warning in the question.
FAN_OUT_WARNING = 1.05


@dataclass
class Table:
    """One loaded file, waiting to be joined."""

    name: str
    frame: pd.DataFrame
    profile: DatasetProfile
    path: Path | None = None

    @property
    def rows(self) -> int:
        return len(self.frame)

    def describe(self) -> str:
        return (
            f"{self.name}: {self.rows:,} rows, {self.frame.shape[1]} columns "
            f"({', '.join(str(name) for name in list(self.frame.columns)[:5])}"
            + (", …" if self.frame.shape[1] > 5 else "")
            + ")"
        )


@dataclass
class JoinCandidate:
    """A proposed relationship between two tables."""

    left: str
    right: str
    left_column: str
    right_column: str
    overlap: float
    right_is_unique: bool
    left_rows: int
    matched_rows: int
    expected_rows: int

    @property
    def fan_out(self) -> float:
        """How much the row count would grow. 1.0 means it would not."""
        return self.expected_rows / self.left_rows if self.left_rows else 1.0

    @property
    def is_safe(self) -> bool:
        return self.right_is_unique and self.fan_out <= FAN_OUT_WARNING

    @property
    def score(self) -> float:
        """How confident we are that this is the intended relationship."""
        confidence = self.overlap
        if self.right_is_unique:
            confidence += 0.25
        if _names_agree(self.left_column, self.right_column):
            confidence += 0.15
        return confidence

    def describe(self) -> str:
        left_readable = self.left_column.replace("_", " ")
        right_readable = self.right_column.replace("_", " ")
        same = left_readable == right_readable

        naming = (
            f"{left_readable}" if same
            else f"{left_readable} in {self.left} and {right_readable} in {self.right}"
        )
        lines = [
            f"Match on {naming}: {self.overlap:.0%} of the values in {self.left} "
            f"are found in {self.right}."
        ]
        if not self.right_is_unique:
            lines.append(
                f"Careful: {right_readable} repeats in {self.right}, so this "
                f"would turn {self.left_rows:,} rows into {self.expected_rows:,} "
                "and every total after it would be inflated."
            )
        elif self.matched_rows < self.left_rows:
            missing = self.left_rows - self.matched_rows
            lines.append(
                f"{missing:,} rows in {self.left} have no match and would keep "
                "their existing columns with the new ones left empty."
            )
        return " ".join(lines)


def _names_agree(left: str, right: str) -> bool:
    normalise = lambda name: name.casefold().replace("_", "").replace(" ", "")  # noqa: E731
    return normalise(left) == normalise(right)


def _key_columns(table: Table, minimum: int = MIN_KEY_VALUES) -> list[str]:
    """Columns that could plausibly identify a row in another table."""
    usable: list[str] = []
    for column in table.profile.columns:
        if column.name not in table.frame.columns:
            continue
        if column.role in (Role.MEASURE, Role.DATETIME, Role.CONSTANT):
            continue
        if column.unique_count < minimum:
            continue
        usable.append(column.name)
    return usable


def _overlap(left: pd.Series, right: pd.Series) -> tuple[float, int]:
    """Share of the left column's values present in the right, and how many rows."""
    left_values = left.dropna().astype(str).str.strip()
    right_values = set(right.dropna().astype(str).str.strip())
    if left_values.empty or not right_values:
        return 0.0, 0

    matched = left_values.isin(right_values)
    distinct = set(left_values.unique())
    if not distinct:
        return 0.0, 0
    share = len(distinct & right_values) / len(distinct)
    return share, int(matched.sum())


def find_candidates(base: Table, other: Table) -> list[JoinCandidate]:
    """Every plausible way two tables relate, best first."""
    candidates: list[JoinCandidate] = []

    for left_name in _key_columns(base):
        for right_name in _key_columns(other, MIN_LOOKUP_VALUES):
            left_series = base.frame[left_name]
            right_series = other.frame[right_name]

            # Comparing a column of names against a column of codes wastes
            # time; require the same broad kind of value.
            if pd.api.types.is_numeric_dtype(left_series) != pd.api.types.is_numeric_dtype(
                right_series
            ):
                continue

            share, matched = _overlap(left_series, right_series)
            if share < MIN_OVERLAP:
                continue

            right_unique = bool(right_series.dropna().is_unique)
            expected = _expected_rows(base.frame, left_name, other.frame, right_name)

            candidates.append(
                JoinCandidate(
                    left=base.name,
                    right=other.name,
                    left_column=left_name,
                    right_column=right_name,
                    overlap=share,
                    right_is_unique=right_unique,
                    left_rows=len(base.frame),
                    matched_rows=matched,
                    expected_rows=expected,
                )
            )

    candidates.sort(key=lambda item: item.score, reverse=True)
    return candidates


def _expected_rows(
    left: pd.DataFrame, left_column: str, right: pd.DataFrame, right_column: str
) -> int:
    """How many rows a left join would produce, computed without doing it.

    Counting first matters: discovering the fan-out by performing the join
    means building the inflated frame in memory before finding out it was
    wrong.
    """
    counts = right[right_column].dropna().astype(str).str.strip().value_counts()
    keys = left[left_column].astype(str).str.strip()
    multiples = keys.map(counts).fillna(1).clip(lower=1)
    return int(multiples.sum())


def choose_base(tables: list[Table]) -> Table:
    """The table the others attach to.

    The fact table - the one with the most rows and the most numbers to total.
    Joining a lookup onto transactions is right; joining transactions onto a
    lookup produces a table where every total is meaningless.
    """

    def weight(table: Table) -> tuple[int, int]:
        measures = sum(
            1 for column in table.profile.columns if column.role is Role.MEASURE
        )
        return (len(table.frame), measures)

    return max(tables, key=weight)


@dataclass
class JoinResult:
    frame: pd.DataFrame
    description: str
    rows_before: int
    rows_after: int
    columns_added: list[str] = field(default_factory=list)

    @property
    def fanned_out(self) -> bool:
        return self.rows_after > self.rows_before


def apply_join(
    base: pd.DataFrame, other: Table, candidate: JoinCandidate
) -> JoinResult:
    """Attach one table to another on a chosen relationship."""
    before = len(base)
    right = other.frame.copy()

    # Both sides are compared as trimmed text, exactly as the overlap was
    # measured. Joining on the raw values would silently miss the pairs the
    # measurement counted.
    left_key = "__join_left"
    right_key = "__join_right"
    working = base.copy()
    working[left_key] = working[candidate.left_column].astype(str).str.strip()
    right[right_key] = right[candidate.right_column].astype(str).str.strip()

    # Columns that exist on both sides would collide; the base table wins,
    # since it is the one the user has been looking at.
    overlapping = [
        name
        for name in right.columns
        if name in working.columns and name != right_key
    ]
    right = right.drop(columns=overlapping)

    merged = working.merge(
        right, how="left", left_on=left_key, right_on=right_key, suffixes=("", "_from_" + other.name)
    ).drop(columns=[left_key, right_key], errors="ignore")

    added = [name for name in merged.columns if name not in base.columns]
    after = len(merged)

    detail = ""
    if after > before:
        detail = (
            f" The row count rose from {before:,} to {after:,} because "
            f"{candidate.right_column} repeats in {other.name}; totals now count "
            "some records more than once."
        )
    elif overlapping:
        detail = (
            f" {len(overlapping)} column(s) already existed and were kept from "
            "the original file."
        )

    return JoinResult(
        frame=merged,
        description=(
            f"Joined {other.name} on {candidate.left_column}, adding "
            f"{len(added)} column(s): {', '.join(added[:6])}"
            + ("…" if len(added) > 6 else "")
            + "."
            + detail
        ),
        rows_before=before,
        rows_after=after,
        columns_added=added,
    )
