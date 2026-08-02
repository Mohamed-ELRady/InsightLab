"""Business facts as data, not just sentences.

A fact stored as free text can be read into a prompt and nothing else. It cannot
be tested against new data, applied automatically, compared with a later
statement, or shown to have expired. Every memory feature is capped by that.

So a fact may also carry a :class:`Claim`: a small structured form saying what
it asserts about which column. The sentence stays - it is what the user reads
and what they wrote - and the structure drives behaviour underneath it.

A claim is only ever a *reading* of what the user said. It is shown back to them
in plain words so they can correct it, and a fact with no claim still works
exactly as it did before.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

import pandas as pd

Kind = Literal["threshold", "peak_period", "exclusion", "target", "ranking"]

#: Comparisons a threshold claim can make.
OPERATORS = {
    ">": "is more than",
    ">=": "is at least",
    "<": "is less than",
    "<=": "is at most",
    "=": "is",
    "!=": "is not",
}


@dataclass
class Claim:
    """The testable part of something the owner told us."""

    kind: Kind
    column: str = ""
    operator: str = ">"
    value: Any = None
    label: str = ""
    #: For a peak claim: the period the owner says is highest, e.g. "November".
    period: str = ""
    #: How the value should be aggregated before testing. Sum for money.
    aggregation: str = "sum"

    def describe(self) -> str:
        """What we understood, in the owner's language, for them to correct."""
        readable = self.column.replace("_", " ")
        if self.kind == "threshold":
            word = OPERATORS.get(self.operator, self.operator)
            label = f" is counted as {self.label}" if self.label else ""
            return f"Any record where {readable} {word} {self.value:,g}{label}."
        if self.kind == "peak_period":
            subject = f" for {readable}" if self.column else ""
            return f"{self.period} is the strongest period{subject}."
        if self.kind == "exclusion":
            return f"Records where {readable} is {self.value} are left out."
        if self.kind == "target":
            word = OPERATORS.get(self.operator, self.operator)
            return f"The target for {readable} {word} {self.value:,g}."
        if self.kind == "ranking":
            return f"{self.value} leads on {readable}."
        return ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "column": self.column,
            "operator": self.operator,
            "value": self.value,
            "label": self.label,
            "period": self.period,
            "aggregation": self.aggregation,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Claim | None":
        if not isinstance(raw, dict) or not raw.get("kind"):
            return None
        try:
            return cls(
                kind=raw["kind"],
                column=str(raw.get("column", "")),
                operator=str(raw.get("operator", ">")),
                value=raw.get("value"),
                label=str(raw.get("label", "")),
                period=str(raw.get("period", "")),
                aggregation=str(raw.get("aggregation", "sum")),
            )
        except (KeyError, TypeError):
            return None


@dataclass
class Test:
    """What the data says about a claim."""

    claim: Claim
    holds: bool
    detail: str
    observed: str = ""
    expected: str = ""
    testable: bool = True
    affected_rows: int = 0

    @property
    def contradicts(self) -> bool:
        return self.testable and not self.holds


#: Month names, for reading a peak claim.
MONTHS = {
    name.casefold(): index
    for index, name in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ],
        start=1,
    )
}

QUARTERS = {"q1": 1, "q2": 2, "q3": 3, "q4": 4}


def evaluate_claim(claim: Claim, frame: pd.DataFrame, profile) -> Test:
    """Check a claim against the data in front of us.

    A claim that cannot be tested on this file is not a contradiction - it is
    simply out of scope here, and saying otherwise would train the user to
    ignore the warnings.
    """
    if claim.kind == "threshold":
        return _test_threshold(claim, frame)
    if claim.kind == "peak_period":
        return _test_peak(claim, frame, profile)
    if claim.kind == "exclusion":
        return _test_exclusion(claim, frame)
    return Test(
        claim=claim,
        holds=True,
        detail="",
        testable=False,
    )


def _missing(claim: Claim, reason: str) -> Test:
    return Test(claim=claim, holds=True, detail=reason, testable=False)


def _test_threshold(claim: Claim, frame: pd.DataFrame) -> Test:
    """A threshold holds if it actually divides the data into two groups.

    "VIP customers spend more than 5,000" is contradicted when nobody clears
    5,000, or when everybody does - either way the rule no longer separates
    anyone, which is worth telling the owner.
    """
    if claim.column not in frame.columns:
        return _missing(claim, f"There is no {claim.column} column in this file.")

    values = pd.to_numeric(frame[claim.column], errors="coerce").dropna()
    if values.empty:
        return _missing(claim, f"{claim.column} holds no numbers in this file.")

    try:
        threshold = float(claim.value)
    except (TypeError, ValueError):
        return _missing(claim, "That rule has no number in it to test.")

    comparisons = {
        ">": values > threshold,
        ">=": values >= threshold,
        "<": values < threshold,
        "<=": values <= threshold,
        "=": values == threshold,
        "!=": values != threshold,
    }
    mask = comparisons.get(claim.operator, values > threshold)
    matching = int(mask.sum())
    share = matching / len(values)

    if matching == 0:
        return Test(
            claim=claim,
            holds=False,
            detail=(
                f"No record in this file meets that rule. The highest "
                f"{claim.column.replace('_', ' ')} here is {values.max():,.0f}."
            ),
            observed=f"0 of {len(values):,} records",
            expected="at least some",
            affected_rows=0,
        )
    if share > 0.98:
        return Test(
            claim=claim,
            holds=False,
            detail=(
                f"Almost every record meets that rule ({share:.0%}), so it is not "
                "separating anything in this file."
            ),
            observed=f"{matching:,} of {len(values):,} records",
            expected="a distinct group",
            affected_rows=matching,
        )
    return Test(
        claim=claim,
        holds=True,
        detail=f"{matching:,} records meet it ({share:.0%}).",
        observed=f"{matching:,} of {len(values):,} records",
        affected_rows=matching,
    )


def _test_peak(claim: Claim, frame: pd.DataFrame, profile) -> Test:
    """Test "peak season starts in November" against when the peak actually is."""
    from ..analysis.exploration import resolve_columns
    from ..analysis.profiling import try_parse_datetime

    columns = resolve_columns(frame, profile)
    measure = claim.column if claim.column in frame.columns else columns.primary_measure
    if not measure or not columns.primary_date:
        return _missing(claim, "This file has no date column to check a season against.")

    series = frame[columns.primary_date]
    if not pd.api.types.is_datetime64_any_dtype(series):
        parsed = try_parse_datetime(series)
        if parsed is None:
            return _missing(claim, "The dates in this file could not be read.")
        series = parsed

    stated = claim.period.strip().casefold()
    if stated in MONTHS:
        grouped = (
            pd.to_numeric(frame[measure], errors="coerce")
            .groupby(series.dt.month)
            .sum()
        )
        expected_key = MONTHS[stated]
        naming = {index: name for name, index in MONTHS.items()}
    elif stated in QUARTERS:
        grouped = (
            pd.to_numeric(frame[measure], errors="coerce")
            .groupby(series.dt.quarter)
            .sum()
        )
        expected_key = QUARTERS[stated]
        naming = {index: f"q{index}" for index in QUARTERS.values()}
    else:
        return _missing(claim, f'"{claim.period}" is not a month or quarter we can check.')

    grouped = grouped.dropna()
    if len(grouped) < 3:
        return _missing(claim, "This file does not cover enough periods to tell.")

    actual_key = int(grouped.idxmax())
    if actual_key == expected_key:
        return Test(
            claim=claim,
            holds=True,
            detail=f"{claim.period.title()} is the strongest period here too.",
            observed=claim.period.title(),
            expected=claim.period.title(),
        )

    actual = str(naming.get(actual_key, actual_key)).title()
    stated_total = float(grouped.get(expected_key, 0.0))
    actual_total = float(grouped.max())
    return Test(
        claim=claim,
        holds=False,
        detail=(
            f"In this file the strongest period is {actual} at "
            f"{actual_total:,.0f}, not {claim.period.title()}, which comes in at "
            f"{stated_total:,.0f}."
        ),
        observed=actual,
        expected=claim.period.title(),
    )


def _test_exclusion(claim: Claim, frame: pd.DataFrame) -> Test:
    """Test whether rows the owner said to exclude are still present."""
    if claim.column not in frame.columns:
        return _missing(claim, f"There is no {claim.column} column in this file.")

    wanted = claim.value if isinstance(claim.value, list) else [claim.value]
    wanted = {str(value).casefold() for value in wanted if value is not None}
    if not wanted:
        return _missing(claim, "That rule names no values to exclude.")

    present = frame[claim.column].astype(str).str.casefold().isin(wanted)
    count = int(present.sum())
    if count == 0:
        return Test(
            claim=claim,
            holds=True,
            detail="None of those records are in this file.",
            affected_rows=0,
        )
    return Test(
        claim=claim,
        holds=False,
        detail=(
            f"{count:,} records in this file still have "
            f"{claim.column.replace('_', ' ')} set to "
            f"{', '.join(sorted(wanted))}, even though you said to leave them out."
        ),
        observed=f"{count:,} records",
        expected="none",
        affected_rows=count,
    )


# ---------------------------------------------------------------------------
# Reading a claim out of a sentence, with no model
# ---------------------------------------------------------------------------

_THRESHOLD = re.compile(
    r"(?P<label>[\w\-]+(?:\s+[\w\-]+){0,2})\s+(?:are|is)\b"
    r"[^.]{0,60}?\b"
    r"(?P<operator>exceeds?|over|above|more than|greater than|at least|"
    r"under|below|less than|at most|no more than)\b"
    r"\s*[\$£€]?\s*(?P<value>\d[\d,]*(?:\.\d+)?)",
    re.IGNORECASE,
)

#: Any month or quarter name appearing near a "peak" word. Far more robust than
#: trying to parse the sentence structure, which varies endlessly.
_PEAK = re.compile(
    r"\b(?:peak|busiest|strongest|highest|best)\b[^.]{0,40}?\b(?P<period>"
    + "|".join(list(MONTHS) + list(QUARTERS))
    + r")\b",
    re.IGNORECASE,
)

_IGNORE = re.compile(
    r"\b(?:ignore|exclude|leave out|don't count|do not count|skip)\b\s+"
    r"(?:all\s+|any\s+|the\s+)?(?P<value>[\w\-]+)",
    re.IGNORECASE,
)

_OPERATOR_WORDS = {
    "exceed": ">", "exceeds": ">", "over": ">", "above": ">", "more than": ">",
    "greater than": ">", "at least": ">=",
    "under": "<", "below": "<", "less than": "<",
    "at most": "<=", "no more than": "<=",
}


def read_claim(
    statement: str,
    columns: list[str],
    frame: pd.DataFrame | None = None,
) -> Claim | None:
    """Extract a testable claim from a sentence, without a model.

    Deliberately narrow, and deliberately silent when unsure. A wrong
    structured reading is worse than none, because a claim gets applied
    automatically to later runs and tested against later files: a
    misunderstanding becomes a recurring false alarm. So nothing is returned
    unless the reading can be checked against the data in front of us.
    """
    text = " ".join(statement.split())

    match = _PEAK.search(text)
    if match:
        period = match.group("period").strip().casefold()
        if period in MONTHS or period in QUARTERS:
            return Claim(kind="peak_period", period=period.title())

    match = _THRESHOLD.search(text)
    if match:
        column = _numeric_column(text, columns, frame)
        if column:
            return Claim(
                kind="threshold",
                column=column,
                operator=_OPERATOR_WORDS.get(match.group("operator").casefold(), ">"),
                value=float(match.group("value").replace(",", "")),
                label=match.group("label").strip().title(),
            )

    match = _IGNORE.search(text)
    if match:
        token = match.group("value").strip()
        found = _column_holding(token, columns, frame)
        if found:
            column, actual = found
            return Claim(kind="exclusion", column=column, value=actual)
    return None


def _numeric_column(
    text: str, columns: list[str], frame: pd.DataFrame | None
) -> str | None:
    """The numeric column a threshold sentence is about."""
    candidates = [
        name
        for name in columns
        if frame is None
        or (name in frame.columns and pd.api.types.is_numeric_dtype(frame[name]))
    ]
    return _closest_column(text, candidates)


def _column_holding(
    token: str, columns: list[str], frame: pd.DataFrame | None
) -> tuple[str, str] | None:
    """Find the column that actually contains a value like ``token``.

    "Ignore cancelled invoices" names the value, not the column, and the value
    the user typed rarely matches the spelling in the file. Rather than guess a
    column from the sentence and store the user's spelling - producing a rule
    that silently matches nothing - the real value is looked up, and no claim is
    made if it is not there.
    """
    if frame is None:
        return None

    wanted = token.casefold().rstrip("s")
    best: tuple[int, str, str] | None = None

    for name in columns:
        if name not in frame.columns:
            continue
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_datetime64_any_dtype(
            series
        ):
            continue
        values = series.dropna().astype(str).unique()
        if len(values) > 60:
            continue
        for value in values:
            normalised = str(value).casefold().rstrip("s")
            if normalised == wanted or normalised.startswith(wanted):
                score = len(wanted)
                if best is None or score > best[0]:
                    best = (score, name, str(value))
    return (best[1], best[2]) if best else None


def _closest_column(text: str, columns: list[str]) -> str | None:
    """The column a sentence is most likely talking about."""
    lowered = text.casefold()
    best: tuple[int, str] | None = None
    for name in columns:
        readable = name.replace("_", " ").casefold()
        if readable in lowered or name.casefold() in lowered:
            score = len(readable)
            if best is None or score > best[0]:
                best = (score, name)
    return best[1] if best else None
