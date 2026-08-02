"""Detecting when a difference is explained by something else entirely.

Two ways a group comparison misleads, both invisible to any check a
non-analyst can run:

**Reversal.** "Channel A converts better than channel B" holds overall and
reverses inside every region. This is Simpson's paradox, and the overall number
is the one that is wrong.

**Mix.** "Wholesale customers spend more" is true, but only because wholesale
orders are for more units. The difference is real and the explanation is not the
one the reader will reach for.

Both are computed exactly, with no model involved. The decomposition below is
algebraically exact: within-group effect plus mix effect equals the total
difference, always.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .cleaning import UNUSUAL_SUFFIX

#: A subgroup smaller than this cannot support a comparison of its own, so it
#: is excluded from the reversal check rather than voting on it.
MIN_SUBGROUP_ROWS = 15

#: A subgroup must hold at least this share of the data before its reversal
#: counts. Stops one tiny slice overturning a solid overall finding.
MIN_SUBGROUP_SHARE = 0.05

#: When the mix effect accounts for more than this share of the total
#: difference, the third variable is the better explanation.
MIX_DOMINANCE = 0.5

#: Reversal in at least this share of the weight is a genuine paradox rather
#: than one odd subgroup.
REVERSAL_SHARE = 0.6

Kind = Literal["reversal", "mix", "none"]


@dataclass
class Confound:
    """A third variable that changes how a comparison should be read."""

    kind: Kind
    group_column: str
    measure: str
    left: str
    right: str
    condition_column: str
    overall_difference: float
    within_difference: float
    mix_difference: float
    reversed_share: float = 0.0
    subgroups: list[str] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.subgroups is None:
            self.subgroups = []

    @property
    def mix_share(self) -> float:
        """How much of the gap is the composition rather than the groups."""
        total = abs(self.overall_difference)
        return abs(self.mix_difference) / total if total else 0.0

    @property
    def mix_explains_it(self) -> bool:
        """Whether the composition, rather than the groups, drives the gap."""
        return self.mix_share >= MIX_DOMINANCE

    def describe(self) -> str:
        """The finding rewritten to lead with the variable that explains it."""
        readable_group = self.group_column.replace("_", " ")
        readable_measure = self.measure.replace("_", " ")
        readable_condition = self.condition_column.replace("_", " ")

        if self.kind == "reversal":
            listed = ", ".join(self.subgroups[:4])
            # Two very different stories share the reversal symptom, and the
            # decomposition is what tells them apart. Claiming the composition
            # explains a gap that is 92% within-group would be false.
            if self.mix_explains_it:
                return (
                    f"Read this one carefully. Overall, {self.left} looks ahead "
                    f"of {self.right} on {readable_measure}, but once you split "
                    f"by {readable_condition} the order flips: inside {listed} "
                    f"it is {self.right} that is ahead. The overall comparison "
                    f"is being driven by how {readable_group} and "
                    f"{readable_condition} are distributed against each other, "
                    "not by a real advantage. Treat the split figures as the "
                    "true ones."
                )
            return (
                f"{self.left} is ahead of {self.right} on {readable_measure} "
                f"overall, and the gap is a genuine one rather than an artefact "
                f"of how the records are distributed. But it does not hold "
                f"everywhere: inside {listed} the order is reversed. Whatever "
                f"action you take on this should be checked against "
                f"{readable_condition} first."
            )
        return (
            f"The gap between {self.left} and {self.right} on {readable_measure} "
            f"is mostly about {readable_condition}, not about {readable_group}. "
            f"{self.mix_share:.0%} of the difference disappears once you compare "
            f"like with like on {readable_condition}. The remaining "
            f"{1 - self.mix_share:.0%} is the genuine difference between the two "
            "groups."
        )


def _weighted_means(
    frame: pd.DataFrame, group_column: str, measure: str, condition: str
) -> pd.DataFrame | None:
    """Mean and share of the measure per (group, condition) cell."""
    working = pd.DataFrame(
        {
            "group": frame[group_column].astype(str),
            "condition": frame[condition].astype(str),
            "value": pd.to_numeric(frame[measure], errors="coerce"),
        }
    ).dropna()
    if working.empty:
        return None

    cells = (
        working.groupby(["group", "condition"], observed=True)["value"]
        .agg(["mean", "count"])
        .reset_index()
    )
    return cells if not cells.empty else None


def decompose(
    frame: pd.DataFrame,
    group_column: str,
    measure: str,
    left: str,
    right: str,
    condition: str,
) -> Confound | None:
    """Split the gap between two groups into a within part and a mix part.

    The identity is exact::

        mean(left) - mean(right) = within + mix

    where ``within`` is the gap that survives comparing like with like on the
    conditioning variable, and ``mix`` is the part caused purely by the two
    groups having different compositions.
    """
    cells = _weighted_means(frame, group_column, measure, condition)
    if cells is None:
        return None

    left_cells = cells[cells["group"] == left].set_index("condition")
    right_cells = cells[cells["group"] == right].set_index("condition")
    shared = sorted(set(left_cells.index) & set(right_cells.index))
    if len(shared) < 2:
        return None

    left_total = float(left_cells.loc[shared, "count"].sum())
    right_total = float(right_cells.loc[shared, "count"].sum())
    if not left_total or not right_total:
        return None

    within = 0.0
    mix = 0.0
    reversed_weight = 0.0
    total_weight = 0.0
    reversed_groups: list[str] = []

    overall_left = float(
        (left_cells.loc[shared, "mean"] * left_cells.loc[shared, "count"]).sum()
        / left_total
    )
    overall_right = float(
        (right_cells.loc[shared, "mean"] * right_cells.loc[shared, "count"]).sum()
        / right_total
    )
    overall = overall_left - overall_right
    direction = np.sign(overall)

    for level in shared:
        left_share = float(left_cells.loc[level, "count"]) / left_total
        right_share = float(right_cells.loc[level, "count"]) / right_total
        left_mean = float(left_cells.loc[level, "mean"])
        right_mean = float(right_cells.loc[level, "mean"])

        average_share = (left_share + right_share) / 2
        average_mean = (left_mean + right_mean) / 2

        within += average_share * (left_mean - right_mean)
        mix += (left_share - right_share) * average_mean

        rows = float(left_cells.loc[level, "count"] + right_cells.loc[level, "count"])
        if (
            left_cells.loc[level, "count"] >= MIN_SUBGROUP_ROWS
            and right_cells.loc[level, "count"] >= MIN_SUBGROUP_ROWS
        ):
            total_weight += rows
            if direction and np.sign(left_mean - right_mean) == -direction:
                reversed_weight += rows
                reversed_groups.append(str(level))

    reversed_share = reversed_weight / total_weight if total_weight else 0.0

    return Confound(
        kind="none",
        group_column=group_column,
        measure=measure,
        left=left,
        right=right,
        condition_column=condition,
        overall_difference=overall,
        within_difference=within,
        mix_difference=mix,
        reversed_share=reversed_share,
        subgroups=reversed_groups,
    )


def _is_artefact(name: str) -> bool:
    """Whether a column is our own bookkeeping rather than a business fact.

    The cleaning stage adds ``<measure>_is_unusual`` markers when the owner
    chooses to keep outliers. Those columns describe a decision we made about
    the data, not something that happens in the business, and because they are
    defined by the size of a measure they will always appear to "explain" a
    difference in that measure or in anything correlated with it. Telling an
    owner that their wholesale revenue gap is explained by which rows we flagged
    is circular and useless.
    """
    return name.casefold().endswith(UNUSUAL_SUFFIX)


def _is_derived_from(name: str, measure: str) -> bool:
    """Whether a column was computed from the measure being compared.

    ``revenue_tier`` is quartiles of revenue, so conditioning a revenue
    comparison on it is a tautology dressed up as an explanation.
    """
    if not measure:
        return False
    stem = measure.casefold()
    lowered = name.casefold()
    if lowered == stem:
        return True
    return lowered.startswith(f"{stem}_") or lowered.endswith(f"_{stem}")


def _usable_conditions(
    frame: pd.DataFrame, exclude: set[str], limit: int = 6, measure: str = ""
) -> list[str]:
    """Categorical columns worth conditioning on."""
    candidates: list[tuple[int, str]] = []
    for name in frame.columns:
        if (
            name in exclude
            or _is_artefact(str(name))
            or _is_derived_from(str(name), measure)
        ):
            continue
        series = frame[name]
        if pd.api.types.is_numeric_dtype(series) and series.nunique() > 12:
            continue
        distinct = int(series.dropna().nunique())
        if 2 <= distinct <= 15 and len(frame) / distinct >= MIN_SUBGROUP_ROWS:
            candidates.append((distinct, str(name)))
    # Fewer levels first: those give the largest, most trustworthy subgroups.
    return [name for _, name in sorted(candidates)][:limit]


def check(
    frame: pd.DataFrame,
    group_column: str,
    measure: str,
    left: str,
    right: str,
) -> Confound | None:
    """Find the third variable that most changes how this comparison reads.

    Returns the single worst offender rather than every candidate: a finding
    carrying four caveats is a finding nobody reads.
    """
    conditions = _usable_conditions(
        frame, exclude={group_column, measure}, measure=measure
    )
    if not conditions:
        return None

    best: Confound | None = None
    for condition in conditions:
        result = decompose(frame, group_column, measure, left, right, condition)
        if result is None:
            continue

        share = len(frame[frame[condition].notna()]) / max(len(frame), 1)
        if share < MIN_SUBGROUP_SHARE:
            continue

        if result.reversed_share >= REVERSAL_SHARE and result.subgroups:
            result.kind = "reversal"
            # A reversal outranks any mix effect, and the first one found on the
            # fewest-levels column is the clearest to explain.
            return result

        if result.mix_share >= MIX_DOMINANCE:
            result.kind = "mix"
            if best is None or result.mix_share > best.mix_share:
                best = result

    return best


def check_extremes(
    frame: pd.DataFrame, group_column: str, measure: str, *, how: str = "mean"
) -> Confound | None:
    """Run the check on the comparison a ranked bar chart implies."""
    working = pd.DataFrame(
        {
            "group": frame[group_column].astype(str),
            "value": pd.to_numeric(frame[measure], errors="coerce"),
        }
    ).dropna()
    if working.empty:
        return None

    ranked = working.groupby("group", observed=True)["value"].agg(how).sort_values(
        ascending=False
    )
    if len(ranked) < 2:
        return None

    return check(frame, group_column, measure, str(ranked.index[0]), str(ranked.index[-1]))
