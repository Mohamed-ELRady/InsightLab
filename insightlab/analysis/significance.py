"""Whether a difference is big enough to be worth saying out loud.

With forty rows per region, a six per cent gap between two regions is noise.
The product would otherwise present it in exactly the same voice as a sixty per
cent gap across four thousand rows, and a business owner has no way to tell the
two apart.

Everything here is a bootstrap or an exact count. That avoids a scipy dependency
and, more importantly, avoids assuming a distribution that skewed business data
almost never has.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

#: Resamples per bootstrap. Enough for a stable 95% interval, cheap enough to
#: run on every comparison in a run.
RESAMPLES = 2000

#: Confidence level for the reported interval.
CONFIDENCE = 0.95

#: Fewer rows than this in either group and no comparison is worth making,
#: however large the gap looks.
MIN_GROUP_ROWS = 20

#: A relative gap smaller than this is treated as no practical difference even
#: when a large sample makes it statistically detectable. Statistical
#: significance is not the same as mattering.
MIN_RELATIVE_GAP = 0.05

# Keep bootstrap allocations bounded.  The previous ``(2000, group_size)``
# arrays could briefly exceed a gigabyte on an otherwise modest upload.  This
# cap changes neither the samples nor the interval; it only calculates their
# means in cache-friendly chunks.
MAX_BOOTSTRAP_VALUES = 2_000_000

Verdict = Literal["solid", "weak", "noise", "too_few"]


@dataclass
class Comparison:
    """The result of testing one difference between two groups."""

    left: str
    right: str
    left_value: float
    right_value: float
    left_rows: int
    right_rows: int
    difference: float
    relative_difference: float
    interval: tuple[float, float]
    verdict: Verdict

    @property
    def is_reportable(self) -> bool:
        """Whether this difference should be stated as a finding at all."""
        return self.verdict in ("solid", "weak")

    @property
    def confidence(self) -> str:
        """Maps onto the confidence levels an insight already carries."""
        return {"solid": "high", "weak": "medium"}.get(self.verdict, "low")

    def describe(self) -> str:
        """One plain sentence about how far this difference can be trusted."""
        if self.verdict == "too_few":
            return (
                f"There are only {min(self.left_rows, self.right_rows)} records in "
                f"the smaller of these two groups, which is too few to tell them "
                "apart with any confidence."
            )
        if self.verdict == "noise":
            return (
                f"The gap between {self.left} and {self.right} is small enough "
                "that it could easily be chance rather than a real difference."
            )
        if self.verdict == "weak":
            return (
                f"{self.left} is ahead of {self.right} by "
                f"{abs(self.relative_difference):.0%}, but the records are few "
                "enough that the true gap could be anywhere between "
                f"{self.interval[0]:,.0f} and {self.interval[1]:,.0f}."
            )
        return (
            f"{self.left} is ahead of {self.right} by "
            f"{abs(self.relative_difference):.0%}, and the gap is large enough "
            "relative to the variation in the data to be a real difference "
            "rather than chance."
        )


def _bootstrap_difference(
    left: np.ndarray, right: np.ndarray, *, resamples: int = RESAMPLES
) -> tuple[float, float]:
    """A confidence interval for the difference in means, by resampling.

    The seed is fixed so the same data always produces the same interval - a
    finding that changes wording between two runs of the same file would be
    indefensible.
    """
    generator = np.random.default_rng(12345)
    def sampled_means(values: np.ndarray) -> np.ndarray:
        per_chunk = max(1, MAX_BOOTSTRAP_VALUES // max(len(values), 1))
        means = np.empty(resamples, dtype=float)
        for start in range(0, resamples, per_chunk):
            stop = min(resamples, start + per_chunk)
            draws = generator.choice(
                values, size=(stop - start, len(values)), replace=True
            )
            means[start:stop] = draws.mean(axis=1)
        return means

    # Left is completed before right, matching the historical RNG order and
    # therefore preserving exact intervals for datasets that fit one chunk.
    differences = sampled_means(left) - sampled_means(right)

    tail = (1 - CONFIDENCE) / 2
    return (
        float(np.quantile(differences, tail)),
        float(np.quantile(differences, 1 - tail)),
    )


def _normal_mean_difference_interval(
    left: np.ndarray, right: np.ndarray, difference: float
) -> tuple[float, float]:
    """A fast large-sample estimate used only for decisive comparisons."""
    left_variance = float(left.var(ddof=1)) if len(left) > 1 else 0.0
    right_variance = float(right.var(ddof=1)) if len(right) > 1 else 0.0
    standard_error = np.sqrt(
        left_variance / max(len(left), 1) + right_variance / max(len(right), 1)
    )
    margin = 1.96 * float(standard_error)
    return difference - margin, difference + margin


def compare_groups(
    frame: pd.DataFrame, group_column: str, measure: str, left: str, right: str
) -> Comparison | None:
    """Test whether two groups genuinely differ on a measure."""
    if group_column not in frame.columns or measure not in frame.columns:
        return None

    groups = frame[group_column].astype(str)
    values = pd.to_numeric(frame[measure], errors="coerce")

    left_values = values[groups == left].dropna().to_numpy()
    right_values = values[groups == right].dropna().to_numpy()
    if len(left_values) == 0 or len(right_values) == 0:
        return None

    left_mean = float(left_values.mean())
    right_mean = float(right_values.mean())
    difference = left_mean - right_mean
    base = abs(right_mean) if right_mean else abs(left_mean)
    relative = difference / base if base else 0.0

    if len(left_values) < MIN_GROUP_ROWS or len(right_values) < MIN_GROUP_ROWS:
        return Comparison(
            left, right, left_mean, right_mean,
            len(left_values), len(right_values),
            difference, relative, (float("nan"), float("nan")), "too_few",
        )

    estimate = _normal_mean_difference_interval(left_values, right_values, difference)

    # This verdict is mathematically fixed by the practical-effect rule below,
    # regardless of what a 2,000-resample bootstrap says.  Avoiding that work is
    # especially valuable for very large groups and does not change the finding.
    if abs(relative) < MIN_RELATIVE_GAP:
        return Comparison(
            left, right, left_mean, right_mean,
            len(left_values), len(right_values),
            difference, relative, estimate, "noise",
        )

    # For a large, unambiguous effect the normal and bootstrap intervals are
    # indistinguishable for the decision we make.  Keep a generous 20% safety
    # band around the solid/weak boundary; borderline findings still take the
    # full deterministic bootstrap path.
    estimate_width = estimate[1] - estimate[0]
    estimate_excludes_zero = estimate[0] > 0 or estimate[1] < 0
    if (
        min(len(left_values), len(right_values)) >= 200
        and estimate_excludes_zero
        and estimate_width <= abs(difference) * 0.8
    ):
        return Comparison(
            left, right, left_mean, right_mean,
            len(left_values), len(right_values),
            difference, relative, estimate, "solid",
        )

    interval = _bootstrap_difference(left_values, right_values)

    # The interval straddling zero means the direction of the difference is not
    # established - it could run the other way.
    straddles_zero = interval[0] <= 0 <= interval[1]
    if straddles_zero or abs(relative) < MIN_RELATIVE_GAP:
        verdict: Verdict = "noise"
    else:
        # "Solid" has to mean you know roughly how big the gap is, not merely
        # which way it runs. An interval of 24 to 79 around a difference of 50
        # establishes the direction and leaves the size uncertain by threefold,
        # which is a weak finding however confident the sign is. So the
        # uncertainty must be smaller than the effect itself.
        width = interval[1] - interval[0]
        verdict = "solid" if width <= abs(difference) else "weak"

    return Comparison(
        left, right, left_mean, right_mean,
        len(left_values), len(right_values),
        difference, relative, interval, verdict,
    )


def compare_extremes(
    frame: pd.DataFrame, group_column: str, measure: str, *, how: str = "mean"
) -> Comparison | None:
    """Test the best group against the worst - the comparison a bar chart implies.

    A ranked bar chart makes an implicit claim that the top bar is really ahead
    of the bottom one. This is the test of that claim.
    """
    if group_column not in frame.columns or measure not in frame.columns:
        return None

    working = pd.DataFrame(
        {"group": frame[group_column].astype(str),
         "value": pd.to_numeric(frame[measure], errors="coerce")}
    ).dropna()
    if working.empty:
        return None

    ranked = working.groupby("group", observed=True)["value"].agg(how).sort_values(
        ascending=False
    )
    if len(ranked) < 2:
        return None

    return compare_groups(
        frame, group_column, measure, str(ranked.index[0]), str(ranked.index[-1])
    )


@dataclass
class ShareComparison:
    """Whether two counts differ by more than sampling would explain."""

    label: str
    count: int
    total: int
    share: float
    interval: tuple[float, float]

    def describe(self) -> str:
        return (
            f"{self.label} accounts for {self.share:.0%} of records. Given "
            f"{self.total:,} records in total, the true share is somewhere "
            f"between {self.interval[0]:.0%} and {self.interval[1]:.0%}."
        )


def share_interval(count: int, total: int) -> ShareComparison | None:
    """A confidence interval around a proportion, using the Wilson method.

    Wilson rather than the textbook normal interval because business data
    routinely has shares near zero or one, where the normal interval produces
    impossible bounds like -4%.
    """
    if total <= 0 or count < 0 or count > total:
        return None

    z = 1.96  # 95%
    proportion = count / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    spread = (
        z
        * np.sqrt(proportion * (1 - proportion) / total + z**2 / (4 * total**2))
        / denominator
    )
    return ShareComparison(
        label="",
        count=count,
        total=total,
        share=proportion,
        interval=(max(0.0, centre - spread), min(1.0, centre + spread)),
    )


def enough_rows(frame: pd.DataFrame, group_column: str, minimum: int = MIN_GROUP_ROWS) -> bool:
    """Whether every group is large enough for a comparison to mean anything."""
    if group_column not in frame.columns:
        return False
    counts = frame[group_column].value_counts()
    return bool(len(counts) >= 2 and counts.min() >= minimum)
