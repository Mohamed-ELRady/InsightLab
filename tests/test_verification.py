"""The checks a conclusion has to survive.

These are the tests that matter most in the suite. Everything else protects the
product from being broken; these protect the user from being told something
false in a confident voice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insightlab.analysis import confounding, significance
from insightlab.core.grounding import (
    FactBase,
    build_fact_base,
    check,
    extract_figures,
    strip_unverified,
)
from insightlab.core.state import Chart, Insight, PipelineState


# ---------------------------------------------------------------------------
# B4 — significance
# ---------------------------------------------------------------------------


def two_groups(left_mean, right_mean, spread, rows, seed=1):
    generator = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "g": ["a"] * rows + ["b"] * rows,
            "v": list(generator.normal(left_mean, spread, rows))
            + list(generator.normal(right_mean, spread, rows)),
        }
    )


class TestSignificance:
    def test_a_large_clear_gap_is_solid(self):
        result = significance.compare_extremes(two_groups(160, 100, 10, 200), "g", "v")
        assert result.verdict == "solid"
        assert result.confidence == "high"
        assert result.is_reportable

    def test_a_gap_inside_the_noise_is_rejected(self):
        result = significance.compare_extremes(two_groups(101, 100, 40, 200), "g", "v")
        assert result.verdict == "noise"
        assert not result.is_reportable

    def test_too_few_rows_is_rejected_however_large_the_gap(self):
        # A tenfold gap on eight rows a side is still not evidence.
        result = significance.compare_extremes(two_groups(1000, 100, 10, 8), "g", "v")
        assert result.verdict == "too_few"
        assert not result.is_reportable
        assert "too few" in result.describe()

    def test_a_real_but_imprecise_gap_is_weak_not_solid(self):
        result = significance.compare_extremes(two_groups(150, 100, 90, 60), "g", "v")
        assert result.verdict in ("weak", "noise")
        if result.verdict == "weak":
            assert result.confidence == "medium"

    def test_a_tiny_relative_gap_is_noise_even_with_a_huge_sample(self):
        # Statistically detectable is not the same as worth telling someone.
        result = significance.compare_extremes(two_groups(100.5, 100, 1, 5000), "g", "v")
        assert result.verdict == "noise"

    def test_the_verdict_is_stable_across_repeated_runs(self):
        # A finding that changes wording between two runs of the same file
        # would be indefensible, so the bootstrap seed is fixed.
        frame = two_groups(150, 100, 30, 100)
        first = significance.compare_extremes(frame, "g", "v")
        second = significance.compare_extremes(frame, "g", "v")
        assert first.interval == second.interval
        assert first.verdict == second.verdict

    def test_the_share_interval_never_leaves_the_valid_range(self):
        # The textbook normal interval produces impossible bounds near zero.
        result = significance.share_interval(1, 500)
        assert result.interval[0] >= 0
        assert result.interval[1] <= 1


# ---------------------------------------------------------------------------
# B3 — confounding
# ---------------------------------------------------------------------------


def simpsons_frame() -> pd.DataFrame:
    """A textbook paradox: A beats B in every region, loses overall."""
    generator = np.random.default_rng(7)
    rows = []
    for region, base in (("Cairo", 100), ("Aswan", 500)):
        a_rows = 400 if region == "Cairo" else 40
        b_rows = 40 if region == "Cairo" else 400
        rows += [
            {"channel": "A", "region": region, "revenue": value}
            for value in generator.normal(base + 40, 15, a_rows)
        ]
        rows += [
            {"channel": "B", "region": region, "revenue": value}
            for value in generator.normal(base, 15, b_rows)
        ]
    return pd.DataFrame(rows)


class TestConfounding:
    def test_the_decomposition_is_exact(self):
        result = confounding.decompose(
            simpsons_frame(), "channel", "revenue", "B", "A", "region"
        )
        assert result.within_difference + result.mix_difference == pytest.approx(
            result.overall_difference
        )

    def test_a_reversal_is_detected(self):
        result = confounding.check_extremes(simpsons_frame(), "channel", "revenue")
        assert result is not None
        assert result.kind == "reversal"
        assert result.condition_column == "region"
        assert set(result.subgroups) == {"Cairo", "Aswan"}

    def test_a_mix_driven_reversal_says_the_split_figures_are_the_true_ones(self):
        result = confounding.check_extremes(simpsons_frame(), "channel", "revenue")
        assert result.mix_explains_it
        assert "split figures as the true ones" in result.describe()

    def test_a_within_driven_reversal_does_not_blame_the_composition(self):
        # The two cases share a symptom and have opposite explanations. Saying
        # the composition drives a gap that is 90% within-group would be false.
        result = confounding.Confound(
            kind="reversal",
            group_column="channel",
            measure="revenue",
            left="Store",
            right="Phone",
            condition_column="segment",
            overall_difference=1000.0,
            within_difference=920.0,
            mix_difference=80.0,
            reversed_share=0.7,
            subgroups=["Retail"],
        )
        assert not result.mix_explains_it
        assert "genuine one" in result.describe()
        assert "not by a real advantage" not in result.describe()

    def test_a_clean_comparison_produces_no_confound(self):
        generator = np.random.default_rng(3)
        frame = pd.DataFrame(
            {
                "g": ["a"] * 300 + ["b"] * 300,
                "other": list(generator.choice(["x", "y"], 600)),
                "v": list(generator.normal(200, 15, 300))
                + list(generator.normal(100, 15, 300)),
            }
        )
        assert confounding.check_extremes(frame, "g", "v") is None

    def test_our_own_outlier_markers_are_never_offered_as_an_explanation(self):
        # revenue_is_unusual is defined by the size of revenue, so it will
        # always appear to explain a revenue gap. That is circular, and it is
        # our own bookkeeping rather than anything about the business.
        generator = np.random.default_rng(5)
        frame = pd.DataFrame(
            {
                "segment": ["a"] * 300 + ["b"] * 300,
                "revenue": list(generator.normal(500, 100, 300))
                + list(generator.normal(100, 100, 300)),
            }
        )
        frame["revenue_is_unusual"] = frame["revenue"] > 550
        frame["cost_is_unusual"] = frame["revenue"] > 540

        result = confounding.check_extremes(frame, "segment", "revenue")
        assert result is None or not result.condition_column.endswith("_is_unusual")

    def test_a_column_derived_from_the_measure_is_never_the_explanation(self):
        generator = np.random.default_rng(5)
        frame = pd.DataFrame(
            {
                "segment": ["a"] * 300 + ["b"] * 300,
                "revenue": list(generator.normal(500, 100, 300))
                + list(generator.normal(100, 100, 300)),
            }
        )
        frame["revenue_tier"] = pd.qcut(frame["revenue"], 4, labels=list("abcd")).astype(str)

        result = confounding.check_extremes(frame, "segment", "revenue")
        assert result is None or result.condition_column != "revenue_tier"


# ---------------------------------------------------------------------------
# B1 — numeric grounding
# ---------------------------------------------------------------------------


class TestExtraction:
    @pytest.mark.parametrize(
        ("text", "value", "percentage"),
        [
            ("45%", 45.0, True),
            ("2.8M", 2_800_000.0, False),
            ("1,234,567", 1_234_567.0, False),
            ("3.5K", 3_500.0, False),
            ("-3.7%", -3.7, True),
            ("2 billion", 2_000_000_000.0, False),
        ],
    )
    def test_business_number_formats_are_understood(self, text, value, percentage):
        figures = extract_figures(f"The figure was {text} last year.")
        assert len(figures) == 1
        assert figures[0].value == pytest.approx(value)
        assert figures[0].is_percentage is percentage

    def test_several_figures_in_one_sentence_are_all_found(self):
        figures = extract_figures("Revenue grew 45% to 2.8M from 1,234,567.")
        assert [figure.value for figure in figures] == [45.0, 2_800_000.0, 1_234_567.0]


class TestGrounding:
    def test_a_figure_in_the_data_is_verified(self):
        facts = FactBase()
        facts.add(2_764_183)
        assert check("Revenue was 2,764,183.", facts).is_clean

    def test_a_figure_that_is_not_in_the_data_is_caught(self):
        facts = FactBase()
        facts.add(2_764_183)
        result = check("Revenue was 5,000,000.", facts)
        assert not result.is_clean
        assert result.unverified[0].value == 5_000_000

    def test_the_products_own_rounding_is_accepted(self):
        # The charts render 1,858,741 as "1.9M". A check tighter than the
        # product's own display precision would reject its own output.
        facts = FactBase()
        facts.add(1_858_740.91)
        for written in ("1.9M", "1.86M", "1,858,741"):
            assert check(f"It was {written}.", facts).is_clean, written

    def test_a_rounding_too_coarse_to_be_ours_is_still_rejected(self):
        facts = FactBase()
        facts.add(1_858_740.91)
        assert not check("It was 2M.", facts).is_clean

    def test_years_and_small_counts_are_not_treated_as_claims(self):
        facts = FactBase()
        assert check("Across 5 regions in 2024, sales rose.", facts).is_clean

    def test_a_percentage_matches_a_stored_fraction(self):
        facts = FactBase()
        facts.add(0.291)
        assert check("The margin is 29.1%.", facts).is_clean

    def test_only_the_offending_sentence_is_removed(self):
        facts = FactBase()
        facts.add(2_800_000)
        facts.add(1_234_567)
        result = strip_unverified(
            "Revenue reached 2.8M. Growth was 45% year on year. "
            "Last year it was 1,234,567.",
            facts,
        )
        assert "2.8M" in result.text
        assert "1,234,567" in result.text
        assert "45%" not in result.text
        assert len(result.removed_sentences) == 1

    def test_shares_of_a_chart_total_are_treated_as_real_facts(self):
        # A model quoting "52% of revenue" is quoting arithmetic the reader
        # would do themselves, so it must not be flagged as invented.
        facts = FactBase()
        facts.add_frame(pd.DataFrame({"group": ["a", "b"], "value": [520.0, 480.0]}))
        assert check("That group is 52% of the total.", facts).is_clean

    def test_the_fact_base_is_built_from_the_whole_run(self, sample_frame):
        from insightlab.analysis.profiling import profile_dataset

        state = PipelineState()
        state.frame = sample_frame
        state.profile = profile_dataset(sample_frame)
        state.charts = [
            Chart(
                id="c",
                title="t",
                kind="bar",
                description="d",
                table=pd.DataFrame({"g": ["a"], "v": [123456.0]}),
            )
        ]
        facts = build_fact_base(state)

        assert len(facts) > 0
        assert check("The value was 123,456.", facts).is_clean
        assert not check("The value was 999,999,999.", facts).is_clean


# ---------------------------------------------------------------------------
# B2 — the verifier as a whole
# ---------------------------------------------------------------------------


class TestVerifier:
    @pytest.fixture
    def state_with_noise(self, sample_frame):
        from insightlab.analysis.profiling import profile_dataset

        state = PipelineState()
        state.frame = sample_frame
        state.profile = profile_dataset(sample_frame)
        return state

    def test_an_unsupported_figure_is_stripped_and_the_finding_demoted(
        self, state_with_noise
    ):
        from insightlab.agents.verification import Verifier
        from insightlab.core.reasoning import ReasoningEngine

        insight = Insight(
            title="Made up",
            result="Revenue grew by 999,999,999 this year.",
            evidence="From the data.",
            interpretation="Growth.",
            confidence="high",
            action="Celebrate.",
        )
        kept = Verifier(ReasoningEngine()).verify(state_with_noise, [insight])

        # Nothing survives with its only claim removed.
        assert kept == []

    def test_a_supported_finding_passes_through_untouched(self, state_with_noise):
        from insightlab.agents.verification import Verifier
        from insightlab.core.reasoning import ReasoningEngine

        total = state_with_noise.frame["revenue"].sum()
        insight = Insight(
            title="Real",
            result=f"Total revenue is {total:,.0f}.",
            evidence="Summed from every row.",
            interpretation="That is the top line.",
            confidence="high",
            action="Compare it with your books.",
        )
        kept = Verifier(ReasoningEngine()).verify(state_with_noise, [insight])

        assert len(kept) == 1
        assert kept[0].confidence == "high"
        assert kept[0].grounded

    def test_removals_are_written_to_the_log_not_done_silently(self, state_with_noise):
        from insightlab.agents.verification import Verifier
        from insightlab.core.activity_log import EventKind
        from insightlab.core.reasoning import ReasoningEngine

        Verifier(ReasoningEngine()).verify(
            state_with_noise,
            [
                Insight(
                    title="Made up",
                    result="Revenue grew by 999,999,999.",
                    evidence="",
                    interpretation="",
                )
            ],
        )
        warnings = state_with_noise.log.of_kind(EventKind.WARNING)
        assert warnings, "a removal must be recorded, not hidden"

    def test_demote_never_falls_below_the_floor(self):
        from insightlab.agents.verification import demote

        assert demote("high") == "medium"
        assert demote("high", 2) == "low"
        assert demote("low") == "low"
        assert demote("low", 5) == "low"
