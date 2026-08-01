"""The deterministic analysis layer.

These tests guard the numbers. If profiling calls a month number a measure, or a
bar chart describes a mean as a share of a total, the user is shown something
false - which matters more than any interface bug.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from insightlab.analysis import cleaning, exploration, features, metrics, theme
from insightlab.analysis.profiling import (
    detect_role,
    profile_dataset,
    try_parse_datetime,
)
from insightlab.core.state import Role


class TestRoleDetection:
    @pytest.mark.parametrize(
        ("name", "values", "expected"),
        [
            ("order_id", [f"ORD-{n}" for n in range(50)], Role.IDENTIFIER),
            ("revenue", list(np.linspace(10, 900, 50)), Role.MEASURE),
            ("region", ["Cairo", "Giza"] * 25, Role.CATEGORY),
            ("currency", ["EGP"] * 50, Role.CONSTANT),
            ("is_paid", [True, False] * 25, Role.BOOLEAN),
            ("order_date", pd.date_range("2024-01-01", periods=50).astype(str).tolist(), Role.DATETIME),
        ],
    )
    def test_columns_are_read_as_what_they_mean(self, name, values, expected):
        assert detect_role(pd.Series(values), name) is expected

    def test_a_month_name_column_is_a_group_not_a_date(self):
        # pandas parses "January" happily, but a column of month names is a set
        # of groups to compare, not a timeline.
        months = pd.Series(["January", "February", "March"] * 10)
        assert detect_role(months, "month_name") is Role.CATEGORY

    def test_calendar_parts_are_never_measures(self):
        # Averaging a month number is meaningless, so year/month/quarter stay
        # on the category side whatever they are stored as.
        assert detect_role(pd.Series([1, 2, 3] * 10), "month") is Role.CATEGORY
        assert detect_role(pd.Series([2023, 2024] * 15), "year") is Role.CATEGORY

    def test_a_column_of_integers_is_not_parsed_as_timestamps(self):
        assert try_parse_datetime(pd.Series(["1", "2", "3"] * 10)) is None

    def test_real_dates_still_parse(self):
        parsed = try_parse_datetime(pd.Series(["2024-01-05", "2024-02-11"] * 10))
        assert parsed is not None and parsed.notna().all()


class TestProfiling:
    def test_the_profile_matches_the_frame(self, sample_frame, sample_profile):
        assert sample_profile.row_count == len(sample_frame)
        assert sample_profile.column_count == sample_frame.shape[1]
        assert len(sample_profile.columns) == sample_frame.shape[1]

    def test_duplicates_are_counted(self, sample_profile):
        assert sample_profile.duplicate_rows > 0

    def test_a_constant_column_is_called_out(self, sample_profile):
        currency = sample_profile.column("currency")
        assert currency.role is Role.CONSTANT
        assert "same value" in currency.note

    def test_missing_rates_are_reported(self, sample_frame, sample_profile):
        region = sample_profile.column("region")
        assert region.missing_count == int(sample_frame["region"].isna().sum())
        assert 0 < region.missing_rate < 1


class TestDuplicates:
    def test_duplicates_are_found_and_removed(self, sample_frame):
        report = cleaning.find_duplicates(sample_frame)
        assert report.found

        operation = cleaning.drop_duplicates(sample_frame)
        assert operation.rows_removed == report.total
        assert not operation.frame.duplicated().any()

    def test_merging_keeps_the_totals(self, sample_frame, sample_profile):
        before = sample_frame["revenue"].sum()
        operation = cleaning.merge_duplicates(sample_frame, sample_profile)

        assert len(operation.frame) < len(sample_frame)
        assert operation.frame["revenue"].sum() == pytest.approx(before)

    def test_deleting_reduces_the_totals(self, sample_frame):
        before = sample_frame["revenue"].sum()
        operation = cleaning.drop_duplicates(sample_frame)
        assert operation.frame["revenue"].sum() < before


class TestOutliers:
    def test_extremes_are_flagged_with_their_bounds(self):
        values = list(np.linspace(100, 300, 60)) + [50_000.0, 60_000.0]
        report = cleaning.find_outliers(pd.DataFrame({"revenue": values}), "revenue")

        assert report is not None
        assert report.count == 2
        assert report.high_values[:2] == [60_000.0, 50_000.0]
        assert report.lower_bound < 100 < report.upper_bound < 50_000

    def test_a_column_needs_enough_rows_before_we_ask(self, sample_frame):
        small = sample_frame.head(10)
        assert cleaning.find_outliers(small, "revenue") is None

    def test_capping_keeps_every_row(self, sample_frame):
        report = cleaning.find_outliers(sample_frame, "revenue")
        assert report is not None

        operation = cleaning.cap_outliers(sample_frame, report)
        assert len(operation.frame) == len(sample_frame)
        assert operation.frame["revenue"].max() <= report.upper_bound + 1e-6

    def test_removing_drops_exactly_the_flagged_rows(self, sample_frame):
        report = cleaning.find_outliers(sample_frame, "revenue")
        operation = cleaning.remove_outliers(sample_frame, report)
        assert operation.rows_removed == report.count

    def test_flagging_changes_no_value(self, sample_frame):
        report = cleaning.find_outliers(sample_frame, "revenue")
        operation = cleaning.flag_outliers(sample_frame, report)

        assert "revenue_is_unusual" in operation.frame.columns
        assert operation.frame["revenue"].equals(sample_frame["revenue"])

    def test_a_skewed_column_is_not_offered_as_a_problem(self, sample_frame, sample_profile):
        # Where a sixth of the rows are "outliers", that is the distribution,
        # not a data-entry problem, and asking about it wastes attention.
        for report in cleaning.outlier_candidates(sample_frame, sample_profile):
            assert report.share <= cleaning.MAX_OUTLIER_SHARE


class TestColumnOperations:
    def test_filling_removes_every_blank(self, sample_frame):
        operation = cleaning.fill_missing(sample_frame, "cost", "median")
        assert operation.frame["cost"].isna().sum() == 0
        assert operation.cells_changed == int(sample_frame["cost"].isna().sum())

    def test_filling_a_category_uses_a_label(self, sample_frame):
        operation = cleaning.fill_missing(sample_frame, "region", "Unknown")
        assert "Unknown" in set(operation.frame["region"])

    def test_a_column_with_no_blanks_is_left_alone(self, sample_frame):
        operation = cleaning.fill_missing(sample_frame, "order_id", "median")
        assert operation.cells_changed == 0

    def test_renaming_moves_the_values(self, sample_frame):
        operation = cleaning.rename_column(sample_frame, "region", "sales_region")
        assert "sales_region" in operation.frame.columns
        assert "region" not in operation.frame.columns

    def test_converting_reports_what_could_not_be_read(self):
        frame = pd.DataFrame({"amount": ["10", "20", "not a number"]})
        operation = cleaning.convert_column(frame, "amount", "number")
        assert operation.details["unreadable"] == 1

    def test_filtering_removes_the_named_values(self, sample_frame):
        operation = cleaning.filter_rows(sample_frame, "order_status", ["Cancelled"])
        assert "Cancelled" not in set(operation.frame["order_status"])
        assert operation.rows_removed > 0

    def test_a_constant_column_is_suggested_for_removal(self, sample_profile):
        action, _, payload = cleaning.suggest_column_action(
            sample_profile.column("currency"), sample_profile.row_count
        )
        assert action == "drop"
        assert payload["action"] == "drop"

    def test_a_healthy_column_is_left_alone(self, sample_profile):
        action, _, _ = cleaning.suggest_column_action(
            sample_profile.column("product_category"), sample_profile.row_count
        )
        assert action == "keep"


class TestFeatures:
    def test_every_suggestion_builds_the_columns_it_promises(
        self, sample_frame, sample_profile
    ):
        frame = sample_frame.copy()
        frame["order_date"] = pd.to_datetime(frame["order_date"])
        profile = profile_dataset(frame)

        for suggestion in features.suggest_features(frame, profile):
            result, created = suggestion.apply(frame)
            assert created, f"{suggestion.id} created nothing"
            assert set(suggestion.creates) <= set(result.columns)

    def test_profit_is_revenue_minus_cost(self, sample_frame, sample_profile):
        frame = sample_frame.dropna(subset=["cost"]).copy()
        profile = profile_dataset(frame)
        suggestion = next(
            item for item in features.suggest_features(frame, profile) if item.id == "profit"
        )
        result, _ = suggestion.apply(frame)

        expected = frame["revenue"] - frame["cost"]
        assert result["profit"].equals(expected)

    def test_a_custom_rule_becomes_a_real_column(self):
        frame = pd.DataFrame({"total_spend": [100, 6000, 4999, 5001]})
        result, created = features.apply_custom_rule(
            frame,
            "customer_class",
            {
                "source": "total_spend",
                "bands": [{"upto": 5000, "label": "Regular"}],
                "otherwise": "VIP",
            },
        )
        assert created == "customer_class"
        assert list(result["customer_class"]) == ["Regular", "VIP", "Regular", "VIP"]

    def test_an_unknown_source_column_changes_nothing(self):
        frame = pd.DataFrame({"a": [1, 2]})
        result, created = features.apply_custom_rule(
            frame, "tier", {"source": "missing", "bands": [{"upto": 1, "label": "x"}]}
        )
        assert created == ""
        assert result.equals(frame)


@pytest.fixture
def enriched(sample_frame):
    """The sample data after feature engineering, as the charts see it."""
    frame = sample_frame.copy()
    frame["order_date"] = pd.to_datetime(frame["order_date"])
    profile = profile_dataset(frame)
    for suggestion in features.suggest_features(frame, profile):
        frame, _ = suggestion.apply(frame)
    return frame, profile_dataset(frame)


class TestExploration:
    def test_only_answerable_areas_are_offered(self, enriched):
        frame, profile = enriched
        axes = exploration.available_axes(frame, profile)
        assert set(axes) <= set(exploration.AXES)
        assert "regions" in axes, "the sample has a region column"

    def test_an_area_with_no_columns_is_not_offered(self):
        frame = pd.DataFrame({"amount": np.linspace(1, 100, 60)})
        profile = profile_dataset(frame)
        assert "regions" not in exploration.available_axes(frame, profile)

    def test_every_chart_carries_a_figure_a_description_and_a_table(self, enriched):
        frame, profile = enriched
        charts = exploration.build_charts(
            frame, profile, exploration.available_axes(frame, profile)
        )
        assert charts
        for chart in charts:
            assert chart.figure_json, f"{chart.id} has no figure"
            assert chart.description, f"{chart.id} has no description"
            assert chart.table is not None, f"{chart.id} has no table fallback"

    def test_chart_identifiers_are_unique(self, enriched):
        frame, profile = enriched
        charts = exploration.build_charts(
            frame, profile, exploration.available_axes(frame, profile)
        )
        identifiers = [chart.id for chart in charts]
        assert len(identifiers) == len(set(identifiers))

    def test_a_counted_bar_chart_does_not_claim_to_show_the_measure(self, enriched):
        frame, profile = enriched
        chart = exploration.category_bar(
            frame, "order_status", "revenue", "operations", how="count"
        )
        assert chart.title.startswith("Number of rows")
        assert "% of the total" not in chart.description

    def test_an_averaged_bar_chart_does_not_talk_about_shares(self, enriched):
        # A percentage column summed, or an average described as a share of a
        # total, would both be arithmetic nonsense presented as a finding.
        frame, profile = enriched
        chart = exploration.category_bar(
            frame, "product_category", "profit_margin_pct", "profits", how="mean"
        )
        assert "% of the total" not in chart.description
        assert "highest" in chart.description

    def test_calendar_categories_keep_their_own_order(self, enriched):
        frame, profile = enriched
        chart = exploration.category_bar(frame, "month_name", "revenue", "sales")
        months = list(chart.table["month_name"])
        assert months[:3] == ["January", "February", "March"]
        assert theme.OTHER_LABEL not in months

    def test_folding_names_the_bucket_and_does_not_call_it_smallest(self):
        labels, values = theme.fold_to_other(
            [f"g{n}" for n in range(12)], [12 - n for n in range(12)], limit=8
        )
        assert labels[-1] == theme.OTHER_LABEL
        assert len(labels) == 8
        assert values[-1] == sum(range(1, 6))

    def test_calendar_parts_stay_out_of_the_correlation_matrix(self, enriched):
        frame, profile = enriched
        columns = exploration.resolve_columns(frame, profile)
        assert "month" not in columns.measures
        assert "year" not in columns.measures


class TestMetrics:
    def test_counting_customers_uses_the_identity_column(self, enriched):
        # Both customer_id and customer_segment match the word "customer". The
        # segment column has three values; counting it would report three
        # customers for a file with hundreds.
        frame, profile = enriched
        columns = exploration.resolve_columns(frame, profile)
        assert columns.entity_named("customer") == "customer_id"
        assert columns.category_named("customer") == "customer_segment"

    def test_the_headline_figures_match_the_data(self, enriched):
        frame, profile = enriched
        computed = {kpi.name: kpi for kpi in metrics.compute_kpis(frame, profile)}

        assert computed["Total revenue"].value == pytest.approx(frame["revenue"].sum())
        assert computed["Active customers"].value == frame["customer_id"].nunique()
        assert computed["Number of records"].value == len(frame)

    def test_the_margin_is_profit_over_revenue(self, enriched):
        frame, profile = enriched
        computed = {kpi.name: kpi for kpi in metrics.compute_kpis(frame, profile)}
        expected = frame["profit"].sum() / frame["revenue"].sum() * 100
        assert computed["Profit margin"].value == pytest.approx(expected)

    def test_every_figure_explains_itself(self, enriched):
        frame, profile = enriched
        for kpi in metrics.compute_kpis(frame, profile):
            assert kpi.formula, f"{kpi.name} has no formula"
            assert kpi.interpretation, f"{kpi.name} has no interpretation"

    def test_a_custom_ratio_is_expressed_as_a_percentage(self, enriched):
        frame, _ = enriched
        kpi = metrics.custom_kpi(
            frame, "Cost ratio", "cost", "revenue", as_percentage=True
        )
        assert kpi.display_value.endswith("%")
        assert kpi.value == pytest.approx(
            frame["cost"].sum() / frame["revenue"].sum() * 100
        )

    def test_a_measure_naming_a_missing_column_returns_nothing(self, enriched):
        frame, _ = enriched
        assert metrics.custom_kpi(frame, "Nope", "not_a_column") is None


class TestPalette:
    def test_the_categorical_order_is_never_cycled(self):
        # A ninth series must fold into "Other" rather than reusing a hue, so
        # the slot lookup clamps instead of wrapping.
        last = theme.series_colour(len(theme.CATEGORICAL_LIGHT) - 1)
        assert theme.series_colour(99) == last

    def test_light_and_dark_are_the_same_slots_restepped(self):
        assert len(theme.CATEGORICAL_LIGHT) == len(theme.CATEGORICAL_DARK)

    def test_the_sequential_ramp_is_ordered_light_to_dark(self):
        def luminance(hex_colour: str) -> float:
            red, green, blue = (
                int(hex_colour[index : index + 2], 16) for index in (1, 3, 5)
            )
            return 0.2126 * red + 0.7152 * green + 0.0722 * blue

        levels = [luminance(step) for step in theme.SEQUENTIAL_BLUE]
        assert levels == sorted(levels, reverse=True)

    def test_status_colours_are_not_series_colours(self):
        assert not set(theme.STATUS.values()) & set(theme.CATEGORICAL_LIGHT)
