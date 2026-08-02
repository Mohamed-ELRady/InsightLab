"""Dark mode.

Figures are built once against the light palette and stored, so every colour a
builder set has to move when the mode does. The ones easiest to miss are the
ones that fail worst: a value label left at the light secondary ink is
near-invisible on the dark surface, and the surface-coloured ring that
separates overlapping dots becomes a white halo.
"""

from __future__ import annotations

import plotly.io as plotly_io
import pytest

from insightlab.analysis import theme
from insightlab.analysis.exploration import build_charts
from insightlab.analysis.features import suggest_features
from insightlab.analysis.profiling import profile_dataset


@pytest.fixture
def charts(sample_frame):
    import pandas as pd

    frame = sample_frame.copy()
    frame["order_date"] = pd.to_datetime(frame["order_date"])
    profile = profile_dataset(frame)
    for suggestion in suggest_features(frame, profile):
        frame, _ = suggestion.apply(frame)
    profile = profile_dataset(frame)
    return {
        chart.id: chart
        # regions is included because it is the axis that builds the
        # two-category heatmap, which is the sequential-ramp case.
        for chart in build_charts(
            frame, profile, ["sales", "products", "profits", "regions"]
        )
    }


def trace_colours(figure) -> list[str]:
    """Every colour a builder set on a trace, flattened."""
    found: list[str] = []
    for trace in figure.data:
        marker = getattr(trace, "marker", None)
        if marker is not None:
            for value in (
                getattr(marker, "color", None),
                getattr(getattr(marker, "line", None), "color", None),
            ):
                if isinstance(value, str):
                    found.append(value.casefold())
        for holder in ("line", "textfont"):
            item = getattr(trace, holder, None)
            colour = getattr(item, "color", None) if item is not None else None
            if isinstance(colour, str):
                found.append(colour.casefold())
    return found


LIGHT_ONLY = {
    theme.LIGHT[role].casefold()
    for role in ("surface", "primary", "secondary", "grid", "axis")
} | {colour.casefold() for colour in theme.CATEGORICAL_LIGHT}


class TestPalette:
    def test_the_two_modes_are_the_same_slots_restepped(self):
        assert len(theme.CATEGORICAL_LIGHT) == len(theme.CATEGORICAL_DARK)

    def test_a_series_colour_differs_between_modes(self):
        assert theme.series_colour(0, dark=False) != theme.series_colour(0, dark=True)

    def test_the_muted_ink_is_deliberately_shared(self):
        # It is the one token chosen to read on both surfaces, so nothing needs
        # to swap it and the swap table must not contain it.
        assert theme.LIGHT["muted"] == theme.DARK["muted"]

    def test_the_swap_table_has_no_ambiguous_entries(self):
        # A hex appearing twice on the same side of the mapping would make the
        # remap non-deterministic.
        for tokens in (theme.LIGHT, theme.DARK):
            values = [
                tokens[role]
                for role in ("surface", "primary", "secondary", "grid", "axis")
            ]
            assert len(values) == len(set(values))


class TestRetheming:
    def test_no_light_colour_survives_the_switch_to_dark(self, charts):
        for chart in charts.values():
            figure = plotly_io.from_json(chart.figure_json)
            theme.retheme(figure, dark=True)

            leftovers = [
                colour
                for colour in trace_colours(figure)
                # The dark secondary happens to equal the light axis colour, so
                # it is excluded by value rather than flagged as a leftover.
                if colour in LIGHT_ONLY
                and colour != theme.DARK["secondary"].casefold()
            ]
            assert not leftovers, f"{chart.id} kept {leftovers}"

    def test_bar_value_labels_are_legible_on_the_dark_surface(self, charts):
        bar = next(chart for chart in charts.values() if chart.kind == "bar")
        figure = plotly_io.from_json(bar.figure_json)
        assert figure.data[0].textfont.color == theme.LIGHT["secondary"]

        theme.retheme(figure, dark=True)
        assert figure.data[0].textfont.color == theme.DARK["secondary"]

    def test_the_ring_around_overlapping_dots_follows_the_surface(self, charts):
        scatter = next(
            (chart for chart in charts.values() if chart.kind == "scatter"), None
        )
        if scatter is None:
            pytest.skip("this sample produced no scatter chart")

        figure = plotly_io.from_json(scatter.figure_json)
        theme.retheme(figure, dark=True)
        for trace in figure.data:
            ring = getattr(getattr(trace, "marker", None), "line", None)
            if ring is not None and isinstance(ring.color, str):
                assert ring.color == theme.DARK["surface"], "a white halo on dark"

    def test_a_diverging_scale_takes_the_dark_neutral_midpoint(self, charts):
        figure = plotly_io.from_json(charts["correlation"].figure_json)
        before = figure.data[0].colorscale
        midpoint_before = before[len(before) // 2][1]

        theme.retheme(figure, dark=True)
        after = figure.data[0].colorscale
        midpoint_after = after[len(after) // 2][1]

        assert midpoint_before != midpoint_after
        assert midpoint_after == theme.DIVERGING_DARK[2][1]

    def test_a_sequential_ramp_is_left_alone(self, charts):
        # Only the diverging scale has a neutral midpoint that has to change
        # with the surface; the blue ramp reads the same on both.
        heatmap = next(
            (
                chart
                for chart in charts.values()
                if chart.kind == "heatmap" and chart.id != "correlation"
            ),
            None,
        )
        if heatmap is None:
            pytest.skip("this sample produced no sequential heatmap")

        figure = plotly_io.from_json(heatmap.figure_json)
        before = figure.data[0].colorscale
        theme.retheme(figure, dark=True)
        assert figure.data[0].colorscale == before

    def test_switching_there_and_back_returns_the_original(self, charts):
        for chart in charts.values():
            figure = plotly_io.from_json(chart.figure_json)
            original = trace_colours(figure)

            theme.retheme(figure, dark=True)
            theme.retheme(figure, dark=False)

            assert trace_colours(figure) == original, chart.id


class TestInterface:
    def test_the_stylesheet_uses_the_right_accent_for_each_mode(self):
        from insightlab.app.theming import stylesheet

        assert theme.series_colour(0, False) in stylesheet(False, False)
        assert theme.series_colour(0, True) in stylesheet(False, True)

    def test_both_modes_are_configured_for_streamlit(self):
        # Without a [theme.dark] section the app's own theme switch drops from
        # our design into Streamlit's default dark, which the charts were never
        # validated against.
        import tomllib
        from pathlib import Path

        config = Path(__file__).resolve().parents[1] / ".streamlit" / "config.toml"
        settings = tomllib.loads(config.read_text(encoding="utf-8"))

        assert settings["theme"]["backgroundColor"].casefold() == theme.LIGHT["surface"]
        assert (
            settings["theme"]["dark"]["backgroundColor"].casefold()
            == theme.DARK["surface"]
        )

    def test_the_chat_chart_is_built_in_the_mode_it_is_shown_in(self, sample_frame):
        import pandas as pd

        from insightlab.analysis.query import QueryPlan, QueryResult
        from insightlab.app.conversation import _figure

        result = QueryResult(
            plan=QueryPlan(measure="revenue"),
            table=pd.DataFrame({"region": ["Cairo", "Giza"], "Total revenue": [5.0, 3.0]}),
            headline="",
            row_count=2,
            chart_kind="bar",
        )
        light = _figure(result, dark=False)
        dark = _figure(result, dark=True)

        assert light.data[0].marker.color == theme.series_colour(0, False)
        assert dark.data[0].marker.color == theme.series_colour(0, True)
        assert dark.data[0].textfont.color == theme.DARK["secondary"]
