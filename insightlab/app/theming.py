"""Screen presentation: theme detection and the small visual pieces.

Charts are stored once, built against the light palette. Dark mode is a
selected set of steps from the same ramps rather than an automatic inversion, so
switching modes remaps each series slot instead of flipping colours.
"""

from __future__ import annotations

import plotly.io as plotly_io
import streamlit as st

from ..analysis import theme
from ..core.state import Chart, Kpi

#: Confidence levels mapped onto the reserved status colours. These are never
#: used for a data series.
CONFIDENCE_COLOURS = {
    "high": theme.STATUS["good"],
    "medium": theme.STATUS["warning"],
    "low": theme.STATUS["serious"],
}

CONFIDENCE_LABELS = {
    "high": "Strong evidence",
    "medium": "Worth checking",
    "low": "Treat as a hint",
}


def is_dark() -> bool:
    """Whether the viewer is in dark mode, defaulting to light if unknown."""
    try:
        return getattr(st.context.theme, "type", "light") == "dark"
    except Exception:  # noqa: BLE001 - older hosts do not expose the context
        return False


def show_chart(chart: Chart, *, key: str = "") -> None:
    """Render a chart with its explanation and a table fallback.

    Three of the light-mode series colours sit below 3:1 against the surface, so
    every chart ships with the numbers available in text - which is also what a
    screen reader and a user who wants to copy the figures need.
    """
    if not chart.figure_json:
        return

    figure = plotly_io.from_json(chart.figure_json)
    if is_dark():
        theme.retheme(figure, dark=True)

    st.markdown(f"**{chart.title}**")
    st.plotly_chart(
        figure,
        width="stretch",
        key=key or f"chart_{chart.id}",
        config={"displayModeBar": False},
    )
    st.caption(chart.description)

    if chart.table is not None and not chart.table.empty:
        with st.expander("See the numbers behind this chart"):
            st.dataframe(chart.table, width="stretch", hide_index=True)


def kpi_tiles(kpis: list[Kpi], columns: int = 4) -> None:
    """A row of headline figures, each with its formula underneath."""
    if not kpis:
        return
    for start in range(0, len(kpis), columns):
        row = st.columns(columns)
        for slot, kpi in zip(row, kpis[start : start + columns]):
            with slot:
                st.metric(
                    label=kpi.name,
                    value=kpi.display_value,
                    delta=kpi.trend or None,
                    help=f"{kpi.formula}\n\n{kpi.interpretation}",
                )
                st.caption(kpi.formula)


def confidence_badge(confidence: str) -> str:
    """A label for how far a finding can be trusted.

    The word carries the meaning and the colour only reinforces it, so the
    badge still reads correctly in greyscale or for a colour-blind reader.
    """
    colour = CONFIDENCE_COLOURS.get(confidence, theme.STATUS["warning"])
    label = CONFIDENCE_LABELS.get(confidence, confidence.title())
    return (
        f'<span style="display:inline-block;padding:2px 9px;border-radius:11px;'
        f'border:1px solid {colour};color:{colour};font-size:0.74rem;'
        f'font-weight:600;letter-spacing:0.01em;">{label}</span>'
    )


def stylesheet() -> str:
    """Page-level styling, written for both light and dark surfaces."""
    return """
    <style>
      .il-card {
        border: 1px solid rgba(137,135,129,0.28);
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.9rem;
      }
      .il-card h4 { margin: 0 0 0.35rem 0; font-size: 1.02rem; }
      .il-muted { color: #898781; font-size: 0.86rem; }
      .il-question {
        font-size: 1.12rem;
        font-weight: 600;
        line-height: 1.45;
        margin-bottom: 0.5rem;
      }
      .il-fact {
        border-left: 3px solid #2a78d6;
        padding: 0.3rem 0 0.3rem 0.7rem;
        margin-bottom: 0.55rem;
        font-size: 0.88rem;
      }
      .il-tag {
        display: inline-block;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #898781;
      }
    </style>
    """
