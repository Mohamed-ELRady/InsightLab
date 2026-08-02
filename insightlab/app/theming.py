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

#: Keys into the string catalogue, so the badge follows the run's language.
CONFIDENCE_KEYS = {
    "high": "confidence.high",
    "medium": "confidence.medium",
    "low": "confidence.low",
}


def is_dark() -> bool:
    """Whether the viewer is in dark mode, defaulting to light if unknown."""
    try:
        return getattr(st.context.theme, "type", "light") == "dark"
    except Exception:  # noqa: BLE001 - older hosts do not expose the context
        return False


def show_chart(chart: Chart, *, key: str = "", language=None) -> None:
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
        from ..core.language import DEFAULT, translate

        with st.expander(translate("chart.numbers", language or DEFAULT)):
            st.dataframe(chart.table, width="stretch", hide_index=True)


def kpi_tiles(kpis: list[Kpi], columns: int = 4) -> None:
    """A row of headline figures, each with its formula underneath.

    Deliberately no delta badge. Streamlit colours a delta by parsing it as a
    number, and a figure that already carries its own sign - a growth rate of
    -3.7% - would get a green upward badge beside it. A falling number shown in
    green is worse than no badge at all.
    """
    if not kpis:
        return
    for start in range(0, len(kpis), columns):
        row = st.columns(columns)
        for slot, kpi in zip(row, kpis[start : start + columns]):
            with slot:
                st.metric(
                    label=kpi.name,
                    value=kpi.display_value,
                    help=f"{kpi.formula}\n\n{kpi.interpretation}",
                )
                st.caption(kpi.formula)


def confidence_badge(confidence: str, language=None) -> str:
    """A label for how far a finding can be trusted.

    The word carries the meaning and the colour only reinforces it, so the
    badge still reads correctly in greyscale or for a colour-blind reader.
    """
    from ..core.language import DEFAULT, translate

    colour = CONFIDENCE_COLOURS.get(confidence, theme.STATUS["warning"])
    key = CONFIDENCE_KEYS.get(confidence)
    label = translate(key, language or DEFAULT) if key else confidence.title()
    return (
        f'<span style="display:inline-block;padding:2px 9px;border-radius:11px;'
        f'border:1px solid {colour};color:{colour};font-size:0.74rem;'
        f'font-weight:600;letter-spacing:0.01em;">{label}</span>'
    )


def stylesheet(rtl: bool = False, dark: bool = False) -> str:
    """Page-level styling, written for both light and dark surfaces.

    The accent on a saved fact is the categorical blue, which is a different
    step in each mode; the muted ink is the same in both by design. Anything
    with a surface behind it uses a translucent grey so it sits correctly on
    either without a second rule.
    """
    direction = _RTL_RULES if rtl else ""
    accent = theme.series_colour(0, dark)
    muted = theme.tokens(dark)["muted"]
    return direction + f"""
    <style>
      .il-card {{
        border: 1px solid rgba(137,135,129,0.28);
        border-radius: 12px;
        padding: 1.1rem 1.25rem;
        margin-bottom: 0.9rem;
      }}
      .il-card h4 {{ margin: 0 0 0.35rem 0; font-size: 1.02rem; }}
      .il-muted {{ color: {muted}; font-size: 0.86rem; }}
      .il-question {{
        font-size: 1.12rem;
        font-weight: 600;
        line-height: 1.45;
        margin-bottom: 0.5rem;
      }}
      .il-fact {{
        border-inline-start: 3px solid {accent};
        padding-block: 0.3rem;
        padding-inline-start: 0.7rem;
        margin-bottom: 0.55rem;
        font-size: 0.88rem;
      }}
      .il-tag {{
        display: inline-block;
        font-size: 0.7rem;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: {muted};
      }}
    </style>
    """


#: Applied when the run is in a right-to-left language. Streamlit lays out
#: left-to-right, so the direction is flipped at the document level and the
#: pieces that must stay left-to-right - charts, tables and code - are flipped
#: back. Numbers and dates read the same way in both directions, so the tables
#: are deliberately left alone.
_RTL_RULES = """
    <style>
      .stApp, [data-testid="stSidebar"] { direction: rtl; text-align: right; }
      [data-testid="stMarkdownContainer"] { text-align: right; }
      .stPlotlyChart, [data-testid="stDataFrame"], [data-testid="stTable"],
      pre, code { direction: ltr; text-align: left; }
      [data-testid="stMetricValue"], [data-testid="stMetricLabel"] {
        direction: rtl; text-align: right;
      }
      [data-testid="stChatInput"] textarea { direction: rtl; text-align: right; }
      .stRadio [data-testid="stMarkdownContainer"] { text-align: right; }
      ul, ol { padding-inline-start: 1.4rem; padding-inline-end: 0; }
    </style>
"""
