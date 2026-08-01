"""One visual system for every chart in the product.

Charts are read by people who are not analysts, so identity is never carried by
colour alone: series get a legend and, where they fit, direct labels, and every
chart is also available as a table. The categorical order is fixed and never
cycled - a ninth series folds into "Other" rather than reusing a hue.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

#: Fixed categorical order. Never reorder, never cycle past the end.
CATEGORICAL_LIGHT = (
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)

CATEGORICAL_DARK = (
    "#3987e5",
    "#d95926",
    "#199e70",
    "#c98500",
    "#d55181",
    "#008300",
    "#9085e9",
    "#e66767",
)

#: Blue ramp, light to dark, for continuous magnitude.
SEQUENTIAL_BLUE = (
    "#cde2fb",
    "#b7d3f6",
    "#9ec5f4",
    "#86b6ef",
    "#6da7ec",
    "#5598e7",
    "#3987e5",
    "#2a78d6",
    "#256abf",
    "#1c5cab",
    "#184f95",
    "#104281",
    "#0d366b",
)

#: Blue to red through a neutral grey, for values that have a sign.
DIVERGING_LIGHT = (
    (0.0, "#0d366b"),
    (0.25, "#3987e5"),
    (0.5, "#f0efec"),
    (0.75, "#e34948"),
    (1.0, "#8c1c1c"),
)

DIVERGING_DARK = (
    (0.0, "#104281"),
    (0.25, "#3987e5"),
    (0.5, "#383835"),
    (0.75, "#e66767"),
    (1.0, "#a52a2a"),
)

#: Reserved for state, never for a series.
STATUS = {
    "good": "#0ca30c",
    "warning": "#fab219",
    "serious": "#ec835a",
    "critical": "#d03b3b",
}

LIGHT = {
    "surface": "#fcfcfb",
    "primary": "#0b0b0b",
    "secondary": "#52514e",
    "muted": "#898781",
    "grid": "#e1e0d9",
    "axis": "#c3c2b7",
    "categorical": CATEGORICAL_LIGHT,
    "diverging": DIVERGING_LIGHT,
}

DARK = {
    "surface": "#1a1a19",
    "primary": "#ffffff",
    "secondary": "#c3c2b7",
    "muted": "#898781",
    "grid": "#2c2c2a",
    "axis": "#383835",
    "categorical": CATEGORICAL_DARK,
    "diverging": DIVERGING_DARK,
}

FONT_FAMILY = 'system-ui, -apple-system, "Segoe UI", sans-serif'

#: Series past this many fold into an "Other" group rather than taking a
#: ninth hue. Scatter-style charts compare every pair at once, so they stop
#: at three, which is the number that stays separable for colour-blind readers.
MAX_SERIES = 8
MAX_SERIES_ALL_PAIRS = 3

OTHER_LABEL = "Other"


def tokens(dark: bool = False) -> dict[str, Any]:
    return DARK if dark else LIGHT


def series_colour(index: int, dark: bool = False) -> str:
    """Colour for slot ``index``, clamped rather than cycled."""
    palette = tokens(dark)["categorical"]
    return palette[min(index, len(palette) - 1)]


def style(
    figure: go.Figure,
    *,
    dark: bool = False,
    show_legend: bool | None = None,
    x_title: str = "",
    y_title: str = "",
    height: int = 380,
) -> go.Figure:
    """Apply the house chrome to a finished figure.

    Backgrounds stay transparent so the same figure sits correctly on the app
    surface and on a report page. Axis ink is the mode-invariant muted grey, so
    a figure built for light mode does not become unreadable in dark mode.
    """
    palette = tokens(dark)
    if show_legend is None:
        show_legend = len(figure.data) > 1

    figure.update_layout(
        template="none",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY, size=13, color=palette["secondary"]),
        margin=dict(l=8, r=8, t=8, b=8),
        height=height,
        showlegend=show_legend,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(size=12, color=palette["secondary"]),
            bgcolor="rgba(0,0,0,0)",
        ),
        hoverlabel=dict(font_family=FONT_FAMILY, font_size=12),
        colorway=list(palette["categorical"]),
        bargap=0.28,
        barcornerradius=4,
    )
    # automargin everywhere: these figures are rendered side by side in a
    # two-column dashboard, where a long category name would otherwise be
    # clipped to a single letter.
    figure.update_xaxes(
        title_text=x_title,
        showgrid=False,
        zeroline=False,
        automargin=True,
        linecolor=palette["axis"],
        tickfont=dict(color=palette["muted"], size=11),
        title_font=dict(color=palette["muted"], size=11),
    )
    figure.update_yaxes(
        title_text=y_title,
        showgrid=True,
        gridcolor=palette["grid"],
        gridwidth=1,
        zeroline=False,
        automargin=True,
        linecolor="rgba(0,0,0,0)",
        tickfont=dict(color=palette["muted"], size=11),
        title_font=dict(color=palette["muted"], size=11),
    )
    return figure


def retheme(figure: go.Figure, dark: bool) -> go.Figure:
    """Re-apply chrome for the opposite mode without rebuilding the figure.

    Dark mode is a selected set of steps from the same ramps, not an automatic
    inversion, so the categorical colours are remapped slot for slot.
    """
    source = CATEGORICAL_LIGHT if dark else CATEGORICAL_DARK
    target = CATEGORICAL_DARK if dark else CATEGORICAL_LIGHT
    mapping = {a.casefold(): b for a, b in zip(source, target)}

    for trace in figure.data:
        marker = getattr(trace, "marker", None)
        if marker is not None and isinstance(getattr(marker, "color", None), str):
            replacement = mapping.get(marker.color.casefold())
            if replacement:
                trace.marker.color = replacement
        line = getattr(trace, "line", None)
        if line is not None and isinstance(getattr(line, "color", None), str):
            replacement = mapping.get(line.color.casefold())
            if replacement:
                trace.line.color = replacement

    return style(
        figure,
        dark=dark,
        show_legend=figure.layout.showlegend,
        x_title=figure.layout.xaxis.title.text or "",
        y_title=figure.layout.yaxis.title.text or "",
        height=figure.layout.height or 380,
    )


def fold_to_other(labels: list[str], values: list[float], limit: int = MAX_SERIES):
    """Keep the top ``limit - 1`` groups and total the rest as "Other".

    A ninth colour is never invented, so anything past the palette becomes one
    honest bucket instead of a hue the reader cannot name.
    """
    if len(labels) <= limit:
        return labels, values
    keep = limit - 1
    remainder = sum(values[keep:])
    return labels[:keep] + [OTHER_LABEL], values[:keep] + [remainder]
