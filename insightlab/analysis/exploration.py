"""Chart building for the exploration stage.

Charts are chosen by the job the data has to do, not by what looks impressive:
magnitude comparisons become bars, change over time becomes a line, the shape of
a single measure becomes a histogram, and the relationship between measures
becomes a diverging heatmap. Every chart carries a table alongside it, which is
both the accessibility fallback and, for many business users, the thing they
actually copy into a spreadsheet.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from ..core.state import Chart, DatasetProfile, Role
from . import theme
from .profiling import try_parse_datetime

#: The business questions a user can ask us to concentrate on.
AXES: dict[str, str] = {
    "sales": "Sales and revenue",
    "customers": "Customers",
    "products": "Products and services",
    "marketing": "Marketing and channels",
    "profits": "Profit and margins",
    "regions": "Regions and locations",
    "operations": "Operations and fulfilment",
}

#: How many bars a comparison chart shows before the tail is folded into Other.
TOP_N = 8

#: Below this many points a monthly trend line is not worth drawing.
MIN_TREND_POINTS = 4

#: Calendar parts stored as integers. They are labels, not quantities, so they
#: are excluded anywhere a number is treated as a magnitude.
CALENDAR_NUMBERS = frozenset({"year", "month", "day", "week", "day_of_month"})


@dataclass
class Columns:
    """The columns each chart builder needs, resolved once per run."""

    dates: list[str]
    measures: list[str]
    categories: list[str]
    booleans: list[str]
    labels: list[str]
    distinct: dict[str, int]
    primary_measure: str | None
    primary_date: str | None

    def category_named(self, *keywords: str) -> str | None:
        """A column with few enough groups to put on a chart axis."""
        return _first_named(self.categories, keywords)

    def entity_named(self, *keywords: str) -> str | None:
        """The column that names individual things, not groups of them.

        A file often has both ``customer_id`` and ``customer_segment``. Counting
        customers means counting the former; the latter has three values and
        would report three customers. Among the columns whose names match, the
        one with the most distinct values is the individual-level one, so that
        is what wins. Chart builders want :meth:`category_named` instead, since
        they need few enough groups to fit on an axis.
        """
        matches = [
            name
            for name in self.labels
            if any(keyword in name.casefold() for keyword in keywords)
        ]
        if not matches:
            return None
        return max(matches, key=lambda name: self.distinct.get(name, 0))

    def measure_named(self, *keywords: str) -> str | None:
        return _first_named(self.measures, keywords)


def _first_named(names: list[str], keywords: tuple[str, ...]) -> str | None:
    for name in names:
        lowered = name.casefold()
        if any(keyword in lowered for keyword in keywords):
            return name
    return None


def resolve_columns(frame: pd.DataFrame, profile: DatasetProfile) -> Columns:
    """Work out which columns are usable for charting."""
    present = set(frame.columns)
    dates = [
        column.name
        for column in profile.columns
        if column.role is Role.DATETIME and column.name in present
    ]
    # Calendar parts profile as numeric measures because that is how they are
    # stored, but averaging a month number is meaningless - they belong on the
    # category side only.
    measures = [
        column.name
        for column in profile.columns
        if column.role is Role.MEASURE
        and column.name in present
        and column.name not in CALENDAR_NUMBERS
    ]
    categories = [
        column.name
        for column in profile.columns
        if column.role is Role.CATEGORY
        and column.name in present
        and 1 < column.unique_count <= 60
    ]
    booleans = [
        column.name
        for column in profile.columns
        if column.role is Role.BOOLEAN and column.name in present
    ]
    # Anything that names a thing rather than measures one, with no cap on how
    # many distinct values it holds.
    labels = [
        column.name
        for column in profile.columns
        if column.role in (Role.CATEGORY, Role.IDENTIFIER, Role.TEXT, Role.BOOLEAN)
        and column.name in present
    ]

    # Columns added during feature engineering are not in the profile yet.
    for name in ("profit", "profit_margin_pct", "net_revenue", "revenue_per_unit"):
        if name in present and name not in measures:
            measures.append(name)
    for name in ("month_name", "quarter", "day_of_week", "revenue_tier", "year"):
        if name in present and name not in categories:
            categories.append(name)

    primary = None
    for keywords in (("revenue",), ("sales", "total"), ("amount", "value"), ("profit",)):
        for name in measures:
            if any(keyword in name.casefold() for keyword in keywords):
                primary = name
                break
        if primary:
            break
    if primary is None and measures:
        primary = measures[0]

    return Columns(
        dates=dates,
        measures=measures,
        categories=categories,
        booleans=booleans,
        labels=labels,
        distinct={
            column.name: column.unique_count
            for column in profile.columns
            if column.name in present
        },
        primary_measure=primary,
        primary_date=dates[0] if dates else None,
    )


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _datetime(frame: pd.DataFrame, column: str) -> pd.Series:
    series = frame[column]
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    parsed = try_parse_datetime(series)
    return parsed if parsed is not None else pd.to_datetime(series, errors="coerce")


def _compact(value: float) -> str:
    """Format a number the way a business person writes it."""
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.1f}B"
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:,.1f}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:,.1f}K"
    if magnitude >= 10:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _wrap(figure: go.Figure, chart_id: str, title: str, kind: str, description: str,
          axis: str, table: pd.DataFrame) -> Chart:
    return Chart(
        id=chart_id,
        title=title,
        kind=kind,
        description=description,
        axis=axis,
        figure_json=figure.to_json(),
        table=table,
    )


# ---------------------------------------------------------------------------
# Individual chart builders
# ---------------------------------------------------------------------------


def trend_chart(
    frame: pd.DataFrame, date_column: str, measure: str, axis: str
) -> Chart | None:
    """Total of ``measure`` per month, as a line."""
    dates = _datetime(frame, date_column)
    values = _numeric(frame, measure)
    working = pd.DataFrame({"period": dates.dt.to_period("M"), "value": values}).dropna()
    if working.empty:
        return None

    grouped = working.groupby("period", observed=True)["value"].sum().sort_index()
    if len(grouped) < MIN_TREND_POINTS:
        return None

    labels = [str(period) for period in grouped.index]
    figure = go.Figure(
        go.Scatter(
            x=labels,
            y=grouped.to_numpy(),
            mode="lines+markers",
            # Straight segments, not a spline: a curve between two months would
            # invent values for weeks that were never measured.
            line=dict(color=theme.series_colour(0), width=2),
            marker=dict(size=8, color=theme.series_colour(0)),
            hovertemplate="%{x}<br>%{y:,.0f}<extra></extra>",
            name=measure,
        )
    )
    theme.style(figure, y_title=measure.replace("_", " "), show_legend=False)

    first, last = float(grouped.iloc[0]), float(grouped.iloc[-1])
    change = ((last - first) / first * 100) if first else 0.0
    peak_period = str(grouped.idxmax())
    direction = "grew" if change > 2 else ("fell" if change < -2 else "stayed flat")
    description = (
        f"{measure.replace('_', ' ').title()} {direction} by {abs(change):.0f}% between "
        f"{labels[0]} and {labels[-1]}, peaking in {peak_period} at "
        f"{_compact(float(grouped.max()))}."
    )

    table = grouped.reset_index()
    table.columns = ["Month", measure]
    table["Month"] = table["Month"].astype(str)
    return _wrap(
        figure,
        f"trend_{measure}",
        f"{measure.replace('_', ' ').title()} over time",
        "line",
        description,
        axis,
        table,
    )


#: Categories that have a natural order of their own. Ranking these by size
#: hides the very pattern the user is looking for, so they keep calendar order
#: and are never folded into an "Other" bucket.
ORDERED_CATEGORIES: dict[str, list[str]] = {
    "month_name": [
        "January", "February", "March", "April", "May", "June",
        "July", "August", "September", "October", "November", "December",
    ],
    "day_of_week": [
        "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday",
    ],
    "quarter": ["Q1", "Q2", "Q3", "Q4"],
    "revenue_tier": ["Small", "Medium", "Large", "Very large"],
}


def category_bar(
    frame: pd.DataFrame,
    category: str,
    measure: str,
    axis: str,
    *,
    how: str = "sum",
    chart_id: str | None = None,
) -> Chart | None:
    """Compare a measure across the groups of one category."""
    values = _numeric(frame, measure)
    working = pd.DataFrame({"group": frame[category].astype(str), "value": values}).dropna()
    if working.empty or working["group"].nunique() < 2:
        return None

    grouped = working.groupby("group", observed=True)["value"].agg(how)

    natural_order = ORDERED_CATEGORIES.get(category)
    if natural_order:
        # Every bar is the same colour, so there is no palette pressure here -
        # showing all twelve months in order is more useful than a top eight.
        present = [name for name in natural_order if name in grouped.index]
        extras = [name for name in grouped.index if name not in natural_order]
        grouped = grouped.reindex(present + extras)
        labels = list(grouped.index)
        totals = [float(value) for value in grouped]
        folded = False
    else:
        grouped = grouped.sort_values(ascending=False)
        labels, totals = theme.fold_to_other(
            list(grouped.index), [float(value) for value in grouped], TOP_N
        )
        folded = theme.OTHER_LABEL in labels

    readable_measure = measure.replace("_", " ")
    readable_category = category.replace("_", " ")
    # What the bar length actually means. Summing a percentage would be
    # nonsense, so the wording has to follow the aggregation, not the column.
    if how == "count":
        quantity_label = "Number of rows"
        title = f"Number of rows by {readable_category}"
        value_label = "Rows"
    elif how == "mean":
        quantity_label = f"Average {readable_measure}"
        title = f"Average {readable_measure} by {readable_category}"
        value_label = f"Average {measure}"
    elif how == "median":
        quantity_label = f"Typical {readable_measure}"
        title = f"Typical {readable_measure} by {readable_category}"
        value_label = f"Median {measure}"
    else:
        quantity_label = f"Total {readable_measure}"
        title = f"{readable_measure.title()} by {readable_category}"
        value_label = f"Total {measure}"

    # Horizontal bars so long category names stay readable. Ranked charts put
    # the largest group at the top; naturally ordered ones keep their own order
    # running downwards.
    figure = go.Figure(
        go.Bar(
            x=totals[::-1],
            y=labels[::-1],
            orientation="h",
            marker=dict(color=theme.series_colour(0), cornerradius=4),
            text=[_compact(value) for value in totals[::-1]],
            textposition="outside",
            textfont=dict(size=11, color=theme.LIGHT["secondary"]),
            hovertemplate="%{y}<br>%{x:,.2f}<extra></extra>",
        )
    )
    theme.style(
        figure,
        show_legend=False,
        x_title=quantity_label,
        height=max(260, 46 * len(labels) + 80),
    )
    figure.update_xaxes(showgrid=True, gridcolor=theme.LIGHT["grid"])
    figure.update_yaxes(showgrid=False)

    # Describe the ranking using the named groups only. "Other" is a bucket of
    # leftovers, so calling it the smallest group would be false.
    named = [
        (label, value)
        for label, value in zip(labels, totals)
        if label != theme.OTHER_LABEL
    ]
    ranked = sorted(named, key=lambda item: item[1], reverse=True)
    top_label, top_value = ranked[0]
    bottom_label, bottom_value = ranked[-1]

    if how == "sum":
        total = sum(totals)
        share = (top_value / total * 100) if total else 0.0
        lead = (
            f"{top_label} leads on {readable_measure} with {_compact(top_value)}, "
            f"which is {share:.0f}% of the total."
        )
    elif how == "count":
        lead = f"{top_label} appears most often, in {_compact(top_value)} rows."
    else:
        lead = (
            f"{top_label} has the highest {quantity_label.lower()} at "
            f"{_compact(top_value)}."
        )

    tail = f" {bottom_label} is the lowest at {_compact(bottom_value)}."
    if folded:
        hidden = len(grouped) - (len(labels) - 1)
        tail += (
            f" The remaining {hidden} smaller groups are combined into "
            f'"{theme.OTHER_LABEL}".'
        )

    table = pd.DataFrame({category: labels, value_label: totals})
    return _wrap(
        figure,
        chart_id or f"bar_{measure}_by_{category}_{how}",
        title,
        "bar",
        lead + tail,
        axis,
        table,
    )


def distribution(frame: pd.DataFrame, measure: str, axis: str) -> Chart | None:
    """The shape of a single measure, as a histogram."""
    values = _numeric(frame, measure).dropna()
    if len(values) < 20 or values.nunique() < 5:
        return None

    figure = go.Figure(
        go.Histogram(
            x=values,
            nbinsx=32,
            marker=dict(color=theme.series_colour(0), cornerradius=4),
            hovertemplate="%{x}<br>%{y} rows<extra></extra>",
        )
    )
    theme.style(
        figure,
        show_legend=False,
        x_title=measure.replace("_", " "),
        y_title="Number of rows",
    )

    median = float(values.median())
    mean = float(values.mean())
    gap = "well above" if mean > median * 1.25 else (
        "close to" if abs(mean - median) < median * 0.1 else "above"
    )
    description = (
        f"Half of all values for {measure.replace('_', ' ')} sit below "
        f"{_compact(median)}. The average is {_compact(mean)}, {gap} the middle, "
        "which means a smaller number of large values is pulling it up."
        if mean > median
        else (
            f"Half of all values for {measure.replace('_', ' ')} sit below "
            f"{_compact(median)} and the average is {_compact(mean)}."
        )
    )

    quantiles = values.quantile([0, 0.25, 0.5, 0.75, 1.0])
    table = pd.DataFrame(
        {
            "Point": ["Lowest", "25% below", "Middle", "25% above", "Highest"],
            measure: [round(float(value), 2) for value in quantiles],
        }
    )
    return _wrap(
        figure,
        f"dist_{measure}",
        f"How {measure.replace('_', ' ')} is spread",
        "histogram",
        description,
        axis,
        table,
    )


def correlation_heatmap(
    frame: pd.DataFrame, measures: list[str], axis: str = "general"
) -> Chart | None:
    """Which measures move together, on a diverging blue-to-red scale."""
    # Calendar parts are stored as numbers but are labels, not quantities.
    # Correlating "month" with revenue would produce a number that reads as
    # meaningful and is not, so they are kept out of the matrix.
    usable = [
        name
        for name in measures
        if name in frame.columns and name not in CALENDAR_NUMBERS
    ]
    numeric = frame[usable].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.loc[:, numeric.nunique() > 1]
    if numeric.shape[1] < 3:
        return None
    numeric = numeric.iloc[:, :10]

    matrix = numeric.corr(numeric_only=True).round(2)
    labels = [name.replace("_", " ") for name in matrix.columns]

    figure = go.Figure(
        go.Heatmap(
            z=matrix.to_numpy(),
            x=labels,
            y=labels,
            zmin=-1,
            zmax=1,
            colorscale=[list(stop) for stop in theme.DIVERGING_LIGHT],
            xgap=2,
            ygap=2,
            text=matrix.to_numpy(),
            texttemplate="%{text:.2f}",
            textfont=dict(size=10),
            hovertemplate="%{y} and %{x}<br>%{z:.2f}<extra></extra>",
            colorbar=dict(
                thickness=10,
                outlinewidth=0,
                tickfont=dict(size=10, color=theme.LIGHT["muted"]),
            ),
        )
    )
    theme.style(figure, show_legend=False, height=max(320, 44 * len(labels) + 120))
    figure.update_yaxes(showgrid=False, autorange="reversed", automargin=True)
    figure.update_xaxes(tickangle=-30, automargin=True)

    pairs: list[tuple[float, str, str]] = []
    for row in range(len(matrix)):
        for column in range(row + 1, len(matrix)):
            pairs.append(
                (float(matrix.iat[row, column]), matrix.index[row], matrix.columns[column])
            )
    pairs.sort(key=lambda item: abs(item[0]), reverse=True)
    if pairs and abs(pairs[0][0]) >= 0.5:
        value, left, right = pairs[0]
        direction = "rise together" if value > 0 else "move in opposite directions"
        description = (
            f"{left.replace('_', ' ')} and {right.replace('_', ' ')} {direction} "
            f"most strongly (score {value:+.2f}, where +1 means they move exactly "
            "together and -1 means exactly opposite)."
        )
    else:
        description = (
            "No two measures move together strongly, which means each one is "
            "telling you something the others are not."
        )

    table = matrix.reset_index().rename(columns={"index": "Measure"})
    return _wrap(
        figure, "correlation", "Which measures move together", "heatmap",
        description, axis, table,
    )


def box_by_category(
    frame: pd.DataFrame, category: str, measure: str, axis: str
) -> Chart | None:
    """Spread of a measure within each group, not just the average."""
    working = pd.DataFrame(
        {"group": frame[category].astype(str), "value": _numeric(frame, measure)}
    ).dropna()
    if working.empty:
        return None

    order = (
        working.groupby("group", observed=True)["value"].median().sort_values(ascending=False)
    )
    groups = list(order.index)[: theme.MAX_SERIES]
    if len(groups) < 2:
        return None

    figure = go.Figure()
    for index, group in enumerate(groups):
        figure.add_trace(
            go.Box(
                y=working.loc[working["group"] == group, "value"],
                name=str(group),
                marker=dict(color=theme.series_colour(index), size=5),
                line=dict(width=2),
                boxpoints=False,
                hovertemplate="%{y:,.2f}<extra>" + str(group) + "</extra>",
            )
        )
    theme.style(
        figure,
        show_legend=False,
        y_title=measure.replace("_", " "),
        height=400,
    )

    top, bottom = groups[0], groups[-1]
    description = (
        f"A typical {measure.replace('_', ' ')} is highest in {top} "
        f"({_compact(float(order.iloc[0]))}) and lowest in {bottom} "
        f"({_compact(float(order.loc[bottom]))}). The height of each box shows how "
        "much variation there is inside the group, which the averages hide."
    )

    summary = (
        working[working["group"].isin(groups)]
        .groupby("group", observed=True)["value"]
        .agg(["count", "median", "mean", "min", "max"])
        .round(2)
        .reset_index()
    )
    summary.columns = [category, "Rows", "Middle", "Average", "Lowest", "Highest"]
    return _wrap(
        figure,
        f"box_{measure}_by_{category}",
        f"Spread of {measure.replace('_', ' ')} within each {category.replace('_', ' ')}",
        "box",
        description,
        axis,
        summary,
    )


def two_category_heatmap(
    frame: pd.DataFrame, row_category: str, column_category: str, measure: str, axis: str
) -> Chart | None:
    """A measure across two categories at once, on a single blue ramp."""
    working = pd.DataFrame(
        {
            "row": frame[row_category].astype(str),
            "column": frame[column_category].astype(str),
            "value": _numeric(frame, measure),
        }
    ).dropna()
    if working.empty:
        return None

    pivot = working.pivot_table(
        index="row", columns="column", values="value", aggfunc="sum", observed=True
    )
    if pivot.shape[0] < 2 or pivot.shape[1] < 2:
        return None
    pivot = pivot.loc[
        pivot.sum(axis=1).sort_values(ascending=False).index[:12],
        pivot.sum(axis=0).sort_values(ascending=False).index[:12],
    ]

    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=[str(value) for value in pivot.columns],
            y=[str(value) for value in pivot.index],
            colorscale=[
                [index / (len(theme.SEQUENTIAL_BLUE) - 1), colour]
                for index, colour in enumerate(theme.SEQUENTIAL_BLUE)
            ],
            xgap=2,
            ygap=2,
            hovertemplate="%{y} / %{x}<br>%{z:,.0f}<extra></extra>",
            colorbar=dict(
                thickness=10,
                outlinewidth=0,
                tickfont=dict(size=10, color=theme.LIGHT["muted"]),
            ),
        )
    )
    theme.style(figure, show_legend=False, height=max(320, 40 * pivot.shape[0] + 120))
    # Largest group at the top, where reading starts.
    figure.update_yaxes(showgrid=False, autorange="reversed", automargin=True)
    figure.update_xaxes(automargin=True)

    stacked = pivot.stack()
    best_row, best_column = stacked.idxmax()
    description = (
        f"The strongest combination is {best_row} with {best_column}, at "
        f"{_compact(float(stacked.max()))} of {measure.replace('_', ' ')}. "
        "Dark cells are where the business is concentrated; pale cells are gaps."
    )

    table = pivot.round(2).reset_index().rename(columns={"row": row_category})
    return _wrap(
        figure,
        f"heat_{measure}_{row_category}_{column_category}",
        f"{measure.replace('_', ' ').title()} by "
        f"{row_category.replace('_', ' ')} and {column_category.replace('_', ' ')}",
        "heatmap",
        description,
        axis,
        table,
    )


def scatter_relationship(
    frame: pd.DataFrame, x_measure: str, y_measure: str, axis: str,
    category: str | None = None,
) -> Chart | None:
    """Two measures against each other.

    Every point is compared with every other, so the series cap here is three,
    not eight - beyond that the colours stop being reliably distinguishable.
    """
    columns = {"x": _numeric(frame, x_measure), "y": _numeric(frame, y_measure)}
    if category and category in frame.columns:
        columns["group"] = frame[category].astype(str)
    working = pd.DataFrame(columns).dropna()
    if len(working) < 20:
        return None

    figure = go.Figure()
    if "group" in working.columns:
        top_groups = list(working["group"].value_counts().index[: theme.MAX_SERIES_ALL_PAIRS])
        working["group"] = np.where(
            working["group"].isin(top_groups), working["group"], theme.OTHER_LABEL
        )
        ordered = top_groups + (
            [theme.OTHER_LABEL] if (working["group"] == theme.OTHER_LABEL).any() else []
        )
        for index, group in enumerate(ordered):
            subset = working[working["group"] == group]
            colour = (
                theme.LIGHT["muted"]
                if group == theme.OTHER_LABEL
                else theme.series_colour(index)
            )
            figure.add_trace(
                go.Scatter(
                    x=subset["x"],
                    y=subset["y"],
                    mode="markers",
                    name=str(group),
                    marker=dict(
                        size=8,
                        color=colour,
                        opacity=0.7,
                        line=dict(width=2, color=theme.LIGHT["surface"]),
                    ),
                    hovertemplate="%{x:,.2f} / %{y:,.2f}<extra>" + str(group) + "</extra>",
                )
            )
    else:
        figure.add_trace(
            go.Scatter(
                x=working["x"],
                y=working["y"],
                mode="markers",
                marker=dict(
                    size=8,
                    color=theme.series_colour(0),
                    opacity=0.7,
                    line=dict(width=2, color=theme.LIGHT["surface"]),
                ),
                hovertemplate="%{x:,.2f} / %{y:,.2f}<extra></extra>",
            )
        )

    theme.style(
        figure,
        x_title=x_measure.replace("_", " "),
        y_title=y_measure.replace("_", " "),
        show_legend="group" in working.columns,
    )
    figure.update_xaxes(showgrid=True, gridcolor=theme.LIGHT["grid"])

    correlation = float(working["x"].corr(working["y"]))
    strength = (
        "move closely together"
        if abs(correlation) > 0.7
        else "are loosely related" if abs(correlation) > 0.35 else "are barely related"
    )
    description = (
        f"{x_measure.replace('_', ' ').title()} and {y_measure.replace('_', ' ')} "
        f"{strength} (score {correlation:+.2f}). Each dot is one row of your data."
    )

    table = working.head(200).rename(columns={"x": x_measure, "y": y_measure}).round(2)
    return _wrap(
        figure,
        f"scatter_{x_measure}_{y_measure}",
        f"{x_measure.replace('_', ' ').title()} against {y_measure.replace('_', ' ')}",
        "scatter",
        description,
        axis,
        table,
    )


def missing_values_chart(profile: DatasetProfile, axis: str = "general") -> Chart | None:
    """How complete each column is - the first thing to check before trusting a number."""
    incomplete = [
        column for column in profile.columns if column.missing_rate > 0
    ]
    if not incomplete:
        return None
    incomplete.sort(key=lambda column: column.missing_rate, reverse=True)
    incomplete = incomplete[:TOP_N]

    names = [column.name.replace("_", " ") for column in incomplete]
    shares = [round(column.missing_rate * 100, 1) for column in incomplete]

    figure = go.Figure(
        go.Bar(
            x=shares[::-1],
            y=names[::-1],
            orientation="h",
            marker=dict(color=theme.STATUS["warning"], cornerradius=4),
            text=[f"{share}%" for share in shares[::-1]],
            textposition="outside",
            textfont=dict(size=11, color=theme.LIGHT["secondary"]),
            hovertemplate="%{y}<br>%{x}% empty<extra></extra>",
        )
    )
    theme.style(
        figure,
        show_legend=False,
        x_title="Percent of rows with no value",
        height=max(240, 44 * len(names) + 80),
    )
    figure.update_xaxes(showgrid=True, gridcolor=theme.LIGHT["grid"])
    figure.update_yaxes(showgrid=False)

    description = (
        f"{incomplete[0].name.replace('_', ' ')} is the least complete column, with "
        f"{shares[0]}% of rows empty. Any total or average using it is based on the "
        "rows that do have a value."
    )
    table = pd.DataFrame({"Column": names, "Percent empty": shares})
    return _wrap(
        figure, "missing_values", "How complete each column is", "bar",
        description, axis, table,
    )


# ---------------------------------------------------------------------------
# Axis plans
# ---------------------------------------------------------------------------


def _axis_plan(
    axis: str, frame: pd.DataFrame, columns: Columns
) -> list[Chart]:
    """Build the charts that answer one business question."""
    charts: list[Chart] = []
    measure = columns.primary_measure
    date = columns.primary_date

    def add(chart: Chart | None) -> None:
        if chart is not None and not any(item.id == chart.id for item in charts):
            charts.append(chart)

    if axis == "sales":
        if date and measure:
            add(trend_chart(frame, date, measure, axis))
        for name in ("month_name", "quarter", "day_of_week"):
            if name in columns.categories and measure:
                add(category_bar(frame, name, measure, axis))
                break
        channel = columns.category_named("channel", "source", "medium")
        if channel and measure:
            add(category_bar(frame, channel, measure, axis))
        if measure:
            add(distribution(frame, measure, axis))

    elif axis == "customers":
        customer = columns.category_named("customer", "client", "account", "segment")
        if customer and measure:
            add(category_bar(frame, customer, measure, axis))
        segment = columns.category_named("segment", "tier", "class", "type")
        if segment and measure:
            add(box_by_category(frame, segment, measure, axis))
        value_column = columns.measure_named("customer_total_value")
        if value_column:
            add(distribution(frame, value_column, axis))
        if segment and measure and date:
            add(trend_chart(frame, date, measure, axis))

    elif axis == "products":
        product = columns.category_named("product", "category", "item", "sku", "service")
        if product and measure:
            add(category_bar(frame, product, measure, axis))
            margin = columns.measure_named("margin")
            if margin:
                add(category_bar(frame, product, margin, axis, how="mean"))
            add(box_by_category(frame, product, measure, axis))
        quantity = columns.measure_named("quantity", "qty", "units")
        if quantity and measure:
            add(scatter_relationship(frame, quantity, measure, axis, product))

    elif axis == "marketing":
        channel = columns.category_named("channel", "source", "campaign", "medium")
        if channel and measure:
            add(category_bar(frame, channel, measure, axis))
            add(box_by_category(frame, channel, measure, axis))
            product = columns.category_named("product", "category", "item")
            if product:
                add(two_category_heatmap(frame, channel, product, measure, axis))
        discount = columns.measure_named("discount")
        if discount and measure:
            add(scatter_relationship(frame, discount, measure, axis))

    elif axis == "profits":
        profit = columns.measure_named("profit") or measure
        margin = columns.measure_named("margin")
        if profit and date:
            add(trend_chart(frame, date, profit, axis))
        product = columns.category_named("product", "category", "item", "service")
        if product and profit:
            add(category_bar(frame, product, profit, axis))
        if margin:
            add(distribution(frame, margin, axis))
            if product:
                add(box_by_category(frame, product, margin, axis))
        if profit and measure and profit != measure:
            add(scatter_relationship(frame, measure, profit, axis))

    elif axis == "regions":
        region = columns.category_named("region", "city", "country", "state", "branch", "area")
        if region and measure:
            add(category_bar(frame, region, measure, axis))
            add(box_by_category(frame, region, measure, axis))
            product = columns.category_named("product", "category", "segment")
            if product:
                add(two_category_heatmap(frame, region, product, measure, axis))
            if date:
                add(trend_chart(frame, date, measure, axis))

    elif axis == "operations":
        status = columns.category_named("status", "state", "stage", "shipping", "delivery")
        if status and measure:
            add(category_bar(frame, status, measure, axis, how="count"))
        if "day_of_week" in columns.categories and measure:
            add(category_bar(frame, "day_of_week", measure, axis))
        quantity = columns.measure_named("quantity", "qty", "units")
        if quantity:
            add(distribution(frame, quantity, axis))
        if date and quantity:
            add(trend_chart(frame, date, quantity, axis))

    return charts


def general_charts(
    frame: pd.DataFrame, profile: DatasetProfile, columns: Columns
) -> list[Chart]:
    """The charts worth showing whatever the user asked to focus on."""
    charts: list[Chart] = []
    if columns.primary_date and columns.primary_measure:
        chart = trend_chart(frame, columns.primary_date, columns.primary_measure, "general")
        if chart:
            charts.append(chart)
    if columns.primary_measure:
        chart = distribution(frame, columns.primary_measure, "general")
        if chart:
            charts.append(chart)
    chart = correlation_heatmap(frame, columns.measures, "general")
    if chart:
        charts.append(chart)
    chart = missing_values_chart(profile, "general")
    if chart:
        charts.append(chart)
    return charts


def build_charts(
    frame: pd.DataFrame,
    profile: DatasetProfile,
    axes: list[str],
    *,
    include_general: bool = True,
) -> list[Chart]:
    """Every chart for the requested business questions, deduplicated."""
    columns = resolve_columns(frame, profile)
    charts: list[Chart] = []
    seen: set[str] = set()

    def extend(new_charts: list[Chart]) -> None:
        for chart in new_charts:
            if chart.id not in seen:
                seen.add(chart.id)
                charts.append(chart)

    if include_general:
        extend(general_charts(frame, profile, columns))
    for axis in axes:
        if axis in AXES:
            extend(_axis_plan(axis, frame, columns))
    return charts


def available_axes(frame: pd.DataFrame, profile: DatasetProfile) -> list[str]:
    """Business questions this dataset can actually answer.

    Offering the user a "regions" analysis when there is no location column
    would waste their time, so the menu is filtered to what the data supports.
    """
    columns = resolve_columns(frame, profile)
    supported: list[str] = []
    if columns.primary_measure:
        supported.append("sales")
    if columns.category_named("customer", "client", "account", "segment"):
        supported.append("customers")
    if columns.category_named("product", "category", "item", "sku", "service"):
        supported.append("products")
    if columns.category_named("channel", "source", "campaign", "medium") or columns.measure_named(
        "discount"
    ):
        supported.append("marketing")
    if columns.measure_named("profit", "margin", "cost"):
        supported.append("profits")
    if columns.category_named("region", "city", "country", "state", "branch", "area"):
        supported.append("regions")
    if columns.category_named("status", "state", "stage") or columns.measure_named(
        "quantity", "qty", "units"
    ):
        supported.append("operations")
    return supported or ["sales"]
