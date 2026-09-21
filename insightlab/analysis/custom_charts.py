"""User-directed charts built safely from one or two selected columns.

The automatic analysis still chooses charts from the meaning it inferred.  This
module covers the complementary case: the user has a specific comparison in
mind and should be able to draw it without writing code or changing the data.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from . import theme
from .profiling import try_parse_datetime
from ..core.state import Chart

ChartKind = Literal[
    "bar", "line", "area", "scatter", "histogram", "box", "pie", "heatmap"
]

CHART_KINDS: tuple[ChartKind, ...] = (
    "bar", "line", "area", "scatter", "histogram", "box", "pie", "heatmap"
)
AGGREGATIONS = ("sum", "mean", "median", "count")
MAX_GROUPS = 20
MAX_POINTS = 600


class CustomChartError(ValueError):
    """A clear, user-correctable chart configuration problem."""


def numeric_columns(frame: pd.DataFrame) -> list[str]:
    """Columns containing enough real numbers to use as a value axis."""
    result: list[str] = []
    for column in frame.columns:
        series = frame[column]
        if pd.api.types.is_bool_dtype(series):
            continue
        non_empty = int(series.notna().sum())
        if non_empty == 0:
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() >= min(non_empty, max(3, math.ceil(non_empty * 0.7))):
            result.append(str(column))
    return result


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise CustomChartError(f"Column {column!r} does not exist.")
    values = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    if values.notna().sum() < 2:
        raise CustomChartError(f"{column} does not contain enough numeric values.")
    return values


def _name(column: str) -> str:
    return str(column).replace("_", " ")


def _grouped(
    frame: pd.DataFrame,
    group_column: str,
    value_column: str | None,
    aggregation: str,
    *,
    max_groups: int = MAX_GROUPS,
) -> pd.DataFrame:
    groups = frame[group_column].where(frame[group_column].notna(), "(empty)").astype(str)
    if value_column:
        values = _numeric(frame, value_column)
        working = pd.DataFrame({"group": groups, "value": values}).dropna(subset=["value"])
        grouped = working.groupby("group", observed=True)["value"].agg(aggregation)
    else:
        grouped = groups.value_counts(dropna=False)
    grouped = grouped.sort_values(ascending=False)
    if len(grouped) > max_groups:
        head = grouped.iloc[: max_groups - 1]
        tail = grouped.iloc[max_groups - 1 :]
        # Totals and counts are additive, so the tail has an honest "Other"
        # value. An average of group averages (or medians) would be false
        # without the original group weights, so those views show the top
        # groups only.
        grouped = (
            pd.concat([head, pd.Series({theme.OTHER_LABEL: float(tail.sum())})])
            if aggregation in {"sum", "count"} or not value_column
            else grouped.iloc[:max_groups]
        )
    result = grouped.rename("value").reset_index()
    result.columns = ["group", "value"]
    return result


def _time_or_values(series: pd.Series) -> tuple[pd.Series, bool]:
    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, errors="coerce"), True
    parsed = try_parse_datetime(series)
    if parsed is not None:
        return parsed, True
    return series, False


def _line_table(
    frame: pd.DataFrame, x_column: str, y_column: str, aggregation: str
) -> pd.DataFrame:
    x_values, is_time = _time_or_values(frame[x_column])
    working = pd.DataFrame({"x": x_values, "y": _numeric(frame, y_column)}).dropna()
    if working.empty:
        raise CustomChartError("The selected columns have no rows that can be plotted together.")
    if is_time:
        span = working["x"].max() - working["x"].min()
        if span > pd.Timedelta(days=180):
            working["x"] = working["x"].dt.to_period("M").dt.to_timestamp()
        elif span > pd.Timedelta(days=3):
            working["x"] = working["x"].dt.floor("D")
    grouped = working.groupby("x", observed=True)["y"].agg(aggregation).sort_index()
    if len(grouped) > MAX_POINTS:
        positions = np.linspace(0, len(grouped) - 1, MAX_POINTS, dtype=int)
        grouped = grouped.iloc[np.unique(positions)]
    if len(grouped) < 2:
        raise CustomChartError("This chart needs at least two distinct values on the X axis.")
    return grouped.reset_index()


def _copy(arabic: bool, english: str, arabic_text: str) -> str:
    return arabic_text if arabic else english


def build_custom_chart(
    frame: pd.DataFrame,
    *,
    chart_id: str,
    kind: ChartKind,
    x_column: str,
    y_column: str | None = None,
    aggregation: str = "sum",
    title: str = "",
    language_code: str = "en",
) -> Chart:
    """Build a chart and its accessible table from the user's exact choices."""
    if frame is None or frame.empty:
        raise CustomChartError("There is no data to chart.")
    if kind not in CHART_KINDS:
        raise CustomChartError("Choose a supported chart type.")
    if x_column not in frame.columns:
        raise CustomChartError("Choose a valid first column.")
    if aggregation not in AGGREGATIONS:
        raise CustomChartError("Choose a supported calculation.")

    ar = language_code == "ar"
    x_name = _name(x_column)
    y_name = _name(y_column) if y_column else ""
    figure = go.Figure()
    table: pd.DataFrame
    description: str
    group_column = ""
    measure = y_column or ""
    used_aggregation = aggregation

    if kind == "histogram":
        values = _numeric(frame, x_column).dropna()
        figure.add_trace(go.Histogram(
            x=values, nbinsx=min(40, max(10, int(math.sqrt(len(values))))),
            marker=dict(color=theme.series_colour(0), cornerradius=4),
            hovertemplate="%{x}<br>%{y}<extra></extra>",
        ))
        theme.style(figure, x_title=x_name, y_title=_copy(ar, "Rows", "عدد الصفوف"), show_legend=False)
        table = pd.DataFrame({
            _copy(ar, "Statistic", "الإحصاء"): [_copy(ar, "Rows", "الصفوف"), _copy(ar, "Average", "المتوسط"), _copy(ar, "Median", "الوسيط"), _copy(ar, "Minimum", "الأدنى"), _copy(ar, "Maximum", "الأعلى")],
            _copy(ar, "Value", "القيمة"): [len(values), values.mean(), values.median(), values.min(), values.max()],
        }).round(3)
        default_title = _copy(ar, f"Distribution of {x_name}", f"توزيع {x_name}")
        description = _copy(ar, f"Shows the shape of {x_name} across {len(values):,} usable rows.", f"يوضح شكل توزيع {x_name} عبر {len(values):,} صفًا صالحًا.")
        measure = x_column
        used_aggregation = "distribution"

    elif kind == "scatter":
        if not y_column:
            raise CustomChartError("A scatter chart needs two numeric columns.")
        working = pd.DataFrame({x_column: _numeric(frame, x_column), y_column: _numeric(frame, y_column)}).dropna()
        if len(working) > MAX_POINTS:
            working = working.sample(MAX_POINTS, random_state=42).sort_index()
        if len(working) < 2:
            raise CustomChartError("The selected columns do not have enough matching numeric rows.")
        figure.add_trace(go.Scatter(
            x=working[x_column], y=working[y_column], mode="markers",
            marker=dict(size=8, color=theme.series_colour(0), opacity=0.7, line=dict(width=1, color=theme.LIGHT["surface"])),
            hovertemplate="%{x:,.3f} / %{y:,.3f}<extra></extra>",
        ))
        theme.style(figure, x_title=x_name, y_title=y_name, show_legend=False)
        figure.update_xaxes(showgrid=True, gridcolor=theme.LIGHT["grid"])
        score = working[x_column].corr(working[y_column])
        table = working.head(500).round(3)
        default_title = _copy(ar, f"{x_name} against {y_name}", f"{x_name} مقابل {y_name}")
        description = _copy(ar, f"Each point is one row. The correlation score is {score:+.2f}.", f"كل نقطة تمثل صفًا واحدًا، ودرجة الارتباط بين العمودين {score:+.2f}.")
        used_aggregation = "none"

    elif kind in {"line", "area"}:
        if not y_column:
            raise CustomChartError("This chart needs a numeric Y column.")
        table = _line_table(frame, x_column, y_column, aggregation)
        fill = "tozeroy" if kind == "area" else None
        figure.add_trace(go.Scatter(
            x=table["x"], y=table["y"], mode="lines+markers", fill=fill,
            line=dict(color=theme.series_colour(0), width=2),
            marker=dict(size=6, color=theme.series_colour(0)),
            hovertemplate="%{x}<br>%{y:,.3f}<extra></extra>",
        ))
        theme.style(figure, x_title=x_name, y_title=y_name, show_legend=False)
        table.columns = [x_column, y_column]
        word = _copy(ar, "Area", "مساحة") if kind == "area" else _copy(ar, "Trend", "تغيّر")
        default_title = _copy(ar, f"{word} of {y_name} by {x_name}", f"{word} {y_name} حسب {x_name}")
        description = _copy(ar, f"Shows {aggregation} {y_name} across the ordered values of {x_name}.", f"يوضح {y_name} عبر القيم المرتبة في {x_name} باستخدام التجميع المختار.")
        group_column = x_column

    elif kind == "bar":
        table = _grouped(frame, x_column, y_column, aggregation)
        labels, values = table["group"].tolist(), table["value"].tolist()
        figure.add_trace(go.Bar(
            x=values[::-1], y=labels[::-1], orientation="h",
            marker=dict(color=theme.series_colour(0), cornerradius=4),
            hovertemplate="%{y}<br>%{x:,.3f}<extra></extra>",
        ))
        theme.style(figure, x_title=y_name or _copy(ar, "Rows", "عدد الصفوف"), show_legend=False, height=max(300, 34 * len(labels) + 100))
        figure.update_yaxes(showgrid=False)
        table.columns = [x_column, y_column or _copy(ar, "Rows", "عدد الصفوف")]
        default_title = _copy(ar, f"{y_name or 'Rows'} by {x_name}", f"{y_name or 'عدد الصفوف'} حسب {x_name}")
        description = _copy(ar, f"Compares {y_name or 'row count'} across {len(labels)} groups in {x_name}.", f"يقارن {y_name or 'عدد الصفوف'} بين {len(labels)} مجموعة في {x_name}.")
        group_column = x_column
        if not y_column:
            used_aggregation = "count"

    elif kind == "pie":
        table = _grouped(
            frame, x_column, y_column, aggregation, max_groups=theme.MAX_SERIES
        )
        labels, values = table["group"].tolist(), table["value"].tolist()
        figure.add_trace(go.Pie(
            labels=labels, values=values, hole=0.42,
            marker=dict(colors=[theme.series_colour(i) for i in range(len(labels))]),
            textinfo="percent+label", hovertemplate="%{label}<br>%{value:,.3f} · %{percent}<extra></extra>",
        ))
        theme.style(figure, show_legend=False, height=430)
        table.columns = [x_column, y_column or _copy(ar, "Rows", "عدد الصفوف")]
        default_title = _copy(ar, f"Share of {y_name or 'rows'} by {x_name}", f"حصة {y_name or 'الصفوف'} حسب {x_name}")
        description = _copy(ar, f"Shows how the total is divided across {len(labels)} groups in {x_name}.", f"يوضح كيفية توزيع الإجمالي بين {len(labels)} مجموعة في {x_name}.")
        group_column = x_column
        if not y_column:
            used_aggregation = "count"

    elif kind == "box":
        values = _numeric(frame, x_column)
        if y_column:
            working = pd.DataFrame({"group": frame[y_column].astype(str), "value": values}).dropna()
            groups = working.groupby("group", observed=True)["value"].median().sort_values(ascending=False).index[: theme.MAX_SERIES]
            for index, group in enumerate(groups):
                figure.add_trace(go.Box(y=working.loc[working["group"] == group, "value"], name=str(group), boxpoints=False, marker_color=theme.series_colour(index)))
            table = working[working["group"].isin(groups)].groupby("group", observed=True)["value"].agg(["count", "median", "mean", "min", "max"]).round(3).reset_index()
            table.rename(columns={"group": y_column}, inplace=True)
            group_column = y_column
            default_title = _copy(ar, f"Spread of {x_name} by {y_name}", f"انتشار {x_name} حسب {y_name}")
            description = _copy(ar, f"Compares the middle and spread of {x_name} inside each {y_name} group.", f"يقارن الوسيط ومدى انتشار {x_name} داخل مجموعات {y_name}.")
        else:
            clean = values.dropna()
            figure.add_trace(go.Box(y=clean, name=x_name, boxpoints="outliers", marker_color=theme.series_colour(0)))
            table = pd.DataFrame({"count": [clean.count()], "median": [clean.median()], "mean": [clean.mean()], "min": [clean.min()], "max": [clean.max()]}).round(3)
            default_title = _copy(ar, f"Spread of {x_name}", f"انتشار {x_name}")
            description = _copy(ar, f"Shows the middle, spread and unusual values of {x_name}.", f"يوضح الوسيط ومدى الانتشار والقيم غير المعتادة في {x_name}.")
        theme.style(figure, y_title=x_name, show_legend=False, height=420)
        measure = x_column
        used_aggregation = "median"

    else:  # heatmap: two categorical columns, counted together.
        if not y_column:
            raise CustomChartError("A heatmap needs two columns.")
        working = pd.DataFrame({"row": frame[x_column].astype(str), "column": frame[y_column].astype(str)})
        top_rows = working["row"].value_counts().index[:15]
        top_columns = working["column"].value_counts().index[:15]
        working = working[working["row"].isin(top_rows) & working["column"].isin(top_columns)]
        pivot = pd.crosstab(working["row"], working["column"])
        if pivot.shape[0] < 2 or pivot.shape[1] < 2:
            raise CustomChartError("Each selected column needs at least two distinct values.")
        figure.add_trace(go.Heatmap(
            z=pivot.to_numpy(), x=[str(value) for value in pivot.columns], y=[str(value) for value in pivot.index],
            colorscale=[[i / (len(theme.SEQUENTIAL_BLUE) - 1), colour] for i, colour in enumerate(theme.SEQUENTIAL_BLUE)],
            xgap=2, ygap=2, hovertemplate="%{y} / %{x}<br>%{z} rows<extra></extra>",
        ))
        theme.style(figure, show_legend=False, height=max(360, 34 * len(pivot) + 100))
        figure.update_yaxes(showgrid=False, autorange="reversed")
        table = pivot.reset_index().rename(columns={"row": x_column})
        default_title = _copy(ar, f"{x_name} with {y_name}", f"{x_name} مع {y_name}")
        description = _copy(ar, "Darker cells contain more rows for that combination.", "الخلايا الأغمق تحتوي على عدد أكبر من الصفوف لهذا التقاطع.")
        group_column = x_column
        measure = ""
        used_aggregation = "count"

    return Chart(
        id=chart_id,
        title=title.strip() or default_title,
        kind=kind,
        description=description,
        axis="custom",
        figure_json=figure.to_json(),
        table=table,
        group_column=group_column,
        measure=measure,
        aggregation=used_aggregation,
    )
