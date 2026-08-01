"""Key performance indicators computed from whatever columns exist.

A KPI here is not just a number: it carries the formula it came from, so the
user can check it against their own books, and a sentence saying what the number
means for them. A figure with no interpretation is what makes dashboards get
ignored.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from ..core.state import DatasetProfile, Kpi, Role
from .exploration import Columns, resolve_columns
from .profiling import try_parse_datetime


@dataclass
class KpiDefinition:
    """A KPI we can offer, and the test for whether this dataset supports it."""

    id: str
    name: str
    why: str
    supported: Callable[[Columns, pd.DataFrame], bool]
    compute: Callable[[Columns, pd.DataFrame], Kpi | None]


def _money(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f}B"
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:,.2f}M"
    if magnitude >= 1_000:
        return f"{value:,.0f}"
    return f"{value:,.2f}"


def _count(value: float) -> str:
    return f"{value:,.0f}"


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="coerce")


def _monthly_totals(frame: pd.DataFrame, columns: Columns) -> pd.Series | None:
    if not columns.primary_date or not columns.primary_measure:
        return None
    series = frame[columns.primary_date]
    if not pd.api.types.is_datetime64_any_dtype(series):
        parsed = try_parse_datetime(series)
        if parsed is None:
            return None
        series = parsed
    working = pd.DataFrame(
        {"period": series.dt.to_period("M"), "value": _numeric(frame, columns.primary_measure)}
    ).dropna()
    if working.empty:
        return None
    return working.groupby("period", observed=True)["value"].sum().sort_index()


# ---------------------------------------------------------------------------
# Definitions
# ---------------------------------------------------------------------------


def _total_revenue(columns: Columns, frame: pd.DataFrame) -> Kpi | None:
    measure = columns.primary_measure
    if not measure:
        return None
    total = float(_numeric(frame, measure).sum())
    return Kpi(
        name="Total revenue",
        value=total,
        display_value=_money(total),
        formula=f"Sum of every value in {measure}",
        interpretation=(
            f"This is everything the business brought in across the {len(frame):,} "
            "rows in this file. It is the headline figure, but it says nothing "
            "about what was kept after costs."
        ),
    )


def _total_profit(columns: Columns, frame: pd.DataFrame) -> Kpi | None:
    profit = columns.measure_named("profit")
    if not profit or profit not in frame.columns:
        return None
    total = float(_numeric(frame, profit).sum())
    revenue = columns.primary_measure
    share = ""
    if revenue:
        revenue_total = float(_numeric(frame, revenue).sum())
        if revenue_total:
            share = f" That is {total / revenue_total:.0%} of everything that came in."
    return Kpi(
        name="Total profit",
        value=total,
        display_value=_money(total),
        formula=f"Sum of every value in {profit}",
        interpretation=(
            "This is what the business actually kept after the recorded costs." + share
        ),
    )


def _profit_margin(columns: Columns, frame: pd.DataFrame) -> Kpi | None:
    profit = columns.measure_named("profit")
    revenue = columns.primary_measure
    if not profit or not revenue or profit == revenue:
        return None
    revenue_total = float(_numeric(frame, revenue).sum())
    if not revenue_total:
        return None
    margin = float(_numeric(frame, profit).sum()) / revenue_total * 100
    return Kpi(
        name="Profit margin",
        value=margin,
        display_value=f"{margin:.1f}%",
        unit="%",
        formula=f"Total {profit} divided by total {revenue}, as a percentage",
        interpretation=(
            f"For every 100 that comes in, {margin:.0f} is kept. This is the number "
            "to watch when sales grow: rising revenue with a falling margin means "
            "the growth is being bought rather than earned."
        ),
    )


def _average_order(columns: Columns, frame: pd.DataFrame) -> Kpi | None:
    measure = columns.primary_measure
    if not measure or frame.empty:
        return None
    values = _numeric(frame, measure).dropna()
    if values.empty:
        return None
    average = float(values.mean())
    median = float(values.median())
    return Kpi(
        name="Average order value",
        value=average,
        display_value=_money(average),
        formula=f"Total {measure} divided by the number of rows",
        interpretation=(
            f"The typical order is {_money(median)}, while the average is "
            f"{_money(average)}. The gap between them is caused by a small number "
            "of large orders, so the average alone would overstate a normal sale."
            if average > median * 1.15
            else "Orders cluster closely around this figure, so it is a fair "
            "description of a normal sale."
        ),
    )


def _order_count(columns: Columns, frame: pd.DataFrame) -> Kpi | None:
    return Kpi(
        name="Number of records",
        value=float(len(frame)),
        display_value=_count(len(frame)),
        formula="Count of rows after cleaning",
        interpretation=(
            "Every figure on this page is calculated from these rows. If this "
            "number is far from what you expected, the file may be filtered or "
            "incomplete."
        ),
    )


def _customer_count(columns: Columns, frame: pd.DataFrame) -> Kpi | None:
    customer = columns.entity_named("customer", "client", "account", "buyer")
    if not customer or customer not in frame.columns:
        return None
    unique = int(frame[customer].nunique())
    if unique < 2:
        return None
    per_customer = len(frame) / unique
    return Kpi(
        name="Active customers",
        value=float(unique),
        display_value=_count(unique),
        formula=f"Count of distinct values in {customer}",
        interpretation=(
            f"{unique:,} different customers appear in this data, averaging "
            f"{per_customer:.1f} orders each."
        ),
    )


def _repeat_rate(columns: Columns, frame: pd.DataFrame) -> Kpi | None:
    customer = columns.entity_named("customer", "client", "account", "buyer")
    if not customer or customer not in frame.columns:
        return None
    counts = frame[customer].value_counts()
    if len(counts) < 5:
        return None
    repeat = float((counts > 1).sum() / len(counts) * 100)
    return Kpi(
        name="Repeat customer rate",
        value=repeat,
        display_value=f"{repeat:.1f}%",
        unit="%",
        formula=f"Share of values in {customer} that appear more than once",
        interpretation=(
            f"{repeat:.0f}% of customers came back at least once. Winning a new "
            "customer usually costs several times more than keeping an existing "
            "one, so this number moving is worth more than it looks."
        ),
    )


def _growth_rate(columns: Columns, frame: pd.DataFrame) -> Kpi | None:
    monthly = _monthly_totals(frame, columns)
    if monthly is None or len(monthly) < 4:
        return None

    half = len(monthly) // 2
    earlier = float(monthly.iloc[:half].mean())
    later = float(monthly.iloc[half:].mean())
    if not earlier:
        return None
    growth = (later - earlier) / earlier * 100
    direction = "up" if growth > 0 else "down"
    return Kpi(
        name="Growth rate",
        value=growth,
        display_value=f"{growth:+.1f}%",
        unit="%",
        trend=direction,
        formula=(
            "Average month in the second half of the period compared with the "
            "average month in the first half"
        ),
        interpretation=(
            f"A typical month in the recent half of this period was {abs(growth):.0f}% "
            f"{direction} on a typical month in the earlier half. Comparing halves "
            "rather than the first and last month stops one unusual month from "
            "setting the story."
        ),
    )


def _best_month(columns: Columns, frame: pd.DataFrame) -> Kpi | None:
    monthly = _monthly_totals(frame, columns)
    if monthly is None or len(monthly) < 3:
        return None
    peak = monthly.idxmax()
    value = float(monthly.max())
    average = float(monthly.mean())
    lift = (value / average - 1) * 100 if average else 0
    return Kpi(
        name="Strongest month",
        value=value,
        display_value=str(peak),
        formula=f"Month with the highest total {columns.primary_measure}",
        interpretation=(
            f"{peak} brought in {_money(value)}, which is {lift:.0f}% above an "
            "average month. Knowing when the peak falls is what makes stock and "
            "staffing decisions possible in advance."
        ),
    )


def _units_sold(columns: Columns, frame: pd.DataFrame) -> Kpi | None:
    quantity = columns.measure_named("quantity", "qty", "units")
    if not quantity or quantity not in frame.columns:
        return None
    total = float(_numeric(frame, quantity).sum())
    return Kpi(
        name="Units sold",
        value=total,
        display_value=_count(total),
        formula=f"Sum of {quantity}",
        interpretation=(
            "The volume side of the business. Read alongside revenue: rising "
            "revenue on flat units means prices went up, not that you sold more."
        ),
    )


def _concentration(columns: Columns, frame: pd.DataFrame) -> Kpi | None:
    customer = columns.entity_named("customer", "client", "account", "buyer")
    measure = columns.primary_measure
    if not customer or not measure or customer not in frame.columns:
        return None
    totals = _numeric(frame, measure).groupby(frame[customer]).sum().sort_values(
        ascending=False
    )
    if len(totals) < 10:
        return None
    grand_total = float(totals.sum())
    if not grand_total:
        return None
    top_count = max(1, int(len(totals) * 0.1))
    share = float(totals.iloc[:top_count].sum()) / grand_total * 100
    return Kpi(
        name="Top 10% customer share",
        value=share,
        display_value=f"{share:.1f}%",
        unit="%",
        formula=(
            f"Total {measure} from the largest 10% of customers, divided by the "
            "total from everyone"
        ),
        interpretation=(
            f"The biggest {top_count:,} customers account for {share:.0f}% of the "
            "business. The higher this is, the more a single departure hurts."
        ),
    )


DEFINITIONS: tuple[KpiDefinition, ...] = (
    KpiDefinition(
        "total_revenue",
        "Total revenue",
        "the headline figure everything else is measured against",
        lambda columns, frame: columns.primary_measure is not None,
        _total_revenue,
    ),
    KpiDefinition(
        "total_profit",
        "Total profit",
        "what was actually kept, which revenue alone never shows",
        lambda columns, frame: columns.measure_named("profit") is not None,
        _total_profit,
    ),
    KpiDefinition(
        "profit_margin",
        "Profit margin",
        "whether growth is being earned or bought",
        lambda columns, frame: columns.measure_named("profit") is not None,
        _profit_margin,
    ),
    KpiDefinition(
        "average_order",
        "Average order value",
        "the lever that raises revenue without finding new customers",
        lambda columns, frame: columns.primary_measure is not None,
        _average_order,
    ),
    KpiDefinition(
        "growth_rate",
        "Growth rate",
        "the direction of travel, not just the current position",
        lambda columns, frame: columns.primary_date is not None,
        _growth_rate,
    ),
    KpiDefinition(
        "best_month",
        "Strongest month",
        "when to have stock and staff ready",
        lambda columns, frame: columns.primary_date is not None,
        _best_month,
    ),
    KpiDefinition(
        "customer_count",
        "Active customers",
        "the size of the base the business depends on",
        lambda columns, frame: columns.entity_named("customer", "client", "account")
        is not None,
        _customer_count,
    ),
    KpiDefinition(
        "repeat_rate",
        "Repeat customer rate",
        "whether customers come back, which is cheaper than finding new ones",
        lambda columns, frame: columns.entity_named("customer", "client", "account")
        is not None,
        _repeat_rate,
    ),
    KpiDefinition(
        "concentration",
        "Top 10% customer share",
        "how exposed the business is to losing one big account",
        lambda columns, frame: columns.entity_named("customer", "client", "account")
        is not None,
        _concentration,
    ),
    KpiDefinition(
        "units_sold",
        "Units sold",
        "volume, so price rises are not mistaken for growth",
        lambda columns, frame: columns.measure_named("quantity", "qty", "units")
        is not None,
        _units_sold,
    ),
    KpiDefinition(
        "record_count",
        "Number of records",
        "the base every other figure is calculated from",
        lambda columns, frame: True,
        _order_count,
    ),
)


def available_kpis(frame: pd.DataFrame, profile: DatasetProfile) -> list[KpiDefinition]:
    """KPI definitions this dataset can actually support."""
    columns = resolve_columns(frame, profile)
    return [item for item in DEFINITIONS if item.supported(columns, frame)]


def compute_kpis(
    frame: pd.DataFrame, profile: DatasetProfile, chosen: list[str] | None = None
) -> list[Kpi]:
    """Calculate the chosen KPIs, skipping any that turn out not to apply."""
    columns = resolve_columns(frame, profile)
    wanted = set(chosen) if chosen else None
    results: list[Kpi] = []
    for definition in DEFINITIONS:
        if wanted is not None and definition.id not in wanted:
            continue
        if not definition.supported(columns, frame):
            continue
        try:
            kpi = definition.compute(columns, frame)
        except (ValueError, TypeError, ZeroDivisionError):
            kpi = None
        if kpi is not None:
            results.append(kpi)
    return results


def custom_kpi(
    frame: pd.DataFrame,
    name: str,
    numerator: str,
    denominator: str | None = None,
    *,
    as_percentage: bool = False,
) -> Kpi | None:
    """Build a KPI the user described, as a total or a ratio of two columns."""
    if numerator not in frame.columns:
        return None
    top = float(_numeric(frame, numerator).sum())

    if denominator is None:
        return Kpi(
            name=name,
            value=top,
            display_value=_money(top),
            formula=f"Sum of {numerator}",
            interpretation="Defined by you for this business.",
        )

    if denominator not in frame.columns:
        return None
    bottom = float(_numeric(frame, denominator).sum())
    if not bottom:
        return None
    ratio = top / bottom * (100 if as_percentage else 1)
    return Kpi(
        name=name,
        value=ratio,
        display_value=f"{ratio:.1f}%" if as_percentage else _money(ratio),
        unit="%" if as_percentage else "",
        formula=f"Total {numerator} divided by total {denominator}"
        + (", as a percentage" if as_percentage else ""),
        interpretation="Defined by you for this business.",
    )
