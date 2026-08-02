"""New measures derived from the columns the user already has.

The point of this module is that a business question is rarely answerable from
raw columns. "Which months are strongest" needs a month column; "which products
actually make money" needs a margin. Each suggestion below explains itself in
the user's terms and knows how to build itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..core.state import DatasetProfile, Role
from .profiling import MONEY_NAME_PATTERN, try_parse_datetime

Builder = Callable[[pd.DataFrame], pd.DataFrame]


@dataclass
class FeatureSuggestion:
    """One proposed new column, with the reason a business would want it."""

    id: str
    name: str
    question: str
    reason: str
    builder: Builder = field(repr=False, default=lambda frame: frame)
    creates: list[str] = field(default_factory=list)

    def apply(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
        """Build the columns and report which ones actually appeared."""
        before = set(frame.columns)
        result = self.builder(frame)
        created = [name for name in result.columns if name not in before]
        return result, created


# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------


def _find(profile: DatasetProfile, *keywords: str, role: Role | None = None) -> str | None:
    """First column whose name contains any keyword, optionally by role."""
    for column in profile.columns:
        if role is not None and column.role is not role:
            continue
        lowered = column.name.casefold()
        if any(keyword in lowered for keyword in keywords):
            return column.name
    return None


def _revenue_column(profile: DatasetProfile) -> str | None:
    for keywords in (("revenue",), ("sales", "total"), ("amount", "value")):
        found = _find(profile, *keywords, role=Role.MEASURE)
        if found:
            return found
    return None


def _first_datetime(frame: pd.DataFrame, profile: DatasetProfile) -> str | None:
    for column in profile.columns:
        if column.role is Role.DATETIME and column.name in frame.columns:
            return column.name
    return None


def _as_datetime(frame: pd.DataFrame, column: str) -> pd.Series:
    series = frame[column]
    if pd.api.types.is_datetime64_any_dtype(series):
        return series
    parsed = try_parse_datetime(series)
    if parsed is not None:
        return parsed
    return pd.to_datetime(series, errors="coerce", format="mixed")


# ---------------------------------------------------------------------------
# Suggestions
# ---------------------------------------------------------------------------


def _calendar_builder(date_column: str) -> Builder:
    def build(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        dates = _as_datetime(result, date_column)
        result["year"] = dates.dt.year
        result["quarter"] = "Q" + dates.dt.quarter.astype("Int64").astype(str)
        result["month"] = dates.dt.month
        result["month_name"] = dates.dt.month_name()
        result["day_of_week"] = dates.dt.day_name()
        result["is_weekend"] = dates.dt.dayofweek.isin([4, 5])
        return result

    return build


def _profit_builder(revenue: str, cost: str, quantity: str | None = None) -> Builder:
    def build(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        revenue_values = pd.to_numeric(result[revenue], errors="coerce")
        cost_values = pd.to_numeric(result[cost], errors="coerce")

        # A per-unit cost against a whole-order revenue would overstate profit
        # by the order size. This turns up as soon as cost arrives from a
        # product file, where it is almost always per unit.
        if quantity and quantity in result.columns:
            cost_values = cost_values * pd.to_numeric(
                result[quantity], errors="coerce"
            )

        result["profit"] = revenue_values - cost_values
        with np.errstate(divide="ignore", invalid="ignore"):
            margin = np.where(
                revenue_values > 0, (revenue_values - cost_values) / revenue_values, np.nan
            )
        result["profit_margin_pct"] = np.round(margin * 100, 2)
        return result

    return build


def _net_revenue_builder(revenue: str, discount: str) -> Builder:
    def build(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        revenue_values = pd.to_numeric(result[revenue], errors="coerce")
        discount_values = pd.to_numeric(result[discount], errors="coerce").fillna(0)
        # A discount column may hold 0.15 or 15; both mean fifteen per cent.
        if discount_values.max() is not None and float(discount_values.max()) > 1.5:
            discount_values = discount_values / 100.0
        result["net_revenue"] = (revenue_values * (1 - discount_values)).round(2)
        return result

    return build


def _tier_builder(revenue: str) -> Builder:
    def build(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        values = pd.to_numeric(result[revenue], errors="coerce")
        try:
            result["revenue_tier"] = pd.qcut(
                values,
                q=[0, 0.25, 0.5, 0.75, 1.0],
                labels=["Small", "Medium", "Large", "Very large"],
                duplicates="drop",
            ).astype(object)
        except ValueError:
            # Not enough distinct values to cut into four groups.
            result["revenue_tier"] = "Unclassified"
        return result

    return build


def _customer_value_builder(customer: str, revenue: str) -> Builder:
    def build(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        values = pd.to_numeric(result[revenue], errors="coerce")
        grouped = values.groupby(result[customer])
        result["customer_total_value"] = grouped.transform("sum").round(2)
        result["customer_order_count"] = grouped.transform("count")
        result["customer_average_order"] = grouped.transform("mean").round(2)
        return result

    return build


def _unit_economics_builder(revenue: str, quantity: str) -> Builder:
    def build(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        revenue_values = pd.to_numeric(result[revenue], errors="coerce")
        quantity_values = pd.to_numeric(result[quantity], errors="coerce")
        result["revenue_per_unit"] = np.where(
            quantity_values > 0, (revenue_values / quantity_values).round(2), np.nan
        )
        return result

    return build


def _age_band_builder(age: str) -> Builder:
    def build(frame: pd.DataFrame) -> pd.DataFrame:
        result = frame.copy()
        values = pd.to_numeric(result[age], errors="coerce")
        result["age_band"] = pd.cut(
            values,
            bins=[0, 24, 34, 44, 54, 64, 200],
            labels=["Under 25", "25-34", "35-44", "45-54", "55-64", "65+"],
        ).astype(object)
        return result

    return build


def suggest_features(
    frame: pd.DataFrame, profile: DatasetProfile
) -> list[FeatureSuggestion]:
    """Everything worth adding to this particular dataset."""
    suggestions: list[FeatureSuggestion] = []

    date_column = _first_datetime(frame, profile)
    if date_column:
        suggestions.append(
            FeatureSuggestion(
                id="calendar",
                name="Calendar breakdown",
                question="When does the business do well?",
                reason=(
                    f"Splitting {date_column} into year, quarter, month and day of "
                    "week is what makes it possible to see seasons, quiet periods "
                    "and which days of the week carry the business."
                ),
                builder=_calendar_builder(date_column),
                creates=["year", "quarter", "month", "month_name", "day_of_week", "is_weekend"],
            )
        )

    revenue = _revenue_column(profile)
    cost = _find(profile, "cost", "cogs", "expense", role=Role.MEASURE)
    if revenue and cost:
        quantity_column = _find(profile, "quantity", "qty", "units", role=Role.MEASURE)
        # "unit cost" is a rate, not an amount. Subtracting it from an order
        # total overstates profit by the order size, and this turns up the
        # moment cost arrives from a product file, where it is almost always
        # stated per unit.
        per_unit = "unit" in cost.casefold() or "per_" in cost.casefold()
        multiplier = quantity_column if per_unit and quantity_column else None
        formula = (
            f"{revenue} minus {cost} times {multiplier}"
            if multiplier
            else f"{revenue} minus {cost}"
        )
        suggestions.append(
            FeatureSuggestion(
                id="profit",
                name="Profit and profit margin",
                question="Which sales actually make money?",
                reason=(
                    f"{revenue} tells you what came in, not what you kept. Profit "
                    f"({formula}) and the margin as a percentage often "
                    "rank products in the opposite order to revenue alone."
                ),
                builder=_profit_builder(revenue, cost, multiplier),
                creates=["profit", "profit_margin_pct"],
            )
        )

    discount = _find(profile, "discount", role=Role.MEASURE)
    if revenue and discount:
        suggestions.append(
            FeatureSuggestion(
                id="net_revenue",
                name="Revenue after discount",
                question="What did the discounts really cost?",
                reason=(
                    "Comparing revenue before and after discount shows how much of "
                    "the headline figure is being given away."
                ),
                builder=_net_revenue_builder(revenue, discount),
                creates=["net_revenue"],
            )
        )

    if revenue:
        suggestions.append(
            FeatureSuggestion(
                id="revenue_tier",
                name="Order size band",
                question="How do small and large orders differ?",
                reason=(
                    "Sorting orders into small, medium, large and very large makes "
                    "it possible to see whether the big orders come from different "
                    "regions, channels or customers than the small ones."
                ),
                builder=_tier_builder(revenue),
                creates=["revenue_tier"],
            )
        )

    customer = _find(profile, "customer", "client", "account", "buyer")
    if customer and revenue and customer != revenue:
        suggestions.append(
            FeatureSuggestion(
                id="customer_value",
                name="Customer value",
                question="Who are the customers worth keeping?",
                reason=(
                    "Totalling each customer's spend, order count and average order "
                    "turns a list of transactions into a view of the customer base, "
                    "which is what a retention or loyalty decision needs."
                ),
                builder=_customer_value_builder(customer, revenue),
                creates=[
                    "customer_total_value",
                    "customer_order_count",
                    "customer_average_order",
                ],
            )
        )

    quantity = _find(profile, "quantity", "qty", "units", role=Role.MEASURE)
    if revenue and quantity:
        suggestions.append(
            FeatureSuggestion(
                id="unit_economics",
                name="Revenue per unit",
                question="Is the price holding up?",
                reason=(
                    "Revenue per unit separates selling more from selling at a "
                    "higher price - two very different reasons for growth."
                ),
                builder=_unit_economics_builder(revenue, quantity),
                creates=["revenue_per_unit"],
            )
        )

    age = _find(profile, "age", role=Role.MEASURE)
    if age:
        suggestions.append(
            FeatureSuggestion(
                id="age_band",
                name="Age groups",
                question="Which age groups buy from you?",
                reason=(
                    "Individual ages are too fine to act on. Grouping them into "
                    "bands gives segments a marketing decision can be made about."
                ),
                builder=_age_band_builder(age),
                creates=["age_band"],
            )
        )

    return suggestions


def apply_custom_rule(
    frame: pd.DataFrame, column: str, rule: dict[str, Any]
) -> tuple[pd.DataFrame, str]:
    """Build a classification column from a rule the user described.

    ``rule`` holds a ``source`` column, a list of ``bands`` as
    ``{"upto": number, "label": text}`` sorted ascending, and a final
    ``otherwise`` label. This is how a statement like "VIP customers spend more
    than 5,000" becomes an actual column.
    """
    source = rule.get("source")
    bands = rule.get("bands") or []
    otherwise = rule.get("otherwise", "Other")
    if not source or source not in frame.columns or not bands:
        return frame, ""

    result = frame.copy()
    values = pd.to_numeric(result[source], errors="coerce")
    labels = pd.Series(otherwise, index=result.index, dtype=object)

    remaining = pd.Series(True, index=result.index)
    for band in sorted(bands, key=lambda item: float(item.get("upto", 0))):
        threshold = float(band.get("upto", 0))
        label = str(band.get("label", "Band"))
        mask = remaining & (values <= threshold)
        labels[mask] = label
        remaining &= ~mask

    result[column] = labels
    return result, column
