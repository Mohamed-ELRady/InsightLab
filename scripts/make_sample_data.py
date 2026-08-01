"""Generate the sample retail dataset used by the tests and the demo.

The data is deliberately imperfect. It carries duplicate rows, missing values,
a constant column, a handful of genuine high-value orders that look like
outliers, and dates stored as text - the same problems a real business file
arrives with.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROWS = 1200
SEED = 20240501

REGIONS = ["Cairo", "Alexandria", "Giza", "Mansoura", "Aswan"]
CATEGORIES = ["Electronics", "Furniture", "Groceries", "Clothing", "Stationery"]
CHANNELS = ["Store", "Online", "Phone"]
SEGMENTS = ["Retail", "Wholesale", "Corporate"]


def build() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)

    dates = pd.to_datetime("2023-01-01") + pd.to_timedelta(
        rng.integers(0, 730, ROWS), unit="D"
    )
    category = rng.choice(CATEGORIES, ROWS, p=[0.22, 0.15, 0.3, 0.23, 0.10])
    region = rng.choice(REGIONS, ROWS, p=[0.35, 0.2, 0.2, 0.15, 0.10])
    channel = rng.choice(CHANNELS, ROWS, p=[0.5, 0.38, 0.12])
    segment = rng.choice(SEGMENTS, ROWS, p=[0.6, 0.25, 0.15])

    base_price = {
        "Electronics": 2400,
        "Furniture": 1800,
        "Groceries": 120,
        "Clothing": 450,
        "Stationery": 60,
    }
    unit_price = np.array([base_price[item] for item in category], dtype=float)
    unit_price *= rng.lognormal(0.0, 0.28, ROWS)

    quantity = rng.integers(1, 9, ROWS)
    # Wholesale buys in bulk.
    quantity = np.where(segment == "Wholesale", quantity * rng.integers(3, 9, ROWS), quantity)

    # A November and December lift, so the seasonality question has an answer.
    month = dates.month.to_numpy()
    seasonal = np.where(np.isin(month, [11, 12]), rng.uniform(1.25, 1.6, ROWS), 1.0)
    unit_price *= seasonal

    revenue = np.round(unit_price * quantity, 2)
    cost = np.round(revenue * rng.uniform(0.55, 0.85, ROWS), 2)

    frame = pd.DataFrame(
        {
            "order_id": [f"ORD-{100000 + index}" for index in range(ROWS)],
            "order_date": dates.strftime("%Y-%m-%d"),
            "customer_id": [f"CUST-{rng.integers(1, 320):04d}" for _ in range(ROWS)],
            "customer_segment": segment,
            "region": region,
            "product_category": category,
            "sales_channel": channel,
            "quantity": quantity,
            "unit_price": np.round(unit_price, 2),
            "revenue": revenue,
            "cost": cost,
            "discount_pct": np.round(rng.uniform(0, 0.25, ROWS), 3),
            "order_status": rng.choice(
                ["Completed", "Cancelled", "Returned"], ROWS, p=[0.88, 0.08, 0.04]
            ),
            "currency": "EGP",
        }
    )

    # A few genuinely large corporate orders.
    for index in rng.choice(ROWS, 8, replace=False):
        frame.loc[index, "quantity"] = int(rng.integers(60, 160))
        frame.loc[index, "revenue"] = round(
            float(frame.loc[index, "unit_price"]) * float(frame.loc[index, "quantity"]), 2
        )
        frame.loc[index, "cost"] = round(float(frame.loc[index, "revenue"]) * 0.7, 2)
        frame.loc[index, "customer_segment"] = "Corporate"

    # Missing values, as any real export has.
    for column, share in (("region", 0.04), ("discount_pct", 0.09), ("cost", 0.03)):
        blanks = rng.choice(ROWS, int(ROWS * share), replace=False)
        frame.loc[blanks, column] = np.nan

    # Duplicated rows from a double export.
    duplicates = frame.sample(24, random_state=SEED)
    frame = pd.concat([frame, duplicates], ignore_index=True)

    return frame.sample(frac=1.0, random_state=SEED).reset_index(drop=True)


if __name__ == "__main__":
    from pathlib import Path

    target = Path(__file__).resolve().parents[1] / "data" / "samples"
    target.mkdir(parents=True, exist_ok=True)

    data = build()
    data.to_csv(target / "retail_sales.csv", index=False)
    data.to_excel(target / "retail_sales.xlsx", index=False)
    print(f"Wrote {len(data)} rows to {target}")
