"""Shared fixtures.

Every test runs offline. That is not only for speed: it is the configuration in
which the fallback paths are exercised, and those are the paths that keep the
product working for a user with no credentials.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

os.environ["INSIGHTLAB_OFFLINE"] = "1"

from insightlab.analysis.profiling import profile_dataset  # noqa: E402
from insightlab.core.config import reset_settings_cache  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = PROJECT_ROOT / "data" / "samples" / "retail_sales.csv"


@pytest.fixture(autouse=True)
def offline_settings(tmp_path, monkeypatch):
    """Force offline mode and a throwaway workspace for every test."""
    monkeypatch.setenv("INSIGHTLAB_OFFLINE", "1")
    monkeypatch.setenv("INSIGHTLAB_WORKSPACE", str(tmp_path / "runs"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    reset_settings_cache()
    yield
    reset_settings_cache()


@pytest.fixture
def sample_path() -> Path:
    if not SAMPLE_CSV.exists():
        pytest.skip("Run scripts/make_sample_data.py to generate the sample data.")
    return SAMPLE_CSV


@pytest.fixture
def sample_frame(sample_path) -> pd.DataFrame:
    return pd.read_csv(sample_path)


@pytest.fixture
def sample_profile(sample_frame):
    return profile_dataset(sample_frame)


@pytest.fixture
def messy_frame() -> pd.DataFrame:
    """A small frame carrying one of every problem the cleaner handles."""
    return pd.DataFrame(
        {
            "invoice_id": ["A1", "A2", "A3", "A4", "A5", "A5"],
            "sale_date": [
                "2024-01-05",
                "2024-02-11",
                "2024-03-02",
                "2024-11-20",
                "2024-12-01",
                "2024-12-01",
            ],
            "customer_name": ["Ali", "Sara", "Ali", "Omar", "Sara", "Sara"],
            "product_category": ["Tools", "Tools", "Paint", "Paint", "Tools", "Tools"],
            "revenue": [100.0, 150.0, None, 90_000.0, 120.0, 120.0],
            "cost": [60.0, 90.0, 70.0, 60_000.0, 80.0, 80.0],
            "country": ["EG", "EG", "EG", "EG", "EG", "EG"],
        }
    )
