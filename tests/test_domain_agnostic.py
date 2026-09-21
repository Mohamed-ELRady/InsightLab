from __future__ import annotations

import pandas as pd
import pytest

from insightlab.agents.data_understanding import DataUnderstandingAgent
from insightlab.agents.analyst import AnalystAgent
from insightlab.agents.memory_agent import MemoryAgent
from insightlab.analysis.exploration import available_axes, build_charts
from insightlab.analysis.metrics import available_kpis, compute_kpis
from insightlab.core.reasoning import ReasoningEngine
from insightlab.core.state import Chart, Kpi, PipelineState, RunMode, StageStatus
from insightlab.core.business_memory import Fact
from insightlab.core.evaluation import evaluate_run


def earthquake_state() -> PipelineState:
    rows = 120
    state = PipelineState(mode=RunMode.AUTONOMOUS)
    state.source_name = "earthquake_dataset.csv"
    state.frame = pd.DataFrame({
        "time": pd.date_range("2024-01-01", periods=rows, freq="10D").astype(str),
        "latitude": [-33.0 + (index % 12) for index in range(rows)],
        "longitude": [-71.0 + (index % 9) for index in range(rows)],
        "depth": [5.0 + (index % 40) for index in range(rows)],
        "mag": [3.0 + (index % 25) / 10 for index in range(rows)],
        "magType": ["ml", "mb", "mw"] * 40,
        "place": [f"Region {index % 6}" for index in range(rows)],
        "status": ["reviewed", "automatic"] * 60,
        "horizontalError": [0.2 + (index % 5) / 10 for index in range(rows)],
        "depthError": [0.5 + (index % 7) / 10 for index in range(rows)],
    })
    flow = DataUnderstandingAgent(ReasoningEngine()).run(state)
    try:
        while True:
            next(flow)
    except StopIteration:
        pass
    return state


def test_earthquake_schema_is_understood_before_analysis():
    state = earthquake_state()
    reading = state.understanding

    assert reading.domain_family == "earth_science"
    assert reading.subject == "earthquake events"
    assert reading.primary_measure == "mag"
    assert reading.primary_date == "time"
    assert reading.aggregation_for("mag") == "mean"
    assert "strongest recorded event" in reading.hook.casefold()
    assert "business" not in reading.summary.casefold()
    assert "revenue" not in reading.summary.casefold()


def test_earthquake_analysis_never_invents_business_kpis_or_sums_magnitude():
    state = earthquake_state()
    definitions = available_kpis(state.frame, state.profile, state.understanding)
    kpis = compute_kpis(state.frame, state.profile, None, state.understanding)
    axes = available_axes(state.frame, state.profile, state.understanding)
    charts = build_charts(
        state.frame, state.profile, axes, understanding=state.understanding
    )

    names = {item.name for item in definitions} | {item.name for item in kpis}
    assert "Total revenue" not in names
    assert "Average order value" not in names
    assert "Number of earthquake events" in names
    assert {"time", "distributions", "relationships"}.issubset(axes)
    assert not any(
        chart.measure in {"mag", "latitude", "longitude", "depth"}
        and chart.aggregation == "sum"
        for chart in charts
    )
    assert any(
        chart.kind == "scatter"
        and "mag" in chart.id and "depth" in chart.id
        for chart in charts
    )


def test_signed_coordinates_are_not_reported_as_invalid_negatives():
    state = earthquake_state()
    assert "negative" not in state.profile.column("latitude").note
    assert "negative" not in state.profile.column("longitude").note


def test_follow_up_questions_use_domain_measure_and_non_commercial_language():
    state = earthquake_state()
    questions = AnalystAgent(ReasoningEngine()).suggest(state)

    joined = " ".join(questions).casefold()
    assert "latitude" not in joined
    assert "brings in" not in joined
    assert "revenue" not in joined
    assert "earthquake frequency" in joined
    assert "mag" in joined


def test_legacy_business_memory_is_not_relevant_to_earthquake_domain():
    state = earthquake_state()
    old_rule = Fact("Exclude Cancelled order_status rows from revenue totals")
    domain_rule = Fact("Earthquake magnitudes above 7 need separate review")

    assert not MemoryAgent._relevant_to_domain(old_rule, state)
    assert MemoryAgent._relevant_to_domain(domain_rule, state)


def test_quality_gate_detects_commercial_leakage_and_unsafe_scientific_sums():
    state = earthquake_state()
    state.stage_status = {key: StageStatus.DONE for key in state.stage_status}
    state.kpis = [Kpi(
        name="Total revenue", value=10, display_value="10",
        formula="Sum of revenue", interpretation="Commercial total",
    )]
    state.charts = [Chart(
        id="bad-mag-sum", title="Magnitude total", kind="bar",
        description="Invalid total", measure="mag", aggregation="sum",
    )]

    result = evaluate_run(state)

    assert result.metrics["domain_kpi_violation_count"] == 1
    assert result.metrics["unsafe_aggregation_count"] == 1
    assert result.metrics["semantic_consistency"] == 0.0
    assert result.score < 100


@pytest.mark.parametrize(("columns", "expected_family"), [
    ({"player": ["A", "B"], "team": ["X", "Y"], "goals": [1, 2]}, "sports"),
    ({"vehicle": ["v1", "v2"], "route": ["r1", "r2"], "speed": [40, 55]}, "transport"),
    ({"crop": ["corn", "wheat"], "soil": ["clay", "sand"], "yield": [4.2, 3.8]}, "agriculture"),
    ({"sample": ["s1", "s2"], "treatment": ["A", "control"], "concentration": [1.2, 0.9]}, "laboratory"),
    ({"ticker": ["AAA", "BBB"], "close": [101.0, 52.0], "volume": [1000, 2000]}, "markets"),
])
def test_additional_domains_are_not_forced_into_business(columns, expected_family):
    state = PipelineState(mode=RunMode.AUTONOMOUS)
    state.source_name = f"{expected_family}.csv"
    state.frame = pd.DataFrame(columns)
    flow = DataUnderstandingAgent(ReasoningEngine()).run(state)
    try:
        while True:
            next(flow)
    except StopIteration:
        pass

    assert state.understanding.domain_family == expected_family
    assert not state.understanding.is_business
    assert "business" not in state.understanding.summary.casefold()


def test_unknown_schema_stays_general_and_asks_for_row_meaning():
    state = PipelineState(mode=RunMode.AUTONOMOUS)
    state.source_name = "mystery.csv"
    state.frame = pd.DataFrame({"alpha": [1, 2], "beta": ["x", "y"]})
    flow = DataUnderstandingAgent(ReasoningEngine()).run(state)
    decision = next(flow)

    assert state.understanding.domain_family == "general"
    assert decision.topic
    assert "business" not in decision.question.casefold()
