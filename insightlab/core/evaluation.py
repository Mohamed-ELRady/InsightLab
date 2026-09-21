"""Deterministic run evaluation with privacy-reduced structural cases."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .state import PipelineState, StageStatus


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    metrics: dict[str, Any]
    cases: list[dict[str, Any]]


def evaluate_run(state: PipelineState) -> EvaluationResult:
    """Score observable quality without asking a model to grade itself.

    Cases intentionally contain no title, result, evidence, action, column
    value, file name, or raw row. They can be reused for trend analysis without
    turning the evaluation store into a second copy of the user's dataset.
    """
    statuses = list(state.stage_status.values())
    settled = sum(status in {StageStatus.DONE, StageStatus.SKIPPED} for status in statuses)
    completion = settled / max(1, len(statuses))
    insights = state.insights
    grounded = sum(bool(item.grounded) for item in insights) / max(1, len(insights))
    evidence = sum(bool(item.evidence.strip()) for item in insights) / max(1, len(insights))
    actionable = sum(bool(item.action.strip()) for item in insights) / max(1, len(insights))
    chart_linked = sum(bool(item.chart_id and state.chart(item.chart_id)) for item in insights) / max(1, len(insights))
    error_rate = len(state.errors) / max(1, len(statuses))

    understanding = state.understanding
    if understanding.domain_family == "general" and not understanding.summary:
        # Legacy/test states made before the understanding phase do not assert
        # a domain and therefore cannot contradict it.
        domain_understanding = 1.0
    else:
        signals = (
            understanding.domain_family not in {"", "general"},
            bool(understanding.subject.strip()),
            bool(understanding.row_represents.strip()),
            bool(understanding.primary_measure or understanding.primary_category),
            understanding.confidence in {"medium", "high"},
        )
        domain_understanding = sum(signals) / len(signals)

    unsafe_aggregations = 0
    for chart in state.charts:
        if not chart.measure or chart.aggregation != "sum":
            continue
        expected = understanding.measure_aggregations.get(chart.measure)
        if expected and expected != "sum":
            unsafe_aggregations += 1

    commercial_terms = {
        "revenue", "profit", "margin", "order value", "customers", "units sold",
        "الإيرادات", "الأرباح", "هامش الربح", "العملاء", "قيمة الطلب", "الوحدات المباعة",
    }
    domain_kpi_violations = 0
    if not understanding.is_business and understanding.domain_family != "general":
        for kpi in state.kpis:
            text = f"{kpi.name} {kpi.formula} {kpi.interpretation}".casefold()
            if any(term.casefold() in text for term in commercial_terms):
                domain_kpi_violations += 1
    semantic_consistency = 1.0 if not (unsafe_aggregations or domain_kpi_violations) else 0.0

    score = 100 * (
        0.20 * completion + 0.20 * grounded + 0.15 * evidence
        + 0.10 * actionable + 0.10 * chart_linked
        + 0.15 * domain_understanding + 0.10 * semantic_consistency
        - 0.25 * error_rate
    )
    score = round(max(0.0, min(100.0, score)), 1)
    metrics = {
        "completion_rate": round(completion, 3),
        "grounded_insight_rate": round(grounded, 3),
        "evidence_coverage": round(evidence, 3),
        "action_coverage": round(actionable, 3),
        "chart_link_rate": round(chart_linked, 3),
        "domain_understanding": round(domain_understanding, 3),
        "semantic_consistency": round(semantic_consistency, 3),
        "unsafe_aggregation_count": unsafe_aggregations,
        "domain_kpi_violation_count": domain_kpi_violations,
        "error_count": len(state.errors),
        "insight_count": len(insights),
        "chart_count": len(state.charts),
        "kpi_count": len(state.kpis),
    }
    cases = [
        {
            "position": index,
            "axis": item.axis,
            "confidence": item.confidence,
            "grounded": bool(item.grounded),
            "has_evidence": bool(item.evidence.strip()),
            "has_action": bool(item.action.strip()),
            "has_chart": bool(item.chart_id and state.chart(item.chart_id)),
        }
        for index, item in enumerate(insights)
    ]
    return EvaluationResult(score=score, metrics=metrics, cases=cases)
