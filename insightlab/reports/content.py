"""The report's content, assembled once and rendered three ways.

PDF, PowerPoint and Word are three presentations of the same material, so the
material is built here and each renderer only decides how to lay it out. That is
what stops the Word version quietly saying something different from the slides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import plotly.io as plotly_io

from ..core.activity_log import EventKind
from ..core.state import PipelineState, RunMode, StageStatus

#: Width and height in pixels for chart images embedded in documents.
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 460

#: Charts beyond this many make a document nobody finishes.
MAX_CHART_IMAGES = 10


@dataclass
class Section:
    """One block of the report."""

    title: str
    paragraphs: list[str] = field(default_factory=list)
    bullets: list[str] = field(default_factory=list)
    table: tuple[list[str], list[list[str]]] | None = None
    chart_ids: list[str] = field(default_factory=list)


@dataclass
class ReportContent:
    """Everything a renderer needs, in order."""

    title: str
    subtitle: str
    sections: list[Section]
    chart_images: dict[str, bytes] = field(default_factory=dict)


def render_chart_images(state: PipelineState) -> dict[str, bytes]:
    """Turn the stored plotly figures into PNGs for the documents.

    Image export needs a browser engine that is not guaranteed to be present, so
    a failure here drops the picture and keeps the text rather than losing the
    whole report.
    """
    images: dict[str, bytes] = {}
    for chart in state.charts[:MAX_CHART_IMAGES]:
        if not chart.figure_json:
            continue
        try:
            figure = plotly_io.from_json(chart.figure_json)
            figure.update_layout(
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                margin=dict(l=70, r=40, t=30, b=60),
            )
            # Bars and heatmaps grow with the number of categories, so they
            # keep the height the chart builder asked for.
            height = IMAGE_HEIGHT
            if chart.kind in ("bar", "heatmap"):
                height = max(IMAGE_HEIGHT, int(figure.layout.height or IMAGE_HEIGHT))

            images[chart.id] = figure.to_image(
                format="png", width=IMAGE_WIDTH, height=height, scale=2
            )
        except Exception:  # noqa: BLE001 - a missing picture must not lose the report
            continue
    return images


def build_content(state: PipelineState) -> ReportContent:
    """Assemble the whole report from the finished run."""
    sections = [
        _executive_summary(state),
        _about_the_data(state),
        _what_we_changed(state),
        _headline_figures(state),
        _findings(state),
        _charts(state),
        _what_you_told_us(state),
        _recommendations(state),
        _next_steps(state),
        _how_this_was_produced(state),
    ]
    return ReportContent(
        title=f"Analysis of {state.source_name or 'your data'}",
        subtitle=state.started_at.strftime("Prepared %d %B %Y"),
        sections=[section for section in sections if _has_content(section)],
        chart_images=render_chart_images(state),
    )


def _has_content(section: Section) -> bool:
    return bool(section.paragraphs or section.bullets or section.table or section.chart_ids)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _executive_summary(state: PipelineState) -> Section:
    paragraphs = []
    if state.profile.summary:
        paragraphs.append(state.profile.summary)

    if state.kpis:
        headline = ", ".join(
            f"{kpi.name.lower()} of {kpi.display_value}" for kpi in state.kpis[:3]
        )
        paragraphs.append(f"The headline position is {headline}.")

    high = [item for item in state.insights if item.confidence == "high"]
    if high:
        paragraphs.append(
            f"{len(high)} of the {len(state.insights)} conclusions below rest on "
            "large, clear differences in the data and can be acted on directly. "
            "The rest are worth checking before committing to them."
        )
    elif state.insights:
        paragraphs.append(
            f"{len(state.insights)} conclusions were drawn. None of them rests on "
            "a difference large enough to act on without checking first."
        )

    return Section(
        title="Executive summary",
        paragraphs=paragraphs,
        bullets=[
            f"{item.title}: {item.result}" for item in state.insights[:4]
        ],
    )


def _about_the_data(state: PipelineState) -> Section:
    profile = state.profile
    paragraphs = [
        f"The analysis was run on {state.source_name or 'the supplied file'}"
        + (f" ({state.source_format})" if state.source_format else "")
        + f", which holds {profile.row_count:,} rows across {profile.column_count} "
        "columns after cleaning."
    ]
    if state.load_notes:
        paragraphs.append(" ".join(state.load_notes))

    rows = [
        [
            column.name,
            column.role.value,
            f"{column.missing_rate:.0%}",
            f"{column.unique_count:,}",
            column.note or "-",
        ]
        for column in profile.columns
    ]
    return Section(
        title="About the data",
        paragraphs=paragraphs,
        table=(["Column", "Holds", "Empty", "Distinct values", "Notes"], rows),
    )


def _what_we_changed(state: PipelineState) -> Section:
    operations = state.log.data_operations()
    if not operations:
        return Section(
            title="What we changed",
            paragraphs=["The data was analysed exactly as supplied. Nothing was changed."],
        )

    bullets = []
    for event in operations:
        detail = event.details
        before, after = detail.get("rows_before"), detail.get("rows_after")
        suffix = ""
        # The first entry is the load itself, which has nothing before it, so
        # "0 rows became 1,224" would be noise rather than information.
        if before and after is not None and before != after:
            suffix = f" ({before:,} rows became {after:,})"
        bullets.append(f"{event.message}{suffix}")

    return Section(
        title="What we changed",
        paragraphs=[
            "Every change made to your data is listed here in the order it "
            "happened. The cleaned copy is saved alongside this report, and your "
            "original file was never modified."
        ],
        bullets=bullets,
    )


def _headline_figures(state: PipelineState) -> Section:
    if not state.kpis:
        return Section(title="Headline figures")
    rows = [
        [kpi.name, kpi.display_value, kpi.formula, kpi.interpretation]
        for kpi in state.kpis
    ]
    return Section(
        title="Headline figures",
        paragraphs=[
            "Each figure below is calculated from your own columns. The formula "
            "is given so you can reconcile it against your own records."
        ],
        table=(["Measure", "Value", "How it is calculated", "What it means"], rows),
    )


def _findings(state: PipelineState) -> Section:
    if not state.insights:
        return Section(title="What the data shows")

    paragraphs = [
        "Each finding below states what the data shows, the figure it rests on, "
        "what it means for the business, and how far it can be trusted. This is a "
        "single file of records, so a finding can show that two things move "
        "together but never that one caused the other."
    ]
    bullets = []
    for insight in state.insights:
        parts = [f"{insight.title} [confidence: {insight.confidence}]", insight.result]
        if insight.evidence:
            parts.append(f"Evidence: {insight.evidence}")
        if insight.interpretation:
            parts.append(f"What this means: {insight.interpretation}")
        if insight.action:
            parts.append(f"Suggested action: {insight.action}")
        bullets.append("\n".join(parts))

    return Section(title="What the data shows", paragraphs=paragraphs, bullets=bullets)


def _charts(state: PipelineState) -> Section:
    if not state.charts:
        return Section(title="Charts")
    return Section(
        title="Charts",
        paragraphs=[
            "Each chart carries a sentence saying what it shows, so it can be "
            "read without going back to the analysis."
        ],
        chart_ids=[chart.id for chart in state.charts[:MAX_CHART_IMAGES]],
    )


def _what_you_told_us(state: PipelineState) -> Section:
    if not state.memory:
        return Section(title="What you told us about your business")
    return Section(
        title="What you told us about your business",
        paragraphs=[
            "These are the facts about your business that shaped this analysis. "
            "They are saved with the project and will be applied automatically to "
            "future analyses of newer data."
        ],
        bullets=[f"{fact.statement} ({fact.category})" for fact in state.memory],
    )


def _recommendations(state: PipelineState) -> Section:
    actions = [
        insight.action
        for insight in state.insights
        if insight.action and insight.confidence in ("high", "medium")
    ]
    if not actions:
        return Section(title="Recommendations")
    return Section(
        title="Recommendations",
        paragraphs=[
            "These follow directly from the findings above, ordered by how far "
            "the evidence behind them can be trusted."
        ],
        bullets=list(dict.fromkeys(actions)),
    )


def _next_steps(state: PipelineState) -> Section:
    steps = []

    weak = [column for column in state.profile.columns if column.missing_rate > 0.2]
    if weak:
        steps.append(
            f"Improve how {', '.join(column.name for column in weak[:3])} is "
            "recorded. Anything calculated from these columns currently rests on "
            "the rows that happen to have a value."
        )

    low = [item for item in state.insights if item.confidence == "low"]
    if low:
        steps.append(
            f"Check the {len(low)} lower-confidence findings against what you "
            "know before acting on them."
        )

    if not state.focus_axes:
        steps.append(
            "Run the analysis again with a specific business question in mind. "
            "A focused run produces sharper findings than a general one."
        )
    else:
        steps.append(
            "Run this again when you have newer data. Everything you told us "
            "about the business is remembered, so the next run starts where this "
            "one left off rather than asking again."
        )

    if state.mode is RunMode.AUTONOMOUS:
        steps.append(
            "This run was automatic, so every decision was taken on our judgement "
            "rather than yours. Reviewing the decision log is worth the time "
            "before the findings are shared."
        )

    return Section(title="Next steps", bullets=steps)


def _how_this_was_produced(state: PipelineState) -> Section:
    decisions = state.answers
    automatic = sum(1 for _, answer in decisions if answer.automatic)
    manual = len(decisions) - automatic

    paragraphs = [
        f"The analysis ran in {state.mode.value} mode and passed through "
        f"{len(decisions)} decision points. {manual} were answered by you and "
        f"{automatic} were taken automatically."
    ]

    failed = [
        stage
        for stage, status in state.stage_status.items()
        if status is StageStatus.FAILED
    ]
    if failed:
        paragraphs.append(
            f"These stages did not complete: {', '.join(failed)}. The findings "
            "above do not include anything they would have produced."
        )

    bullets = [
        f"{answer.answered_at:%H:%M} - {decision.topic}: {answer.describe()}"
        + (" (taken automatically)" if answer.automatic else "")
        for decision, answer in decisions
    ]

    warnings = [
        event.message
        for event in state.log.of_kind(EventKind.WARNING, EventKind.ERROR)
    ]
    if warnings:
        bullets.append("Warnings raised during the run: " + "; ".join(warnings))

    return Section(
        title="How this analysis was produced",
        paragraphs=paragraphs,
        bullets=bullets,
    )


def summary_rows(state: PipelineState) -> list[list[str]]:
    """The activity log as rows, for the appendix of a document."""
    return [
        [
            f"{event.at:%H:%M:%S}",
            event.stage,
            event.kind.value.replace("_", " "),
            event.message,
        ]
        for event in state.log
    ]
