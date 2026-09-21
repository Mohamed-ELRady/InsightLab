"""The report's content, assembled once and rendered three ways.

PDF, PowerPoint and Word are three presentations of the same material, so the
material is built here and each renderer only decides how to lay it out. That is
what stops the Word version quietly saying something different from the slides.
"""

from __future__ import annotations

import hashlib
import io
import tempfile
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import plotly.io as plotly_io

from ..core.activity_log import EventKind
from ..core.language import (
    answer_description,
    kpi_formula,
    kpi_meaning,
    kpi_name,
    profile_note,
    role_name,
    stage_title,
)
from ..core.state import PipelineState, RunMode, StageStatus

#: Width and height in pixels for chart images embedded in documents.
IMAGE_WIDTH = 1000
IMAGE_HEIGHT = 460

#: Charts beyond this many make a document nobody finishes.
MAX_CHART_IMAGES = 10

# Repeated analyses of the same export produce byte-for-byte identical Plotly
# figures.  Keeping a small process-local cache removes the browser startup and
# rasterisation cost on reruns without persisting user data anywhere new.
MAX_IMAGE_CACHE_BYTES = 64 * 1024 * 1024
_IMAGE_CACHE: OrderedDict[str, bytes] = OrderedDict()
_IMAGE_CACHE_BYTES = 0
_IMAGE_CACHE_LOCK = threading.Lock()
_KALEIDO_LOCK = threading.Lock()


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


def _cached_image(key: str) -> bytes | None:
    with _IMAGE_CACHE_LOCK:
        image = _IMAGE_CACHE.get(key)
        if image is not None:
            _IMAGE_CACHE.move_to_end(key)
        return image


def _remember_image(key: str, image: bytes) -> None:
    global _IMAGE_CACHE_BYTES
    if len(image) > MAX_IMAGE_CACHE_BYTES:
        return
    with _IMAGE_CACHE_LOCK:
        previous = _IMAGE_CACHE.pop(key, None)
        if previous is not None:
            _IMAGE_CACHE_BYTES -= len(previous)
        _IMAGE_CACHE[key] = image
        _IMAGE_CACHE_BYTES += len(image)
        while _IMAGE_CACHE_BYTES > MAX_IMAGE_CACHE_BYTES and _IMAGE_CACHE:
            _, removed = _IMAGE_CACHE.popitem(last=False)
            _IMAGE_CACHE_BYTES -= len(removed)


def _run_kaleido_batch(prepared, paths: list[Path]) -> None:
    """Render a variable-size batch through a process-wide warm browser."""
    import kaleido

    jobs = [
        {
            "fig": figure,
            "path": path,
            "opts": {
                "format": "png",
                "width": IMAGE_WIDTH,
                "height": height,
                "scale": 2,
            },
        }
        for (_, figure, height, _), path in zip(prepared, paths)
    ]
    # Kaleido's sync server owns one long-lived browser in a daemon thread.
    # Starting Chrome was the dominant report cost; keeping it warm makes later
    # analyses fast even when their charts differ and the image cache misses.
    # The queue behind the server serialises concurrent Streamlit sessions.
    with _KALEIDO_LOCK:
        kaleido.start_sync_server(n=min(2, len(jobs)), silence_warnings=True)
        kaleido._global_server.call_function(  # noqa: SLF001 - Kaleido's documented server bridge
            "write_fig_from_object", jobs, cancel_on_error=True
        )


def render_chart_images(state: PipelineState) -> dict[str, bytes]:
    """Turn the stored plotly figures into PNGs for the documents.

    Image export needs a browser engine that is not guaranteed to be present, so
    a failure here drops the picture and keeps the text rather than losing the
    whole report.
    """
    images: dict[str, bytes] = {}
    prepared: list[tuple[str, Any, int, str]] = []
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

            cache_key = hashlib.sha256(
                (
                    chart.figure_json
                    + f"|png|{IMAGE_WIDTH}|{height}|2|white-report-v1"
                ).encode("utf-8")
            ).hexdigest()
            cached = _cached_image(cache_key)
            if cached is not None:
                images[chart.id] = cached
                continue
            prepared.append((chart.id, figure, height, cache_key))
        except Exception:  # noqa: BLE001 - a missing picture must not lose the report
            continue

    if not prepared:
        return images

    state.report_progress("report", 0.28, "progress.report.charts")
    try:
        # Plotly 6/Kaleido 1 can keep one browser process alive for a batch.
        # The old one-image-at-a-time path paid browser startup cost for every
        # chart (roughly 18 seconds for nine images in profiling).
        with tempfile.TemporaryDirectory(prefix="insightlab-charts-") as temporary:
            paths = [Path(temporary) / f"chart-{index}.png" for index in range(len(prepared))]
            # Direct Kaleido use avoids Plotly's compatibility wrapper and lets
            # every image share the same already-open browser process.
            _run_kaleido_batch(prepared, paths)
            for (chart_id, _, _, cache_key), path in zip(prepared, paths):
                if path.exists():
                    rendered = path.read_bytes()
                    images[chart_id] = rendered
                    _remember_image(cache_key, rendered)
    except Exception:  # noqa: BLE001 - preserve compatibility with older Kaleido
        for chart_id, figure, height, cache_key in prepared:
            try:
                output = io.BytesIO()
                figure.write_image(
                    output, format="png", width=IMAGE_WIDTH, height=height, scale=2
                )
                rendered = output.getvalue()
                images[chart_id] = rendered
                _remember_image(cache_key, rendered)
            except Exception:  # noqa: BLE001
                continue
    state.report_progress("report", 0.62, "progress.report.charts")
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
        title=(
            f"تحليل {state.source_name or 'بياناتك'}"
            if state.language.code == "ar"
            else f"Analysis of {state.source_name or 'your data'}"
        ),
        subtitle=(
            f"أُعِدّ في {state.started_at:%Y-%m-%d}"
            if state.language.code == "ar"
            else state.started_at.strftime("Prepared %d %B %Y")
        ),
        sections=[section for section in sections if _has_content(section)],
        chart_images=render_chart_images(state),
    )


def _has_content(section: Section) -> bool:
    return bool(section.paragraphs or section.bullets or section.table or section.chart_ids)


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------


def _executive_summary(state: PipelineState) -> Section:
    ar = state.language.code == "ar"
    paragraphs = []
    if state.profile.summary:
        if ar:
            paragraphs.append(
                f"يتضمن التحليل {state.profile.row_count:,} صفًا و{state.profile.column_count} عمودًا بعد التنظيف."
            )
        else:
            paragraphs.append(state.profile.summary)

    if state.kpis:
        headline = ("، " if ar else ", ").join(
            (f"{kpi_name(kpi.name, state.language)}: {kpi.display_value}" if ar else f"{kpi.name.lower()} of {kpi.display_value}")
            for kpi in state.kpis[:3]
        )
        paragraphs.append((f"أهم المؤشرات: {headline}." if ar else f"The headline position is {headline}."))

    high = [item for item in state.insights if item.confidence == "high"]
    if high:
        paragraphs.append((
            f"يستند {len(high)} من أصل {len(state.insights)} استنتاجات إلى فروق واضحة ويمكن التعامل معها مباشرة، بينما تحتاج النتائج الأخرى إلى مراجعة."
            if ar else
            f"{len(high)} of the {len(state.insights)} conclusions below rest on "
            "large, clear differences in the data and can be acted on directly. "
            "The rest are worth checking before committing to them."
        ))
    elif state.insights:
        paragraphs.append((
            f"تم استخلاص {len(state.insights)} استنتاجات، لكنها تحتاج إلى مراجعة قبل اتخاذ قرار مباشر."
            if ar else
            f"{len(state.insights)} conclusions were drawn. None of them rests on "
            "a difference large enough to act on without checking first."
        ))

    return Section(
        title="الملخص التنفيذي" if ar else "Executive summary",
        paragraphs=paragraphs,
        bullets=[
            f"{item.title}: {item.result}" for item in state.insights[:4]
        ],
    )


def _about_the_data(state: PipelineState) -> Section:
    ar = state.language.code == "ar"
    profile = state.profile
    paragraphs = [(
        f"تم إجراء التحليل على {state.source_name or 'الملف المرفوع'}"
        + (f" ({state.source_format})" if state.source_format else "")
        + f"، ويحتوي بعد التنظيف على {profile.row_count:,} صفًا و{profile.column_count} عمودًا."
        if ar else
        f"The analysis was run on {state.source_name or 'the supplied file'}"
        + (f" ({state.source_format})" if state.source_format else "")
        + f", which holds {profile.row_count:,} rows across {profile.column_count} "
        "columns after cleaning."
    )]
    if state.load_notes:
        paragraphs.append(" ".join(state.load_notes))

    rows = [
        [
            column.name,
            role_name(column.role.value, state.language),
            f"{column.missing_rate:.0%}",
            f"{column.unique_count:,}",
            profile_note(column.note, state.language) or "-",
        ]
        for column in profile.columns
    ]
    return Section(
        title="عن البيانات" if ar else "About the data",
        paragraphs=paragraphs,
        table=(([
            "العمود", "نوع المحتوى", "القيم الفارغة", "القيم المختلفة", "ملاحظات"
        ] if ar else ["Column", "Holds", "Empty", "Distinct values", "Notes"]), rows),
    )


def _what_we_changed(state: PipelineState) -> Section:
    ar = state.language.code == "ar"
    operations = state.log.data_operations()
    if not operations:
        return Section(
            title="التعديلات على البيانات" if ar else "What we changed",
            paragraphs=["تم تحليل البيانات كما رُفعت بالضبط من غير أي تعديل." if ar else "The data was analysed exactly as supplied. Nothing was changed."],
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
        title="التعديلات على البيانات" if ar else "What we changed",
        paragraphs=[(
            "يعرض هذا القسم كل تعديل بالترتيب. النسخة المنظّفة محفوظة بجانب التقرير، ولم يتم تعديل الملف الأصلي."
            if ar else
            "Every change made to your data is listed here in the order it "
            "happened. The cleaned copy is saved alongside this report, and your "
            "original file was never modified."
        )],
        bullets=bullets,
    )


def _headline_figures(state: PipelineState) -> Section:
    ar = state.language.code == "ar"
    if not state.kpis:
        return Section(title="المؤشرات الرئيسية" if ar else "Headline figures")
    rows = [
        [
            kpi_name(kpi.name, state.language),
            kpi.display_value,
            kpi_formula(kpi.formula, state.language),
            kpi_meaning(kpi.name, kpi.interpretation, state.language),
        ]
        for kpi in state.kpis
    ]
    return Section(
        title="المؤشرات الرئيسية" if ar else "Headline figures",
        paragraphs=[(
            "كل مؤشر محسوب من أعمدة ملفك، ومعه طريقة الحساب حتى تقدر تراجعه مع سجلاتك."
            if ar else
            "Each figure below is calculated from your own columns. The formula "
            "is given so you can reconcile it against your own records."
        )],
        table=(([
            "المؤشر", "القيمة", "طريقة الحساب", "ماذا يعني"
        ] if ar else ["Measure", "Value", "How it is calculated", "What it means"]), rows),
    )


def _findings(state: PipelineState) -> Section:
    ar = state.language.code == "ar"
    if not state.insights:
        return Section(title="ما توضحه البيانات" if ar else "What the data shows")

    paragraphs = [(
        "كل نتيجة تعرض ما توضحه البيانات والرقم الداعم ومعناها ودرجة الثقة. البيانات توضح الارتباط لكنها لا تثبت السببية."
        if ar else
        "Each finding below states what the data shows, the figure it rests on, "
        "what it means in this domain, and how far it can be trusted. This is a "
        "single file of records, so a finding can show that two things move "
        "together but never that one caused the other."
    )]
    bullets = []
    for insight in state.insights:
        confidence = {"high": "مرتفعة", "medium": "متوسطة", "low": "منخفضة"}.get(insight.confidence, insight.confidence)
        parts = [
            (f"{insight.title} [الثقة: {confidence}]" if ar else f"{insight.title} [confidence: {insight.confidence}]"),
            insight.result,
        ]
        if insight.evidence:
            parts.append((f"الدليل: {insight.evidence}" if ar else f"Evidence: {insight.evidence}"))
        if insight.interpretation:
            parts.append((f"المعنى: {insight.interpretation}" if ar else f"What this means: {insight.interpretation}"))
        if insight.action:
            parts.append((f"الإجراء المقترح: {insight.action}" if ar else f"Suggested action: {insight.action}"))
        bullets.append("\n".join(parts))

    return Section(title="ما توضحه البيانات" if ar else "What the data shows", paragraphs=paragraphs, bullets=bullets)


def _charts(state: PipelineState) -> Section:
    ar = state.language.code == "ar"
    if not state.charts:
        return Section(title="الرسومات البيانية" if ar else "Charts")
    return Section(
        title="الرسومات البيانية" if ar else "Charts",
        paragraphs=[(
            "كل رسم معه شرح مختصر لما يعرضه حتى يمكن فهمه من داخل التقرير مباشرة."
            if ar else
            "Each chart carries a sentence saying what it shows, so it can be "
            "read without going back to the analysis."
        )],
        chart_ids=[chart.id for chart in state.charts[:MAX_CHART_IMAGES]],
    )


def _what_you_told_us(state: PipelineState) -> Section:
    ar = state.language.code == "ar"
    if not state.memory:
        return Section(title="ما أخبرتنا به عن البيانات" if ar else "What you told us about this data")
    return Section(
        title="ما أخبرتنا به عن البيانات" if ar else "What you told us about this data",
        paragraphs=[(
            "هذه المعلومات أثرت على التحليل وهي محفوظة مع المشروع لتطبيقها تلقائيًا في التحليلات القادمة."
            if ar else
            "These are the project and domain facts that shaped this analysis. "
            "They are saved with the project and will be applied automatically to "
            "future analyses of newer data."
        )],
        bullets=[f"{fact.statement} ({fact.category})" for fact in state.memory],
    )


def _recommendations(state: PipelineState) -> Section:
    ar = state.language.code == "ar"
    actions = [
        insight.action
        for insight in state.insights
        if insight.action and insight.confidence in ("high", "medium")
    ]
    if not actions:
        return Section(title="التوصيات" if ar else "Recommendations")
    return Section(
        title="التوصيات" if ar else "Recommendations",
        paragraphs=[(
            "التوصيات مرتبطة مباشرة بالنتائج السابقة ومرتبة حسب قوة الأدلة."
            if ar else
            "These follow directly from the findings above, ordered by how far "
            "the evidence behind them can be trusted."
        )],
        bullets=list(dict.fromkeys(actions)),
    )


def _next_steps(state: PipelineState) -> Section:
    ar = state.language.code == "ar"
    steps = []

    weak = [column for column in state.profile.columns if column.missing_rate > 0.2]
    if weak:
        steps.append(
            (f"حسّن طريقة تسجيل {', '.join(column.name for column in weak[:3])}؛ الحسابات الحالية تعتمد فقط على الصفوف التي تحتوي على قيمة.")
            if ar else
            (f"Improve how {', '.join(column.name for column in weak[:3])} is recorded. Anything calculated from these columns currently rests on the rows that happen to have a value.")
        )

    low = [item for item in state.insights if item.confidence == "low"]
    if low:
        steps.append(
            (f"راجع النتائج منخفضة الثقة وعددها {len(low)} اعتمادًا على معرفتك بالنشاط قبل اتخاذ قرار.")
            if ar else (f"Check the {len(low)} lower-confidence findings against what you know before acting on them.")
        )

    if not state.focus_axes:
        steps.append(
            "أعد التحليل مع سؤال محدد عن هذه البيانات؛ التركيز على سؤال واضح ينتج نتائج أدق."
            if ar else "Run the analysis again with a specific question about this data. A focused run produces sharper findings than a general one."
        )
    else:
        steps.append(
            "أعد التحليل عند توفر بيانات أحدث. كل معلومات المشروع محفوظة، لذلك سيبدأ التحليل القادم من حيث انتهى هذا التحليل."
            if ar else "Run this again when you have newer data. Everything you established about the project and domain is remembered, so the next run starts where this one left off rather than asking again."
        )

    if state.mode is RunMode.AUTONOMOUS:
        steps.append(
            "هذا التحليل تم تلقائيًا، لذلك راجع سجل القرارات قبل مشاركة النتائج."
            if ar else "This run was automatic, so every decision was taken on our judgement rather than yours. Reviewing the decision log is worth the time before the findings are shared."
        )

    return Section(title="الخطوات التالية" if ar else "Next steps", bullets=steps)


def _how_this_was_produced(state: PipelineState) -> Section:
    ar = state.language.code == "ar"
    decisions = state.answers
    automatic = sum(1 for _, answer in decisions if answer.automatic)
    manual = len(decisions) - automatic

    paragraphs = [(
        f"مرّ التحليل عبر {len(decisions)} نقاط قرار؛ أجبت أنت عن {manual} منها، وتم اتخاذ {automatic} تلقائيًا."
        if ar else
        f"The analysis ran in {state.mode.value} mode and passed through "
        f"{len(decisions)} decision points. {manual} were answered by you and "
        f"{automatic} were taken automatically."
    )]

    failed = [
        stage
        for stage, status in state.stage_status.items()
        if status is StageStatus.FAILED
    ]
    if failed:
        paragraphs.append(
            (f"لم تكتمل المراحل التالية: {', '.join(stage_title(stage, state.language) for stage in failed)}. النتائج لا تتضمن أي مخرجات منها.")
            if ar else (f"These stages did not complete: {', '.join(failed)}. The findings above do not include anything they would have produced.")
        )

    bullets = [
        f"{answer.answered_at:%H:%M} - {decision.topic}: {answer_description(answer, state.language)}"
        + ((" (تم تلقائيًا)" if ar else " (taken automatically)") if answer.automatic else "")
        for decision, answer in decisions
    ]

    warnings = [
        event.message
        for event in state.log.of_kind(EventKind.WARNING, EventKind.ERROR)
    ]
    if warnings:
        bullets.append(("تحذيرات أثناء التحليل: " if ar else "Warnings raised during the run: ") + "; ".join(warnings))

    return Section(
        title="كيف تم إعداد التحليل" if ar else "How this analysis was produced",
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
