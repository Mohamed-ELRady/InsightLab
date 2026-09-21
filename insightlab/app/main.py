"""InsightLab web application.

Run with:  streamlit run insightlab/app/main.py

The whole run lives in Streamlit's session state, including the supervisor's
generator, so the pipeline genuinely pauses between decisions rather than being
restarted and replayed on every interaction.
"""

from __future__ import annotations

import sys
from copy import deepcopy
from collections import OrderedDict
from pathlib import Path

import streamlit as st

if __package__ in (None, ""):  # pragma: no cover - direct `streamlit run` entry
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from insightlab.agents.supervisor import Supervisor, build_revision_agents
from insightlab.analysis.exploration import AXES
from insightlab.app import (
    chatgpt_panel,
    conversation,
    custom_chart_panel,
    decision_panel,
    project_memory_panel,
    provider_panel,
    theming,
)
from insightlab.core.config import PROJECT_ROOT
from insightlab.core.language import (
    ENGLISH,
    LANGUAGES,
    answer_description,
    artefact_name,
    axis_label,
    kpi_formula,
    kpi_meaning,
    kpi_name,
    profile_note,
    role_name,
    stage_title,
)
from insightlab.core.language import translate as _t
from insightlab.core.reasoning import ReasoningEngine
from insightlab.core.state import (
    STAGES,
    PipelineState,
    RunMode,
    StageStatus,
)
from insightlab.core.project_memory import ProjectMemoryStore
from insightlab.core.activity_log import EventKind
from insightlab.core.storage import list_runs

STATUS_MARKS = {
    StageStatus.PENDING: "○",
    StageStatus.RUNNING: "◐",
    StageStatus.DONE: "●",
    StageStatus.SKIPPED: "◌",
    StageStatus.FAILED: "✕",
}

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"


def language():
    """The language this session is in.

    Read from the run once one exists, and from the picker before that, so the
    landing page is already translated when the user changes it.
    """
    state = st.session_state.get("state")
    if state is not None:
        return state.language
    return LANGUAGES.get(st.session_state.get("language_code", "en"), ENGLISH)


def t(key: str) -> str:
    return _t(key, language())


def tf(key: str, **values) -> str:
    """Translate and interpolate a UI string in one place."""
    return t(key).format(**values)


def progress_text(value: float, stage: str = "", detail: str = "") -> str:
    """A visible numeric percentage plus a plain description of current work."""
    percent = max(1, min(100, round(value * 100)))
    shown_percent = str(percent)
    symbol = "%"
    if language().code == "ar":
        shown_percent = shown_percent.translate(str.maketrans("0123456789", "٠١٢٣٤٥٦٧٨٩"))
        symbol = "٪"
    label = t(detail) if detail else stage_title(stage, language()) if stage else t("progress.preparing")
    return f"{shown_percent}{symbol} — {label}"


def model_description(settings) -> str:
    """Localise the model summary without changing provider configuration."""
    if settings.offline:
        return t("sidebar.offline_model")
    if issue := settings.configuration_issue:
        if language().code == "ar":
            lowered = issue.casefold()
            if "is not set" in lowered:
                issue = "لم يتم إدخال مفتاح API"
            elif "model id" in lowered:
                issue = "مطلوب معرّف للنموذج"
            elif "base url" in lowered:
                issue = "مطلوب رابط API الأساسي"
            elif "sign in with chatgpt" in lowered:
                issue = "سجّل الدخول بحساب ChatGPT أولًا"
            elif "offline mode" in lowered:
                issue = "وضع العمل بدون اتصال مفعّل"
            elif "no model provider" in lowered:
                issue = "لم يتم إعداد أي مزوّد للنموذج"
        return f"{settings.provider.label} · {issue}"
    fallback_count = max(0, len(settings.candidate_settings()) - 1)
    if fallback_count:
        suffix = (
            f" · {fallback_count} مسار احتياطي"
            if language().code == "ar"
            else f" · {fallback_count} fallback(s)"
        )
    else:
        suffix = ""
    return f"{settings.provider.label} · {settings.llm_model}{suffix}"


def profile_summary(state: PipelineState) -> str:
    """Describe the profile in the UI language from stable numeric fields."""
    if language().code != "ar":
        return state.profile.summary
    counts: dict[str, int] = {}
    for column in state.profile.columns:
        counts[column.role.value] = counts.get(column.role.value, 0) + 1
    breakdown = "، ".join(
        f"{count} {role_name(role, language())}"
        for role, count in sorted(counts.items())
        if count
    )
    return (
        f"{state.profile.row_count:,} صفًا و{state.profile.column_count} عمودًا"
        + (f" ({breakdown})" if breakdown else "")
        + "."
    )
ACCEPTED = ["csv", "tsv", "txt", "xlsx", "xlsm", "xls"]


def main() -> None:
    st.set_page_config(
        page_title="InsightLab",
        page_icon="◧",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(
        theming.stylesheet(language().rtl, theming.is_dark()), unsafe_allow_html=True
    )

    sidebar()
    if "supervisor" not in st.session_state:
        landing()
    else:
        run_view()


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------


def sidebar() -> None:
    with st.sidebar:
        st.markdown("### InsightLab")

        if "supervisor" not in st.session_state:
            chosen = st.segmented_control(
                "اللغة / Language",
                options=list(LANGUAGES),
                format_func=lambda code: LANGUAGES[code].name,
                default=st.session_state.get("language_code", "en"),
                label_visibility="collapsed",
            )
            if chosen and chosen != st.session_state.get("language_code"):
                st.session_state.language_code = chosen
                st.rerun()

        st.caption(t("app.tagline"))
        st.divider()

        active_reasoning = (
            st.session_state.supervisor.reasoning
            if "supervisor" in st.session_state else None
        )
        provider_panel.render(language(), reasoning=active_reasoning)
        st.divider()

        if "supervisor" in st.session_state:
            stage_list()
            st.divider()
            model_usage_panel()
            st.divider()
            project_memory_panel.render_sidebar(st.session_state.state, language())
            st.divider()
            if st.button(t("sidebar.new_run"), width="stretch"):
                reset()
                st.rerun()
        else:
            settings = provider_panel.current_settings()
            if not settings.llm_available:
                st.caption(t("sidebar.no_credentials"))

        st.divider()
        appearance()

        previous = list_runs()
        if previous:
            st.divider()
            st.markdown(f"**{t('sidebar.earlier')}**")
            for run in previous[:5]:
                st.caption(run.name)


def appearance() -> None:
    """Say which mode is active and where the switch is.

    Streamlit has no API for setting the theme from inside the app - it lives
    in the built-in menu, which almost nobody finds. Both modes are configured,
    so the switch works properly; this just points at it.
    """
    dark = theming.is_dark()
    current = t("sidebar.appearance.dark" if dark else "sidebar.appearance.light")
    st.markdown(f"**{t('sidebar.appearance')}** · {current}")
    st.caption(t("sidebar.appearance.hint"))


def model_usage_panel() -> None:
    """Show real request savings without guessing at provider token prices."""
    reasoning = st.session_state.supervisor.reasoning
    ar = language().code == "ar"
    active = reasoning.active_settings
    st.markdown(f"**{active.provider.label} · {active.llm_model}**")
    if reasoning.configured_route_count > 1:
        st.caption(
            (f"المسار النشط {reasoning.active_route_number} من {reasoning.configured_route_count}"
             if ar else f"Active route {reasoning.active_route_number} of {reasoning.configured_route_count}")
        )
    st.caption(f"{'طلبات التحليل' if ar else 'Analysis requests'}: {reasoning.call_count}")
    st.caption(f"{'طلبات أُعيد استخدامها' if ar else 'Reused answers'}: {reasoning.cache_hits}")
    used_tokens = reasoning.estimated_input_tokens + reasoning.estimated_output_tokens
    st.caption(
        (f"تقدير التوكنز المستخدمة: {used_tokens:,} · الموفَّرة: {reasoning.saved_tokens:,}"
         if ar else f"Estimated tokens used: {used_tokens:,} · saved: {reasoning.saved_tokens:,}")
    )
    if reasoning.failover_log:
        with st.expander("سجل التحويل التلقائي" if ar else "Automatic failover log"):
            for event in reasoning.failover_log:
                st.warning(event)
    if active.llm_provider == "chatgpt" and reasoning.chatgpt_client:
        if st.button(
            "تسجيل الخروج / قطع الاتصال" if ar else "Sign out / disconnect",
            key="chatgpt_run_sign_out",
            width="stretch",
        ):
            chatgpt_panel.disconnect()
            st.rerun()
    if not reasoning.available:
        st.caption("كل اتصالات الذكاء الاصطناعي غير متاحة؛ التحليل الإحصائي مكمل." if ar else "All AI routes are unavailable; deterministic analysis remains active.")


def stage_list() -> None:
    state: PipelineState = st.session_state.state
    supervisor: Supervisor = st.session_state.supervisor

    st.markdown(f"**{t('sidebar.progress')}**")
    for key, title in STAGES:
        status = state.stage_status[key]
        mark = STATUS_MARKS[status]
        active = supervisor.current_stage == key and status is StageStatus.RUNNING
        line = f"{mark}  {stage_title(key, language())}"
        if active:
            st.markdown(f"**{line}**")
        elif status is StageStatus.FAILED:
            st.markdown(f":red[{line}]")
        elif status in (StageStatus.DONE, StageStatus.SKIPPED):
            st.caption(line)
        else:
            st.markdown(f":gray[{line}]")


# ---------------------------------------------------------------------------
# Landing
# ---------------------------------------------------------------------------


def landing() -> None:
    st.title(t("landing.title"))
    st.markdown(t("landing.intro"))

    left, right = st.columns([3, 2], gap="large")

    with left:
        uploaded = st.file_uploader(
            t("landing.files"),
            type=ACCEPTED,
            accept_multiple_files=True,
            help=t("landing.files.help"),
        )
        if uploaded and len(uploaded) > 1:
            st.caption(tf("landing.files.multiple", count=len(uploaded)))

        mode_label = st.radio(
            t("landing.mode"),
            [
                t("landing.mode.interactive"),
                t("landing.mode.autonomous"),
            ],
            captions=[
                t("landing.mode.interactive.help"),
                t("landing.mode.autonomous.help"),
            ],
        )
        mode = (
            RunMode.INTERACTIVE
            if mode_label == t("landing.mode.interactive")
            else RunMode.AUTONOMOUS
        )

        project_name = project_memory_panel.project_picker(language())

        if st.button(
            t("landing.start"),
            type="primary",
            disabled=not uploaded or not project_name.strip(),
            width="stretch",
        ):
            begin(uploaded, mode, project_name)
            st.rerun()

    with right:
        st.markdown(f"#### {t('landing.deliverables')}")
        for item in (
            t("landing.deliverable.cleaned"),
            t("landing.deliverable.insights"),
            t("landing.deliverable.kpis"),
            t("landing.deliverable.dashboards"),
            t("landing.deliverable.reports"),
            t("landing.deliverable.memory"),
        ):
            st.markdown(f"- {item}")

        st.divider()
        st.markdown(f"#### {t('landing.areas')}")
        universal_axes = (
            "time", "comparisons", "distributions", "relationships", "locations", "quality"
        )
        st.caption("، ".join(axis_label(axis, language()) for axis in universal_axes) + ".")

        sample = PROJECT_ROOT / "data" / "samples" / "retail_sales.csv"
        if sample.exists():
            st.divider()
            if st.button(t("landing.sample"), width="stretch"):
                # Honours whichever mode was selected above, so the sample
                # behaves exactly like a real upload.
                begin(sample, mode, project_name)
                st.rerun()


def begin(
    uploaded,
    mode: RunMode,
    project_name: str = "My project",
    directive: str = "",
    parent_run_id: str = "",
    revision_kind: str = "",
) -> None:
    """Save the upload, build the run and take the first step."""
    state = PipelineState(mode=mode, language=language())
    state.analysis_directive = directive.strip()
    state.parent_run_id = parent_run_id
    state.revision_kind = revision_kind
    progress_bar = st.progress(0.01, text=progress_text(0.01, detail="progress.preparing"))

    if isinstance(uploaded, Path) or (
        isinstance(uploaded, list) and uploaded and all(isinstance(item, Path) for item in uploaded)
    ):
        saved = [uploaded] if isinstance(uploaded, Path) else uploaded
        progress_bar.progress(0.08, text=progress_text(0.08, detail="progress.load.detect"))
    else:
        # The uploader returns a list when several files are allowed, and a
        # single object when only one was chosen.
        incoming = uploaded if isinstance(uploaded, list) else [uploaded]
        upload_dir = UPLOAD_DIR / state.run_id
        upload_dir.mkdir(parents=True, exist_ok=True)
        total_bytes = sum(int(getattr(item, "size", 0) or 0) for item in incoming)
        written_bytes = 0
        last_percent = -1
        saved = []
        for item in incoming:
            target = upload_dir / Path(item.name).name
            item.seek(0)
            with target.open("wb") as handle:
                while chunk := item.read(4 * 1024 * 1024):
                    handle.write(chunk)
                    written_bytes += len(chunk)
                    share = written_bytes / max(total_bytes, written_bytes, 1)
                    visible = 0.01 + (0.07 * share)
                    percent = int(visible * 100)
                    # Updating Streamlit is much slower than writing another
                    # chunk. Refresh only when the user-visible percentage
                    # changes, while still reporting the final byte exactly.
                    if percent != last_percent or written_bytes >= total_bytes:
                        progress_bar.progress(
                            visible,
                            text=progress_text(visible, detail="progress.saving"),
                        )
                        last_percent = percent
            saved.append(target)

    state.source_path = saved[0]
    state.extra_paths = saved[1:]
    memory_store = ProjectMemoryStore(state.workspace)
    project = memory_store.ensure_project(project_name)
    legacy_memory = st.session_state.get("legacy_memory_import", "")
    if legacy_memory:
        try:
            memory_store.import_legacy_memory(project.id, Path(legacy_memory))
        except OSError:
            pass
    state.project_id = project.id
    state.project_name = project.name
    # Load early so an interactive run can show that the chosen project was
    # found. Data understanding never reads this memory; MemoryAgent filters it
    # by detected domain and columns before any later analytical stage uses it.
    state.memory = memory_store.load_memory(project.id)
    state.project_preferences = memory_store.preferences(project.id)
    state.learning_profile = memory_store.adaptive_profile(project.id)
    active_policy = memory_store.active_policy(project.id)
    state.improvement_policy = active_policy.rules
    state.improvement_policy_version = active_policy.version
    memory_store.record_run(
        state.run_id, project.id, str(saved[0].name), state.started_at
    )
    if state.parent_run_id:
        state.log.record(
            EventKind.NOTE,
            "revision",
            "Started a full rebuild from the original source files.",
            previous_run_id=state.parent_run_id,
            instruction=state.analysis_directive,
        )

    settings = provider_panel.current_settings()
    response_cache = st.session_state.setdefault("llm_response_cache", OrderedDict())
    reasoning = ReasoningEngine(
        settings=settings,
        language=state.language,
        response_cache=response_cache,
        chatgpt_client=st.session_state.get("chatgpt_client"),
    )
    supervisor = Supervisor(state, reasoning=reasoning)
    st.session_state.state = state
    st.session_state.supervisor = supervisor

    def update(overall: float, stage: str, detail: str) -> None:
        # File transfer owns the first 8%; the ten analysis stages own the rest.
        visible = min(1.0, 0.08 + 0.92 * overall)
        progress_bar.progress(
            visible,
            text=progress_text(visible, stage=stage, detail=detail),
        )

    supervisor.set_progress_callback(update)
    try:
        st.session_state.pending = supervisor.start()
    finally:
        supervisor.set_progress_callback(None)


def revise_current_results(previous: PipelineState, directive: str) -> None:
    """Create a new result revision without repeating load, cleaning or features.

    The prepared dataframe is copied into a fresh run, so the original result
    remains auditable and downloadable. Only the five output-producing stages
    run again, using the user's correction as an explicit instruction.
    """
    state = PipelineState(mode=previous.mode, language=previous.language)
    state.analysis_directive = directive.strip()
    state.parent_run_id = previous.run_id
    state.revision_kind = "revise"

    # Stable project and learning context.
    state.project_id = previous.project_id
    state.project_name = previous.project_name
    state.project_preferences = dict(previous.project_preferences)
    state.learning_profile = deepcopy(previous.learning_profile)
    state.improvement_policy = dict(previous.improvement_policy)
    state.improvement_policy_version = previous.improvement_policy_version

    # Preserve the exact prepared input but not prior outputs. A shallow pandas
    # copy avoids doubling a large file in memory; downstream stages are
    # read-only with respect to these frames.
    state.source_path = previous.source_path
    state.extra_paths = list(previous.extra_paths)
    state.source_name = previous.source_name
    state.source_format = previous.source_format
    state.load_notes = list(previous.load_notes)
    state.raw_frame = previous.raw_frame.copy(deep=False) if previous.raw_frame is not None else None
    state.frame = previous.frame.copy(deep=False) if previous.frame is not None else None
    state.profile = deepcopy(previous.profile)
    state._frame_revision = previous._frame_revision
    state._profile_revision = previous._profile_revision
    state.understanding = deepcopy(previous.understanding)
    state.engineered_columns = list(previous.engineered_columns)
    state.comparison = previous.comparison
    state.fingerprint = previous.fingerprint

    store = ProjectMemoryStore(state.workspace)
    # The completed run already filtered memory by dataset columns and detected
    # domain. Reuse that safe view; reloading the whole project here could leak
    # an old sales rule into a scientific revision.
    state.memory = deepcopy(previous.memory)
    store.record_run(
        state.run_id, state.project_id, state.source_name, state.started_at
    )

    preserved = {"load", "understand", "recall", "clean", "features"}
    for stage in preserved:
        prior_status = previous.stage_status.get(stage, StageStatus.DONE)
        state.stage_status[stage] = (
            prior_status if prior_status in {StageStatus.DONE, StageStatus.SKIPPED}
            else StageStatus.DONE
        )
        state.stage_progress[stage] = 1.0
    state.log.record(
        EventKind.NOTE,
        "revision",
        "Started a focused revision from the prepared data.",
        previous_run_id=previous.run_id,
        instruction=state.analysis_directive,
    )

    # Reuse the model route and response cache; prompt caching still keys on
    # the directive, so unchanged work is free while requested changes rerun.
    reasoning = st.session_state.supervisor.reasoning
    supervisor = Supervisor(
        state,
        agents=build_revision_agents(reasoning),
        reasoning=reasoning,
    )
    st.session_state.state = state
    st.session_state.supervisor = supervisor
    progress_bar = st.progress(
        supervisor.progress(),
        text=progress_text(supervisor.progress(), detail="progress.preparing"),
    )
    supervisor.set_progress_callback(
        lambda value, stage, detail: progress_bar.progress(
            value, text=progress_text(value, stage=stage, detail=detail)
        )
    )
    try:
        st.session_state.pending = supervisor.start()
    finally:
        supervisor.set_progress_callback(None)


def reset() -> None:
    for key in ("state", "supervisor", "pending", "conversation",
                "suggested_questions", "attribution", "understanding_seen"):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_view() -> None:
    state: PipelineState = st.session_state.state
    supervisor: Supervisor = st.session_state.supervisor

    current_progress = supervisor.progress()
    progress_bar = st.progress(
        current_progress,
        text=(progress_text(1.0, detail="complete") if supervisor.finished else
              progress_text(current_progress, stage=supervisor.current_stage)),
    )

    pending = st.session_state.get("pending")
    if pending is not None:
        if state.understanding.summary:
            already_seen = st.session_state.get("understanding_seen", False)
            understanding_card(state, compact=already_seen)
            st.session_state.understanding_seen = True
        st.subheader(stage_title(pending.stage, language()))
        with st.container(border=True):
            answer = decision_panel.render(pending, language())
            decision_panel.render_evidence(pending, language())
        if answer is not None:
            supervisor.set_progress_callback(
                lambda value, stage, detail: progress_bar.progress(
                    value, text=progress_text(value, stage=stage, detail=detail)
                )
            )
            try:
                st.session_state.pending = supervisor.resolve(answer)
            finally:
                supervisor.set_progress_callback(None)
            st.rerun()
        show_progress_so_far(state)
        return

    if state.errors:
        for error in state.errors:
            st.error(error)

    results(state)


def show_progress_so_far(state: PipelineState) -> None:
    """A glance at what has been established, while a decision is open."""
    if not state.kpis and not state.charts:
        return
    st.divider()
    st.caption(t("run.found_so_far"))
    if state.kpis:
        theming.kpi_tiles(state.kpis[:4], language=language())


def understanding_card(state: PipelineState, *, compact: bool = False) -> None:
    """Keep the initial reading visible as the workflow moves to stage two."""
    understanding = state.understanding
    if not understanding.summary:
        return

    surface = (
        st.expander(t("understanding.title"), expanded=False)
        if compact
        else st.container(border=True)
    )
    with surface:
        if not compact:
            st.markdown(f"#### {understanding.dataset_title or t('understanding.title')}")
        st.caption(
            t(
                "understanding.phase1"
                if state.stage_status["understand"] is StageStatus.RUNNING
                else "understanding.ready"
            )
        )
        if understanding.hook:
            st.info(understanding.hook)
        st.markdown(understanding.summary)
        labels = {
            "high": t("confidence.high"),
            "medium": t("confidence.medium"),
            "low": t("confidence.low"),
        }
        details = []
        if understanding.business_domain:
            details.append(
                f"**{t('understanding.domain')}:** {understanding.business_domain}"
            )
        if understanding.row_represents:
            details.append(
                f"**{t('understanding.row')}:** {understanding.row_represents}"
            )
        details.append(
            f"**{t('understanding.confidence')}:** "
            f"{labels.get(understanding.confidence, understanding.confidence)}"
        )
        st.caption(" · ".join(details))
        if understanding.analysis_goal:
            st.markdown(f"**{t('understanding.goal')}:** {understanding.analysis_goal}")
        if understanding.suggested_questions:
            st.markdown(f"**{t('understanding.questions')}:**")
            for question in understanding.suggested_questions:
                st.markdown(f"- {question}")

        answered = [question for question in understanding.questions if question.answered]
        if answered:
            with st.expander(t("understanding.answers")):
                for question in answered:
                    st.markdown(f"- **{question.question}** — {question.answer}")


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def results(state: PipelineState) -> None:
    project_memory_panel.persist_completed_run(state)
    st.title(tf("results.title", name=state.source_name))
    st.caption(tf("results.project", name=state.project_name))
    if state.parent_run_id:
        action = (
            ("إعادة بناء كاملة من الملف الأصلي" if state.revision_kind == "rebuild" else "تعديل النتائج الحالية")
            if language().code == "ar" else
            ("Full rebuild from the original file" if state.revision_kind == "rebuild" else "Focused revision of the current results")
        )
        st.success(
            (f"تم تطبيق طلبك — {action}." if language().code == "ar" else f"Your request was applied — {action}.")
        )
        if state.analysis_directive:
            st.caption(
                ("طلبك: " if language().code == "ar" else "Your request: ")
                + state.analysis_directive
            )
    understanding_card(state)
    if state.profile.summary:
        st.markdown(profile_summary(state))

    tabs = st.tabs(
        [
            t("tab.ask"),
            t("tab.dashboards"),
            t("tab.insights"),
            t("tab.kpis"),
            t("tab.charts"),
            t("tab.data"),
            t("tab.log"),
            t("tab.downloads"),
            "Memory" if language().code != "ar" else "الذاكرة",
            "Improvement" if language().code != "ar" else "التحسين",
        ]
    )

    supervisor = st.session_state.supervisor
    with tabs[0]:
        conversation.render(state, supervisor)
        st.divider()
        conversation.render_root_cause(state, supervisor)
    with tabs[1]:
        dashboards_tab(state)
    with tabs[2]:
        insights_tab(state)
    with tabs[3]:
        kpis_tab(state)
    with tabs[4]:
        charts_tab(state)
    with tabs[5]:
        data_tab(state)
    with tabs[6]:
        log_tab(state)
    with tabs[7]:
        downloads_tab(state)
    with tabs[8]:
        project_memory_panel.render_tab(state, language())
    with tabs[9]:
        project_memory_panel.render_improvement_tab(state, language())

    review_action = project_memory_panel.render_run_review(state, language())
    if review_action:
        action, instruction = review_action
        if action == "revise":
            revise_current_results(state, instruction)
        else:
            sources = [state.source_path, *state.extra_paths]
            begin(
                sources,
                state.mode,
                state.project_name,
                directive=instruction,
                parent_run_id=state.run_id,
                revision_kind="rebuild",
            )
        st.rerun()


def dashboards_tab(state: PipelineState) -> None:
    if not state.dashboards:
        st.info(t("dashboard.empty"))
        return

    names = [board.title for board in state.dashboards]
    chosen = st.selectbox(t("dashboard.choose"), names, key="dashboard_pick")
    board = state.dashboards[names.index(chosen)]

    st.caption(board.description)

    kpi_names = [panel.reference for panel in board.panels if panel.kind == "kpi"]
    tiles = [kpi for kpi in (state.kpi(name) for name in kpi_names) if kpi]
    theming.kpi_tiles(tiles, language=language())

    chart_panels = [panel for panel in board.panels if panel.kind == "chart"]
    columns = st.columns(2, gap="large")
    for index, panel in enumerate(chart_panels):
        chart = state.chart(panel.reference)
        if chart is None:
            continue
        with columns[index % 2]:
            theming.show_chart(
                chart, key=f"dash_{board.id}_{chart.id}", language=language()
            )


def insights_tab(state: PipelineState) -> None:
    if not state.insights:
        st.info(t("insights.empty"))
        return

    st.caption(t("insights.about"))

    for insight_index, insight in enumerate(state.insights):
        with st.container(border=True):
            head, badge = st.columns([5, 1])
            with head:
                st.markdown(f"#### {insight.title}")
            with badge:
                st.markdown(
                    theming.confidence_badge(insight.confidence, language()),
                    unsafe_allow_html=True,
                )

            st.markdown(insight.result)
            if insight.evidence:
                st.caption(f"{t('insight.evidence')}: {insight.evidence}")
            if insight.interpretation:
                st.markdown(f"**{t('insight.meaning')}.** {insight.interpretation}")
            if insight.caveat:
                st.warning(f"**{t('insight.caveat')}.** {insight.caveat}")
            if insight.objection:
                st.info(f"**{t('insight.objection')}.** {insight.objection}")
            if insight.action:
                st.success(f"**{t('insight.action')}.** {insight.action}")

            chart = state.chart(insight.chart_id) if insight.chart_id else None
            if chart is not None:
                with st.expander(t("insights.chart")):
                    theming.show_chart(
                        chart,
                        key=f"insight_{insight.chart_id}",
                        language=language(),
                    )
            st.divider()
            project_memory_panel.render_insight_feedback(
                state, insight, language(), item_index=insight_index
            )


def kpis_tab(state: PipelineState) -> None:
    if not state.kpis:
        st.info(t("kpis.empty"))
        return
    theming.kpi_tiles(state.kpis, language=language())
    st.divider()
    for kpi in state.kpis:
        with st.container(border=True):
            st.markdown(f"**{kpi_name(kpi.name, language())}** — {kpi.display_value}")
            st.caption(f"{t('kpis.formula')}: {kpi_formula(kpi.formula, language())}")
            st.markdown(kpi_meaning(kpi.name, kpi.interpretation, language()))
            project_memory_panel.render_simple_feedback(
                state, kpi, language(), item_type="kpi"
            )


def charts_tab(state: PipelineState) -> None:
    custom_chart_panel.render(state, language())
    st.divider()
    if not state.charts:
        st.info(t("charts.empty"))
        return

    areas = sorted({chart.axis for chart in state.charts})
    labels = {
        axis: axis_label(axis, language())
        if axis in AXES else (
            ("رسومك المخصصة" if language().code == "ar" else "Your custom charts")
            if axis == "custom"
            else ("نظرة عامة" if language().code == "ar" else "General overview")
        )
        for axis in areas
    }
    chosen = st.multiselect(
        t("charts.areas"),
        areas,
        default=areas,
        format_func=lambda axis: labels[axis],
        placeholder=("اختر مجالات التحليل" if language().code == "ar" else "Choose areas"),
    )

    shown = [chart for chart in state.charts if chart.axis in chosen]
    columns = st.columns(2, gap="large")
    for index, chart in enumerate(shown):
        with columns[index % 2]:
            theming.show_chart(chart, key=f"all_{chart.id}", language=language())
            project_memory_panel.render_simple_feedback(
                state, chart, language(), item_type="chart"
            )


def data_tab(state: PipelineState) -> None:
    if state.frame is None:
        st.info(t("data.empty"))
        return

    st.markdown(f"**{tf('data.summary', rows=f'{len(state.frame):,}', columns=state.frame.shape[1])}**")
    st.dataframe(state.frame.head(500), width="stretch", height=420)
    if len(state.frame) > 500:
        st.caption(t("data.first_rows"))

    st.divider()
    st.markdown(f"**{t('data.columns')}**")
    import pandas as pd

    st.dataframe(
        pd.DataFrame(
            [
                {
                    t("data.column"): column.name,
                    t("data.role"): role_name(column.role.value, language()),
                    t("data.empty_values"): f"{column.missing_rate:.0%}",
                    t("data.distinct"): column.unique_count,
                    t("data.notes"): profile_note(column.note, language()) or "-",
                }
                for column in state.profile.columns
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def log_tab(state: PipelineState) -> None:
    changes = state.log.data_operations()
    st.markdown(f"**{t('log.changes')}**")
    if not changes:
        st.caption(t("log.no_changes"))
    for event in changes:
        st.markdown(f"- {event.message}")

    st.divider()
    st.markdown(f"**{t('log.decisions')}**")
    for decision, answer in state.answers:
        marker = t("log.automatic" if answer.automatic else "log.your_choice")
        st.markdown(f"- **{decision.topic}** — {answer_description(answer, language())}  *({marker})*")

    st.divider()
    with st.expander(t("log.full")):
        import pandas as pd

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        t("log.time"): f"{event.at:%H:%M:%S}",
                        t("log.stage"): stage_title(event.stage, language()),
                        t("log.event"): event.message,
                    }
                    for event in state.log
                ]
            ),
            width="stretch",
            hide_index=True,
            height=380,
        )


def downloads_tab(state: PipelineState) -> None:
    if not state.artefacts:
        st.info(t("downloads.empty"))
        return

    st.caption(t("downloads.caption"))
    for name, path in state.artefacts.items():
        path = Path(path)
        if not path.exists():
            continue
        st.download_button(
            tf("downloads.button", name=artefact_name(name, language())),
            data=path.read_bytes(),
            file_name=path.name,
            width="stretch",
            key=f"download_{path.name}",
        )

    if state.memory:
        st.divider()
        st.markdown(f"**{t('downloads.memory')}**")
        st.caption(t("downloads.memory.caption"))
        for fact in state.memory:
            st.markdown(f"- {fact.statement}  *({fact.category})*")


if __name__ == "__main__":
    main()
