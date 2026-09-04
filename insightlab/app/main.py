"""InsightLab web application.

Run with:  streamlit run insightlab/app/main.py

The whole run lives in Streamlit's session state, including the supervisor's
generator, so the pipeline genuinely pauses between decisions rather than being
restarted and replayed on every interaction.
"""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

if __package__ in (None, ""):  # pragma: no cover - direct `streamlit run` entry
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from insightlab.agents.supervisor import Supervisor
from insightlab.analysis.exploration import AXES
from insightlab.app import conversation, decision_panel, provider_panel, theming
from insightlab.core.config import PROJECT_ROOT
from insightlab.core.language import ARABIC, ENGLISH, LANGUAGES
from insightlab.core.language import translate as _t
from insightlab.core.reasoning import ReasoningEngine
from insightlab.core.state import (
    STAGES,
    STAGE_TITLES,
    PipelineState,
    RunMode,
    StageStatus,
)
from insightlab.core.storage import list_runs, load_business_memory

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
                "Language",
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

        if "supervisor" in st.session_state:
            stage_list()
            st.divider()
            memory_panel()
            st.divider()
            if st.button(t("sidebar.new_run"), width="stretch"):
                reset()
                st.rerun()
        else:
            settings = provider_panel.current_settings()
            st.markdown(f"**{t('sidebar.model')}**")
            st.caption(settings.describe_llm())
            if not settings.llm_available:
                st.caption(
                    "Without credentials every stage still runs, using the "
                    "built-in statistical rules. The explanations are shorter "
                    "and written from templates rather than tailored to you."
                )
            provider_panel.render(language())

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


def stage_list() -> None:
    state: PipelineState = st.session_state.state
    supervisor: Supervisor = st.session_state.supervisor

    st.markdown(f"**{t('sidebar.progress')}**")
    for key, title in STAGES:
        status = state.stage_status[key]
        mark = STATUS_MARKS[status]
        active = supervisor.current_stage == key and status is StageStatus.RUNNING
        line = f"{mark}  {title}"
        if active:
            st.markdown(f"**{line}**")
        elif status is StageStatus.FAILED:
            st.markdown(f":red[{line}]")
        elif status in (StageStatus.DONE, StageStatus.SKIPPED):
            st.caption(line)
        else:
            st.markdown(f":gray[{line}]")


def memory_panel() -> None:
    """What the owner has told us, and a way to add to it at any time."""
    state: PipelineState = st.session_state.state

    st.markdown(f"**{t('sidebar.memory')}** ({len(state.memory)})")
    if not state.memory:
        st.caption(
            "Nothing yet. Anything you tell us during the analysis is kept here "
            "and applied to every later stage."
        )
    else:
        for fact in list(state.memory)[-6:]:
            st.markdown(
                f'<div class="il-fact">{fact.statement}'
                f'<br><span class="il-tag">{fact.category}</span></div>',
                unsafe_allow_html=True,
            )

    with st.expander(t("sidebar.memory.add")):
        text = st.text_area(
            "Fact",
            key="memory_input",
            height=80,
            label_visibility="collapsed",
            placeholder="For example: our peak season starts in November.",
        )
        category = st.selectbox(
            "Type",
            ["definition", "seasonality", "exclusion", "classification", "target", "context"],
            key="memory_category",
        )
        if st.button(t("sidebar.memory.save"), width="stretch") and text.strip():
            state.remember(text, category=category, stage="user", topic="Added by you")
            st.rerun()


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
            help="CSV, Excel or a tab-separated export. Upload several and we "
            "will work out how they relate - sales, products and customers "
            "usually come out as separate files. Your files are never "
            "modified; everything is done on a copy.",
        )
        if uploaded and len(uploaded) > 1:
            st.caption(
                f"{len(uploaded)} files. We will ask how each one attaches to "
                "the largest before combining them."
            )

        mode_label = st.radio(
            t("landing.mode"),
            [
                t("landing.mode.interactive"),
                t("landing.mode.autonomous"),
            ],
            captions=[
                "We stop and explain each choice, and you decide. This is where "
                "your knowledge of the business goes in.",
                "We use our best judgement at every point and you review the "
                "decisions afterwards. Faster, but the analysis knows nothing "
                "about your business that the file does not say.",
            ],
        )
        mode = (
            RunMode.INTERACTIVE
            if mode_label == t("landing.mode.interactive")
            else RunMode.AUTONOMOUS
        )

        reuse = carry_over_memory()

        if st.button(
            t("landing.start"),
            type="primary",
            disabled=not uploaded,
            width="stretch",
        ):
            begin(uploaded, mode, reuse)
            st.rerun()

    with right:
        st.markdown(f"#### {t('landing.deliverables')}")
        for item in (
            "A cleaned copy of your data, with every change to it listed",
            "The conclusions that matter, each with the figure behind it",
            "Your headline numbers, each with its formula",
            "Dashboards laid out for whoever will open them",
            "A report in PDF, PowerPoint or Word",
            "Everything you told us, saved for the next analysis",
        ):
            st.markdown(f"- {item}")

        st.divider()
        st.markdown(f"#### {t('landing.areas')}")
        st.caption(", ".join(AXES.values()) + ".")

        sample = PROJECT_ROOT / "data" / "samples" / "retail_sales.csv"
        if sample.exists():
            st.divider()
            if st.button(t("landing.sample"), width="stretch"):
                # Honours whichever mode was selected above, so the sample
                # behaves exactly like a real upload.
                begin(sample, mode, reuse)
                st.rerun()


def carry_over_memory() -> Path | None:
    """Offer to start from what a previous run already learned."""
    previous = list_runs()
    candidates = [run / "business_memory.json" for run in previous]
    candidates = [path for path in candidates if path.exists()]
    if not candidates:
        return None

    if not st.checkbox(
        t("landing.reuse")
    ):
        return None

    labels = {str(path): path.parent.name for path in candidates[:10]}
    chosen = st.selectbox(
        "Which analysis",
        list(labels),
        format_func=lambda key: labels[key],
    )
    return Path(chosen)


def begin(uploaded, mode: RunMode, reuse: Path | None) -> None:
    """Save the upload, build the run and take the first step."""
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

    if isinstance(uploaded, Path):
        saved = [uploaded]
    else:
        # The uploader returns a list when several files are allowed, and a
        # single object when only one was chosen.
        incoming = uploaded if isinstance(uploaded, list) else [uploaded]
        saved = []
        for item in incoming:
            target = UPLOAD_DIR / item.name
            target.write_bytes(item.getbuffer())
            saved.append(target)

    state = PipelineState(mode=mode, language=language())
    state.source_path = saved[0]
    state.extra_paths = saved[1:]
    if reuse is not None:
        state.memory = load_business_memory(reuse)

    settings = provider_panel.current_settings()
    reasoning = ReasoningEngine(settings=settings, language=state.language)
    supervisor = Supervisor(state, reasoning=reasoning)
    st.session_state.state = state
    st.session_state.supervisor = supervisor

    # An autonomous run does the whole pipeline inside this one call, which
    # takes the best part of a minute. Without a spinner the page just sits
    # there looking broken.
    if mode is RunMode.AUTONOMOUS:
        with st.spinner(
            "Working through your data - loading, cleaning, exploring and "
            "writing it up. This usually takes under a minute.",
            show_time=True,
        ):
            st.session_state.pending = supervisor.start()
    else:
        st.session_state.pending = supervisor.start()


def reset() -> None:
    for key in ("state", "supervisor", "pending", "conversation",
                "suggested_questions", "attribution"):
        st.session_state.pop(key, None)


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def run_view() -> None:
    state: PipelineState = st.session_state.state
    supervisor: Supervisor = st.session_state.supervisor

    st.progress(
        supervisor.progress(),
        text=(
            supervisor.current_title or "Finished"
            if not supervisor.finished
            else t("complete")
        ),
    )

    pending = st.session_state.get("pending")
    if pending is not None:
        st.subheader(STAGE_TITLES.get(pending.stage, ""))
        with st.container(border=True):
            answer = decision_panel.render(pending, language())
            decision_panel.render_evidence(pending)
        if answer is not None:
            st.session_state.pending = supervisor.resolve(answer)
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
    st.caption("What we have found so far")
    if state.kpis:
        theming.kpi_tiles(state.kpis[:4])


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


def results(state: PipelineState) -> None:
    st.title(f"Analysis of {state.source_name}")
    if state.profile.summary:
        st.markdown(state.profile.summary)

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


def dashboards_tab(state: PipelineState) -> None:
    if not state.dashboards:
        st.info("No dashboard was built for this run.")
        return

    names = [board.title for board in state.dashboards]
    chosen = st.selectbox("Which view", names, key="dashboard_pick")
    board = state.dashboards[names.index(chosen)]

    st.caption(board.description)

    kpi_names = [panel.reference for panel in board.panels if panel.kind == "kpi"]
    tiles = [kpi for kpi in (state.kpi(name) for name in kpi_names) if kpi]
    theming.kpi_tiles(tiles)

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
        st.info("No conclusion was drawn from this data.")
        return

    st.caption(
        "Every conclusion carries the figure it rests on and how far it can be "
        "trusted. This is one file of records, so a finding can show two things "
        "move together but never that one caused the other."
    )

    for insight in state.insights:
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
                with st.expander("See the chart behind this"):
                    theming.show_chart(
                        chart,
                        key=f"insight_{insight.chart_id}",
                        language=language(),
                    )


def kpis_tab(state: PipelineState) -> None:
    if not state.kpis:
        st.info("No headline figure could be calculated from this data.")
        return
    theming.kpi_tiles(state.kpis)
    st.divider()
    for kpi in state.kpis:
        with st.container(border=True):
            st.markdown(f"**{kpi.name}** — {kpi.display_value}")
            st.caption(f"How it is calculated: {kpi.formula}")
            st.markdown(kpi.interpretation)


def charts_tab(state: PipelineState) -> None:
    if not state.charts:
        st.info("No chart was produced.")
        return

    areas = sorted({chart.axis for chart in state.charts})
    labels = {axis: AXES.get(axis, "General overview") for axis in areas}
    chosen = st.multiselect(
        "Areas",
        areas,
        default=areas,
        format_func=lambda axis: labels[axis],
    )

    shown = [chart for chart in state.charts if chart.axis in chosen]
    columns = st.columns(2, gap="large")
    for index, chart in enumerate(shown):
        with columns[index % 2]:
            theming.show_chart(chart, key=f"all_{chart.id}", language=language())


def data_tab(state: PipelineState) -> None:
    if state.frame is None:
        st.info("No data is loaded.")
        return

    st.markdown(
        f"**{len(state.frame):,} rows and {state.frame.shape[1]} columns** after cleaning."
    )
    st.dataframe(state.frame.head(500), width="stretch", height=420)
    if len(state.frame) > 500:
        st.caption("Showing the first 500 rows. The download holds all of them.")

    st.divider()
    st.markdown("**What each column holds**")
    import pandas as pd

    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Column": column.name,
                    "Holds": column.role.value,
                    "Empty": f"{column.missing_rate:.0%}",
                    "Distinct values": column.unique_count,
                    "Notes": column.note or "-",
                }
                for column in state.profile.columns
            ]
        ),
        width="stretch",
        hide_index=True,
    )


def log_tab(state: PipelineState) -> None:
    changes = state.log.data_operations()
    st.markdown("**Every change made to your data**")
    if not changes:
        st.caption("Nothing was changed. The data was analysed exactly as supplied.")
    for event in changes:
        st.markdown(f"- {event.message}")

    st.divider()
    st.markdown("**Every decision taken**")
    for decision, answer in state.answers:
        marker = "automatic" if answer.automatic else "your choice"
        st.markdown(f"- **{decision.topic}** — {answer.describe()}  *({marker})*")

    st.divider()
    with st.expander("The full run log"):
        import pandas as pd

        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Time": f"{event.at:%H:%M:%S}",
                        "Stage": STAGE_TITLES.get(event.stage, event.stage),
                        "What happened": event.message,
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
        st.info("Nothing was saved for this run.")
        return

    st.caption(
        "Everything below is saved on disk as well, so you can come back to it."
    )
    for name, path in state.artefacts.items():
        path = Path(path)
        if not path.exists():
            continue
        st.download_button(
            f"Download the {name.lower()}",
            data=path.read_bytes(),
            file_name=path.name,
            width="stretch",
            key=f"download_{path.name}",
        )

    if state.memory:
        st.divider()
        st.markdown("**What we learned about your business**")
        st.caption(
            "This travels with the project. Start your next analysis from it and "
            "none of these questions get asked again."
        )
        for fact in state.memory:
            st.markdown(f"- {fact.statement}  *({fact.category})*")


if __name__ == "__main__":
    main()
