"""The chat panel: the half of the collaboration the product was missing.

Every answer shows three things — what we understood the question to mean, the
figures, and one sentence of interpretation. The first of those matters most:
when the answer looks wrong, it is usually because the question was read
differently, and showing the reading lets the user correct it in one go instead
of guessing.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ..agents.analyst import Answer
from ..analysis import theme
from ..core.language import DEFAULT, translate
from ..core.state import PipelineState

#: Exchanges kept on screen. Older ones stay in the activity log.
HISTORY_LIMIT = 12


def render(state: PipelineState, supervisor) -> None:
    """Draw the whole conversation area."""
    language = getattr(state, "language", DEFAULT)
    st.markdown(f"#### {translate('ask.title', language)}")
    st.caption(translate("ask.caption", language))

    history = st.session_state.setdefault("conversation", [])

    _suggestions(state, supervisor)

    question = st.chat_input(translate("ask.placeholder", language))
    if question:
        with st.spinner(translate("ask.thinking", language)):
            answer = supervisor.ask(question)
        history.append(answer)
        del history[:-HISTORY_LIMIT]
        st.rerun()

    for index, answer in enumerate(reversed(history)):
        _exchange(answer, index, language)


def _suggestions(state: PipelineState, supervisor) -> None:
    """Offer questions this data can answer, so the box is not intimidating."""
    if "suggested_questions" not in st.session_state:
        with st.spinner("Working out what is worth asking"):
            st.session_state.suggested_questions = supervisor.suggested_questions()

    suggestions = st.session_state.suggested_questions
    if not suggestions:
        return

    st.caption(translate("ask.suggestions", getattr(state, "language", DEFAULT)))
    columns = st.columns(min(len(suggestions), 2))
    for index, question in enumerate(suggestions):
        with columns[index % len(columns)]:
            if st.button(question, key=f"suggested_{index}", width="stretch"):
                with st.spinner("Working it out"):
                    answer = supervisor.ask(question)
                history = st.session_state.setdefault("conversation", [])
                history.append(answer)
                del history[:-HISTORY_LIMIT]
                st.rerun()


def _exchange(answer: Answer, index: int, language=DEFAULT) -> None:
    """One question and its answer."""
    with st.chat_message("user"):
        st.markdown(answer.question)

    with st.chat_message("assistant"):
        if not answer.answered:
            st.warning(answer.refusal)
            return

        # Showing the reading first: when an answer looks wrong it is usually
        # the question that was read differently, and this is what lets the
        # user see that immediately.
        st.caption(f"{translate('ask.understood', language)}: {answer.understood_as}")
        st.markdown(answer.narrative or answer.headline)

        result = answer.result
        if result is None or result.table.empty:
            return

        if result.is_single_value:
            return

        figure = _figure(result)
        if figure is not None:
            st.plotly_chart(
                figure,
                width="stretch",
                key=f"answer_chart_{index}_{id(answer)}",
                config={"displayModeBar": False},
            )

        with st.expander(f"{translate('ask.numbers', language)} ({result.row_count:,})"):
            st.dataframe(result.table, width="stretch", hide_index=True)


def _figure(result) -> go.Figure | None:
    """A chart for an answer, in the same visual system as everything else."""
    table = result.table
    if table.shape[1] < 2 or len(table) < 2:
        return None

    label_column = table.columns[0]
    value_column = table.columns[-1]
    values = pd.to_numeric(table[value_column], errors="coerce")
    if values.isna().all():
        return None

    labels = table[label_column].astype(str).tolist()
    numbers = values.fillna(0).tolist()

    if result.chart_kind == "line":
        figure = go.Figure(
            go.Scatter(
                x=labels,
                y=numbers,
                mode="lines+markers",
                line=dict(color=theme.series_colour(0), width=2),
                marker=dict(size=8, color=theme.series_colour(0)),
                hovertemplate="%{x}<br>%{y:,.2f}<extra></extra>",
            )
        )
        theme.style(figure, show_legend=False, y_title=str(value_column), height=320)
    else:
        shown = labels[:12][::-1]
        amounts = numbers[:12][::-1]
        figure = go.Figure(
            go.Bar(
                x=amounts,
                y=shown,
                orientation="h",
                marker=dict(color=theme.series_colour(0), cornerradius=4),
                text=[f"{value:,.0f}" for value in amounts],
                textposition="outside",
                textfont=dict(size=11, color=theme.LIGHT["secondary"]),
                hovertemplate="%{y}<br>%{x:,.2f}<extra></extra>",
            )
        )
        theme.style(
            figure,
            show_legend=False,
            x_title=str(value_column),
            height=max(240, 40 * len(shown) + 70),
        )
        figure.update_xaxes(
            showgrid=True,
            gridcolor=theme.LIGHT["grid"],
            range=[min(min(amounts), 0), max(amounts) * 1.18] if amounts else None,
        )
        figure.update_yaxes(showgrid=False)

    if theme.tokens(True) and _dark():
        theme.retheme(figure, dark=True)
    return figure


def _dark() -> bool:
    from .theming import is_dark

    return is_dark()


def render_root_cause(state: PipelineState, supervisor) -> None:
    """The "why did that happen" panel."""
    language = getattr(state, "language", DEFAULT)
    st.markdown(f"#### {translate('why.title', language)}")

    attribution = st.session_state.get("attribution", "unset")
    if attribution == "unset":
        with st.spinner("Breaking the movement down"):
            attribution = supervisor.explain_change()
        st.session_state.attribution = attribution

    if attribution is None:
        st.info(
            "Explaining a movement needs a date column and at least three "
            "periods of data. This file does not have both."
        )
        return

    st.markdown(attribution.describe())

    for dimension in attribution.dimensions:
        with st.expander(f"{translate('why.breakdown', language)} {dimension.column.replace('_', ' ')}"):
            st.dataframe(
                pd.DataFrame(
                    {
                        dimension.column: [item.label for item in dimension.contributions],
                        "Contribution": [
                            round(item.change, 2) for item in dimension.contributions
                        ],
                        "Share of the movement": [
                            f"{item.share:.0%}" for item in dimension.contributions
                        ],
                    }
                ),
                width="stretch",
                hide_index=True,
            )
            st.caption(translate("why.exact", language))
