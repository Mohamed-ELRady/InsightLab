"""The decision panel: the same four options, wherever in the pipeline we are.

This is deliberately the only place a decision is ever rendered. Every agent
raises the identical shape, so the owner learns the interaction once and it
never changes underneath them.
"""

from __future__ import annotations

import streamlit as st

from ..core.decision import Answer, Decision
from ..core.language import DEFAULT, translate

#: Fixed labels for the two options that are not a concrete course of action.
CUSTOM_KEY = "decision.custom"
SKIP_KEY = "decision.skip"


def render(decision: Decision, language=DEFAULT) -> Answer | None:
    """Draw the panel. Returns an answer once the owner submits one."""
    st.markdown(f'<span class="il-tag">{decision.topic}</span>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="il-question">{decision.question}</div>', unsafe_allow_html=True
    )
    st.markdown(decision.context)
    st.write("")

    custom_label = translate(CUSTOM_KEY, language)
    skip_label = translate(SKIP_KEY, language)

    labels = [f"{decision.suggestion.label}  ·  {translate('decision.recommended', language)}"]
    labels += [option.label for option in decision.alternatives]
    labels += [custom_label, skip_label]

    choice = st.radio(
        translate("decision.prompt", language),
        options=range(len(labels)),
        format_func=lambda index: labels[index],
        key=f"choice_{decision.id}",
    )

    custom_text = ""
    if choice == 0:
        st.info(decision.suggestion.rationale)
    elif 1 <= choice <= len(decision.alternatives):
        st.info(decision.alternatives[choice - 1].rationale)
    elif labels[choice] == custom_label:
        st.caption(decision.custom_prompt)
        custom_text = st.text_area(
            translate("decision.your_instruction", language),
            key=f"custom_{decision.id}",
            height=110,
            label_visibility="collapsed",
            placeholder=translate("decision.custom_placeholder", language),
        )
    else:
        st.warning(decision.skip_effect)

    if not st.button(translate("decision.continue", language), type="primary", key=f"submit_{decision.id}"):
        return None

    if choice == 0:
        return Answer.accept(decision)
    if 1 <= choice <= len(decision.alternatives):
        return Answer.alternative(decision, choice - 1)
    if labels[choice] == skip_label:
        return Answer.skip(decision)

    if not custom_text.strip():
        st.error(translate("decision.custom_required", language))
        return None
    return Answer.custom(decision, custom_text)


def render_evidence(decision: Decision) -> None:
    """Show the raw figures a decision rests on, for anyone who wants them."""
    if not decision.evidence:
        return
    with st.expander(translate("decision.evidence", DEFAULT)):
        st.json(decision.evidence, expanded=False)
