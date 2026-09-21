"""The controls that turn feedback into reviewed project memory."""

from __future__ import annotations

import json

import pytest
from streamlit.testing.v1 import AppTest

from insightlab.core.project_memory import ProjectMemoryStore


def _state_source(root, project_id: str, body: str) -> str:
    return (
        "from pathlib import Path\n"
        "from insightlab.core.state import PipelineState, Insight\n"
        "from insightlab.core.language import ENGLISH\n"
        "from insightlab.app import project_memory_panel\n"
        f"state = PipelineState(run_id='ui-run', workspace=Path({str(root)!r}))\n"
        f"state.project_id = {project_id!r}\n"
        "state.project_name = 'UI Project'\n"
        + body
    )


def test_incorrect_insight_creates_reviewable_lesson(tmp_path):
    root = tmp_path / "runs"
    store = ProjectMemoryStore(root)
    project = store.ensure_project("UI Project")
    app = AppTest.from_string(
        _state_source(
            root,
            project.id,
            "insight = Insight('Wrong conclusion', 'result', 'evidence', 'meaning', action='act')\n"
            "project_memory_panel.render_insight_feedback(state, insight, ENGLISH)\n",
        )
    ).run()

    key = store.item_key("0", "Wrong conclusion", "evidence", "act")
    app.text_area(key=f"correction_{key}").set_value(
        "Cancelled orders never count as revenue"
    ).run()
    app.text_input(key=f"correction_reason_{key}").set_value(
        "The payment was reversed"
    ).run()
    app.button(key=f"submit_correction_{key}").click().run()

    assert not app.exception
    assert not store.load_memory(project.id), "a correction is not trusted before review"
    candidates = store.list_candidates(project.id, "pending")
    assert [candidate.statement for candidate in candidates] == [
        "Cancelled orders never count as revenue"
    ]


def test_owner_can_edit_and_approve_lesson_from_memory_tab(tmp_path):
    root = tmp_path / "runs"
    store = ProjectMemoryStore(root)
    project = store.ensure_project("UI Project")
    _, candidate_id = store.record_feedback(
        run_id="ui-run", project_id=project.id, item_type="insight", item_key="one",
        rating="incorrect", correction="Initial proposed rule",
    )
    app = AppTest.from_string(
        _state_source(root, project.id, "project_memory_panel.render_tab(state, ENGLISH)\n")
    ).run()

    app.text_area(key=f"candidate_text_{candidate_id}").set_value(
        "The owner-approved reusable rule"
    ).run()
    app.button(key=f"approve_{candidate_id}").click().run()

    assert not app.exception
    assert [fact.statement for fact in store.load_memory(project.id)] == [
        "The owner-approved reusable rule"
    ]
    assert not store.list_candidates(project.id, "pending")


def test_memory_tab_saves_project_preferences(tmp_path):
    root = tmp_path / "runs"
    store = ProjectMemoryStore(root)
    project = store.ensure_project("UI Project")
    app = AppTest.from_string(
        _state_source(root, project.id, "project_memory_panel.render_tab(state, ENGLISH)\n")
    ).run()

    app.selectbox(key="preference_detail").set_value("detailed").run()
    app.selectbox(key="preference_questions").set_value("fewer questions").run()
    app.selectbox(key="preference_audience").set_value("executives").run()
    app.button(key="save_project_preferences").click().run()

    assert not app.exception
    assert store.preferences(project.id) == {
        "explanation_detail": "detailed",
        "question_style": "fewer questions",
        "report_audience": "executives",
    }


@pytest.mark.parametrize(("action", "button_prefix", "rebuild"), [
    ("revise", "run_review_revise_", "false"),
    ("rebuild", "run_review_rebuild_", "true"),
])
def test_overall_review_returns_real_action_and_creates_scoped_lesson(
    tmp_path, action, button_prefix, rebuild
):
    root = tmp_path / "runs"
    store = ProjectMemoryStore(root)
    project = store.ensure_project("Scientific Project")
    app = AppTest.from_string(
        _state_source(
            root,
            project.id,
            "import streamlit as st\n"
            "state.understanding.domain_family = 'earth_science'\n"
            "result = project_memory_panel.render_run_review(state, ENGLISH)\n"
            "if result: st.write('ACTION=' + result[0] + ';REQUEST=' + result[1])\n",
        )
    ).run()

    app.text_area(key="run_review_instruction_ui-run").set_value(
        "Focus on earthquake frequency by region"
    ).run()
    app.button(key=f"{button_prefix}ui-run").click().run()

    assert not app.exception
    assert any(
        f"ACTION={action};REQUEST=Focus on earthquake frequency by region" in item.value
        for item in app.markdown
    )
    feedback = store.feedback_for("ui-run", "run", "overall_result")
    context = json.loads(feedback["context_json"])
    assert feedback["rating"] == "incorrect"
    assert context == {
        "domain_family": "earth_science",
        "rebuild": rebuild,
        "review_action": action,
    }
    assert store.list_candidates(project.id, "pending")
