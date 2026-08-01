"""The Streamlit interface, driven through Streamlit's own test harness.

These are the tests that would catch the interface and the pipeline drifting
apart - a decision the panel cannot render, or a results tab that reads a field
an agent no longer sets.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "insightlab" / "app" / "main.py"
SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "retail_sales.csv"

CONTINUE = "Continue"
CHOICE_LABEL = "What would you like to do?"


def pending(app: AppTest):
    try:
        return app.session_state["pending"]
    except (KeyError, AttributeError):
        return None


@pytest.fixture
def app(tmp_path, monkeypatch) -> AppTest:
    if not SAMPLE.exists():
        pytest.skip("Run scripts/make_sample_data.py to generate the sample data.")
    monkeypatch.setenv("INSIGHTLAB_OFFLINE", "1")
    monkeypatch.setenv("INSIGHTLAB_WORKSPACE", str(tmp_path / "runs"))
    instance = AppTest.from_file(str(APP), default_timeout=300)
    instance.run()
    return instance


def start_sample_run(app: AppTest) -> AppTest:
    [button for button in app.button if button.label == "Try it with sample data"][0].click().run()
    return app


class TestLanding:
    def test_the_page_renders_without_a_run(self, app):
        assert not app.exception
        assert app.title[0].value == "Understand your own data"

    def test_both_modes_are_offered(self, app):
        options = app.radio[0].options
        assert any("Ask me" in option for option in options)
        assert any("automatically" in option for option in options)

    def test_starting_is_blocked_until_a_file_is_chosen(self, app):
        start = [button for button in app.button if button.label == "Start the analysis"][0]
        assert start.disabled


class TestDecisionPanel:
    def test_a_decision_appears_as_soon_as_the_run_starts(self, app):
        start_sample_run(app)

        assert not app.exception
        assert pending(app) is not None
        assert [radio for radio in app.radio if radio.label == CHOICE_LABEL]
        assert [button for button in app.button if button.label == CONTINUE]

    def test_the_panel_always_offers_the_same_four_kinds_of_option(self, app):
        start_sample_run(app)
        decision = pending(app)
        options = [
            radio for radio in app.radio if radio.label == CHOICE_LABEL
        ][0].options

        # suggestion + every alternative + write your own + skip
        assert len(options) == len(decision.alternatives) + 3
        assert "recommended" in options[0]
        assert options[-1] == "Skip this step"
        assert "Tell us how you want this handled" in options[-2]

    def test_the_question_and_its_context_are_both_shown(self, app):
        start_sample_run(app)
        decision = pending(app)
        rendered = " ".join(element.value for element in app.markdown)

        assert decision.question in rendered
        assert decision.context.split("\n")[0][:60] in rendered

    def test_accepting_the_suggestion_moves_to_the_next_decision(self, app):
        start_sample_run(app)
        first = pending(app)

        [button for button in app.button if button.label == CONTINUE][0].click().run()

        assert not app.exception
        second = pending(app)
        assert second is None or second.id != first.id

    def test_choosing_an_alternative_is_recorded_as_the_users_choice(self, app):
        start_sample_run(app)
        decision = pending(app)
        assert decision.alternatives, "this decision should offer alternatives"

        app.radio(key=f"choice_{decision.id}").set_value(1).run()
        [button for button in app.button if button.label == CONTINUE][0].click().run()

        state = app.session_state["state"]
        recorded_decision, answer = state.answers[-1]
        assert recorded_decision.id == decision.id
        assert answer.choice.value == "alternative"
        assert not answer.automatic

    def test_a_custom_instruction_is_saved_to_the_business_memory(self, app):
        start_sample_run(app)
        decision = pending(app)
        custom_index = len(decision.alternatives) + 1

        app.radio(key=f"choice_{decision.id}").set_value(custom_index).run()
        app.text_area(key=f"custom_{decision.id}").set_value(
            "Every row is one delivery to a building site."
        ).run()
        [button for button in app.button if button.label == CONTINUE][0].click().run()

        state = app.session_state["state"]
        statements = [fact.statement for fact in state.memory]
        assert "Every row is one delivery to a building site." in statements

    def test_an_empty_custom_instruction_is_refused(self, app):
        start_sample_run(app)
        decision = pending(app)
        custom_index = len(decision.alternatives) + 1

        app.radio(key=f"choice_{decision.id}").set_value(custom_index).run()
        [button for button in app.button if button.label == CONTINUE][0].click().run()

        assert pending(app).id == decision.id, "the decision must stay open"
        assert app.error, "the user should be told why nothing happened"

    def test_skipping_leaves_the_data_alone(self, app):
        start_sample_run(app)
        decision = pending(app)
        skip_index = len(decision.alternatives) + 2

        app.radio(key=f"choice_{decision.id}").set_value(skip_index).run()
        [button for button in app.button if button.label == CONTINUE][0].click().run()

        state = app.session_state["state"]
        assert state.answers[-1][1].is_skip


class TestCompletedRun:
    @pytest.fixture
    def finished(self, app) -> AppTest:
        start_sample_run(app)
        for _ in range(40):
            if pending(app) is None:
                break
            buttons = [button for button in app.button if button.label == CONTINUE]
            if not buttons:
                break
            buttons[0].click().run()
            assert not app.exception, app.exception
        return app

    def test_the_run_finishes_without_an_error(self, finished):
        assert pending(finished) is None
        assert not finished.exception
        assert not finished.session_state["state"].errors

    def test_every_decision_was_answered_by_the_user(self, finished):
        state = finished.session_state["state"]
        assert state.answers
        assert not any(answer.automatic for _, answer in state.answers)

    def test_all_seven_result_tabs_render(self, finished):
        assert len(finished.tabs) == 7

    def test_the_results_are_all_present(self, finished):
        state = finished.session_state["state"]
        assert state.charts
        assert state.kpis
        assert state.insights
        assert state.dashboards
        assert state.artefacts

    def test_the_downloads_are_offered(self, finished):
        # Download buttons are their own element type, not ordinary buttons.
        labels = [element.label for element in finished.get("download_button")]
        assert labels
        assert any("cleaned data" in label for label in labels)


class TestAutonomousMode:
    def test_the_whole_run_completes_without_a_single_question(self, app):
        app.radio[0].set_value("Run it all automatically").run()
        start_sample_run(app)

        assert pending(app) is None, "autonomous mode must not stop for anything"
        assert not app.exception

        state = app.session_state["state"]
        assert state.insights
        assert all(answer.automatic for _, answer in state.answers)
