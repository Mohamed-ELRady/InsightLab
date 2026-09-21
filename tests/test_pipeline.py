"""The agents and the supervisor driving them."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from insightlab.agents.base import Agent
from insightlab.agents.data_loader import (
    DataLoaderAgent,
    header_looks_wrong,
    sniff_text_format,
)
from insightlab.agents.data_understanding import DataUnderstandingAgent
from insightlab.agents.supervisor import build_default_agents
from insightlab.agents.supervisor import Supervisor
from insightlab.core.decision import Answer, Choice, Option
from insightlab.core.reasoning import ReasoningEngine, parse_json
from insightlab.core.language import ARABIC
from insightlab.core.state import PipelineState, RunMode, StageStatus


def run_autonomously(path: Path) -> PipelineState:
    state = PipelineState(mode=RunMode.AUTONOMOUS)
    state.source_path = path
    return Supervisor(state).run_to_completion()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


class TestLoading:
    def test_a_comma_file_is_recognised(self, sample_path):
        encoding, delimiter = sniff_text_format(sample_path)
        assert delimiter == ","
        assert "utf-8" in encoding

    def test_a_semicolon_file_is_recognised(self, tmp_path):
        path = tmp_path / "export.csv"
        path.write_text("a;b;c\n1;2;3\n4;5;6\n", encoding="utf-8")
        _, delimiter = sniff_text_format(path)
        assert delimiter == ";"

    def test_a_placeholder_header_is_spotted(self):
        frame = pd.DataFrame(
            [[1, 2, 3]], columns=["Sales report 2024", "Unnamed: 1", "Unnamed: 2"]
        )
        assert header_looks_wrong(frame)

    def test_a_real_header_is_left_alone(self, sample_frame):
        assert not header_looks_wrong(sample_frame)

    def test_duplicate_column_names_are_made_unique(self):
        frame = pd.DataFrame([[1, 2, 3]], columns=["total", "total", "total"])
        tidied = DataLoaderAgent._tidy(frame)
        assert list(tidied.columns) == ["total", "total_1", "total_2"]

    def test_blank_column_names_are_given_one(self):
        frame = pd.DataFrame([[1, 2]], columns=["  ", "amount"])
        tidied = DataLoaderAgent._tidy(frame)
        assert tidied.columns[0] == "column_1"

    def test_an_unsupported_file_fails_the_stage_not_the_run(self, tmp_path):
        path = tmp_path / "photo.png"
        path.write_bytes(b"not data")
        state = run_autonomously(path)

        assert state.stage_status["load"] is StageStatus.FAILED
        assert state.errors
        assert "CSV or Excel" in state.errors[0]

    def test_a_missing_file_fails_cleanly(self, tmp_path):
        state = run_autonomously(tmp_path / "nothing.csv")
        assert state.stage_status["load"] is StageStatus.FAILED

    def test_an_excel_workbook_loads(self, tmp_path, sample_frame):
        path = tmp_path / "book.xlsx"
        sample_frame.head(200).to_excel(path, index=False)
        state = run_autonomously(path)

        assert state.stage_status["load"] is StageStatus.DONE
        # raw_frame, not frame: the later stages clean rows out of the latter.
        assert len(state.raw_frame) == 200
        assert state.source_format == "Excel workbook"


# ---------------------------------------------------------------------------
# The supervisor
# ---------------------------------------------------------------------------


class OneQuestionAgent(Agent):
    """Minimal agent used to test the driving loop in isolation."""

    stage = "load"
    key = "test"

    def run(self, state):
        state.begin_stage(self.stage)
        decision = self.decide(
            topic="A question",
            question="Which way?",
            context="Context.",
            suggestion=Option("Left", "Because left.", {"way": "left"}),
            alternatives=[Option("Right", "Because right.", {"way": "right"})],
        )
        answer = yield decision
        state.load_notes.append(answer.payload.get("way", "skipped"))
        state.finish_stage(self.stage)


class TestSupervisor:
    def test_interactive_mode_stops_and_waits(self):
        state = PipelineState(mode=RunMode.INTERACTIVE)
        supervisor = Supervisor(state, agents=[OneQuestionAgent(ReasoningEngine())])

        decision = supervisor.start()
        assert decision is not None
        assert supervisor.pending is decision
        assert not supervisor.finished

        assert supervisor.resolve(Answer.alternative(decision, 0)) is None
        assert supervisor.finished
        assert state.load_notes == ["right"]

    def test_autonomous_mode_answers_for_itself(self):
        state = PipelineState(mode=RunMode.AUTONOMOUS)
        supervisor = Supervisor(state, agents=[OneQuestionAgent(ReasoningEngine())])

        assert supervisor.start() is None
        assert supervisor.finished
        assert state.load_notes == ["left"], "the suggestion should have been taken"

    def test_progress_reports_real_substage_work_and_finishes_at_one_hundred(self):
        class ProgressAgent(Agent):
            stage = "load"
            key = "progress"

            def run(self, state):
                state.begin_stage(self.stage)
                state.report_progress(self.stage, 0.25, "progress.load.read")
                state.report_progress(self.stage, 0.75, "progress.load.tidy")
                state.finish_stage(self.stage)
                return
                yield  # pragma: no cover - make this a generator

        state = PipelineState(mode=RunMode.AUTONOMOUS)
        supervisor = Supervisor(
            state, agents=[ProgressAgent(ReasoningEngine())]
        )
        updates = []
        supervisor.set_progress_callback(
            lambda value, stage, detail: updates.append((value, stage, detail))
        )

        supervisor.run_to_completion()

        values = [value for value, _, _ in updates]
        details = [detail for _, _, detail in updates]
        assert values == sorted(values)
        assert values[-1] == 1.0
        assert "progress.load.read" in details
        assert "progress.load.tidy" in details

    def test_automatic_answers_are_recorded_as_automatic(self):
        state = PipelineState(mode=RunMode.AUTONOMOUS)
        Supervisor(state, agents=[OneQuestionAgent(ReasoningEngine())]).run_to_completion()

        assert len(state.answers) == 1
        decision, answer = state.answers[0]
        assert answer.automatic
        assert answer.choice is Choice.SUGGESTION

    def test_answering_when_nothing_is_pending_is_an_error(self):
        state = PipelineState(mode=RunMode.INTERACTIVE)
        supervisor = Supervisor(state, agents=[OneQuestionAgent(ReasoningEngine())])
        decision = supervisor.start()
        supervisor.resolve(Answer.accept(decision))

        with pytest.raises(RuntimeError):
            supervisor.resolve(Answer.accept(decision))

    def test_a_stale_answer_is_ignored(self):
        state = PipelineState(mode=RunMode.INTERACTIVE)
        supervisor = Supervisor(state, agents=[OneQuestionAgent(ReasoningEngine())])
        decision = supervisor.start()

        stale = Answer.accept(decision)
        stale.decision_id = "not-the-current-one"

        assert supervisor.resolve(stale) is decision, "the decision must stay open"

    def test_a_broken_agent_fails_only_its_own_stage(self):
        class BrokenAgent(Agent):
            stage = "clean"
            key = "broken"

            def run(self, state):
                state.begin_stage(self.stage)
                raise ValueError("something went wrong")
                yield  # pragma: no cover

        state = PipelineState(mode=RunMode.AUTONOMOUS)
        supervisor = Supervisor(
            state,
            agents=[BrokenAgent(ReasoningEngine()), OneQuestionAgent(ReasoningEngine())],
        )
        supervisor.run_to_completion()

        assert state.stage_status["clean"] is StageStatus.FAILED
        assert state.stage_status["load"] is StageStatus.DONE, "the run must continue"

    def test_every_decision_reaches_the_log(self):
        state = PipelineState(mode=RunMode.AUTONOMOUS)
        Supervisor(state, agents=[OneQuestionAgent(ReasoningEngine())]).run_to_completion()

        from insightlab.core.activity_log import EventKind

        assert len(state.log.of_kind(EventKind.DECISION_RAISED)) == 1
        assert len(state.log.of_kind(EventKind.DECISION_ANSWERED)) == 1

    def test_understanding_runs_before_memory_recall(self):
        stages = [
            agent.stage for agent in build_default_agents(ReasoningEngine())[:3]
        ]
        assert stages == ["load", "understand", "recall"]


class TestInitialUnderstanding:
    @staticmethod
    def _agent():
        return DataUnderstandingAgent(ReasoningEngine())

    def test_clear_data_moves_on_without_an_unnecessary_question(self):
        state = PipelineState(mode=RunMode.INTERACTIVE)
        state.source_name = "orders.csv"
        state.frame = pd.DataFrame(
            {
                "order_id": ["A1", "A2", "A3"],
                "sale_date": ["2025-01-01", "2025-01-02", "2025-01-03"],
                "revenue": [100.0, 120.0, 90.0],
                "product_category": ["A", "B", "A"],
            }
        )

        flow = self._agent().run(state)
        with pytest.raises(StopIteration):
            next(flow)

        assert state.stage_status["understand"] is StageStatus.DONE
        assert state.understanding.subject == "sales transactions"
        assert state.understanding.confidence == "high"
        assert state.understanding.questions == []

    def test_ambiguous_data_asks_what_one_row_means(self):
        state = PipelineState(mode=RunMode.INTERACTIVE)
        state.source_name = "export.csv"
        state.frame = pd.DataFrame(
            {"alpha": [1, 2, 3], "beta": [4, 5, 6]}
        )

        decision = next(self._agent().run(state))

        assert decision.topic == "What the data represents"
        assert "what does one row" in decision.question.casefold()
        assert state.understanding.confidence == "low"

    def test_understanding_is_saved_in_the_run_summary(self):
        state = PipelineState(mode=RunMode.INTERACTIVE)
        state.source_name = "orders.csv"
        state.frame = pd.DataFrame(
            {
                "order_id": ["A1", "A2"],
                "sale_amount": [10.0, 20.0],
            }
        )
        flow = self._agent().run(state)
        with pytest.raises(StopIteration):
            next(flow)

        saved = state.summary_dict()["understanding"]
        assert saved["row_represents"]
        assert saved["summary"]

    def test_a_material_model_question_is_asked_and_its_answer_is_remembered(self):
        class QuestionReasoning:
            available = True

            @staticmethod
            def ask_json(*args, **kwargs):
                return {
                    "business_domain": "sales",
                    "subject": "sales transactions",
                    "row_represents": "one order",
                    "confidence": "high",
                    "summary": "This file contains one order per row.",
                    "important_columns": ["order_id", "sale_amount"],
                    "questions": [
                        {
                            "kind": "currency",
                            "question": "Which currency is sale_amount recorded in?",
                            "reason": "The report must label money correctly.",
                            "column": "sale_amount",
                            "suggested_answer": "",
                        }
                    ],
                }

        state = PipelineState(mode=RunMode.INTERACTIVE)
        state.source_name = "orders.csv"
        state.frame = pd.DataFrame(
            {"order_id": ["A1", "A2"], "sale_amount": [10.0, 20.0]}
        )
        flow = DataUnderstandingAgent(QuestionReasoning()).run(state)

        decision = next(flow)
        assert decision.question == "Which currency is sale_amount recorded in?"

        with pytest.raises(StopIteration):
            flow.send(Answer.custom(decision, "Egyptian pounds"))

        assert state.understanding.questions[0].answer == "Egyptian pounds"
        assert any("Egyptian pounds" in fact.statement for fact in state.memory)

    def test_the_offline_first_reading_is_native_arabic(self):
        state = PipelineState(mode=RunMode.INTERACTIVE, language=ARABIC)
        state.source_name = "orders.csv"
        state.frame = pd.DataFrame(
            {
                "order_id": ["A1", "A2"],
                "sale_date": ["2025-01-01", "2025-01-02"],
                "revenue": [10.0, 20.0],
            }
        )
        flow = self._agent().run(state)
        with pytest.raises(StopIteration):
            next(flow)

        assert state.understanding.business_domain == "معاملات المبيعات"
        assert "عملية بيع" in state.understanding.summary
        assert "a sale" not in state.understanding.summary


# ---------------------------------------------------------------------------
# The whole pipeline
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def completed_run(request):
    """One full autonomous run, shared by the tests that only read it."""
    root = Path(__file__).resolve().parents[1]
    sample = root / "data" / "samples" / "retail_sales.csv"
    if not sample.exists():
        pytest.skip("Run scripts/make_sample_data.py to generate the sample data.")

    import os
    import tempfile

    with tempfile.TemporaryDirectory() as workspace:
        os.environ["INSIGHTLAB_OFFLINE"] = "1"
        os.environ["INSIGHTLAB_WORKSPACE"] = workspace
        from insightlab.core.config import reset_settings_cache

        reset_settings_cache()
        yield run_autonomously(sample)
        reset_settings_cache()


class TestFullRun:
    def test_every_stage_completes(self, completed_run):
        assert not completed_run.errors
        for stage, status in completed_run.stage_status.items():
            assert status is StageStatus.DONE, f"{stage} ended as {status.value}"

    def test_it_produces_all_of_the_promised_deliverables(self, completed_run):
        assert completed_run.frame is not None and not completed_run.frame.empty
        assert completed_run.charts
        assert completed_run.kpis
        assert completed_run.insights
        assert completed_run.dashboards
        assert completed_run.log.data_operations()
        assert completed_run.memory

    def test_the_original_data_is_kept_untouched(self, completed_run):
        assert completed_run.raw_frame is not None
        assert len(completed_run.raw_frame) > len(completed_run.frame)

    def test_every_change_to_the_data_is_logged(self, completed_run):
        for event in completed_run.log.data_operations():
            assert event.message.strip()

    def test_every_conclusion_carries_its_five_parts(self, completed_run):
        for insight in completed_run.insights:
            assert insight.title
            assert insight.result
            assert insight.evidence
            assert insight.interpretation
            assert insight.confidence in {"high", "medium", "low"}
            assert insight.action

    def test_an_insight_pointing_at_a_chart_points_at_a_real_one(self, completed_run):
        for insight in completed_run.insights:
            if insight.chart_id:
                assert completed_run.chart(insight.chart_id) is not None

    def test_a_dashboard_only_references_things_that_exist(self, completed_run):
        for board in completed_run.dashboards:
            assert board.panels
            for panel in board.panels:
                if panel.kind == "kpi":
                    assert completed_run.kpi(panel.reference) is not None
                else:
                    assert completed_run.chart(panel.reference) is not None

    def test_the_documents_are_written(self, completed_run):
        names = set(completed_run.artefacts)
        assert "Cleaned data" in names
        assert "Business memory" in names
        assert "PDF document" in names

        for path in completed_run.artefacts.values():
            assert Path(path).exists()
            assert Path(path).stat().st_size > 0

    def test_the_saved_memory_can_start_the_next_run(self, completed_run):
        from insightlab.core.storage import load_business_memory

        path = Path(completed_run.artefacts["Business memory"])
        restored = load_business_memory(path)
        assert len(restored) == len(completed_run.memory)

    def test_the_summary_file_is_valid_json(self, completed_run):
        path = Path(completed_run.artefacts["Run summary"])
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["run_id"] == completed_run.run_id
        assert payload["insights"]
        assert payload["decisions"]


# ---------------------------------------------------------------------------
# Working without a model
# ---------------------------------------------------------------------------


class TestOfflineBehaviour:
    def test_the_engine_reports_that_it_cannot_reason(self):
        engine = ReasoningEngine()
        assert not engine.available
        assert engine.agent("k", _persona()) is None
        assert engine.ask("k", _persona(), "instruction", "output") is None

    def test_asking_for_json_offline_returns_nothing(self):
        engine = ReasoningEngine()
        assert engine.ask_json("k", _persona(), "instruction", "{}") is None

    @pytest.mark.parametrize(
        "reply",
        [
            '{"a": 1}',
            '```json\n{"a": 1}\n```',
            'Here is the answer:\n{"a": 1}\nHope that helps.',
        ],
    )
    def test_json_is_recovered_from_a_wrapped_reply(self, reply):
        assert parse_json(reply) == {"a": 1}

    def test_an_unparseable_reply_returns_nothing(self):
        assert parse_json("no json here at all") is None


def _persona():
    from insightlab.core.reasoning import AgentPersona

    return AgentPersona(role="r", goal="g", backstory="b")
