"""The shared core: decisions, business memory, activity log and state."""

from __future__ import annotations

import pytest

from insightlab.core.activity_log import ActivityLog, EventKind
from insightlab.core.business_memory import BusinessMemory
from insightlab.core.decision import Answer, Choice, Decision, Option
from insightlab.core.state import PipelineState, RunMode, StageStatus


def build_decision() -> Decision:
    return Decision(
        stage="clean",
        topic="Duplicate rows",
        question="What should we do with them?",
        context="24 rows repeat.",
        suggestion=Option("Delete them", "They came from a double export.", {"action": "delete"}),
        alternatives=[
            Option("Keep them", "Two identical sales can happen.", {"action": "keep"}),
            Option("Merge them", "Row count drops, totals stay.", {"action": "merge"}),
        ],
    )


class TestDecisions:
    def test_accepting_the_suggestion_carries_its_payload(self):
        decision = build_decision()
        answer = Answer.accept(decision)

        assert answer.choice is Choice.SUGGESTION
        assert answer.payload == {"action": "delete"}
        assert not answer.is_skip

    def test_choosing_an_alternative_carries_that_payload(self):
        decision = build_decision()
        answer = Answer.alternative(decision, 1)

        assert answer.payload == {"action": "merge"}
        assert "Merge them" in answer.describe()

    def test_a_custom_answer_has_no_payload_of_its_own(self):
        decision = build_decision()
        answer = Answer.custom(decision, "  Two identical invoices are real.  ")

        assert answer.is_custom
        assert answer.text == "Two identical invoices are real."
        assert answer.payload == {}

    def test_skipping_selects_nothing(self):
        decision = build_decision()
        answer = Answer.skip(decision)

        assert answer.is_skip
        assert answer.selected is None
        assert answer.describe() == "Skipped."

    def test_an_out_of_range_alternative_selects_nothing(self):
        decision = build_decision()
        answer = Answer.alternative(decision, 99)

        assert answer.selected is None
        assert answer.payload == {}

    def test_automatic_answers_are_marked_as_such(self):
        decision = build_decision()
        assert Answer.accept(decision, automatic=True).automatic
        assert not Answer.accept(decision).automatic


class TestBusinessMemory:
    def test_a_fact_is_recorded_once(self):
        memory = BusinessMemory()
        first = memory.remember("VIP customers spend over 5000", category="definition")
        second = memory.remember("vip customers SPEND over 5000")

        assert first is not None
        assert second is None, "the same fact must not be stored twice"
        assert len(memory) == 1

    def test_blank_statements_are_ignored(self):
        memory = BusinessMemory()
        assert memory.remember("   ") is None
        assert len(memory) == 0

    def test_an_unknown_category_falls_back_to_context(self):
        memory = BusinessMemory()
        fact = memory.remember("Something", category="nonsense")
        assert fact.category == "context"

    def test_the_prompt_block_is_empty_when_nothing_is_known(self):
        assert BusinessMemory().as_prompt_block() == ""

    def test_memory_survives_a_round_trip(self):
        memory = BusinessMemory()
        memory.remember("Peak season starts in November", category="seasonality")
        memory.remember("Ignore cancelled invoices", category="exclusion")

        restored = BusinessMemory.from_list(memory.to_list())

        assert len(restored) == 2
        assert [fact.category for fact in restored] == ["seasonality", "exclusion"]

    def test_forgetting_removes_only_the_named_fact(self):
        memory = BusinessMemory()
        keep = memory.remember("Keep this")
        drop = memory.remember("Drop this")

        assert memory.forget(drop.id)
        assert not memory.forget("nonexistent")
        assert [fact.id for fact in memory] == [keep.id]


class TestActivityLog:
    def test_events_keep_their_order_and_details(self):
        log = ActivityLog()
        log.record(EventKind.NOTE, "clean", "Looked at duplicates", count=24)
        log.record(EventKind.DATA_CHANGED, "clean", "Removed 24 rows")

        assert len(log) == 2
        assert log.events()[0].details == {"count": 24}
        assert len(log.data_operations()) == 1

    def test_the_log_survives_a_round_trip(self):
        log = ActivityLog()
        log.record(EventKind.WARNING, "load", "Odd delimiter")
        restored = ActivityLog.from_list(log.to_list())

        assert restored.events()[0].kind is EventKind.WARNING


class TestPipelineState:
    def test_a_run_is_only_complete_once_every_stage_is_settled(self):
        state = PipelineState(mode=RunMode.INTERACTIVE)
        assert not state.is_complete

        for stage in state.stage_status:
            state.finish_stage(stage)
        assert state.is_complete

    def test_a_failed_stage_still_counts_as_settled(self):
        state = PipelineState()
        for stage in state.stage_status:
            state.finish_stage(stage)
        state.fail_stage("clean", "something broke")

        assert state.is_complete
        assert state.stage_status["clean"] is StageStatus.FAILED
        assert state.errors == ["clean: something broke"]

    def test_recording_a_fact_writes_it_to_the_log(self):
        state = PipelineState()
        state.remember("Peak season is November", category="seasonality", stage="clean")

        assert len(state.memory) == 1
        assert len(state.log.of_kind(EventKind.FACT_RECORDED)) == 1

    def test_changing_the_frame_logs_the_row_counts(self, sample_frame):
        state = PipelineState()
        state.set_frame(sample_frame, "load", "Loaded the file")
        state.set_frame(sample_frame.head(10), "clean", "Removed rows")

        change = state.log.data_operations()[-1]
        assert change.details["rows_before"] == len(sample_frame)
        assert change.details["rows_after"] == 10

    def test_a_profile_is_reused_until_the_frame_changes(self, sample_frame):
        from insightlab.analysis.profiling import ensure_profile

        state = PipelineState()
        state.set_frame(sample_frame, "load", "Loaded")
        first = ensure_profile(state)

        assert ensure_profile(state) is first

        state.set_frame(sample_frame.head(10), "clean", "Removed rows")
        second = ensure_profile(state)
        assert second is not first
        assert second.row_count == 10

    def test_the_summary_is_json_safe(self, sample_frame):
        import json

        state = PipelineState()
        state.set_frame(sample_frame, "load", "Loaded")
        state.remember("A fact")

        json.dumps(state.summary_dict(), default=str)  # must not raise
