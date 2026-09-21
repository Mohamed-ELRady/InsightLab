"""Project isolation, versioned memory, and the approval-gated learning loop."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import pandas as pd

from insightlab.agents.memory_agent import MemoryAgent
from insightlab.core.business_memory import BusinessMemory
from insightlab.core.config import Settings
from insightlab.core.project_memory import ProjectMemoryStore
from insightlab.core.evaluation import evaluate_run
from insightlab.core.reasoning import ReasoningEngine
from insightlab.core.state import Chart, Insight, Kpi, PipelineState, StageStatus


def store(tmp_path):
    return ProjectMemoryStore(tmp_path / "memory")


def test_projects_are_stable_and_isolated(tmp_path):
    repository = store(tmp_path)
    sales = repository.ensure_project("Sales")
    same_sales = repository.ensure_project("sales")
    inventory = repository.ensure_project("Inventory")
    repository.add_memory(sales.id, "Exclude cancelled orders", category="exclusion")

    assert same_sales.id == sales.id
    assert inventory.id != sales.id
    assert [fact.statement for fact in repository.load_memory(sales.id)] == ["Exclude cancelled orders"]
    assert not repository.load_memory(inventory.id)


def test_only_active_memory_is_retrieved(tmp_path):
    repository = store(tmp_path)
    project = repository.ensure_project("Business")
    keep = repository.add_memory(project.id, "Keep this")
    drop = repository.add_memory(project.id, "Disable this")
    repository.update_memory(drop, status="archived")

    assert [fact.id for fact in repository.load_memory(project.id)] == [keep]
    assert {item.status for item in repository.list_memory(project.id)} == {"active", "archived"}


def test_exact_duplicate_does_not_create_two_memories(tmp_path):
    repository = store(tmp_path)
    project = repository.ensure_project("Business")
    first = repository.add_memory(project.id, "A confirmed rule")
    second = repository.add_memory(project.id, "a confirmed rule")

    assert first == second
    assert len(repository.list_memory(project.id)) == 1


def test_edits_are_versioned_and_reversible_in_the_audit_history(tmp_path):
    repository = store(tmp_path)
    project = repository.ensure_project("Business")
    item_id = repository.add_memory(project.id, "Old threshold", category="target")
    repository.update_memory(item_id, statement="New threshold", reason="Owner corrected it")
    repository.update_memory(item_id, status="archived", reason="No longer applies")

    item = next(item for item in repository.list_memory(project.id) if item.id == item_id)
    history = repository.history(item_id)
    assert item.statement == "New threshold"
    assert item.version == 3
    assert [entry["version"] for entry in history] == [3, 2, 1]
    assert history[0]["reason"] == "No longer applies"
    assert history[-1]["snapshot"]["statement"] == "Old threshold"

    assert repository.restore_version(item_id, 1)
    restored = next(item for item in repository.list_memory(project.id) if item.id == item_id)
    assert restored.statement == "Old threshold"
    assert restored.status == "active"
    assert restored.version == 4
    assert repository.history(item_id)[0]["change_kind"] == "restored"


def test_pipeline_memory_changes_sync_without_losing_ids(tmp_path):
    repository = store(tmp_path)
    project = repository.ensure_project("Business")
    first_id = repository.add_memory(project.id, "Threshold is 100", category="target")
    second_id = repository.add_memory(project.id, "Temporary note")
    memory = repository.load_memory(project.id)
    assert memory.replace(first_id, "Threshold is 200")
    assert memory.forget(second_id)
    created = memory.remember("A new rule", category="definition")

    repository.sync_memory(project.id, memory)

    active = repository.load_memory(project.id)
    assert {fact.statement for fact in active} == {"Threshold is 200", "A new rule"}
    assert next(fact for fact in active if fact.statement == "Threshold is 200").id == first_id
    assert created.id in {fact.id for fact in active}
    archived = next(item for item in repository.list_memory(project.id) if item.id == second_id)
    assert archived.status == "archived"


def test_incorrect_feedback_creates_pending_lesson_not_active_memory(tmp_path):
    repository = store(tmp_path)
    project = repository.ensure_project("Business")
    _, candidate_id = repository.record_feedback(
        run_id="run-1", project_id=project.id, item_type="insight", item_key="one",
        rating="incorrect", correction="Refunded orders do not count as revenue",
        reason="They are fully reversed", category="exclusion",
    )

    assert candidate_id
    assert not repository.load_memory(project.id)
    candidates = repository.list_candidates(project.id, "pending")
    assert [candidate.statement for candidate in candidates] == ["Refunded orders do not count as revenue"]
    assert repository.metrics(project.id)["pending_lessons"] == 1


def test_approved_lesson_becomes_active_memory(tmp_path):
    repository = store(tmp_path)
    project = repository.ensure_project("Business")
    _, candidate_id = repository.record_feedback(
        run_id="run-1", project_id=project.id, item_type="insight", item_key="one",
        rating="incorrect", correction="Original correction",
    )
    memory_id = repository.review_candidate(
        candidate_id, approve=True, edited_statement="Approved reusable rule"
    )

    assert memory_id
    memory = repository.load_memory(project.id)
    assert [fact.statement for fact in memory] == ["Approved reusable rule"]
    assert memory.facts()[0].source_type == "user_correction"
    assert not repository.list_candidates(project.id, "pending")
    assert repository.list_candidates(project.id, "approved")[0].statement == "Approved reusable rule"


def test_approved_run_correction_is_scoped_to_its_detected_domain(tmp_path):
    repository = store(tmp_path)
    project = repository.ensure_project("Mixed data")
    _, candidate_id = repository.record_feedback(
        run_id="quake-run", project_id=project.id, item_type="run",
        item_key="overall", rating="incorrect",
        correction="Magnitude 6 and above is severe for this workflow",
        context={"domain_family": "earth_science", "review_action": "revise"},
    )

    repository.review_candidate(candidate_id, approve=True)

    fact = repository.load_memory(project.id).facts()[0]
    assert fact.source_topic.startswith("domain:earth_science|")


def test_rejected_lesson_never_reaches_memory(tmp_path):
    repository = store(tmp_path)
    project = repository.ensure_project("Business")
    _, candidate_id = repository.record_feedback(
        run_id="run-1", project_id=project.id, item_type="insight", item_key="one",
        rating="incorrect", correction="Do not learn this",
    )
    repository.review_candidate(candidate_id, approve=False)

    assert not repository.load_memory(project.id)
    assert repository.list_candidates(project.id, "rejected")[0].statement == "Do not learn this"


def test_changing_feedback_replaces_a_pending_lesson(tmp_path):
    repository = store(tmp_path)
    project = repository.ensure_project("Business")
    first_feedback, first_candidate = repository.record_feedback(
        run_id="run-1", project_id=project.id, item_type="insight", item_key="one",
        rating="incorrect", correction="First correction",
    )
    second_feedback, second_candidate = repository.record_feedback(
        run_id="run-1", project_id=project.id, item_type="insight", item_key="one",
        rating="incorrect", correction="Better correction",
    )

    assert first_feedback == second_feedback
    assert first_candidate != second_candidate
    pending = repository.list_candidates(project.id, "pending")
    assert [candidate.statement for candidate in pending] == ["Better correction"]


def test_positive_and_negative_feedback_do_not_create_lessons(tmp_path):
    repository = store(tmp_path)
    project = repository.ensure_project("Business")
    repository.record_feedback(
        run_id="run-1", project_id=project.id, item_type="insight", item_key="one", rating="useful"
    )
    repository.record_feedback(
        run_id="run-1", project_id=project.id, item_type="insight", item_key="two", rating="not_useful"
    )

    assert not repository.list_candidates(project.id)
    assert repository.metrics(project.id)["useful"] == 1


def test_feedback_builds_an_explainable_project_scoped_adaptive_profile(tmp_path):
    repository = store(tmp_path)
    first = repository.ensure_project("First")
    second = repository.ensure_project("Second")
    repository.record_feedback(
        run_id="run-1", project_id=first.id, item_type="insight", item_key="one",
        rating="useful", context={"axis": "sales", "confidence": "high", "secret": "ignored"},
    )
    repository.record_feedback(
        run_id="run-1", project_id=first.id, item_type="chart", item_key="two",
        rating="not_useful", context={"axis": "operations", "chart_kind": "bar"},
    )
    repository.record_feedback(
        run_id="run-1", project_id=first.id, item_type="kpi", item_key="three",
        rating="useful", context={"kpi": "Total revenue"},
    )

    profile = repository.adaptive_profile(first.id)
    assert profile["axes"] == {"sales": 2.0, "operations": -1.0}
    assert profile["kpis"] == {"total revenue": 2.0}
    assert profile["chart_kinds"] == {"bar": -1.0}
    assert profile["feedback_count"] == 3
    assert repository.adaptive_profile(second.id)["feedback_count"] == 0
    with repository._connect() as db:
        context = json.loads(db.execute(
            "SELECT context_json FROM feedback WHERE item_key='one'"
        ).fetchone()[0])
    assert "secret" not in context


def test_quality_evaluation_stores_only_structural_sanitized_cases(tmp_path):
    repository = store(tmp_path)
    project = repository.ensure_project("Business")
    state = PipelineState(workspace=repository.root)
    state.project_id = project.id
    state.charts = [Chart(id="chart-1", title="Sensitive chart title", kind="bar", description="private")]
    state.kpis = [Kpi(name="Revenue", value=10, display_value="10", formula="private formula")]
    state.insights = [Insight(
        title="Sensitive conclusion", result="Sensitive result", evidence="Sensitive evidence",
        interpretation="Sensitive meaning", action="Sensitive action", chart_id="chart-1",
    )]
    state.stage_status = {key: StageStatus.DONE for key in state.stage_status}
    evaluation = evaluate_run(state)
    repository.record_evaluation(
        state.run_id, project.id, score=evaluation.score,
        metrics=evaluation.metrics, cases=evaluation.cases,
    )

    assert evaluation.score == 100.0
    saved = repository.evaluations(project.id)[0]
    encoded = json.dumps(saved["cases"])
    assert "Sensitive" not in encoded
    assert saved["cases"][0] == {
        "axis": "general", "confidence": "medium", "grounded": True,
        "has_action": True, "has_chart": True, "has_evidence": True, "position": 0,
    }


def test_policy_requires_validation_can_activate_and_roll_back(tmp_path):
    repository = store(tmp_path)
    project = repository.ensure_project("Business")
    original = repository.active_policy(project.id)
    candidate = repository.create_policy_candidate(
        project.id, "Prefer learned axes",
        {**original.rules, "axis_weight": 2.0},
    )

    assert not repository.activate_policy(candidate.id)
    shadow = repository.validate_policy(candidate.id)
    assert shadow["guardrails_passed"] is True
    assert shadow["mode"] == "offline_shadow"
    assert shadow["uses_raw_data"] is False
    assert repository.activate_policy(candidate.id)
    assert repository.active_policy(project.id).version == 2
    assert repository.rollback_policy(project.id)
    assert repository.active_policy(project.id).version == 1


def test_shadow_gate_rejects_policy_that_scores_historical_preferences_worse(tmp_path):
    repository = store(tmp_path)
    project = repository.ensure_project("Business")
    repository.record_feedback(
        run_id="one", project_id=project.id, item_type="insight", item_key="good",
        rating="useful", context={"axis": "sales"},
    )
    repository.record_feedback(
        run_id="one", project_id=project.id, item_type="chart", item_key="bad",
        rating="not_useful", context={"chart_kind": "pie"},
    )
    candidate = repository.create_policy_candidate(project.id, "Worse", {
        "axis_weight": 0.0, "kpi_weight": 1.0,
        "chart_weight": 3.0, "confidence_weight": 1.0,
    })

    result = repository.validate_policy(candidate.id)
    assert result["quality_gate_passed"] is False
    assert result["alignment_delta"] < 0
    assert not repository.activate_policy(candidate.id)
    assert next(item for item in repository.list_policies(project.id) if item.id == candidate.id).status == "rejected"


def test_runs_are_attached_to_their_project(tmp_path):
    repository = store(tmp_path)
    project = repository.ensure_project("Business")
    repository.record_run(
        "run-1", project.id, "sales.csv", datetime.now(timezone.utc), completed=False
    )
    repository.record_run(
        "run-1", project.id, "sales.csv", datetime.now(timezone.utc), completed=True
    )
    with repository._connect() as db:
        row = db.execute("SELECT * FROM analysis_runs WHERE run_id='run-1'").fetchone()
    assert row["project_id"] == project.id
    assert row["completed_at"]


def test_memory_column_filter_is_deterministic_not_semantic_guesswork(tmp_path):
    repository = store(tmp_path)
    project = repository.ensure_project("Business")
    repository.add_memory(project.id, "Revenue rule", scope="dataset", columns=["revenue"])
    repository.add_memory(project.id, "Customer rule", scope="dataset", columns=["customer_id"])
    repository.add_memory(project.id, "Global project context")

    assert {fact.statement for fact in repository.load_memory(project.id)} == {"Global project context"}
    retrieved = repository.load_memory(project.id, columns=["Revenue"])
    assert {fact.statement for fact in retrieved} == {"Revenue rule", "Global project context"}


def test_project_preferences_are_isolated_and_updated(tmp_path):
    repository = store(tmp_path)
    first = repository.ensure_project("First")
    second = repository.ensure_project("Second")
    repository.set_preferences(first.id, {"explanation_detail": "concise", "report_audience": "team"})
    repository.set_preferences(first.id, {"explanation_detail": "detailed"})

    assert repository.preferences(first.id) == {
        "explanation_detail": "detailed", "report_audience": "team"
    }
    assert repository.preferences(second.id) == {}


def test_expired_memory_is_versioned_and_not_retrieved(tmp_path):
    repository = store(tmp_path)
    project = repository.ensure_project("Business")
    item_id = repository.add_memory(
        project.id, "A time-limited campaign rule", valid_to="2000-01-01"
    )

    assert not repository.load_memory(project.id)
    item = next(item for item in repository.list_memory(project.id) if item.id == item_id)
    assert item.status == "expired"
    assert item.version == 2
    assert repository.history(item_id)[0]["reason"] == "Validity period ended"


def test_recall_adds_only_matching_dataset_memory_without_losing_current_facts(tmp_path):
    root = tmp_path / "memory"
    repository = ProjectMemoryStore(root)
    project = repository.ensure_project("Business")
    repository.add_memory(project.id, "Project-wide context")
    repository.add_memory(project.id, "Revenue-specific rule", scope="dataset", columns=["revenue"])
    repository.add_memory(project.id, "Customer-only rule", scope="dataset", columns=["customer_id"])

    state = PipelineState(workspace=root)
    state.project_id = project.id
    state.project_name = project.name
    state.memory = repository.load_memory(project.id)
    state.remember("Learned earlier in this same run", stage="understand")
    state.frame = pd.DataFrame({"revenue": [10, 20]})
    agent = MemoryAgent(ReasoningEngine(Settings(workspace=root, offline=True)))
    list(agent.run(state))

    statements = {fact.statement for fact in state.memory}
    assert statements == {
        "Project-wide context", "Revenue-specific rule", "Learned earlier in this same run"
    }


def test_legacy_run_memory_can_be_imported_once_without_duplicates(tmp_path):
    repository = store(tmp_path)
    project = repository.ensure_project("Migrated")
    legacy = BusinessMemory()
    legacy.remember("Legacy rule", category="definition")
    path = tmp_path / "business_memory.json"
    path.write_text(json.dumps(legacy.to_list()), encoding="utf-8")

    assert repository.import_legacy_memory(project.id, path) == 1
    assert repository.import_legacy_memory(project.id, path) == 0
    assert [fact.statement for fact in repository.load_memory(project.id)] == ["Legacy rule"]
