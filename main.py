"""Command-line entry point for an autonomous run.

    python main.py data/samples/retail_sales.csv

Interactive mode lives in the web interface, since it is a conversation:

    streamlit run insightlab/app/main.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from insightlab.agents.supervisor import Supervisor
from insightlab.core.config import get_settings
from insightlab.core.evaluation import evaluate_run
from insightlab.core.project_memory import ProjectMemoryStore
from insightlab.core.state import PipelineState, RunMode, StageStatus

MARKS = {
    StageStatus.DONE: "done",
    StageStatus.SKIPPED: "skipped",
    StageStatus.FAILED: "FAILED",
    StageStatus.PENDING: "not reached",
    StageStatus.RUNNING: "interrupted",
}


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="insightlab",
        description="Analyse a data file end to end and write up the findings.",
    )
    parser.add_argument("file", type=Path, help="CSV or Excel file to analyse")
    parser.add_argument(
        "--memory",
        type=Path,
        default=None,
        help="business_memory.json from an earlier run, to start from what it learned",
    )
    parser.add_argument(
        "--project",
        default="CLI analyses",
        help="durable memory project shared with later CLI or web analyses",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="only print the final summary"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(argv)

    if not arguments.file.exists():
        print(f"No such file: {arguments.file}", file=sys.stderr)
        return 2

    settings = get_settings()
    if not arguments.quiet:
        print(f"InsightLab - analysis model: {settings.describe_llm()}")
        print(f"Reading {arguments.file}\n")

    state = PipelineState(mode=RunMode.AUTONOMOUS, workspace=settings.workspace)
    state.source_path = arguments.file
    memory_store = ProjectMemoryStore(settings.workspace)
    project = memory_store.ensure_project(arguments.project)
    if arguments.memory is not None:
        imported = memory_store.import_legacy_memory(project.id, arguments.memory)
        if not arguments.quiet:
            print(f"Imported {imported} new fact(s) from a previous run.\n")

    state.project_id = project.id
    state.project_name = project.name
    state.memory = memory_store.load_memory(project.id)
    state.project_preferences = memory_store.preferences(project.id)
    state.learning_profile = memory_store.adaptive_profile(project.id)
    active_policy = memory_store.active_policy(project.id)
    state.improvement_policy = active_policy.rules
    state.improvement_policy_version = active_policy.version
    memory_store.record_run(
        state.run_id, project.id, arguments.file.name, state.started_at
    )

    Supervisor(state).run_to_completion()

    memory_store.sync_memory(project.id, state.memory)
    memory_store.record_run(
        state.run_id, project.id, state.source_name or arguments.file.name,
        state.started_at, completed=state.is_complete,
    )
    evaluation = evaluate_run(state)
    memory_store.record_evaluation(
        state.run_id, project.id, score=evaluation.score,
        metrics=evaluation.metrics, cases=evaluation.cases,
    )

    if not arguments.quiet:
        for stage, status in state.stage_status.items():
            print(f"  {stage:<12} {MARKS[status]}")
        print()

    print(f"{len(state.insights)} conclusions, {len(state.kpis)} headline figures.\n")
    for insight in state.insights[:5]:
        print(f"- {insight.title} [{insight.confidence}]")
        print(f"  {insight.result}")
        if insight.action:
            print(f"  Do this: {insight.action}")
        print()

    if state.artefacts:
        print("Saved:")
        for name, path in state.artefacts.items():
            print(f"  {name}: {path}")

    if state.errors:
        print("\nProblems during the run:", file=sys.stderr)
        for error in state.errors:
            print(f"  {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
