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
from insightlab.core.state import PipelineState, RunMode, StageStatus
from insightlab.core.storage import load_business_memory

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

    state = PipelineState(mode=RunMode.AUTONOMOUS)
    state.source_path = arguments.file
    if arguments.memory is not None:
        state.memory = load_business_memory(arguments.memory)
        if not arguments.quiet:
            print(f"Carried over {len(state.memory)} facts from a previous run.\n")

    Supervisor(state).run_to_completion()

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
