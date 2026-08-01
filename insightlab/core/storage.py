"""Persistence for a run's artefacts.

Each run gets its own directory under the workspace so that two analyses never
overwrite each other:

    data/runs/<run_id>/
        summary.json        full state snapshot, including the activity log
        cleaned_data.csv    the cleaned copy of the user's data
        business_memory.json
        charts/<id>.json    plotly figure definitions
        reports/            generated PDF / PPTX / DOCX
"""

from __future__ import annotations

import json
from pathlib import Path

from .business_memory import BusinessMemory
from .config import get_settings
from .state import PipelineState


class RunWorkspace:
    """Filesystem layout for one run."""

    def __init__(self, run_id: str, root: Path | None = None) -> None:
        base = root or get_settings().workspace
        self.root = Path(base) / run_id
        self.charts_dir = self.root / "charts"
        self.reports_dir = self.root / "reports"

    def prepare(self) -> "RunWorkspace":
        for directory in (self.root, self.charts_dir, self.reports_dir):
            directory.mkdir(parents=True, exist_ok=True)
        return self

    @property
    def summary_path(self) -> Path:
        return self.root / "summary.json"

    @property
    def cleaned_data_path(self) -> Path:
        return self.root / "cleaned_data.csv"

    @property
    def memory_path(self) -> Path:
        return self.root / "business_memory.json"

    def chart_path(self, chart_id: str) -> Path:
        return self.charts_dir / f"{chart_id}.json"

    def report_path(self, filename: str) -> Path:
        return self.reports_dir / filename


def save_run(state: PipelineState) -> RunWorkspace:
    """Write everything the run produced to disk and return the workspace."""
    workspace = RunWorkspace(state.run_id, state.workspace).prepare()

    workspace.summary_path.write_text(
        json.dumps(state.summary_dict(), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    workspace.memory_path.write_text(
        json.dumps(state.memory.to_list(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if state.frame is not None:
        state.frame.to_csv(workspace.cleaned_data_path, index=False)
        state.add_artefact("Cleaned data", workspace.cleaned_data_path, stage="report")

    for chart in state.charts:
        if chart.figure_json:
            workspace.chart_path(chart.id).write_text(chart.figure_json, encoding="utf-8")

    return workspace


def load_business_memory(path: Path) -> BusinessMemory:
    """Restore a memory file saved by an earlier run.

    This is what makes facts reusable across analyses: point a new run at a
    previous run's ``business_memory.json`` and every agent starts already
    knowing the rules of the business.
    """
    if not path.exists():
        return BusinessMemory()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return BusinessMemory()
    if not isinstance(raw, list):
        return BusinessMemory()
    return BusinessMemory.from_list(raw)


def list_runs(root: Path | None = None) -> list[Path]:
    """Every run directory in the workspace, newest first."""
    base = Path(root or get_settings().workspace)
    if not base.exists():
        return []
    runs = [item for item in base.iterdir() if item.is_dir()]
    return sorted(runs, key=lambda path: path.stat().st_mtime, reverse=True)
