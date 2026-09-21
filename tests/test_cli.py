"""Command-line workflow coverage."""

from __future__ import annotations

import sqlite3

import main as cli
from insightlab.core.project_memory import ProjectMemoryStore
from insightlab.core.state import StageStatus


def test_cli_accepts_a_durable_project_name(tmp_path):
    source = tmp_path / "small.csv"
    source.write_text("value\n1\n", encoding="utf-8")

    arguments = cli.parse_arguments([str(source), "--project", "Client Alpha"])

    assert arguments.project == "Client Alpha"


def test_cli_records_the_run_and_evaluation_in_project_memory(
    tmp_path, monkeypatch
):
    source = tmp_path / "small.csv"
    source.write_text("value\n1\n", encoding="utf-8")

    class SettledSupervisor:
        def __init__(self, state):
            self.state = state

        def run_to_completion(self):
            self.state.source_name = source.name
            for stage in self.state.stage_status:
                self.state.stage_status[stage] = StageStatus.SKIPPED
            return self.state

    monkeypatch.setattr(cli, "Supervisor", SettledSupervisor)

    assert cli.main([str(source), "--project", "Client Alpha", "--quiet"]) == 0

    store = ProjectMemoryStore()
    project = store.ensure_project("client alpha")
    assert len(store.evaluations(project.id)) == 1

    with sqlite3.connect(store.path) as connection:
        run = connection.execute(
            "SELECT source_name, completed_at FROM analysis_runs WHERE project_id = ?",
            (project.id,),
        ).fetchone()
    assert run[0] == source.name
    assert run[1] is not None
