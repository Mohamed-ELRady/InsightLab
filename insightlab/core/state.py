"""Pipeline state: the single object every agent reads from and writes to.

The supervisor owns one :class:`PipelineState` per run. Agents never talk to
each other directly - one agent's output becomes the next agent's input by way
of this object.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd

from .activity_log import ActivityLog, EventKind
from .business_memory import BusinessMemory
from .decision import Answer, Decision


class RunMode(str, Enum):
    """Whether the user is consulted at each checkpoint."""

    INTERACTIVE = "interactive"
    AUTONOMOUS = "autonomous"


class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    SKIPPED = "skipped"
    FAILED = "failed"


class Role(str, Enum):
    """What a column means in business terms, not what dtype pandas gave it."""

    IDENTIFIER = "identifier"
    DATETIME = "datetime"
    MEASURE = "measure"
    CATEGORY = "category"
    BOOLEAN = "boolean"
    TEXT = "text"
    CONSTANT = "constant"
    UNKNOWN = "unknown"


#: Pipeline order. The supervisor walks these in sequence.
STAGES: tuple[tuple[str, str], ...] = (
    ("load", "Loading your data"),
    ("understand", "Understanding the data"),
    ("clean", "Cleaning the data"),
    ("features", "Building new measures"),
    ("explore", "Exploring the data"),
    ("kpis", "Summarising performance"),
    ("insights", "Drawing conclusions"),
    ("dashboard", "Building dashboards"),
    ("report", "Writing the report"),
)

STAGE_TITLES: dict[str, str] = dict(STAGES)


@dataclass
class ColumnProfile:
    """What we know about one column."""

    name: str
    dtype: str
    role: Role = Role.UNKNOWN
    missing_count: int = 0
    missing_rate: float = 0.0
    unique_count: int = 0
    sample_values: list[Any] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "dtype": self.dtype,
            "role": self.role.value,
            "missing_count": self.missing_count,
            "missing_rate": round(self.missing_rate, 4),
            "unique_count": self.unique_count,
            "sample_values": [str(value) for value in self.sample_values],
            "stats": self.stats,
            "note": self.note,
        }


@dataclass
class DatasetProfile:
    """Snapshot of the dataset as a whole."""

    row_count: int = 0
    column_count: int = 0
    columns: list[ColumnProfile] = field(default_factory=list)
    duplicate_rows: int = 0
    memory_mb: float = 0.0
    summary: str = ""

    def column(self, name: str) -> ColumnProfile | None:
        for profile in self.columns:
            if profile.name == name:
                return profile
        return None

    def names_with_role(self, *roles: Role) -> list[str]:
        wanted = set(roles)
        return [column.name for column in self.columns if column.role in wanted]

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_count": self.row_count,
            "column_count": self.column_count,
            "duplicate_rows": self.duplicate_rows,
            "memory_mb": round(self.memory_mb, 3),
            "summary": self.summary,
            "columns": [column.to_dict() for column in self.columns],
        }


@dataclass
class Insight:
    """A business conclusion with the evidence that supports it.

    ``caveat`` carries a confounding variable that changes how the finding
    should be read; ``objection`` carries what an adversarial review of it
    found. Both are shown to the user rather than used to silently delete the
    finding - a reader who can see the objection can judge it.
    """

    title: str
    result: str
    evidence: str
    interpretation: str
    confidence: str = "medium"
    action: str = ""
    axis: str = "general"
    chart_id: str = ""
    caveat: str = ""
    objection: str = ""
    grounded: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "result": self.result,
            "evidence": self.evidence,
            "interpretation": self.interpretation,
            "confidence": self.confidence,
            "action": self.action,
            "axis": self.axis,
            "chart_id": self.chart_id,
            "caveat": self.caveat,
            "objection": self.objection,
            "grounded": self.grounded,
        }


@dataclass
class Kpi:
    """One headline number plus what it means."""

    name: str
    value: float | None
    display_value: str
    formula: str
    interpretation: str = ""
    unit: str = ""
    trend: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "display_value": self.display_value,
            "formula": self.formula,
            "interpretation": self.interpretation,
            "unit": self.unit,
            "trend": self.trend,
        }


@dataclass
class Chart:
    """A figure produced during exploration, kept with its own explanation.

    ``group_column``, ``measure`` and ``aggregation`` record the comparison the
    chart makes, so it can be re-tested for significance and for a confounding
    third variable without having to guess what it was comparing.
    """

    id: str
    title: str
    kind: str
    description: str
    axis: str = "general"
    figure_json: str = ""
    table: pd.DataFrame | None = field(default=None, repr=False)
    group_column: str = ""
    measure: str = ""
    aggregation: str = ""

    @property
    def is_comparison(self) -> bool:
        """Whether this chart claims one group differs from another."""
        return bool(self.group_column and self.measure)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "kind": self.kind,
            "description": self.description,
            "axis": self.axis,
            "group_column": self.group_column,
            "measure": self.measure,
            "aggregation": self.aggregation,
        }


@dataclass
class DashboardPanel:
    """One tile on a dashboard."""

    kind: str  # "kpi" or "chart"
    reference: str  # kpi name or chart id
    width: int = 1


@dataclass
class Dashboard:
    """A named collection of panels aimed at one audience."""

    id: str
    title: str
    audience: str
    description: str
    panels: list[DashboardPanel] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "audience": self.audience,
            "description": self.description,
            "panels": [
                {"kind": panel.kind, "reference": panel.reference, "width": panel.width}
                for panel in self.panels
            ],
        }


class PipelineState:
    """Everything known about one analysis run."""

    def __init__(
        self,
        *,
        mode: RunMode = RunMode.INTERACTIVE,
        run_id: str | None = None,
        workspace: Path | None = None,
    ) -> None:
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.mode = mode
        self.started_at = datetime.now(timezone.utc)
        self.workspace = workspace

        # Source
        self.source_path: Path | None = None
        self.source_name: str = ""
        self.source_format: str = ""
        self.load_notes: list[str] = []

        # Data
        self.raw_frame: pd.DataFrame | None = None
        self.frame: pd.DataFrame | None = None
        self.profile = DatasetProfile()
        self.engineered_columns: list[str] = []

        # Findings
        self.focus_axes: list[str] = []
        self.charts: list[Chart] = []
        self.insights: list[Insight] = []
        self.kpis: list[Kpi] = []
        self.dashboards: list[Dashboard] = []

        # Shared services
        self.memory = BusinessMemory()
        self.log = ActivityLog()

        # Bookkeeping
        self.stage_status: dict[str, StageStatus] = {
            key: StageStatus.PENDING for key, _ in STAGES
        }
        self.answers: list[tuple[Decision, Answer]] = []
        self.artefacts: dict[str, Path] = {}
        self.errors: list[str] = []

    # -- stages ------------------------------------------------------------

    def begin_stage(self, stage: str) -> None:
        self.stage_status[stage] = StageStatus.RUNNING
        self.log.record(
            EventKind.STAGE_STARTED, stage, f"Started: {STAGE_TITLES.get(stage, stage)}"
        )

    def finish_stage(self, stage: str, summary: str = "") -> None:
        self.stage_status[stage] = StageStatus.DONE
        self.log.record(
            EventKind.STAGE_FINISHED,
            stage,
            summary or f"Finished: {STAGE_TITLES.get(stage, stage)}",
        )

    def skip_stage(self, stage: str, reason: str = "") -> None:
        self.stage_status[stage] = StageStatus.SKIPPED
        self.log.record(
            EventKind.STAGE_SKIPPED,
            stage,
            reason or f"Skipped: {STAGE_TITLES.get(stage, stage)}",
        )

    def fail_stage(self, stage: str, reason: str) -> None:
        self.stage_status[stage] = StageStatus.FAILED
        self.errors.append(f"{stage}: {reason}")
        self.log.record(EventKind.ERROR, stage, reason)

    @property
    def is_complete(self) -> bool:
        return all(
            status in (StageStatus.DONE, StageStatus.SKIPPED, StageStatus.FAILED)
            for status in self.stage_status.values()
        )

    # -- data --------------------------------------------------------------

    def set_frame(self, frame: pd.DataFrame, stage: str, description: str) -> None:
        """Replace the working frame and log what changed."""
        before = (0, 0) if self.frame is None else self.frame.shape
        self.frame = frame
        after = frame.shape
        self.log.record(
            EventKind.DATA_CHANGED,
            stage,
            description,
            rows_before=before[0],
            rows_after=after[0],
            columns_before=before[1],
            columns_after=after[1],
        )

    def remember(
        self, statement: str, *, category: str = "context", stage: str = "", topic: str = ""
    ) -> None:
        fact = self.memory.remember(
            statement, category=category, stage=stage, topic=topic
        )
        if fact is not None:
            self.log.record(
                EventKind.FACT_RECORDED,
                stage or "memory",
                f"Recorded: {fact.statement}",
                category=fact.category,
            )

    def record_answer(self, decision: Decision, answer: Answer) -> None:
        self.answers.append((decision, answer))
        self.log.record(
            EventKind.DECISION_ANSWERED,
            decision.stage,
            f"{decision.topic}: {answer.describe()}",
            automatic=answer.automatic,
            question=decision.question,
        )

    def add_artefact(self, name: str, path: Path, stage: str = "report") -> None:
        self.artefacts[name] = path
        self.log.record(
            EventKind.ARTEFACT_CREATED, stage, f"Created {name}", path=str(path)
        )

    def chart(self, chart_id: str) -> Chart | None:
        for chart in self.charts:
            if chart.id == chart_id:
                return chart
        return None

    def kpi(self, name: str) -> Kpi | None:
        for kpi in self.kpis:
            if kpi.name == name:
                return kpi
        return None

    # -- serialisation -----------------------------------------------------

    def summary_dict(self) -> dict[str, Any]:
        """JSON-safe snapshot, used for persistence and for the report."""
        return {
            "run_id": self.run_id,
            "mode": self.mode.value,
            "started_at": self.started_at.isoformat(),
            "source": {
                "name": self.source_name,
                "format": self.source_format,
                "notes": self.load_notes,
            },
            "profile": self.profile.to_dict(),
            "engineered_columns": self.engineered_columns,
            "focus_axes": self.focus_axes,
            "charts": [chart.to_dict() for chart in self.charts],
            "insights": [insight.to_dict() for insight in self.insights],
            "kpis": [kpi.to_dict() for kpi in self.kpis],
            "dashboards": [dashboard.to_dict() for dashboard in self.dashboards],
            "business_memory": self.memory.to_list(),
            "activity_log": self.log.to_list(),
            "decisions": [
                {"decision": decision.to_dict(), "answer": answer.to_dict()}
                for decision, answer in self.answers
            ],
            "stage_status": {
                stage: status.value for stage, status in self.stage_status.items()
            },
            "artefacts": {name: str(path) for name, path in self.artefacts.items()},
            "errors": self.errors,
        }
