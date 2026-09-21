"""Durable, project-scoped memory and a human-approved learning loop.

SQLite is deliberately enough for the first two phases: facts are structured
and filtered by project, so semantic search would add cost before it adds
accuracy. Every write is versioned and model-inferred lessons remain pending
until the owner explicitly approves them.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

from .business_memory import CATEGORIES, BusinessMemory, Fact
from .config import get_settings


MEMORY_STATUSES = ("candidate", "confirmed", "active", "disputed", "superseded", "expired", "archived")
CANDIDATE_STATUSES = ("pending", "approved", "rejected")
FEEDBACK_RATINGS = ("useful", "not_useful", "incorrect")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value: str, limit: int = 4000) -> str:
    return " ".join(str(value or "").split())[:limit]


@dataclass(frozen=True)
class Project:
    id: str
    name: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class MemoryItem:
    id: str
    project_id: str
    statement: str
    category: str
    status: str
    confidence: str
    source_type: str
    source_stage: str
    source_topic: str
    scope: str
    columns: tuple[str, ...]
    disputed: bool
    version: int
    valid_from: str | None
    valid_to: str | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class LearningCandidate:
    id: str
    project_id: str
    statement: str
    category: str
    status: str
    confidence: str
    reason: str
    source_feedback_id: str
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class ImprovementPolicy:
    """A versioned set of safe ranking weights, never executable code."""

    id: str
    project_id: str
    version: int
    name: str
    status: str
    rules: dict[str, float]
    evaluation: dict[str, Any]
    parent_id: str | None
    created_at: str
    activated_at: str | None


class ProjectMemoryStore:
    """Small repository with one short-lived connection per operation."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or get_settings().workspace)
        self.path = self.root / "insightlab_memory.sqlite3"
        self.root.mkdir(parents=True, exist_ok=True)
        self._prepare()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        return connection

    def _prepare(self) -> None:
        with self._connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS projects (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL COLLATE NOCASE UNIQUE,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS memory_items (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    statement TEXT NOT NULL, category TEXT NOT NULL, status TEXT NOT NULL,
                    confidence TEXT NOT NULL, source_type TEXT NOT NULL,
                    source_stage TEXT NOT NULL DEFAULT '', source_topic TEXT NOT NULL DEFAULT '',
                    scope TEXT NOT NULL DEFAULT 'project', columns_json TEXT NOT NULL DEFAULT '[]',
                    claim_json TEXT, disputed INTEGER NOT NULL DEFAULT 0, version INTEGER NOT NULL DEFAULT 1,
                    valid_from TEXT, valid_to TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS memory_project_status ON memory_items(project_id, status);
                CREATE TABLE IF NOT EXISTS memory_versions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id TEXT NOT NULL,
                    version INTEGER NOT NULL, snapshot_json TEXT NOT NULL,
                    change_kind TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '', created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analysis_runs (
                    run_id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    source_name TEXT NOT NULL DEFAULT '', started_at TEXT NOT NULL, completed_at TEXT
                );
                CREATE TABLE IF NOT EXISTS project_preferences (
                    project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    key TEXT NOT NULL, value TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(project_id, key)
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    id TEXT PRIMARY KEY, run_id TEXT NOT NULL, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    item_type TEXT NOT NULL, item_key TEXT NOT NULL, rating TEXT NOT NULL,
                    correction TEXT NOT NULL DEFAULT '', reason TEXT NOT NULL DEFAULT '',
                    context_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
                    UNIQUE(run_id, item_type, item_key)
                );
                CREATE TABLE IF NOT EXISTS learning_candidates (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    statement TEXT NOT NULL, category TEXT NOT NULL, status TEXT NOT NULL,
                    confidence TEXT NOT NULL, reason TEXT NOT NULL DEFAULT '', source_feedback_id TEXT NOT NULL,
                    approved_memory_id TEXT, created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS candidate_project_status ON learning_candidates(project_id, status);
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    score REAL NOT NULL, metrics_json TEXT NOT NULL, cases_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS evaluation_project_created ON evaluation_runs(project_id, created_at);
                CREATE TABLE IF NOT EXISTS improvement_policies (
                    id TEXT PRIMARY KEY, project_id TEXT NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                    version INTEGER NOT NULL, name TEXT NOT NULL, status TEXT NOT NULL,
                    rules_json TEXT NOT NULL, evaluation_json TEXT NOT NULL DEFAULT '{}',
                    parent_id TEXT, created_at TEXT NOT NULL, activated_at TEXT,
                    UNIQUE(project_id, version)
                );
                CREATE INDEX IF NOT EXISTS policy_project_status ON improvement_policies(project_id, status);
                """
            )
            columns = {row["name"] for row in db.execute("PRAGMA table_info(memory_items)")}
            if "valid_from" not in columns:
                db.execute("ALTER TABLE memory_items ADD COLUMN valid_from TEXT")
            if "valid_to" not in columns:
                db.execute("ALTER TABLE memory_items ADD COLUMN valid_to TEXT")
            feedback_columns = {row["name"] for row in db.execute("PRAGMA table_info(feedback)")}
            if "context_json" not in feedback_columns:
                db.execute("ALTER TABLE feedback ADD COLUMN context_json TEXT NOT NULL DEFAULT '{}'")

    # -- projects -----------------------------------------------------

    def ensure_project(self, name: str) -> Project:
        name = _clean(name, 100) or "My project"
        now = _now()
        with self._connect() as db:
            row = db.execute("SELECT * FROM projects WHERE name = ? COLLATE NOCASE", (name,)).fetchone()
            if row is None:
                project_id = uuid.uuid4().hex
                db.execute(
                    "INSERT INTO projects(id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (project_id, name, now, now),
                )
                row = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
            else:
                db.execute("UPDATE projects SET updated_at = ? WHERE id = ?", (now, row["id"]))
        return Project(**dict(row))

    def list_projects(self) -> list[Project]:
        with self._connect() as db:
            rows = db.execute("SELECT * FROM projects ORDER BY updated_at DESC, name").fetchall()
        return [Project(**dict(row)) for row in rows]

    def project(self, project_id: str) -> Project | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        return Project(**dict(row)) if row else None

    def preferences(self, project_id: str) -> dict[str, str]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT key, value FROM project_preferences WHERE project_id=?", (project_id,)
            ).fetchall()
        return {str(row["key"]): str(row["value"]) for row in rows}

    def set_preferences(self, project_id: str, preferences: dict[str, str]) -> None:
        now = _now()
        with self._connect() as db:
            for key, value in preferences.items():
                key, value = _clean(key, 80), _clean(value, 250)
                if not key or not value:
                    continue
                db.execute(
                    "INSERT INTO project_preferences(project_id, key, value, updated_at) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(project_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                    (project_id, key, value, now),
                )

    def import_legacy_memory(self, project_id: str, path: Path) -> int:
        """One-time bridge from the old per-run JSON snapshots."""
        from .storage import load_business_memory

        memory = load_business_memory(Path(path))
        before = len(self.list_memory(project_id))
        self.sync_memory(project_id, memory)
        return max(0, len(self.list_memory(project_id)) - before)

    # -- facts --------------------------------------------------------

    @staticmethod
    def _snapshot(row: sqlite3.Row | dict[str, Any]) -> dict[str, Any]:
        raw = dict(row)
        return {key: raw.get(key) for key in (
            "id", "project_id", "statement", "category", "status", "confidence",
            "source_type", "source_stage", "source_topic", "scope", "columns_json",
            "claim_json", "disputed", "version", "created_at", "updated_at",
            "valid_from", "valid_to",
        )}

    @staticmethod
    def _item(row: sqlite3.Row) -> MemoryItem:
        raw = dict(row)
        columns = tuple(json.loads(raw.pop("columns_json") or "[]"))
        raw.pop("claim_json", None)
        raw["disputed"] = bool(raw["disputed"])
        return MemoryItem(columns=columns, **raw)

    def list_memory(self, project_id: str, *, include_inactive: bool = True) -> list[MemoryItem]:
        self._expire_due(project_id)
        where = "project_id = ?" if include_inactive else "project_id = ? AND status = 'active'"
        with self._connect() as db:
            rows = db.execute(
                f"SELECT * FROM memory_items WHERE {where} ORDER BY updated_at DESC", (project_id,)
            ).fetchall()
        return [self._item(row) for row in rows]

    def load_memory(self, project_id: str, columns: list[str] | None = None) -> BusinessMemory:
        self._expire_due(project_id)
        wanted = {name.casefold() for name in (columns or [])}
        facts: list[Fact] = []
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM memory_items WHERE project_id = ? AND status = 'active' ORDER BY updated_at",
                (project_id,),
            ).fetchall()
        for row in rows:
            related = set(json.loads(row["columns_json"] or "[]"))
            if row["scope"] == "dataset":
                if not wanted or not related or not {name.casefold() for name in related} & wanted:
                    continue
            payload = {
                "id": row["id"], "statement": row["statement"], "category": row["category"],
                "source_stage": row["source_stage"], "source_topic": row["source_topic"],
                "recorded_at": row["created_at"],
                "claim": json.loads(row["claim_json"]) if row["claim_json"] else None,
                "disputed": bool(row["disputed"]), "status": row["status"],
                "confidence": row["confidence"], "source_type": row["source_type"],
                "scope": row["scope"], "columns": list(related), "version": row["version"],
                "updated_at": row["updated_at"],
                "valid_from": row["valid_from"], "valid_to": row["valid_to"],
            }
            facts.append(Fact.from_dict(payload))
        return BusinessMemory(facts)

    def add_memory(
        self, project_id: str, statement: str, *, category: str = "context",
        confidence: str = "high", source_type: str = "user",
        source_stage: str = "user", source_topic: str = "",
        scope: str = "project", columns: list[str] | None = None,
        claim: Any = None, status: str = "active",
        valid_from: str | None = None, valid_to: str | None = None,
    ) -> str:
        statement = _clean(statement)
        if not statement:
            raise ValueError("Memory statement cannot be blank.")
        category = category if category in CATEGORIES else "context"
        status = status if status in MEMORY_STATUSES else "candidate"
        confidence = confidence if confidence in {"high", "medium", "low"} else "low"
        scope = scope if scope in {"project", "dataset"} else "project"
        if scope == "dataset" and not columns:
            raise ValueError("Dataset-scoped memory needs at least one related column.")
        for value in (valid_from, valid_to):
            if value:
                date.fromisoformat(value)
        if valid_from and valid_to and valid_from > valid_to:
            raise ValueError("Memory validity cannot end before it starts.")
        with self._connect() as db:
            existing = db.execute(
                "SELECT id FROM memory_items WHERE project_id = ? AND statement = ? COLLATE NOCASE AND status = 'active'",
                (project_id, statement),
            ).fetchone()
            if existing:
                return str(existing["id"])
            item_id, now = uuid.uuid4().hex[:12], _now()
            claim_json = (
                json.dumps(
                    claim.to_dict() if hasattr(claim, "to_dict") else claim,
                    sort_keys=True,
                )
                if claim else None
            )
            values = (
                item_id, project_id, statement, category, status, confidence, source_type,
                source_stage, source_topic, scope, json.dumps(columns or []), claim_json,
                0, 1, now, now,
                valid_from or None, valid_to or None,
            )
            db.execute(
                "INSERT INTO memory_items(id, project_id, statement, category, status, confidence, source_type, "
                "source_stage, source_topic, scope, columns_json, claim_json, disputed, version, created_at, updated_at, valid_from, valid_to) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values
            )
            self._version(db, item_id, "created", "Initial memory")
        return item_id

    def _version(self, db: sqlite3.Connection, item_id: str, kind: str, reason: str) -> None:
        row = db.execute("SELECT * FROM memory_items WHERE id = ?", (item_id,)).fetchone()
        if row:
            db.execute(
                "INSERT INTO memory_versions(memory_id, version, snapshot_json, change_kind, reason, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (item_id, row["version"], json.dumps(self._snapshot(row), ensure_ascii=False), kind, _clean(reason, 500), _now()),
            )

    def update_memory(self, item_id: str, *, statement: str | None = None,
                      category: str | None = None, status: str | None = None,
                      scope: str | None = None, valid_to: str | None | object = ...,
                      columns: list[str] | None = None,
                      reason: str = "Edited by owner") -> bool:
        with self._connect() as db:
            row = db.execute("SELECT * FROM memory_items WHERE id = ?", (item_id,)).fetchone()
            if row is None:
                return False
            new_statement = _clean(statement) if statement is not None else row["statement"]
            if not new_statement:
                return False
            new_status = status if status in MEMORY_STATUSES else row["status"]
            new_category = category or row["category"]
            new_scope = scope if scope in {"project", "dataset"} else row["scope"]
            new_valid_to = row["valid_to"] if valid_to is ... else (str(valid_to).strip() or None)
            new_columns = row["columns_json"] if columns is None else json.dumps([
                _clean(column, 200) for column in columns if _clean(column, 200)
            ])
            if new_valid_to:
                date.fromisoformat(new_valid_to)
            db.execute(
                "UPDATE memory_items SET statement = ?, category = ?, status = ?, scope = ?, valid_to = ?, columns_json = ?, version = version + 1, updated_at = ? WHERE id = ?",
                (new_statement, new_category, new_status, new_scope, new_valid_to, new_columns, _now(), item_id),
            )
            self._version(db, item_id, "updated", reason)
        return True

    def _expire_due(self, project_id: str) -> None:
        today = date.today().isoformat()
        with self._connect() as db:
            ids = [row["id"] for row in db.execute(
                "SELECT id FROM memory_items WHERE project_id=? AND status='active' AND valid_to IS NOT NULL AND valid_to < ?",
                (project_id, today),
            ).fetchall()]
        for item_id in ids:
            self.update_memory(item_id, status="expired", reason="Validity period ended")

    def history(self, item_id: str) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT version, snapshot_json, change_kind, reason, created_at FROM memory_versions WHERE memory_id = ? ORDER BY version DESC, id DESC",
                (item_id,),
            ).fetchall()
        return [{**dict(row), "snapshot": json.loads(row["snapshot_json"])} for row in rows]

    def restore_version(self, item_id: str, version: int) -> bool:
        """Restore an old snapshot as a new version, never erasing later history."""
        with self._connect() as db:
            current = db.execute("SELECT * FROM memory_items WHERE id=?", (item_id,)).fetchone()
            historic = db.execute(
                "SELECT snapshot_json FROM memory_versions WHERE memory_id=? AND version=? ORDER BY id DESC LIMIT 1",
                (item_id, version),
            ).fetchone()
            if current is None or historic is None:
                return False
            snapshot = json.loads(historic["snapshot_json"])
            db.execute(
                "UPDATE memory_items SET statement=?, category=?, status=?, confidence=?, source_type=?, "
                "source_stage=?, source_topic=?, scope=?, columns_json=?, claim_json=?, disputed=?, "
                "valid_from=?, valid_to=?, version=?, updated_at=? WHERE id=?",
                (
                    snapshot["statement"], snapshot["category"], snapshot["status"],
                    snapshot["confidence"], snapshot["source_type"], snapshot["source_stage"],
                    snapshot["source_topic"], snapshot["scope"], snapshot["columns_json"],
                    snapshot["claim_json"], snapshot["disputed"], snapshot.get("valid_from"),
                    snapshot.get("valid_to"), int(current["version"]) + 1, _now(), item_id,
                ),
            )
            self._version(db, item_id, "restored", f"Restored version {version}")
        return True

    def sync_memory(self, project_id: str, memory: BusinessMemory) -> None:
        """Persist mutations made by pipeline agents without losing history."""
        with self._connect() as db:
            existing = {
                row["id"]: row for row in db.execute(
                    "SELECT * FROM memory_items WHERE project_id = ? AND status = 'active'", (project_id,)
                ).fetchall()
            }
        for fact in memory:
            if fact.id not in existing:
                new_id = self.add_memory(
                    project_id, fact.statement, category=fact.category,
                    confidence=getattr(fact, "confidence", "high"),
                    source_type=getattr(fact, "source_type", "user"),
                    source_stage=fact.source_stage, source_topic=fact.source_topic,
                    scope=getattr(fact, "scope", "project"),
                    columns=getattr(fact, "columns", []), claim=fact.claim,
                    valid_from=getattr(fact, "valid_from", None),
                    valid_to=getattr(fact, "valid_to", None),
                )
                fact.id = new_id
                continue
            row = existing[fact.id]
            claim_json = json.dumps(fact.claim.to_dict(), sort_keys=True) if fact.claim else None
            changed = (
                row["statement"] != fact.statement or row["category"] != fact.category
                or row["claim_json"] != claim_json or bool(row["disputed"]) != fact.disputed
            )
            if changed:
                with self._connect() as db:
                    db.execute(
                        "UPDATE memory_items SET statement=?, category=?, claim_json=?, disputed=?, version=version+1, updated_at=? WHERE id=?",
                        (fact.statement, fact.category, claim_json, int(fact.disputed), _now(), fact.id),
                    )
                    self._version(db, fact.id, "pipeline_update", "Updated during analysis")
        for item_id in memory.forgotten_ids:
            self.update_memory(item_id, status="archived", reason="Removed during analysis")

    # -- runs and feedback --------------------------------------------

    def record_run(self, run_id: str, project_id: str, source_name: str, started_at: datetime,
                   *, completed: bool = False) -> None:
        with self._connect() as db:
            db.execute(
                "INSERT INTO analysis_runs(run_id, project_id, source_name, started_at, completed_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id) DO UPDATE SET source_name=excluded.source_name, completed_at=COALESCE(excluded.completed_at, analysis_runs.completed_at)",
                (run_id, project_id, _clean(source_name, 250), started_at.isoformat(), _now() if completed else None),
            )

    @staticmethod
    def item_key(*parts: str) -> str:
        return hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]

    def feedback_for(self, run_id: str, item_type: str, item_key: str) -> dict[str, Any] | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM feedback WHERE run_id=? AND item_type=? AND item_key=?",
                (run_id, item_type, item_key),
            ).fetchone()
        return dict(row) if row else None

    def record_feedback(self, *, run_id: str, project_id: str, item_type: str,
                        item_key: str, rating: str, correction: str = "",
                        reason: str = "", category: str = "context",
                        context: dict[str, Any] | None = None) -> tuple[str, str | None]:
        if rating not in FEEDBACK_RATINGS:
            raise ValueError("Unsupported feedback rating.")
        correction, reason, now = _clean(correction), _clean(reason, 1000), _now()
        allowed_context = {
            key: _clean(str(value), 200)
            for key, value in (context or {}).items()
            if key in {
                "axis", "kpi", "chart_kind", "confidence", "domain_family",
                "rebuild", "review_action",
            } and value is not None
        }
        context_json = json.dumps(allowed_context, ensure_ascii=False, sort_keys=True)
        if rating == "incorrect" and not correction:
            raise ValueError("A correction is required when an item is incorrect.")
        with self._connect() as db:
            prior = db.execute(
                "SELECT id FROM feedback WHERE run_id=? AND item_type=? AND item_key=?",
                (run_id, item_type, item_key),
            ).fetchone()
            feedback_id = str(prior["id"]) if prior else uuid.uuid4().hex[:12]
            if prior:
                db.execute("DELETE FROM learning_candidates WHERE source_feedback_id=? AND status='pending'", (feedback_id,))
                db.execute(
                    "UPDATE feedback SET rating=?, correction=?, reason=?, context_json=?, created_at=? WHERE id=?",
                    (rating, correction, reason, context_json, now, feedback_id),
                )
            else:
                db.execute(
                    "INSERT INTO feedback(id, run_id, project_id, item_type, item_key, rating, correction, reason, context_json, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (feedback_id, run_id, project_id, item_type, item_key, rating, correction, reason, context_json, now),
                )
            candidate_id = None
            if rating == "incorrect":
                candidate_id = uuid.uuid4().hex[:12]
                db.execute(
                    "INSERT INTO learning_candidates VALUES (?, ?, ?, ?, 'pending', 'high', ?, ?, NULL, ?, ?)",
                    (candidate_id, project_id, correction, category, reason, feedback_id, now, now),
                )
        return feedback_id, candidate_id

    def adaptive_profile(self, project_id: str) -> dict[str, Any]:
        """Aggregate explicit ratings into small, explainable ranking signals."""
        scores: dict[str, dict[str, float]] = {
            "axes": {}, "kpis": {}, "chart_kinds": {}, "confidence": {}
        }
        counts: dict[str, dict[str, int]] = {key: {} for key in scores}
        weights = {"useful": 2.0, "not_useful": -1.0, "incorrect": -2.0}
        mapping = {
            "axis": "axes", "kpi": "kpis", "chart_kind": "chart_kinds",
            "confidence": "confidence",
        }
        total = 0
        with self._connect() as db:
            rows = db.execute(
                "SELECT rating, context_json FROM feedback WHERE project_id=? ORDER BY created_at",
                (project_id,),
            ).fetchall()
        for row in rows:
            try:
                context = json.loads(row["context_json"] or "{}")
            except (TypeError, json.JSONDecodeError):
                continue
            total += 1
            for source, target in mapping.items():
                value = _clean(str(context.get(source, "")), 200).casefold()
                if not value:
                    continue
                scores[target][value] = round(scores[target].get(value, 0.0) + weights[row["rating"]], 3)
                counts[target][value] = counts[target].get(value, 0) + 1
        return {**scores, "counts": counts, "feedback_count": total}

    # -- evaluation and safe policy versions -----------------------------

    def record_evaluation(self, run_id: str, project_id: str, *, score: float,
                          metrics: dict[str, Any], cases: list[dict[str, Any]]) -> None:
        """Persist structural quality signals only; cases must not contain user text."""
        with self._connect() as db:
            db.execute(
                "INSERT INTO evaluation_runs(run_id, project_id, score, metrics_json, cases_json, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(run_id) DO UPDATE SET "
                "score=excluded.score, metrics_json=excluded.metrics_json, cases_json=excluded.cases_json",
                (run_id, project_id, float(score), json.dumps(metrics, sort_keys=True),
                 json.dumps(cases, sort_keys=True), _now()),
            )

    def evaluations(self, project_id: str, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as db:
            rows = db.execute(
                "SELECT run_id, score, metrics_json, cases_json, created_at FROM evaluation_runs "
                "WHERE project_id=? ORDER BY created_at DESC LIMIT ?", (project_id, max(1, limit)),
            ).fetchall()
        return [
            {"run_id": row["run_id"], "score": float(row["score"]),
             "metrics": json.loads(row["metrics_json"]),
             "cases": json.loads(row["cases_json"]), "created_at": row["created_at"]}
            for row in rows
        ]

    @staticmethod
    def default_policy_rules() -> dict[str, float]:
        return {"axis_weight": 1.0, "kpi_weight": 1.0, "chart_weight": 1.0,
                "confidence_weight": 0.25}

    @staticmethod
    def _policy(row: sqlite3.Row) -> ImprovementPolicy:
        return ImprovementPolicy(
            id=row["id"], project_id=row["project_id"], version=int(row["version"]),
            name=row["name"], status=row["status"], rules=json.loads(row["rules_json"]),
            evaluation=json.loads(row["evaluation_json"] or "{}"), parent_id=row["parent_id"],
            created_at=row["created_at"], activated_at=row["activated_at"],
        )

    def active_policy(self, project_id: str) -> ImprovementPolicy:
        with self._connect() as db:
            row = db.execute(
                "SELECT * FROM improvement_policies WHERE project_id=? AND status='active' "
                "ORDER BY version DESC LIMIT 1", (project_id,),
            ).fetchone()
            if row is None:
                now, policy_id = _now(), uuid.uuid4().hex[:12]
                db.execute(
                    "INSERT INTO improvement_policies(id, project_id, version, name, status, rules_json, "
                    "evaluation_json, parent_id, created_at, activated_at) VALUES (?, ?, 1, ?, 'active', ?, '{}', NULL, ?, ?)",
                    (policy_id, project_id, "Safe default", json.dumps(self.default_policy_rules()), now, now),
                )
                row = db.execute("SELECT * FROM improvement_policies WHERE id=?", (policy_id,)).fetchone()
        return self._policy(row)

    def list_policies(self, project_id: str) -> list[ImprovementPolicy]:
        self.active_policy(project_id)
        with self._connect() as db:
            rows = db.execute(
                "SELECT * FROM improvement_policies WHERE project_id=? ORDER BY version DESC",
                (project_id,),
            ).fetchall()
        return [self._policy(row) for row in rows]

    def create_policy_candidate(self, project_id: str, name: str,
                                rules: dict[str, float]) -> ImprovementPolicy:
        clean_rules = self.default_policy_rules()
        for key in clean_rules:
            clean_rules[key] = float(rules.get(key, clean_rules[key]))
            if not 0.0 <= clean_rules[key] <= 3.0:
                raise ValueError("Policy weights must be between 0 and 3.")
        parent = self.active_policy(project_id)
        with self._connect() as db:
            version = int(db.execute(
                "SELECT COALESCE(MAX(version), 0) + 1 FROM improvement_policies WHERE project_id=?",
                (project_id,),
            ).fetchone()[0])
            policy_id, now = uuid.uuid4().hex[:12], _now()
            db.execute(
                "INSERT INTO improvement_policies(id, project_id, version, name, status, rules_json, "
                "evaluation_json, parent_id, created_at) VALUES (?, ?, ?, ?, 'draft', ?, '{}', ?, ?)",
                (policy_id, project_id, version, _clean(name, 100) or f"Policy v{version}",
                 json.dumps(clean_rules, sort_keys=True), parent.id, now),
            )
            row = db.execute("SELECT * FROM improvement_policies WHERE id=?", (policy_id,)).fetchone()
        return self._policy(row)

    def validate_policy(self, policy_id: str) -> dict[str, Any]:
        """Offline shadow check against historical ratings; no model/API call."""
        with self._connect() as db:
            row = db.execute("SELECT * FROM improvement_policies WHERE id=?", (policy_id,)).fetchone()
        if row is None:
            raise ValueError("Policy does not exist.")
        policy = self._policy(row)
        rules = policy.rules
        guardrails = all(0.0 <= float(value) <= 3.0 for value in rules.values())
        profile = self.adaptive_profile(policy.project_id)
        samples = int(profile["feedback_count"])
        group_weights = {
            "axes": "axis_weight", "kpis": "kpi_weight",
            "chart_kinds": "chart_weight", "confidence": "confidence_weight",
        }

        def alignment(candidate_rules: dict[str, float]) -> float:
            positive = 0.0
            negative = 0.0
            for group, weight_key in group_weights.items():
                weight = float(candidate_rules.get(weight_key, 1.0))
                for value in profile[group].values():
                    positive += max(0.0, value) * weight
                    negative += abs(min(0.0, value)) * weight
            return round(100 * positive / max(1.0, positive + negative), 1)

        parent = next(
            (item for item in self.list_policies(policy.project_id) if item.id == policy.parent_id),
            self.active_policy(policy.project_id),
        )
        candidate_alignment = alignment(rules)
        baseline_alignment = alignment(parent.rules)
        quality_gate = samples == 0 or candidate_alignment >= baseline_alignment
        result = {
            "guardrails_passed": guardrails, "historical_feedback": samples,
            "preference_alignment": candidate_alignment,
            "baseline_alignment": baseline_alignment,
            "alignment_delta": round(candidate_alignment - baseline_alignment, 1),
            "quality_gate_passed": quality_gate,
            "mode": "offline_shadow", "uses_raw_data": False,
        }
        status = "validated" if guardrails and quality_gate else "rejected"
        with self._connect() as db:
            db.execute(
                "UPDATE improvement_policies SET status=?, evaluation_json=? WHERE id=? AND status!='active'",
                (status, json.dumps(result, sort_keys=True), policy_id),
            )
        return result

    def activate_policy(self, policy_id: str) -> bool:
        """Owner-only promotion. Drafts cannot bypass the shadow gate."""
        with self._connect() as db:
            row = db.execute("SELECT * FROM improvement_policies WHERE id=?", (policy_id,)).fetchone()
            if row is None or row["status"] != "validated":
                return False
            db.execute(
                "UPDATE improvement_policies SET status='retired' WHERE project_id=? AND status='active'",
                (row["project_id"],),
            )
            db.execute(
                "UPDATE improvement_policies SET status='active', activated_at=? WHERE id=?",
                (_now(), policy_id),
            )
        return True

    def rollback_policy(self, project_id: str) -> bool:
        policies = self.list_policies(project_id)
        active = next((policy for policy in policies if policy.status == "active"), None)
        if active is None or not active.parent_id:
            return False
        with self._connect() as db:
            parent = db.execute("SELECT * FROM improvement_policies WHERE id=?", (active.parent_id,)).fetchone()
            if parent is None:
                return False
            db.execute("UPDATE improvement_policies SET status='retired' WHERE id=?", (active.id,))
            db.execute(
                "UPDATE improvement_policies SET status='active', activated_at=? WHERE id=?",
                (_now(), parent["id"]),
            )
        return True

    def list_candidates(self, project_id: str, status: str | None = None) -> list[LearningCandidate]:
        query, params = "SELECT id, project_id, statement, category, status, confidence, reason, source_feedback_id, created_at, updated_at FROM learning_candidates WHERE project_id=?", [project_id]
        if status in CANDIDATE_STATUSES:
            query += " AND status=?"
            params.append(status)
        query += " ORDER BY created_at DESC"
        with self._connect() as db:
            rows = db.execute(query, params).fetchall()
        return [LearningCandidate(**dict(row)) for row in rows]

    def review_candidate(self, candidate_id: str, *, approve: bool,
                         edited_statement: str | None = None) -> str | None:
        with self._connect() as db:
            row = db.execute("SELECT * FROM learning_candidates WHERE id=?", (candidate_id,)).fetchone()
        if row is None or row["status"] != "pending":
            return None
        if not approve:
            with self._connect() as db:
                db.execute("UPDATE learning_candidates SET status='rejected', updated_at=? WHERE id=?", (_now(), candidate_id))
            return None
        statement = _clean(edited_statement if edited_statement is not None else row["statement"])
        if not statement:
            raise ValueError("Approved memory cannot be blank.")
        with self._connect() as db:
            feedback = db.execute(
                "SELECT context_json FROM feedback WHERE id=?",
                (row["source_feedback_id"],),
            ).fetchone()
        try:
            feedback_context = json.loads(feedback["context_json"] or "{}") if feedback else {}
        except (TypeError, json.JSONDecodeError):
            feedback_context = {}
        domain = _clean(str(feedback_context.get("domain_family", "")), 100)
        topic = f"Approved learning candidate {candidate_id}"
        if domain and domain != "general":
            topic = f"domain:{domain}|{topic}"
        memory_id = self.add_memory(
            row["project_id"], statement, category=row["category"], confidence="high",
            source_type="user_correction", source_stage="feedback",
            source_topic=topic, status="active",
        )
        with self._connect() as db:
            db.execute(
                "UPDATE learning_candidates SET statement=?, status='approved', approved_memory_id=?, updated_at=? WHERE id=?",
                (statement, memory_id, _now(), candidate_id),
            )
        return memory_id

    def metrics(self, project_id: str) -> dict[str, int]:
        self._expire_due(project_id)
        with self._connect() as db:
            active = db.execute("SELECT COUNT(*) FROM memory_items WHERE project_id=? AND status='active'", (project_id,)).fetchone()[0]
            pending = db.execute("SELECT COUNT(*) FROM learning_candidates WHERE project_id=? AND status='pending'", (project_id,)).fetchone()[0]
            corrected = db.execute("SELECT COUNT(*) FROM feedback WHERE project_id=? AND rating='incorrect'", (project_id,)).fetchone()[0]
            useful = db.execute("SELECT COUNT(*) FROM feedback WHERE project_id=? AND rating='useful'", (project_id,)).fetchone()[0]
        return {"active_memory": active, "pending_lessons": pending, "corrections": corrected, "useful": useful}
