"""The agent that answers a question the user asked.

Every other agent in the pipeline runs forwards and asks the user things. This
one runs on demand against a finished analysis and answers them instead, which
is the half of the collaboration the product was missing.

It never calculates. It turns a question into a :class:`QueryPlan`, the plan is
validated against the real schema, pandas executes it, and only then is the
model asked to say one sentence about the result it can see. A question it
cannot express as a plan gets an honest refusal naming what is missing, which
is far more useful than a confident wrong answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ..analysis import query as query_module
from ..analysis.query import PlanError, QueryPlan, QueryResult
from ..core.activity_log import EventKind
from ..core.grounding import build_fact_base, strip_unverified
from ..core.reasoning import AgentPersona, ReasoningEngine
from ..core.state import PipelineState, Role

#: Suggested follow-up questions offered after an analysis.
SUGGESTION_COUNT = 4

ANALYST = AgentPersona(
    role="Data analyst on call",
    goal=(
        "Answer the owner's question from their own data, or say plainly that "
        "the data cannot answer it."
    ),
    backstory=(
        "You have spent years being asked questions across a desk and answering "
        "them in one sentence. You never pad an answer, you never restate the "
        "question back, and when the data cannot answer something you say so "
        "immediately rather than producing a number that looks like an answer."
    ),
)

PLAN_SHAPE = """{
  "measure": "the column being measured, or empty to count records",
  "aggregation": "sum|mean|median|count|min|max|nunique",
  "group_by": ["at most two columns to break the answer down by"],
  "time_grain": "day|week|month|quarter|year, or null if the question is not about time",
  "filters": [{"column": "name", "operator": "=|!=|>|>=|<|<=|in|not in|contains", "value": "..."}],
  "sort_descending": true,
  "limit": 20,
  "answerable": true,
  "reason_if_not": "what the data would need for this to be answerable"
}"""


@dataclass
class Answer:
    """A reply to one question."""

    question: str
    understood_as: str = ""
    headline: str = ""
    narrative: str = ""
    result: QueryResult | None = None
    refusal: str = ""
    facts_used: list[str] = field(default_factory=list)

    @property
    def answered(self) -> bool:
        return self.result is not None and not self.refusal

    @property
    def table(self) -> pd.DataFrame | None:
        return self.result.table if self.result is not None else None


class AnalystAgent:
    """Answers questions against a completed run.

    Not a pipeline agent: it has no stage and yields no decisions. It is
    invoked by :meth:`Supervisor.ask` whenever the user asks something.
    """

    key = "analyst"
    persona = ANALYST

    def __init__(self, reasoning: ReasoningEngine) -> None:
        self.reasoning = reasoning

    # -- answering ---------------------------------------------------------

    def answer(self, state: PipelineState, question: str) -> Answer:
        question = " ".join(question.split())
        reply = Answer(question=question)

        if state.frame is None or state.frame.empty:
            reply.refusal = "There is no data loaded to answer questions about."
            return reply

        state.log.record(EventKind.NOTE, "ask", f"Question asked: {question}")

        raw = self._plan(state, question)
        if raw is None:
            reply.refusal = (
                "That question could not be turned into a calculation over your "
                "data. Try naming the figure you want and how it should be "
                "broken down, for example: revenue by region last quarter."
            )
            self._log_refusal(state, question, reply.refusal)
            return reply

        if not raw.get("answerable", True):
            reply.refusal = str(
                raw.get("reason_if_not")
                or "Your data does not contain what this question needs."
            )
            self._log_refusal(state, question, reply.refusal)
            return reply

        try:
            plan = query_module.validate(raw, state.frame, state.profile)
        except PlanError as error:
            reply.refusal = str(error)
            self._log_refusal(state, question, reply.refusal)
            return reply

        plan.question = question
        result = query_module.run(plan, state.frame, state.profile)

        reply.understood_as = plan.describe()
        reply.result = result
        reply.headline = result.headline
        reply.narrative = self._narrate(state, question, plan, result)

        state.log.record(
            EventKind.NOTE,
            "ask",
            f"Answered: {plan.describe()} -> {result.headline}",
            plan=plan.to_dict(),
        )
        return reply

    def _log_refusal(self, state: PipelineState, question: str, reason: str) -> None:
        state.log.record(
            EventKind.NOTE, "ask", f'Could not answer "{question}": {reason}'
        )

    # -- planning ----------------------------------------------------------

    def _plan(self, state: PipelineState, question: str) -> dict[str, Any] | None:
        """Turn a question into a proposed plan."""
        if not self.reasoning.available:
            return self._plan_without_a_model(state, question)

        parsed = self.reasoning.ask_json(
            self.key,
            self.persona,
            f'A business owner asked: "{question}"\n\n'
            f"{self._schema_block(state)}"
            f"{self._memory_block(state)}\n\n"
            "Express their question as a calculation over these columns. Use "
            "only column names from the list above, exactly as written. If the "
            "data does not contain what the question needs, set answerable to "
            "false and say what is missing - do not substitute a different "
            "column that happens to exist.",
            shape=PLAN_SHAPE,
        )
        if isinstance(parsed, dict):
            return parsed
        return self._plan_without_a_model(state, question)

    def _plan_without_a_model(
        self, state: PipelineState, question: str
    ) -> dict[str, Any] | None:
        """A keyword-matched plan, for when no model is reachable.

        Deliberately conservative: it matches column names the user actually
        typed and nothing else. Guessing at intent with no model is how a tool
        answers a question nobody asked.
        """
        lowered = question.casefold()
        schema = {
            column.name: column.role
            for column in state.profile.columns
            if column.name in state.frame.columns
        }

        def mentioned(role: Role) -> list[str]:
            return [
                name
                for name, column_role in schema.items()
                if column_role is role and name.replace("_", " ").casefold() in lowered
            ]

        measures = mentioned(Role.MEASURE)
        groups = mentioned(Role.CATEGORY)
        if not measures and not groups:
            return None

        aggregation = "sum"
        for word, how in (
            ("average", "mean"), ("mean", "mean"), ("typical", "median"),
            ("median", "median"), ("how many", "count"), ("count", "count"),
            ("highest", "max"), ("lowest", "min"),
        ):
            if word in lowered:
                aggregation = how
                break

        time_grain = None
        for word in ("month", "quarter", "year", "week", "day"):
            if word in lowered:
                time_grain = word
                break

        return {
            "measure": measures[0] if measures else "",
            "aggregation": aggregation,
            "group_by": groups[:2],
            "time_grain": time_grain,
            "filters": [],
            "answerable": True,
        }

    # -- narration ---------------------------------------------------------

    def _narrate(
        self,
        state: PipelineState,
        question: str,
        plan: QueryPlan,
        result: QueryResult,
    ) -> str:
        """One sentence about the answer, grounded in the executed result."""
        if result.table.empty:
            return result.headline

        if not self.reasoning.available:
            return result.headline

        rendered = result.table.head(15).to_csv(index=False).strip()
        text = self.reasoning.ask(
            self.key,
            self.persona,
            f'The owner asked: "{question}"\n\n'
            f"This was calculated from their data:\n{rendered}\n\n"
            f"The headline reading is: {result.headline}"
            f"{self._memory_block(state)}\n\n"
            "Answer their question in one or two sentences. Use only the "
            "figures above - every number you write will be checked against "
            "the data and any that does not match will be removed. Do not "
            "restate the question. Do not describe the table. If the answer "
            "carries an obvious caveat about how few records it rests on, say "
            "so in the same breath.",
            expected="One or two sentences of plain business English.",
        )
        if not text:
            return result.headline

        # The narration is generated prose, so it goes through the same check
        # as anything else generated. A chat answer is exactly where an
        # invented figure would do the most damage.
        facts = build_fact_base(state)
        facts.add_frame(result.table)
        grounded = strip_unverified(text.strip(), facts)
        if not grounded.text.strip():
            return result.headline
        if grounded.removed_sentences:
            state.log.record(
                EventKind.WARNING,
                "ask",
                "Removed an unsupported claim from a chat answer: "
                + " ".join(grounded.removed_sentences)[:200],
            )
        return grounded.text

    # -- suggestions -------------------------------------------------------

    def suggest(self, state: PipelineState) -> list[str]:
        """Questions this dataset can actually answer.

        Every suggestion is validated as a plan before it is offered, so a
        suggested question is never one that then fails.
        """
        if state.frame is None or state.frame.empty:
            return []

        candidates = self._suggest_from_model(state) or self._suggest_by_shape(state)

        usable: list[str] = []
        for question, raw in candidates:
            try:
                query_module.validate(raw, state.frame, state.profile)
            except PlanError:
                continue
            usable.append(question)
            if len(usable) >= SUGGESTION_COUNT:
                break
        return usable

    def _suggest_from_model(
        self, state: PipelineState
    ) -> list[tuple[str, dict[str, Any]]]:
        if not self.reasoning.available:
            return []

        parsed = self.reasoning.ask_json(
            self.key,
            self.persona,
            "A business owner has just been shown an analysis of their data.\n\n"
            f"{self._schema_block(state)}"
            f"{self._memory_block(state)}\n\n"
            f"Suggest {SUGGESTION_COUNT + 2} follow-up questions they would "
            "realistically ask about their own business, and the calculation "
            "each one becomes. Questions must be about what they should do, not "
            "about the data itself. Use only the column names listed.",
            shape=(
                '[{"question": "Which region grew fastest this year?", '
                f'"plan": {PLAN_SHAPE}}}]'
            ),
        )
        if not isinstance(parsed, list):
            return []

        pairs: list[tuple[str, dict[str, Any]]] = []
        for item in parsed:
            if not isinstance(item, dict):
                continue
            question = str(item.get("question", "")).strip()
            plan = item.get("plan")
            if question and isinstance(plan, dict):
                pairs.append((question, plan))
        return pairs

    def _suggest_by_shape(
        self, state: PipelineState
    ) -> list[tuple[str, dict[str, Any]]]:
        """Suggestions built from the columns present, with no model."""
        from ..analysis.exploration import resolve_columns

        columns = resolve_columns(state.frame, state.profile)
        measure = columns.primary_measure
        if not measure:
            return []

        readable = measure.replace("_", " ")
        pairs: list[tuple[str, dict[str, Any]]] = []

        if columns.primary_date:
            pairs.append(
                (
                    f"How has {readable} moved month by month?",
                    {"measure": measure, "aggregation": "sum", "time_grain": "month"},
                )
            )
        for category in columns.categories[:3]:
            pairs.append(
                (
                    f"Which {category.replace('_', ' ')} brings in the most {readable}?",
                    {"measure": measure, "aggregation": "sum", "group_by": [category]},
                )
            )
            pairs.append(
                (
                    f"What is the average {readable} per {category.replace('_', ' ')}?",
                    {"measure": measure, "aggregation": "mean", "group_by": [category]},
                )
            )
        customer = columns.entity_named("customer", "client", "account")
        if customer:
            pairs.append(
                (
                    f"Who are the largest customers by {readable}?",
                    {
                        "measure": measure,
                        "aggregation": "sum",
                        "group_by": [customer],
                        "limit": 10,
                    },
                )
            )
        return pairs

    # -- prompt helpers ----------------------------------------------------

    @staticmethod
    def _schema_block(state: PipelineState) -> str:
        lines = ["These are the columns in their data:"]
        for column in state.profile.columns:
            if state.frame is not None and column.name not in state.frame.columns:
                continue
            detail = f"- {column.name} ({column.role.value})"
            if column.role is Role.CATEGORY and column.stats.get("top_values"):
                values = list(column.stats["top_values"])[:5]
                detail += f", values include: {', '.join(str(v) for v in values)}"
            elif column.role is Role.DATETIME and column.stats.get("earliest"):
                detail += f", {column.stats['earliest']} to {column.stats['latest']}"
            lines.append(detail)

        engineered = [
            name
            for name in state.engineered_columns
            if state.frame is not None and name in state.frame.columns
        ]
        if engineered:
            lines.append(f"Also available, built during the analysis: {', '.join(engineered)}")
        return "\n".join(lines)

    @staticmethod
    def _memory_block(state: PipelineState) -> str:
        block = state.memory.as_prompt_block()
        if not block:
            return ""
        return (
            "\n\nThe owner has told you the following about their business. "
            f"Treat these as true:\n{block}"
        )
