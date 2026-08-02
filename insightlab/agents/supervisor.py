"""Supervisor Agent: owns the run and routes work between the other agents.

The supervisor is the only component that knows whether a run is interactive or
autonomous. Agents raise decisions the same way in both modes; the supervisor
either holds the pipeline open until a human answers, or answers with the
suggestion itself and records that it did so.

It drives the agent generators directly rather than letting them call each
other, which means the whole pipeline can be paused between any two steps and
resumed later without any agent needing to know that happened.
"""

from __future__ import annotations

from typing import Iterator

from ..core.activity_log import EventKind
from ..core.decision import Answer, Decision
from ..core.reasoning import ReasoningEngine
from ..core.state import STAGE_TITLES, PipelineState, RunMode, StageStatus
from .base import Agent, Flow


class Supervisor:
    """Runs a pipeline of agents over one :class:`PipelineState`."""

    def __init__(
        self,
        state: PipelineState,
        agents: list[Agent] | None = None,
        reasoning: ReasoningEngine | None = None,
    ) -> None:
        self.state = state
        self.reasoning = reasoning or ReasoningEngine()
        self.reasoning.set_language(state.language)
        self.agents = agents if agents is not None else build_default_agents(self.reasoning)

        # Not part of the pipeline: it answers questions rather than asking
        # them, and runs against finished state rather than advancing it.
        from .analyst import AnalystAgent

        self.analyst = AnalystAgent(self.reasoning)

        self._flow: Iterator[Decision] | None = None
        self._pending: Decision | None = None
        self._finished = False
        self.current_agent: Agent | None = None

    # -- lifecycle ---------------------------------------------------------

    @property
    def pending(self) -> Decision | None:
        """The decision the pipeline is waiting on, if any."""
        return self._pending

    @property
    def finished(self) -> bool:
        return self._finished

    @property
    def current_stage(self) -> str:
        if self.current_agent is not None:
            return self.current_agent.stage
        return ""

    @property
    def current_title(self) -> str:
        stage = self.current_stage
        return STAGE_TITLES.get(stage, "")

    def progress(self) -> float:
        """Share of stages that have been dealt with, for a progress bar."""
        statuses = list(self.state.stage_status.values())
        if not statuses:
            return 0.0
        settled = sum(
            1
            for status in statuses
            if status in (StageStatus.DONE, StageStatus.SKIPPED, StageStatus.FAILED)
        )
        return settled / len(statuses)

    # -- driving -----------------------------------------------------------

    def start(self) -> Decision | None:
        """Begin the run and return the first decision, if there is one."""
        self._flow = self._pipeline()
        self._pending = None
        self._finished = False
        return self._advance(None)

    def resolve(self, answer: Answer) -> Decision | None:
        """Answer the pending decision and continue to the next one.

        Raises ``RuntimeError`` if there is nothing pending, and ignores an
        answer aimed at a decision that is no longer current - which is what a
        double-submitted form looks like.
        """
        if self._pending is None:
            raise RuntimeError("There is no decision waiting to be answered.")
        if answer.decision_id and answer.decision_id != self._pending.id:
            return self._pending

        answer._decision = self._pending
        self.state.record_answer(self._pending, answer)
        return self._advance(answer)

    # -- questions ---------------------------------------------------------

    def ask(self, question: str):
        """Answer a question against the state, without advancing a stage.

        The pipeline runs forwards only, which is right for an analysis and
        wrong for a conversation. This is the second entry point: it reads the
        finished state, plans a calculation, executes it with pandas and
        returns the answer, touching no stage and consuming no decision. The
        question and its answer are written to the same activity log as
        everything else, so the record of the run stays complete.
        """
        return self.analyst.answer(self.state, question)

    def suggested_questions(self) -> list[str]:
        """Follow-up questions this data can actually answer."""
        return self.analyst.suggest(self.state)

    def explain_change(self, **kwargs):
        """Break a movement down across every dimension. See ``attribution``."""
        from ..analysis import attribution

        return attribution.explain(self.state, **kwargs)

    # -- driving -----------------------------------------------------------

    def run_to_completion(self, max_steps: int = 500) -> PipelineState:
        """Run the whole pipeline, answering every decision automatically.

        Used for autonomous mode and for the tests. The step cap is a guard
        against an agent that never stops raising decisions.
        """
        decision = self.start()
        steps = 0
        while decision is not None and steps < max_steps:
            answer = Answer.accept(decision, automatic=True)
            answer.decision_id = decision.id
            decision = self.resolve(answer)
            steps += 1
        if decision is not None:
            self.state.fail_stage(
                decision.stage,
                "The run was stopped after too many steps without finishing.",
            )
        return self.state

    # -- internals ---------------------------------------------------------

    def _advance(self, answer: Answer | None) -> Decision | None:
        """Push the pipeline forward until it stops or needs a human."""
        if self._flow is None:
            return None

        while True:
            try:
                decision = self._flow.send(answer) if answer is not None else next(self._flow)
            except StopIteration:
                self._pending = None
                self._finished = True
                self.current_agent = None
                return None

            self._pending = decision
            self.state.log.record(
                EventKind.DECISION_RAISED,
                decision.stage,
                f"{decision.topic}: {decision.question}",
                decision_id=decision.id,
            )

            if self.state.mode is RunMode.INTERACTIVE:
                return decision

            # Autonomous mode: take the analyst's own suggestion, record that
            # nobody approved it, and keep going.
            answer = Answer.accept(decision, automatic=True)
            answer.decision_id = decision.id
            self.state.record_answer(decision, answer)

    def _pipeline(self) -> Flow:
        """Walk the agents in order, isolating a failure to its own stage."""
        for agent in self.agents:
            self.current_agent = agent
            try:
                yield from agent.run(self.state)
            except GeneratorExit:  # pragma: no cover - only on early close
                raise
            except Exception as error:  # noqa: BLE001 - one stage must not kill the run
                self.state.fail_stage(
                    agent.stage,
                    f"{STAGE_TITLES.get(agent.stage, agent.stage)} could not be "
                    f"completed: {error}",
                )
        self.current_agent = None


def build_default_agents(reasoning: ReasoningEngine) -> list[Agent]:
    """The standard pipeline, in the order the stages run.

    Imported lazily so that importing the supervisor does not pull in every
    agent module and its dependencies.
    """
    from .dashboard import DashboardAgent
    from .data_cleaning import DataCleaningAgent
    from .data_loader import DataLoaderAgent
    from .data_understanding import DataUnderstandingAgent
    from .eda import ExploratoryAnalysisAgent
    from .feature_engineering import FeatureEngineeringAgent
    from .insight import InsightAgent
    from .memory_agent import MemoryAgent
    from .kpi import KpiAgent
    from .report import ReportAgent

    return [
        DataLoaderAgent(reasoning),
        MemoryAgent(reasoning),
        DataUnderstandingAgent(reasoning),
        DataCleaningAgent(reasoning),
        FeatureEngineeringAgent(reasoning),
        ExploratoryAnalysisAgent(reasoning),
        KpiAgent(reasoning),
        InsightAgent(reasoning),
        DashboardAgent(reasoning),
        ReportAgent(reasoning),
    ]
