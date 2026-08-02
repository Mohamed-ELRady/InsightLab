"""The checks a conclusion has to survive before a user sees it.

Four of them, in order, because each is cheaper than the next and the cheap
ones remove work for the expensive ones:

1. **Grounding** - does every figure in the sentence exist in the data?
2. **Significance** - is the difference bigger than the noise?
3. **Confounding** - does a third variable explain it, or reverse it?
4. **Refutation** - can a hostile reader knock it down?

Nothing here deletes a finding quietly. A finding that fails is either demoted
with the objection attached, so the reader can judge it, or removed with the
removal written to the activity log. Silent deletion would make the product
look more confident than it is, which is the failure mode this whole module
exists to prevent.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ..analysis import confounding, significance
from ..core.activity_log import EventKind
from ..core.grounding import FactBase, build_fact_base, strip_unverified
from ..core.reasoning import AgentPersona, ReasoningEngine
from ..core.state import Insight, PipelineState

#: Below this, a finding is dropped rather than shown with a caveat: there is
#: nothing left worth reading.
DROP_BELOW = "none"

CONFIDENCE_ORDER = ["low", "medium", "high"]

CRITIC = AgentPersona(
    role="Sceptical reviewer",
    goal=(
        "Find the reason each conclusion might be wrong, before the owner acts "
        "on it and finds out the expensive way."
    ),
    backstory=(
        "You are the person in the meeting who asks how many records that is "
        "based on. You have seen confident conclusions built on eleven rows, on "
        "a seasonal effect nobody controlled for, and on a difference well "
        "inside the noise. You are not contrarian for its own sake - when a "
        "finding is solid you say so in three words and move on."
    ),
)


def demote(confidence: str, steps: int = 1) -> str:
    """Lower a confidence level without dropping below the floor."""
    try:
        index = CONFIDENCE_ORDER.index(confidence)
    except ValueError:
        index = 1
    return CONFIDENCE_ORDER[max(0, index - steps)]


@dataclass
class VerificationReport:
    """What the checks did, for the run log and the report appendix."""

    checked: int = 0
    figures_checked: int = 0
    sentences_removed: int = 0
    dropped_as_noise: int = 0
    caveats_added: int = 0
    refuted: int = 0
    demoted: int = 0

    def describe(self) -> str:
        parts = [f"{self.checked} conclusions checked"]
        if self.figures_checked:
            parts.append(f"{self.figures_checked} figures matched against the data")
        if self.sentences_removed:
            parts.append(f"{self.sentences_removed} unsupported claims removed")
        if self.dropped_as_noise:
            parts.append(f"{self.dropped_as_noise} dropped as within the noise")
        if self.caveats_added:
            parts.append(f"{self.caveats_added} given a caveat")
        if self.refuted:
            parts.append(f"{self.refuted} challenged on review")
        if self.demoted:
            parts.append(f"{self.demoted} downgraded")
        return "; ".join(parts) + "."


class Verifier:
    """Runs the four checks over a set of conclusions."""

    def __init__(self, reasoning: ReasoningEngine) -> None:
        self.reasoning = reasoning
        self.report = VerificationReport()

    def verify(self, state: PipelineState, insights: list[Insight]) -> list[Insight]:
        facts = build_fact_base(state)
        surviving: list[Insight] = []

        for insight in insights:
            self.report.checked += 1

            kept = self._ground(state, insight, facts)
            if kept is None:
                continue

            kept = self._qualify(state, kept)
            if kept is None:
                continue

            surviving.append(kept)

        self._refute(state, surviving)
        state.log.record(
            EventKind.NOTE, "insights", f"Verification: {self.report.describe()}"
        )
        return surviving

    # -- 1. grounding ------------------------------------------------------

    def _ground(
        self, state: PipelineState, insight: Insight, facts: FactBase
    ) -> Insight | None:
        """Remove any claim carrying a figure that is not in the data."""
        changed = False
        for field_name in ("result", "evidence", "interpretation", "action"):
            original = getattr(insight, field_name)
            if not original:
                continue

            result = strip_unverified(original, facts)
            self.report.figures_checked += result.checked
            if result.is_clean:
                continue

            self.report.sentences_removed += len(result.removed_sentences)
            changed = True
            setattr(insight, field_name, result.text)
            state.log.record(
                EventKind.WARNING,
                "insights",
                f'Removed an unsupported claim from "{insight.title}": '
                + " ".join(result.removed_sentences)[:200],
                figures=[figure.text for figure in result.unverified],
            )

        if changed:
            insight.grounded = False
            insight.confidence = demote(insight.confidence)
            self.report.demoted += 1

        # A finding whose result did not survive has nothing left to say.
        if not insight.result.strip():
            state.log.record(
                EventKind.WARNING,
                "insights",
                f'Dropped "{insight.title}" entirely: none of its figures were '
                "supported by the data.",
            )
            return None
        return insight

    # -- 2 and 3. significance and confounding -----------------------------

    def _qualify(self, state: PipelineState, insight: Insight) -> Insight | None:
        """Test the comparison the finding rests on, if it rests on one."""
        chart = state.chart(insight.chart_id) if insight.chart_id else None
        if chart is None or not chart.is_comparison or state.frame is None:
            return insight

        frame = state.frame
        if chart.group_column not in frame.columns or chart.measure not in frame.columns:
            return insight

        how = "mean" if chart.aggregation in ("", "sum", "mean") else chart.aggregation
        comparison = significance.compare_extremes(
            frame, chart.group_column, chart.measure, how=how
        )

        if comparison is not None and not comparison.is_reportable:
            # Not a defect in the chart - the chart is accurate. It is the
            # conclusion drawn from it that the data will not carry.
            self.report.dropped_as_noise += 1
            state.log.record(
                EventKind.NOTE,
                "insights",
                f'Dropped "{insight.title}": {comparison.describe()}',
            )
            return None

        if comparison is not None:
            insight.confidence = _least(insight.confidence, comparison.confidence)
            if comparison.verdict == "weak":
                insight.evidence = (
                    f"{insight.evidence} {comparison.describe()}".strip()
                )

        confound = confounding.check_extremes(
            frame, chart.group_column, chart.measure, how=how
        )
        if confound is not None and confound.kind != "none":
            insight.caveat = confound.describe()
            insight.confidence = demote(insight.confidence)
            self.report.caveats_added += 1
            self.report.demoted += 1
            state.log.record(
                EventKind.NOTE,
                "insights",
                f'Caveat added to "{insight.title}": {confound.condition_column} '
                f"changes how it reads ({confound.kind}).",
            )
        return insight

    # -- 4. refutation -----------------------------------------------------

    def _refute(self, state: PipelineState, insights: list[Insight]) -> None:
        """Ask a hostile reviewer to knock each finding down."""
        if not insights or not self.reasoning.available:
            return

        listed = "\n".join(
            f"{index + 1}. {item.title} — {item.result} "
            f"(evidence: {item.evidence}; stated confidence: {item.confidence})"
            for index, item in enumerate(insights)
        )
        context = self._evidence_context(state)

        parsed = self.reasoning.ask_json(
            "critic",
            CRITIC,
            "These conclusions were drawn from a business owner's data and are "
            "about to be put in front of them.\n\n"
            f"{listed}\n\n"
            f"Here is what the data actually contains:\n{context}\n\n"
            "For each one, try to knock it down. The objections that matter "
            "are: too few records to support it; a seasonal or one-off event "
            "that explains it; a third factor that would explain it better; the "
            "difference being small enough to be chance; or the conclusion "
            "claiming cause where the data can only show two things moving "
            "together.\n"
            "Only return the ones you can genuinely challenge. If a conclusion "
            "is solid, leave it out entirely - a list of weak objections against "
            "everything is as useless as no review at all. Never invent a number.",
            shape=(
                '[{"number": 1, "objection": "one sentence saying what is wrong '
                'with it", "severity": "fatal|serious|minor"}]'
            ),
        )

        if not isinstance(parsed, list):
            return

        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("number", 0)) - 1
            except (TypeError, ValueError):
                continue
            if not 0 <= index < len(insights):
                continue

            objection = str(item.get("objection", "")).strip()
            if not objection:
                continue

            severity = str(item.get("severity", "minor")).casefold()
            insight = insights[index]
            insight.objection = objection
            insight.confidence = demote(
                insight.confidence, 2 if severity == "fatal" else 1
            )
            self.report.refuted += 1
            self.report.demoted += 1
            state.log.record(
                EventKind.NOTE,
                "insights",
                f'Review challenged "{insight.title}" ({severity}): {objection}',
            )

    @staticmethod
    def _evidence_context(state: PipelineState) -> str:
        """The row counts a reviewer needs to judge whether a finding holds up."""
        lines = [f"Total records after cleaning: {len(state.frame):,}"]
        if state.frame is None:
            return lines[0]

        for chart in state.charts:
            if not chart.is_comparison or chart.group_column not in state.frame.columns:
                continue
            counts = state.frame[chart.group_column].value_counts().head(8)
            listed = ", ".join(f"{name}: {count:,}" for name, count in counts.items())
            lines.append(f"Records per {chart.group_column}: {listed}")

        dates = [
            column.name
            for column in state.profile.columns
            if column.role.value == "datetime"
        ]
        for name in dates[:1]:
            column = state.profile.column(name)
            if column and column.stats.get("earliest"):
                lines.append(
                    f"Period covered: {column.stats['earliest']} to "
                    f"{column.stats['latest']}"
                )

        # Deduplicate while keeping order - several charts share a group column.
        seen: set[str] = set()
        unique = [line for line in lines if not (line in seen or seen.add(line))]
        return "\n".join(unique[:12])


def _least(first: str, second: str) -> str:
    """The lower of two confidence levels."""
    order = {level: index for index, level in enumerate(CONFIDENCE_ORDER)}
    return first if order.get(first, 1) <= order.get(second, 1) else second
