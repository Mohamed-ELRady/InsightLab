"""KPI Agent: the handful of numbers this business should watch.

A KPI is only useful if the person reading it knows what it means and what to do
when it moves, so each one is presented with the formula behind it and a
sentence of interpretation. The agent also asks whether the company already
tracks figures of its own - a KPI the business already argues about in meetings
beats a textbook one it has never heard of.
"""

from __future__ import annotations

import re

from ..analysis.metrics import available_kpis, compute_kpis, custom_kpi
from ..core.decision import Option
from ..core.reasoning import AgentPersona
from ..core.state import PipelineState, Role
from .base import Agent, Flow

#: Enough to run a business by, few enough to fit on one screen.
HEADLINE_COUNT = 6


class KpiAgent(Agent):
    stage = "kpis"
    key = "kpi"
    title = "Summarising performance"

    persona = AgentPersona(
        role="Performance analyst",
        goal=(
            "Give the owner the few numbers that actually tell them how the "
            "business is doing, each one explained well enough to act on."
        ),
        backstory=(
            "You have seen dashboards with forty metrics that nobody looks at, "
            "and one number on a whiteboard that changed how a company ran. You "
            "always show the formula, because a figure an owner cannot reconcile "
            "against their own books is a figure they will not trust."
        ),
    )

    def run(self, state: PipelineState) -> Flow:
        state.begin_stage(self.stage)

        if state.frame is None or state.frame.empty:
            state.skip_stage(self.stage, "There is no data to measure.")
            return

        supported = available_kpis(state.frame, state.profile)
        if not supported:
            state.skip_stage(
                self.stage,
                "This data does not contain the columns any standard measure "
                "needs.",
            )
            return

        decision = self._selection_decision(state, supported)
        answer = yield decision

        if answer.is_skip:
            chosen = [item.id for item in supported[:HEADLINE_COUNT]]
            self.note(state, "Kept the standard set of measures.")
        elif answer.is_custom:
            self.capture_custom(state, decision, answer, category="target")
            chosen = [item.id for item in supported]
        else:
            chosen = list(answer.payload.get("ids", []))

        state.kpis = compute_kpis(state.frame, state.profile, chosen)

        if answer.is_custom and answer.text:
            self._add_custom(state, answer.text)

        yield from self._ask_targets(state)

        state.finish_stage(
            self.stage, f"Calculated {len(state.kpis)} performance measures."
        )

    # -- selection ---------------------------------------------------------

    def _selection_decision(self, state: PipelineState, supported):
        listing = "\n".join(f"- {item.name}: {item.why}" for item in supported)
        headline = supported[:HEADLINE_COUNT]

        return self.decide(
            topic="Which measures to track",
            question="Which numbers do you want on the front page?",
            context=(
                "Your data supports these measures. Each one is calculated from "
                "your own columns, and the formula is shown next to every figure "
                "so you can check it against your books:\n\n"
                f"{listing}\n\n"
                "Fewer measures on the front page usually means more of them get "
                "acted on."
            ),
            suggestion=Option(
                label=f"Use the main {len(headline)}",
                rationale=(
                    "Covers what came in, what was kept, the direction of travel "
                    "and the customer base - enough to run a week by, few enough "
                    "to read at a glance."
                ),
                payload={"ids": [item.id for item in headline]},
            ),
            alternatives=[
                Option(
                    label=f"Use all {len(supported)}",
                    rationale=(
                        "Nothing is left out. Better when this is a periodic "
                        "review rather than a daily check."
                    ),
                    payload={"ids": [item.id for item in supported]},
                ),
                Option(
                    label="Money only",
                    rationale=(
                        "Revenue, profit and margin. The narrowest useful set "
                        "when the only question is financial."
                    ),
                    payload={
                        "ids": [
                            item.id
                            for item in supported
                            if item.id
                            in {"total_revenue", "total_profit", "profit_margin", "average_order"}
                        ]
                    },
                ),
                Option(
                    label="Customers only",
                    rationale=(
                        "Base size, repeat rate and concentration. Use when the "
                        "question is about retention rather than sales."
                    ),
                    payload={
                        "ids": [
                            item.id
                            for item in supported
                            if item.id in {"customer_count", "repeat_rate", "concentration"}
                        ]
                    },
                ),
            ],
            custom_prompt=(
                "Does your company rely on any specific measures of its own? "
                "Describe them and which columns they come from, for example: "
                "collection rate is payments received divided by revenue."
            ),
            skip_effect="The standard set of measures is used.",
            evidence={"supported": [item.id for item in supported]},
        )

    # -- the company's own measures ----------------------------------------

    def _add_custom(self, state: PipelineState, text: str) -> None:
        """Build a KPI the company already uses, from their description."""
        numeric = [
            column.name
            for column in state.profile.columns
            if column.role is Role.MEASURE
        ]
        if not numeric:
            return

        parsed = self.reason(
            f'A business owner described a measure their company tracks: "{text}"\n\n'
            f"These numeric columns are available: {', '.join(numeric)}."
            f"{self.memory_block(state)}\n\n"
            "Express their measure as a total of one column, or one column "
            "divided by another. Use only column names from the list. If it "
            "cannot be expressed from these columns, return an empty list.",
            shape=(
                '[{"name": "Collection rate", "numerator": "payments", '
                '"denominator": "revenue", "as_percentage": true}]'
            ),
        )

        definitions = parsed if isinstance(parsed, list) else self._guess_custom(text, numeric)

        added = 0
        for item in definitions:
            if not isinstance(item, dict) or not item.get("name"):
                continue
            kpi = custom_kpi(
                state.frame,
                str(item["name"]),
                str(item.get("numerator", "")),
                str(item["denominator"]) if item.get("denominator") else None,
                as_percentage=bool(item.get("as_percentage")),
            )
            if kpi is not None:
                state.kpis.insert(added, kpi)
                added += 1

        if added:
            self.note(state, f"Added {added} measure(s) of your own to the front page.")
        else:
            self.note(
                state,
                "Your measure was saved to the business memory. It could not be "
                "calculated from the columns in this file, so it does not appear "
                "as a figure.",
            )

    @staticmethod
    def _guess_custom(text: str, numeric: list[str]) -> list[dict]:
        """Pull "A divided by B" out of a sentence with no model available."""
        lowered = text.casefold()
        mentioned = [name for name in numeric if name.casefold() in lowered]
        if not mentioned:
            return []

        divides = any(
            word in lowered for word in ("divided by", "per", "ratio", "rate", "%", "percent")
        )
        if divides and len(mentioned) >= 2:
            name = re.split(r"\bis\b|:|=", text)[0].strip() or "Custom measure"
            return [
                {
                    "name": name[:40],
                    "numerator": mentioned[0],
                    "denominator": mentioned[1],
                    "as_percentage": True,
                }
            ]
        name = re.split(r"\bis\b|:|=", text)[0].strip() or "Custom measure"
        return [{"name": name[:40], "numerator": mentioned[0], "denominator": None}]

    # -- targets -----------------------------------------------------------

    def _ask_targets(self, state: PipelineState) -> Flow:
        """Ask what good looks like.

        A number with no target is a fact; a number with a target is a decision.
        Targets are saved to the business memory so later runs can say whether
        the business is on track rather than just reporting the figure again.
        """
        if not state.kpis:
            return

        listing = "\n".join(
            f"- {kpi.name}: {kpi.display_value}" for kpi in state.kpis[:HEADLINE_COUNT]
        )
        decision = self.decide(
            topic="What good looks like",
            question="Do you have targets for any of these?",
            context=(
                "Here is where the business currently stands:\n\n"
                f"{listing}\n\n"
                "If you tell us what you are aiming for, the report can say "
                "whether you are ahead or behind rather than just stating the "
                "figure. Targets are remembered, so the next analysis of newer "
                "data can compare against them."
            ),
            suggestion=Option(
                label="No targets - just report the figures",
                rationale=(
                    "The report states where the business stands without judging "
                    "it against anything."
                ),
                payload={"targets": False},
            ),
            alternatives=[],
            custom_prompt=(
                "List your targets in plain words, for example: we aim for a 35% "
                "margin and 20% growth."
            ),
            skip_effect="The figures are reported without any target to compare against.",
        )
        answer = yield decision

        if answer.is_custom and answer.text:
            self.capture_custom(state, decision, answer, category="target")
            self.note(
                state,
                "Your targets were saved and will appear alongside the figures in "
                "the report.",
            )
