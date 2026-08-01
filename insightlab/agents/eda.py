"""Exploratory Data Analysis Agent: build the charts that answer the question.

There is no such thing as "explore the data" in the abstract. A sales manager
and a CFO looking at the same file want different charts, so this agent asks
which business question is being asked before drawing anything, and only offers
the questions the data can actually answer.
"""

from __future__ import annotations

from ..analysis.exploration import AXES, available_axes, build_charts
from ..core.decision import Option
from ..core.reasoning import AgentPersona
from ..core.state import PipelineState
from .base import Agent, Flow

#: Why each business question is worth asking, in the owner's terms.
AXIS_REASONS = {
    "sales": "where the money comes from and whether it is growing",
    "customers": "who buys, how often, and how much depends on a few of them",
    "products": "which lines carry the business and which quietly drain it",
    "marketing": "which channels are worth the spend",
    "profits": "what is actually kept, not just what comes in",
    "regions": "where the business is strong and where it is missing",
    "operations": "how the work flows and where it stalls",
}


class ExploratoryAnalysisAgent(Agent):
    stage = "explore"
    key = "eda"
    title = "Exploring the data"

    persona = AgentPersona(
        role="Exploratory analyst",
        goal=(
            "Find the patterns in the data that matter to the part of the "
            "business the owner cares about right now."
        ),
        backstory=(
            "You have produced enough reports nobody read to know that a chart is "
            "only useful if it answers a question someone was already asking. So "
            "you ask which question first, and you say what each chart shows in "
            "one sentence rather than leaving the reader to work it out."
        ),
    )

    def run(self, state: PipelineState) -> Flow:
        state.begin_stage(self.stage)

        if state.frame is None or state.frame.empty:
            state.skip_stage(self.stage, "There is no data to explore.")
            return

        supported = available_axes(state.frame, state.profile)
        decision = self._axis_decision(state, supported)
        answer = yield decision

        if answer.is_skip:
            chosen = []
            self.note(
                state,
                "No particular focus was chosen, so only the general overview "
                "charts were produced.",
            )
        elif answer.is_custom:
            self.capture_custom(state, decision, answer, category="context")
            chosen = self._axes_from_text(state, answer.text, supported)
            self.note(
                state,
                "Read your focus as: "
                + (", ".join(AXES[axis] for axis in chosen) if chosen else "a general overview"),
            )
        else:
            chosen = [axis for axis in answer.payload.get("axes", []) if axis in AXES]

        state.focus_axes = chosen
        charts = build_charts(state.frame, state.profile, chosen, include_general=True)
        state.charts = charts

        for chart in charts:
            self.note(
                state,
                f"Built {chart.title.lower()}.",
                chart_id=chart.id,
                axis=chart.axis,
            )

        if not charts:
            state.finish_stage(
                self.stage,
                "No chart could be produced from this data, which usually means "
                "there is no column holding numbers to compare.",
            )
            return

        state.finish_stage(
            self.stage,
            f"Produced {len(charts)} charts across "
            f"{len({chart.axis for chart in charts})} areas of the business.",
        )

    # -- decision ----------------------------------------------------------

    def _axis_decision(self, state: PipelineState, supported: list[str]):
        listing = "\n".join(
            f"- {AXES[axis]}: {AXIS_REASONS.get(axis, '')}" for axis in supported
        )
        recommended = supported[: min(3, len(supported))]
        recommended_names = ", ".join(AXES[axis] for axis in recommended)

        return self.decide(
            topic="What to focus on",
            question="Which part of your business should we look at most closely?",
            context=(
                "Your data supports these lines of enquiry. Only the ones your "
                "columns can actually answer are listed, so nothing here is a "
                "dead end:\n\n"
                f"{listing}\n\n"
                "Whatever you choose, you still get the general overview charts. "
                "Choosing a focus decides where we go deeper."
            ),
            suggestion=Option(
                label=f"Look at {recommended_names}",
                rationale=(
                    "These are the areas your data covers most completely, so "
                    "they will produce the most reliable answers."
                ),
                payload={"axes": recommended},
            ),
            alternatives=[
                Option(
                    label=f"Focus only on {AXES[axis]}",
                    rationale=f"Goes deep on {AXIS_REASONS.get(axis, AXES[axis].lower())}.",
                    payload={"axes": [axis]},
                )
                for axis in supported
            ]
            + [
                Option(
                    label="Look at everything",
                    rationale=(
                        "Produces every chart the data supports. Thorough, but a "
                        "longer report to read."
                    ),
                    payload={"axes": supported},
                )
            ],
            custom_prompt=(
                "Describe the question you actually want answered, for example: "
                "why did sales drop in the second half of the year, or which "
                "customers are worth keeping."
            ),
            skip_effect=(
                "Only the general overview charts are produced - a trend, a "
                "distribution and a summary of data quality."
            ),
            evidence={"supported": supported},
        )

    def _axes_from_text(
        self, state: PipelineState, text: str, supported: list[str]
    ) -> list[str]:
        """Map a free-text question onto the business areas we can chart."""
        chosen = self.reason(
            f'A business owner was asked what to focus on and answered: "{text}"\n\n'
            f"These areas are available: {', '.join(supported)}."
            f"{self.memory_block(state)}\n\n"
            "Return the one to three areas from that list which best match what "
            "they asked for. Use only the exact names from the list.",
            shape='["sales", "customers"]',
        )
        if isinstance(chosen, list):
            matched = [item for item in chosen if item in supported]
            if matched:
                return matched[:3]

        # No model, or an unusable reply: fall back to keyword matching.
        lowered = text.casefold()
        keywords = {
            "sales": ("sale", "revenue", "income", "turnover", "growth"),
            "customers": ("customer", "client", "buyer", "retention", "loyalty", "churn"),
            "products": ("product", "item", "sku", "category", "line", "service"),
            "marketing": ("marketing", "channel", "campaign", "advert", "discount", "promo"),
            "profits": ("profit", "margin", "cost", "loss", "expense"),
            "regions": ("region", "city", "country", "branch", "area", "location", "store"),
            "operations": ("operation", "delivery", "stock", "fulfil", "status", "process"),
        }
        matched = [
            axis
            for axis in supported
            if any(word in lowered for word in keywords.get(axis, ()))
        ]
        return matched[:3] or supported[:2]
