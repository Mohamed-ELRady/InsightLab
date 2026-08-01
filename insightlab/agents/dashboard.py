"""Dashboard Agent: assemble the charts into views built for one reader.

The same charts arranged for a CEO and for an operations manager are two
different products. A CEO wants four numbers and a trend; an operations manager
wants the spread and the exceptions. So the agent asks who this is for and lays
out accordingly, then adds one dashboard per business area explored.
"""

from __future__ import annotations

from ..core.decision import Option
from ..core.reasoning import AgentPersona
from ..core.state import Dashboard, DashboardPanel, PipelineState
from ..analysis.exploration import AXES
from .base import Agent, Flow

#: What each audience opens a dashboard to find out.
AUDIENCES: dict[str, tuple[str, str]] = {
    "ceo": (
        "Chief executive",
        "the direction of the whole business in under a minute",
    ),
    "sales": (
        "Sales manager",
        "which products, regions and channels are carrying the number",
    ),
    "marketing": (
        "Marketing manager",
        "which channels and campaigns return more than they cost",
    ),
    "finance": (
        "Finance director",
        "what is actually kept after costs, and where margin is leaking",
    ),
    "operations": (
        "Operations manager",
        "where volume concentrates and where the process is inconsistent",
    ),
}

#: The KPIs each audience cares about, most important first.
AUDIENCE_KPIS: dict[str, tuple[str, ...]] = {
    "ceo": ("Total revenue", "Total profit", "Profit margin", "Growth rate"),
    "sales": (
        "Total revenue",
        "Average order value",
        "Growth rate",
        "Active customers",
    ),
    "marketing": (
        "Active customers",
        "Repeat customer rate",
        "Average order value",
        "Top 10% customer share",
    ),
    "finance": ("Total revenue", "Total profit", "Profit margin", "Average order value"),
    "operations": ("Units sold", "Number of records", "Average order value", "Strongest month"),
}

#: The business areas each audience's charts are drawn from.
AUDIENCE_AXES: dict[str, tuple[str, ...]] = {
    "ceo": ("general", "sales", "profits"),
    "sales": ("sales", "products", "regions", "customers"),
    "marketing": ("marketing", "customers", "products"),
    "finance": ("profits", "sales", "products"),
    "operations": ("operations", "products", "regions"),
}

#: A dashboard longer than this stops being a dashboard and becomes a report.
MAX_PANELS = 8


class DashboardAgent(Agent):
    stage = "dashboard"
    key = "dashboard"
    title = "Building dashboards"

    persona = AgentPersona(
        role="Dashboard designer",
        goal=(
            "Lay the findings out so the person opening them sees what they need "
            "in the first few seconds, without scrolling or interpreting."
        ),
        backstory=(
            "You have watched executives close dashboards after four seconds and "
            "operations managers dig through them for an hour. You design for "
            "whichever one is actually going to open it, and you never put a "
            "chart on a page just because it exists."
        ),
    )

    def run(self, state: PipelineState) -> Flow:
        state.begin_stage(self.stage)

        if not state.charts and not state.kpis:
            state.skip_stage(self.stage, "There is nothing to put on a dashboard.")
            return

        decision = self._audience_decision(state)
        answer = yield decision

        if answer.is_skip:
            audience = "ceo"
            self.note(state, "Built the standard overview, aimed at a general reader.")
        elif answer.is_custom:
            self.capture_custom(state, decision, answer, category="context")
            audience = self._audience_from_text(answer.text)
            self.note(
                state,
                f"Read your description as a dashboard for a "
                f"{AUDIENCES[audience][0].lower()}.",
            )
        else:
            audience = answer.payload.get("audience", "ceo")

        dashboards = [self._overview(state, audience)]
        dashboards.extend(self._per_area(state))
        state.dashboards = [board for board in dashboards if board.panels]

        for board in state.dashboards:
            self.note(
                state,
                f"Built the {board.title} view with {len(board.panels)} panels.",
                dashboard_id=board.id,
            )

        state.finish_stage(
            self.stage, f"Built {len(state.dashboards)} dashboards."
        )

    # -- decision ----------------------------------------------------------

    def _audience_decision(self, state: PipelineState):
        listing = "\n".join(
            f"- {name}: wants to see {wants}" for name, wants in AUDIENCES.values()
        )
        return self.decide(
            topic="Who the dashboard is for",
            question="Who will actually open this dashboard?",
            context=(
                "The same findings get laid out differently depending on who is "
                "reading them, and the wrong layout is the main reason "
                "dashboards go unused:\n\n"
                f"{listing}\n\n"
                "Whoever you choose, a separate dashboard is still built for each "
                "area we explored, so nothing is lost."
            ),
            suggestion=Option(
                label="A chief executive or business owner",
                rationale=(
                    "Puts four headline figures and the trend at the top, with "
                    "the detail underneath. The safest layout when more than one "
                    "person will open it."
                ),
                payload={"audience": "ceo"},
            ),
            alternatives=[
                Option(
                    label=f"A {name.lower()}",
                    rationale=f"Leads with {wants}.",
                    payload={"audience": key},
                )
                for key, (name, wants) in AUDIENCES.items()
                if key != "ceo"
            ],
            custom_prompt=(
                "Describe who will use this and what decision they need to make "
                "with it."
            ),
            skip_effect="A general overview is built, aimed at a business owner.",
        )

    @staticmethod
    def _audience_from_text(text: str) -> str:
        lowered = text.casefold()
        keywords = {
            "ceo": ("ceo", "owner", "founder", "director", "board", "executive", "chief"),
            "sales": ("sales", "commercial", "account manager", "revenue team"),
            "marketing": ("marketing", "campaign", "brand", "growth team", "advertis"),
            "finance": ("finance", "cfo", "accountant", "controller", "margin", "cost"),
            "operations": ("operations", "logistics", "warehouse", "fulfil", "supply", "stock"),
        }
        for audience, words in keywords.items():
            if any(word in lowered for word in words):
                return audience
        return "ceo"

    # -- layout ------------------------------------------------------------

    def _overview(self, state: PipelineState, audience: str) -> Dashboard:
        """The main dashboard, ordered for the chosen reader."""
        name, wants = AUDIENCES[audience]
        panels: list[DashboardPanel] = []

        wanted_kpis = AUDIENCE_KPIS.get(audience, ())
        chosen = [name for name in wanted_kpis if state.kpi(name) is not None]
        # Top up from whatever else was calculated, so the row is never half empty.
        for kpi in state.kpis:
            if len(chosen) >= 4:
                break
            if kpi.name not in chosen:
                chosen.append(kpi.name)
        panels.extend(DashboardPanel("kpi", kpi_name) for kpi_name in chosen[:4])

        preferred = AUDIENCE_AXES.get(audience, ("general",))
        charts = [chart for chart in state.charts if chart.axis in preferred]
        if not charts:
            charts = list(state.charts)

        # A trend first if there is one: the direction of travel is what the
        # first glance is for.
        charts.sort(key=lambda chart: (chart.kind != "line", chart.axis not in preferred))
        panels.extend(
            DashboardPanel("chart", chart.id)
            for chart in charts[: MAX_PANELS - len(panels)]
        )

        return Dashboard(
            id="overview",
            title=f"Overview for a {name.lower()}",
            audience=name,
            description=(
                f"Built for someone who wants {wants}. The figures across the top "
                "are the state of the business; the charts below show what is "
                "behind them."
            ),
            panels=panels,
        )

    def _per_area(self, state: PipelineState) -> list[Dashboard]:
        """One dashboard per business area that produced charts."""
        by_axis: dict[str, list] = {}
        for chart in state.charts:
            by_axis.setdefault(chart.axis, []).append(chart)

        boards: list[Dashboard] = []
        for axis, charts in by_axis.items():
            if axis == "general":
                continue
            boards.append(
                Dashboard(
                    id=f"area_{axis}",
                    title=AXES.get(axis, axis.title()),
                    audience=AXES.get(axis, axis.title()),
                    description=(
                        f"Everything the data says about "
                        f"{AXES.get(axis, axis).lower()}, in one place."
                    ),
                    panels=[
                        DashboardPanel("chart", chart.id) for chart in charts[:MAX_PANELS]
                    ],
                )
            )
        return boards
