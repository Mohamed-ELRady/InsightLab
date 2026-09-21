"""Exploratory Data Analysis Agent: build charts that fit the dataset's domain.

There is no such thing as "explore the data" in the abstract. A sales manager
and a CFO looking at the same file want different charts, so this agent asks
which business question is being asked before drawing anything, and only offers
the questions the data can actually answer.
"""

from __future__ import annotations

import pandas as pd

from ..analysis.exploration import AXES, available_axes, build_charts
from ..core.decision import Option
from ..core.reasoning import AgentPersona
from ..core.language import axis_label
from ..core.adaptive import rank_objects, rank_values
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
    "time": "how event frequency and measurements change over time",
    "comparisons": "which groups differ and by how much",
    "distributions": "what is typical and which values are unusual",
    "relationships": "which measurements move together",
    "locations": "where observations and events concentrate",
    "quality": "which conclusions need caution because data is incomplete",
}


class ExploratoryAnalysisAgent(Agent):
    stage = "explore"
    key = "eda"
    title = "Exploring the data"

    persona = AgentPersona(
        role="Exploratory analyst",
        goal=(
            "Find the patterns that matter to the user's question in the "
            "dataset's detected domain."
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

        supported = available_axes(state.frame, state.profile, state.understanding)
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

        if state.analysis_directive:
            requested = self._axes_from_text(
                state, state.analysis_directive, supported
            )
            if requested:
                chosen = requested
                self.note(
                    state,
                    "Rebuilt the analysis around the focus requested in the final review: "
                    + ", ".join(AXES[axis] for axis in requested),
                )

        state.focus_axes = chosen
        charts = build_charts(
            state.frame, state.profile, chosen, include_general=True,
            understanding=state.understanding,
        )
        if state.language.code == "ar":
            self._localize_charts(charts)
        state.charts = rank_objects(
            charts, profile=state.learning_profile, policy=state.improvement_policy
        )

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
            f"{len({chart.axis for chart in charts})} analysis areas.",
        )

    @staticmethod
    def _number(value: float) -> str:
        magnitude = abs(value)
        if magnitude >= 1_000_000_000:
            return f"{value / 1_000_000_000:,.1f} مليار"
        if magnitude >= 1_000_000:
            return f"{value / 1_000_000:,.1f} مليون"
        if magnitude >= 1_000:
            return f"{value / 1_000:,.1f} ألف"
        return f"{value:,.2f}"

    @classmethod
    def _localize_charts(cls, charts) -> None:
        """Give deterministic charts native Arabic titles and explanations.

        Column names and values stay untouched so users can still find them in
        the uploaded sheet. Only wording created by InsightLab is translated.
        """
        for chart in charts:
            table = chart.table
            measure = (chart.measure or chart.id.split("_")[1] if "_" in chart.id else "القيمة")
            readable = measure.replace("_", " ")
            group = chart.group_column.replace("_", " ") if chart.group_column else "المجموعة"

            if chart.kind == "line" and table is not None and len(table) >= 2:
                values = pd.to_numeric(table.iloc[:, -1], errors="coerce").dropna()
                first, last = float(values.iloc[0]), float(values.iloc[-1])
                change = ((last - first) / first * 100) if first else 0.0
                direction = "ارتفع" if change > 2 else "انخفض" if change < -2 else "استقر"
                peak_index = int(values.to_numpy().argmax())
                peak_period = str(table.iloc[peak_index, 0])
                chart.title = f"{readable} عبر الزمن"
                chart.description = (
                    f"{direction} {readable} بنسبة {abs(change):.0f}% خلال الفترة، "
                    f"وسجل أعلى قيمة في {peak_period} عند {cls._number(float(values.max()))}."
                )
                table.rename(columns={table.columns[0]: "الفترة"}, inplace=True)
            elif chart.kind == "bar" and table is not None and len(table):
                values = pd.to_numeric(table.iloc[:, -1], errors="coerce")
                valid = values.dropna()
                if not valid.empty:
                    top_index, low_index = valid.idxmax(), valid.idxmin()
                    top = str(table.loc[top_index, table.columns[0]])
                    low = str(table.loc[low_index, table.columns[0]])
                    top_value, low_value = float(valid.loc[top_index]), float(valid.loc[low_index])
                    share = top_value / float(valid.sum()) * 100 if valid.sum() else 0
                    chart.description = (
                        f"تتصدر {top} بقيمة {cls._number(top_value)}"
                        + (f" وتمثل {share:.0f}% من الإجمالي" if chart.aggregation == "sum" else "")
                        + f"، بينما تسجل {low} أقل قيمة عند {cls._number(low_value)}."
                    )
                chart.title = f"{readable} حسب {group}"
            elif chart.kind == "histogram" and table is not None and len(table) >= 3:
                values = pd.to_numeric(table.iloc[:, -1], errors="coerce")
                median = float(values.iloc[2])
                chart.title = f"توزيع {readable}"
                chart.description = f"نصف قيم {readable} تقع عند {cls._number(median)} أو أقل، ويعرض الرسم شكل توزيع كل القيم."
                table.rename(columns={table.columns[0]: "النقطة"}, inplace=True)
            elif chart.id == "correlation":
                chart.title = "العلاقة بين المقاييس"
                chart.description = "يوضح الرسم المقاييس التي تتحرك معًا بقوة؛ الاقتراب من +1 يعني حركة في نفس الاتجاه، والاقتراب من -1 يعني اتجاهين متعاكسين."
                if table is not None and "Measure" in table.columns:
                    table.rename(columns={"Measure": "المقياس"}, inplace=True)
            elif chart.kind == "box" and table is not None and len(table):
                median_column = "median" if "median" in table.columns else table.columns[2]
                medians = pd.to_numeric(table[median_column], errors="coerce")
                top_index, low_index = medians.idxmax(), medians.idxmin()
                top = str(table.loc[top_index, table.columns[0]])
                low = str(table.loc[low_index, table.columns[0]])
                chart.title = f"انتشار {readable} داخل كل {group}"
                chart.description = f"القيمة المعتادة لـ{readable} هي الأعلى في {top} والأقل في {low}. ارتفاع كل صندوق يوضح التفاوت داخل المجموعة الذي قد تخفيه المتوسطات."
            elif chart.kind == "heatmap":
                chart.title = "خريطة تركّز القيم" if chart.id != "correlation" else chart.title
                if chart.id != "correlation":
                    chart.description = "الخلايا الأغمق توضح أين تتركز أعلى القيم، والخلايا الأفتح تكشف الفجوات أو الفرص الأقل استغلالًا."
            elif chart.kind == "scatter" and table is not None and table.shape[1] >= 2:
                left, right = table.columns[:2]
                score = pd.to_numeric(table[left], errors="coerce").corr(
                    pd.to_numeric(table[right], errors="coerce")
                )
                chart.title = f"{str(left).replace('_', ' ')} مقابل {str(right).replace('_', ' ')}"
                chart.description = f"كل نقطة تمثل صفًا واحدًا، ودرجة العلاقة بين المقياسين {score:+.2f}."

            if chart.id == "missing_values" and table is not None and len(table):
                chart.title = "اكتمال بيانات الأعمدة"
                chart.description = f"العمود الأقل اكتمالًا هو {table.iloc[0, 0]}، وتبلغ نسبة القيم الفارغة فيه {table.iloc[0, 1]}%."
                table.rename(columns={"Column": "العمود", "Percent empty": "نسبة القيم الفارغة"}, inplace=True)

    # -- decision ----------------------------------------------------------

    def _axis_decision(self, state: PipelineState, supported: list[str]):
        ar = state.language.code == "ar"
        business = state.understanding.is_business
        supported = rank_values(
            supported, "axes", state.learning_profile, state.improvement_policy
        )
        listing = "\n".join(
            f"- {axis_label(axis, state.language)}"
            + (f": {AXIS_REASONS.get(axis, '')}" if not ar else "")
            for axis in supported
        )
        recommended = supported[: min(3, len(supported))]
        recommended_names = "، ".join(axis_label(axis, state.language) for axis in recommended)

        return self.decide(
            topic="محور التحليل" if ar else "What to focus on",
            question=(
                "أي سؤال تريد أن نركز عليه داخل هذه البيانات؟"
                if ar and not business else
                "أي جزء من نشاطك تريد أن نحلله بتركيز أكبر؟"
                if ar else
                "Which question in this dataset should we examine most closely?"
                if not business else
                "Which part of your business should we look at most closely?"
            ),
            context=(
                ("بياناتك تسمح بتحليل المجالات التالية، وكل مجال ظاهر له أعمدة تدعمه فعلًا:\n\n"
                 f"{listing}\n\nستحصل على رسومات النظرة العامة في كل الحالات، واختيارك يحدد أين نتعمق.")
                if ar else
                ("Your data supports these lines of enquiry. Only the ones your "
                "columns can actually answer are listed, so nothing here is a "
                "dead end:\n\n"
                f"{listing}\n\n"
                "Whatever you choose, you still get the general overview charts. "
                "Choosing a focus decides where we go deeper.")
            ),
            suggestion=Option(
                label=(f"ركّز على {recommended_names}" if ar else f"Look at {recommended_names}"),
                rationale=(
                    "هذه المجالات هي الأكثر اكتمالًا في بياناتك، لذلك نتائجها ستكون الأكثر موثوقية."
                    if ar else "These are the areas your data covers most completely, so they will produce the most reliable answers."
                ),
                payload={"axes": recommended},
            ),
            alternatives=[
                Option(
                    label=((f"ركّز فقط على {axis_label(axis, state.language)}") if ar else f"Focus only on {AXES[axis]}"),
                    rationale=("تحليل متعمق لهذا المجال." if ar else f"Goes deep on {AXIS_REASONS.get(axis, AXES[axis].lower())}."),
                    payload={"axes": [axis]},
                )
                for axis in supported
            ]
            + [
                Option(
                    label="حلّل كل المجالات" if ar else "Look at everything",
                    rationale=(
                        "ينشئ كل الرسومات التي تدعمها البيانات؛ أشمل لكن التقرير سيكون أطول."
                        if ar else "Produces every chart the data supports. Thorough, but a longer report to read."
                    ),
                    payload={"axes": supported},
                )
            ],
            custom_prompt=(
                "اكتب السؤال الذي تريد إجابته بلغتك؛ سنربطه بالأعمدة والمحاور التي تستطيع البيانات إثباتها."
                if ar and not business else
                "اكتب السؤال الذي تريد إجابته، مثل: لماذا انخفضت المبيعات في النصف الثاني؟ أو أي العملاء أهم للاحتفاظ بهم؟"
                if ar else
                "Describe the question you want answered; we will map it to the columns and analyses the data can support."
                if not business else
                "Describe the question you actually want answered, for example: why did sales drop in the second half of the year, or which customers are worth keeping."
            ),
            skip_effect=(
                "سيتم إنشاء رسومات النظرة العامة فقط: الاتجاه والتوزيع وملخص جودة البيانات."
                if ar else "Only the general overview charts are produced - a trend, a distribution and a summary of data quality."
            ),
            evidence={"supported": supported},
        )

    def _axes_from_text(
        self, state: PipelineState, text: str, supported: list[str]
    ) -> list[str]:
        """Map a free-text question onto the business areas we can chart."""
        lowered = text.casefold()
        keywords = {
            "sales": ("sale", "revenue", "income", "turnover", "growth"),
            "customers": ("customer", "client", "buyer", "retention", "loyalty", "churn"),
            "products": ("product", "item", "sku", "category", "line", "service"),
            "marketing": ("marketing", "channel", "campaign", "advert", "discount", "promo"),
            "profits": ("profit", "margin", "cost", "loss", "expense"),
            "regions": ("region", "city", "country", "branch", "area", "location", "store"),
            "operations": ("operation", "delivery", "stock", "fulfil", "status", "process"),
            "time": ("time", "date", "month", "year", "trend", "frequency", "when", "زمن", "وقت", "تغير", "تكرار"),
            "comparisons": ("compare", "group", "type", "category", "difference", "فرق", "مقارنة", "مجموعة", "نوع"),
            "distributions": ("distribution", "typical", "unusual", "outlier", "range", "توزيع", "معتاد", "شاذ", "نطاق"),
            "relationships": ("relationship", "correlation", "affect", "versus", "علاقة", "ارتباط", "مقابل"),
            "locations": ("place", "location", "region", "map", "where", "مكان", "موقع", "منطقة", "أين"),
            "quality": ("missing", "quality", "complete", "error", "ناقص", "جودة", "اكتمال", "خطأ"),
        }
        matched = [
            axis for axis in supported
            if any(word in lowered for word in keywords.get(axis, ()))
        ]
        if matched:
            return matched[:3]

        chosen = self.reason(
            f'A user was asked what to focus on in this dataset and answered: "{text}"\n\n'
            f"These areas are available: {', '.join(supported)}."
            f"{self.memory_block(state)}\n\n"
            "Return the one to three areas from that list which best match what "
            "they asked for. Use only the exact names from the list.",
            shape='["sales", "customers"]',
            max_output_tokens=200,
        )
        if isinstance(chosen, list):
            matched = [item for item in chosen if item in supported]
            if matched:
                return matched[:3]

        # No model, or an unusable reply: fall back to keyword matching.
        return supported[:2]
