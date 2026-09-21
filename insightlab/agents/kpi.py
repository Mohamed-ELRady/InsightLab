"""KPI Agent: the handful of numbers that best describe this dataset.

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
from ..core.adaptive import rank_objects
from ..core.state import PipelineState, Role
from .base import Agent, Flow

#: Enough to run a business by, few enough to fit on one screen.
HEADLINE_COUNT = 6

KPI_NAMES_AR = {
    "Total revenue": "إجمالي الإيرادات",
    "Total profit": "إجمالي الأرباح",
    "Profit margin": "هامش الربح",
    "Average order value": "متوسط قيمة الطلب",
    "Growth rate": "معدل النمو",
    "Strongest month": "أقوى شهر",
    "Active customers": "العملاء النشطون",
    "Repeat customer rate": "معدل تكرار العملاء",
    "Top 10% customer share": "حصة أكبر 10% من العملاء",
    "Units sold": "الوحدات المباعة",
    "Number of records": "عدد السجلات",
    "Number of earthquake events": "عدد الأحداث الزلزالية",
    "Average magnitude": "متوسط قوة الزلازل",
    "Median magnitude": "القيمة الوسيطة لقوة الزلازل",
    "Maximum magnitude": "أقصى قوة مسجلة",
    "Maximum recorded depth": "أقصى عمق مسجل",
}


def _arabic_kpi_name(name: str) -> str:
    if name in KPI_NAMES_AR:
        return KPI_NAMES_AR[name]
    prefixes = {
        "Average ": "متوسط ",
        "Median ": "القيمة الوسيطة لـ",
        "Maximum ": "أقصى قيمة لـ",
        "Number of ": "عدد ",
    }
    for prefix, translated in prefixes.items():
        if name.startswith(prefix):
            return translated + name[len(prefix):]
    return name


def _localize_contextual_kpi(kpi) -> None:
    original_name = kpi.name
    formula = kpi.formula
    kpi.name = _arabic_kpi_name(original_name)
    if formula == "Count of rows after cleaning":
        kpi.formula = "عدد الصفوف بعد التنظيف"
        kpi.interpretation = "عدد الأحداث أو السجلات التي يستند إليها التحليل بعد تطبيق قرارات التنظيف."
        return
    translations = {
        "Average of valid values in ": "متوسط القيم الصالحة في ",
        "Median of valid values in ": "وسيط القيم الصالحة في ",
        "Maximum of valid values in ": "أقصى قيمة صالحة في ",
        "Maximum valid value in ": "أقصى قيمة صالحة في ",
    }
    for prefix, translated in translations.items():
        if formula.startswith(prefix):
            kpi.formula = translated + formula[len(prefix):]
            break
    kpi.interpretation = "ملخص وصفي للقياس داخل هذه البيانات؛ يُقرأ مع التوزيع والسياق وليس كإجمالي قابل للجمع."


class KpiAgent(Agent):
    stage = "kpis"
    key = "kpi"
    title = "Summarising performance"

    persona = AgentPersona(
        role="Domain-aware measurement analyst",
        goal=(
            "Choose the few valid numbers that best describe this dataset in "
            "its detected domain, each explained well enough to use."
        ),
        backstory=(
            "You have seen dashboards with forty metrics that nobody looks at, "
            "and one well-chosen measure that changed how a team understood a "
            "problem. You always show the formula and never sum a non-additive "
            "scientific measurement just because it is numeric."
        ),
    )

    def run(self, state: PipelineState) -> Flow:
        state.begin_stage(self.stage)

        if state.frame is None or state.frame.empty:
            state.skip_stage(self.stage, "There is no data to measure.")
            return

        supported = available_kpis(state.frame, state.profile, state.understanding)
        if not supported:
            state.skip_stage(
                self.stage,
                "This data does not contain the columns any standard measure "
                "needs.",
            )
            return

        supported = rank_objects(
            supported, profile=state.learning_profile,
            policy=state.improvement_policy, name_attr="name",
        )

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

        state.kpis = compute_kpis(
            state.frame, state.profile, chosen, state.understanding
        )
        if state.language.code == "ar" and not state.understanding.is_business:
            for kpi in state.kpis:
                _localize_contextual_kpi(kpi)

        if answer.is_custom and answer.text:
            self._add_custom(state, answer.text)

        state.kpis = rank_objects(
            state.kpis, profile=state.learning_profile,
            policy=state.improvement_policy, name_attr="name",
        )

        self._compare_with_last_time(state)

        yield from self._ask_targets(state)

        summary = f"Calculated {len(state.kpis)} performance measures."
        if state.comparison is not None:
            summary += (
                f" Compared them with your analysis of "
                f"{state.comparison.label()}."
            )
        state.finish_stage(self.stage, summary)

    # -- against last time -------------------------------------------------

    def _compare_with_last_time(self, state: PipelineState) -> None:
        """Put these figures against the last run over the same kind of file.

        A figure on its own is a fact; the same figure against last month is a
        decision. This is what makes the product worth opening a second time.
        """
        from ..analysis import comparison as comparison_module

        if not state.fingerprint:
            state.fingerprint = comparison_module.fingerprint(state.raw_frame)
        previous = comparison_module.find_previous(state, state.workspace)
        if previous is None:
            self.note(
                state,
                "This is the first analysis of a file shaped like this, so there "
                "is nothing yet to compare it against. The next one will be "
                "measured against this.",
            )
            return

        result = comparison_module.compare(state, previous)
        if result is None:
            return

        state.comparison = result
        self.note(state, result.describe(), previous_run=result.previous_run_id)

    # -- selection ---------------------------------------------------------

    def _selection_decision(self, state: PipelineState, supported):
        ar = state.language.code == "ar"
        business = state.understanding.is_business
        listing = "\n".join(
            f"- {_arabic_kpi_name(item.name)}"
            + ("" if ar else f": {item.why}")
            for item in supported
        )
        headline = supported[:HEADLINE_COUNT]

        return self.decide(
            topic="المؤشرات المطلوب متابعتها" if ar else "Which measures to track",
            question="أي أرقام تريد ظهورها في الصفحة الرئيسية؟" if ar else "Which numbers do you want on the front page?",
            context=(
                ("بياناتك تدعم المؤشرات التالية. كل مؤشر محسوب من أعمدتك وتظهر طريقة حسابه بجواره:\n\n"
                 f"{listing}\n\nعدد أقل من المؤشرات الرئيسية يجعل متابعتها واتخاذ قرار منها أسهل.")
                if ar else
                ("Your data supports these measures. Each one is calculated from "
                "your own columns, and the formula is shown next to every figure "
                "so you can check it against your books:\n\n"
                f"{listing}\n\n"
                "Fewer measures on the front page usually means more of them get "
                "acted on.")
            ),
            suggestion=Option(
                label=(f"استخدم أهم {len(headline)} مؤشرات" if ar else f"Use the main {len(headline)}"),
                rationale=(
                    "تغطي الإيرادات والأرباح والاتجاه والعملاء في مجموعة سهلة القراءة."
                    if ar else "Covers what came in, what was kept, the direction of travel and the customer base - enough to run a week by, few enough to read at a glance."
                ),
                payload={"ids": [item.id for item in headline]},
            ),
            alternatives=[
                Option(
                    label=(f"استخدم كل المؤشرات ({len(supported)})" if ar else f"Use all {len(supported)}"),
                    rationale=(
                        "لن يتم استبعاد أي مؤشر؛ مناسب للمراجعة الدورية الشاملة."
                        if ar else "Nothing is left out. Better when this is a periodic review rather than a daily check."
                    ),
                    payload={"ids": [item.id for item in supported]},
                ),
            ] + ([
                Option(
                    label="المؤشرات المالية فقط" if ar else "Money only",
                    rationale=(
                        "الإيرادات والأرباح والهامش فقط."
                        if ar else "Revenue, profit and margin. The narrowest useful set when the only question is financial."
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
                    label="مؤشرات العملاء فقط" if ar else "Customers only",
                    rationale=(
                        "حجم قاعدة العملاء ومعدل التكرار والتركيز."
                        if ar else "Base size, repeat rate and concentration. Use when the question is about retention rather than sales."
                    ),
                    payload={
                        "ids": [
                            item.id
                            for item in supported
                            if item.id in {"customer_count", "repeat_rate", "concentration"}
                        ]
                    },
                ),
            ] if business else []),
            custom_prompt=(
                "هل يوجد مقياس أو حد مهم في هذا المجال؟ اشرحه وحدد الأعمدة المستخدمة."
                if ar and not business else
                "هل يعتمد نشاطك على مؤشر خاص؟ اشرحه وحدد الأعمدة المستخدمة؛ مثل: معدل التحصيل = المدفوعات ÷ الإيرادات."
                if ar else
                "Is there a field-specific measure or threshold you want tracked? Describe it and the columns it uses."
                if not business else
                "Does your company rely on any specific measures of its own? Describe them and which columns they come from, for example: collection rate is payments received divided by revenue."
            ),
            skip_effect=("سيتم استخدام مجموعة المؤشرات القياسية." if ar else "The standard set of measures is used."),
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
            f'A user described a domain-specific measure they want to track: "{text}"\n\n'
            f"These numeric columns are available: {', '.join(numeric)}."
            f"{self.memory_block(state)}\n\n"
            "Express their measure as a total of one column, or one column "
            "divided by another. Use only column names from the list. If it "
            "cannot be expressed from these columns, return an empty list.",
            shape=(
                '[{"name": "Collection rate", "numerator": "payments", '
                '"denominator": "revenue", "as_percentage": true}]'
            ),
            max_output_tokens=600,
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
                "Your measure was saved to the project memory. It could not be "
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
        Targets are saved to project memory so later runs can compare against
        them rather than merely reporting the figure again.
        """
        if not state.kpis:
            return

        ar = state.language.code == "ar"
        business = state.understanding.is_business
        listing = "\n".join(
            f"- {_arabic_kpi_name(kpi.name) if ar else kpi.name}: {kpi.display_value}"
            for kpi in state.kpis[:HEADLINE_COUNT]
        )
        decision = self.decide(
            topic="الأهداف المطلوبة" if ar else "What good looks like",
            question="هل عندك أهداف محددة لأي مؤشر من دول؟" if ar else "Do you have targets for any of these?",
            context=(
                (("ده الوضع الحالي للنشاط:" if business else "ده ملخص القياسات الحالية:")
                 + "\n\n" + f"{listing}\n\n"
                 + "لو شاركت أهدافك أو الحدود المهمة، يقدر التقرير يقارن النتائج بها. هتتحفظ للتحليلات القادمة.")
                if ar else
                (("Here is where the business currently stands:\n\n" if business else "Here are the current measurements:\n\n")
                + f"{listing}\n\n"
                "If you provide a target or meaningful threshold, the report "
                "can compare the result against it rather than only stating the "
                "figure. It is remembered for later analyses.")
            ),
            suggestion=Option(
                label="مفيش أهداف — اعرض الأرقام فقط" if ar else "No targets - just report the figures",
                rationale=(
                    "سيعرض التقرير الوضع الحالي من غير مقارنته بهدف."
                    if ar else "The report states the measurements without judging them against a target."
                ),
                payload={"targets": False},
            ),
            alternatives=[],
            custom_prompt=(
                ("اكتب أهدافك ببساطة؛ مثلًا: نستهدف هامش ربح 35% ونمو 20%." if business
                 else "اكتب الهدف أو الحد المهم ببساطة؛ مثلًا: اعتبر القوة 6 فأكثر حدثًا شديدًا.")
                if ar else
                ("List your targets in plain words, for example: we aim for a 35% margin and 20% growth." if business
                 else "Describe the target or threshold, for example: treat magnitude 6 or above as a severe event.")
            ),
            skip_effect=("سيتم عرض المؤشرات من غير أهداف للمقارنة." if ar else "The figures are reported without any target to compare against."),
        )
        answer = yield decision

        if answer.is_custom and answer.text:
            self.capture_custom(state, decision, answer, category="target")
            self.note(
                state,
                "Your targets were saved and will appear alongside the figures in "
                "the report.",
            )
