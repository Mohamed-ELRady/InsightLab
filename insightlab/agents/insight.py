"""Insight Agent: turn charts into domain-appropriate, evidence-backed conclusions.

Every insight produced here carries five things: the result, the evidence behind
it, what it means for the business, how confident we are, and what to do about
it. An observation without an action is trivia, and a claim without evidence is
not something anyone should reorganise their stock around.

The confidence level is deliberately conservative. This data is a single file,
so nothing here proves cause - only that two things move together.
"""

from __future__ import annotations

from ..core.decision import Option
from ..core.reasoning import AgentPersona
from ..core.adaptive import rank_objects
from ..core.state import Chart, Insight, PipelineState
from .base import Agent, Flow
from .verification import Verifier

#: How many insights to aim for. Beyond this they stop being read.
TARGET_INSIGHTS = 8

VALID_CONFIDENCE = {"high", "medium", "low"}


class InsightAgent(Agent):
    stage = "insights"
    key = "insight"
    title = "Drawing conclusions"

    persona = AgentPersona(
        role="Domain-aware insight analyst",
        goal=(
            "Turn what the charts show into useful domain-appropriate conclusions, "
            "and be honest about which ones are solid and which are hints."
        ),
        backstory=(
            "You have spent years explaining numbers to domain experts who are "
            "not statisticians. You never state a finding without "
            "the figure behind it, you never claim one thing caused another when "
            "the data only shows they moved together, and you always end with what "
            "you would do about it."
        ),
    )

    def run(self, state: PipelineState) -> Flow:
        state.begin_stage(self.stage)

        if not state.charts:
            state.skip_stage(
                self.stage, "There are no charts to draw conclusions from."
            )
            return

        insights = self._from_model(state) or self._from_charts(state)

        # A movement since the last run outranks anything in this file alone,
        # so change findings go to the top rather than competing for a slot.
        changes = self._changes_since_last_time(state)
        insights = changes + insights

        # Nothing generated reaches the user unchecked. Grounding, significance,
        # confounding and refutation all run before anything is kept.
        verifier = Verifier(self.reasoning)
        insights = verifier.verify(state, insights)
        # Keep period-over-period movements first; personalise the remaining
        # findings without weakening that established business priority.
        verified_changes = [item for item in insights if item in changes]
        regular = [item for item in insights if item not in changes]
        insights = verified_changes + rank_objects(
            regular, profile=state.learning_profile, policy=state.improvement_policy
        )
        state.insights = insights[:TARGET_INSIGHTS]

        if not state.insights:
            state.finish_stage(
                self.stage,
                "No conclusion survived checking. The charts are still accurate; "
                "the differences in them are not large enough to draw a "
                "conclusion from.",
            )
            return

        yield from self._challenge(state)

        state.finish_stage(
            self.stage,
            f"Drew {len(state.insights)} conclusions. {verifier.report.describe()}",
        )

    # -- what changed since last time --------------------------------------

    def _changes_since_last_time(self, state: PipelineState) -> list[Insight]:
        """Turn material movements against the previous run into findings.

        These are the only findings in the product that use information from
        outside the current file, which makes them both the most valuable and
        the ones most in need of stating their basis plainly.
        """
        comparison = state.comparison
        if comparison is None:
            return []

        label = comparison.label()
        insights: list[Insight] = []
        for change in comparison.material:
            direction = "risen" if change.difference > 0 else "fallen"
            insights.append(
                Insight(
                    title=f"{change.name} has {direction} since {label}",
                    result=change.describe(label),
                    evidence=(
                        f"Measured the same way in both analyses"
                        + (
                            ", rebased to a monthly figure because the two files "
                            "cover different lengths of time"
                            if change.per_period
                            else ""
                        )
                        + "."
                    ),
                    interpretation=(
                        "A movement between two periods is the one thing a single "
                        "file cannot show you. What it does not say is why - that "
                        "needs either the breakdown below or something you know "
                        "about what happened in between."
                    ),
                    confidence="high" if abs(change.relative) >= 0.15 else "medium",
                    action=(
                        f"Decide whether this movement in {change.name.lower()} "
                        "was something you did, and whether you want it to "
                        "continue."
                    ),
                    axis="general",
                )
            )
        return insights

    # -- generation --------------------------------------------------------

    def _from_model(self, state: PipelineState) -> list[Insight]:
        """Ask the model to interpret the chart findings we computed."""
        if not self.reasoning.available:
            return []

        findings = "\n".join(
            f"{index + 1}. [{chart.axis}] {chart.title}: {chart.description}"
            for index, chart in enumerate(state.charts)
        )
        kpi_lines = "\n".join(
            f"- {kpi.name}: {kpi.display_value}" for kpi in state.kpis
        )

        parsed = self.reason(
            "These are measured findings from the user's dataset. "
            "Every number here is already calculated and correct - your job is "
            "interpretation, not calculation.\n\n"
            f"Domain: {state.understanding.business_domain}\n"
            f"Analysis goal: {state.understanding.analysis_goal}\n"
            f"Findings:\n{findings}\n"
            + (f"\nHeadline figures:\n{kpi_lines}\n" if kpi_lines else "")
            + f"{self.memory_block(state)}\n\n"
            f"Write up to {TARGET_INSIGHTS} conclusions that would change what "
            "this owner does. Rules you must follow:\n"
            "- Quote the actual figure from the findings as the evidence. Never "
            "invent a number that is not above.\n"
            "- Explain what it means in this dataset's actual domain. Never inject business, revenue, customer or sales language unless the domain is commercial.\n"
            "- Do not claim one thing caused another. This is a single file of "
            "records, so it can only show that things move together.\n"
            "- Set confidence to high only when the finding rests on a large, "
            "clear difference; medium when it is suggestive; low when it rests "
            "on few rows or a weak pattern.\n"
            "- Every conclusion ends with one specific action, not a "
            "recommendation to investigate further.\n"
            "- Skip anything that is merely a description of the data.",
            shape=(
                '[{"title": "short headline", "result": "what the data shows", '
                '"evidence": "the figure it rests on", "interpretation": "what it '
                'means for the business", "confidence": "high|medium|low", '
                '"action": "one specific thing to do", "axis": "sales"}]'
            ),
            max_output_tokens=1400,
        )

        if not isinstance(parsed, list):
            return []

        insights: list[Insight] = []
        for item in parsed:
            if not isinstance(item, dict) or not item.get("title"):
                continue
            confidence = str(item.get("confidence", "medium")).casefold()
            insights.append(
                Insight(
                    title=str(item["title"]).strip(),
                    result=str(item.get("result", "")).strip(),
                    evidence=str(item.get("evidence", "")).strip(),
                    interpretation=str(item.get("interpretation", "")).strip(),
                    confidence=confidence if confidence in VALID_CONFIDENCE else "medium",
                    action=str(item.get("action", "")).strip(),
                    axis=str(item.get("axis", "general")).strip() or "general",
                    chart_id=self._match_chart(state, item),
                )
            )
        return insights

    @staticmethod
    def _match_chart(state: PipelineState, item: dict) -> str:
        """Link an insight back to the chart that supports it, if we can."""
        text = " ".join(
            str(item.get(field, "")) for field in ("title", "result", "evidence")
        ).casefold()
        best = ""
        best_score = 0
        for chart in state.charts:
            words = {
                word
                for word in chart.title.casefold().replace("_", " ").split()
                if len(word) > 3
            }
            score = sum(1 for word in words if word in text)
            if score > best_score:
                best, best_score = chart.id, score
        return best if best_score >= 2 else ""

    def _from_charts(self, state: PipelineState) -> list[Insight]:
        """Write the insights ourselves when no model is available.

        Each chart already computed a factual description during exploration, so
        this turns those into the standard five-part shape rather than inventing
        anything new.
        """
        insights: list[Insight] = []
        ar = state.language.code == "ar"
        for chart in state.charts:
            business = state.understanding.is_business
            action, confidence = self._action_for(chart, ar, business)
            insights.append(
                Insight(
                    title=chart.title,
                    result=chart.description,
                    evidence=(
                        f"تم القياس من كل صفوف البيانات وعددها {len(state.frame):,}."
                        if ar else f"Measured from all {len(state.frame):,} rows in your data."
                    ),
                    interpretation=self._interpretation_for(chart, ar, business),
                    confidence=confidence,
                    action=action,
                    axis=chart.axis,
                    chart_id=chart.id,
                )
            )
        # Put the focused findings above the general overview ones.
        insights.sort(key=lambda item: item.axis == "general")
        return insights

    @staticmethod
    def _interpretation_for(chart: Chart, arabic: bool = False, business: bool = True) -> str:
        if chart.kind == "line":
            if arabic:
                return "الاتجاه عبر عدة فترات أهم من شهر واحد؛ الحركة المستمرة اتجاه، أما القفزة المنفردة فغالبًا حدث استثنائي."
            return (
                "The direction matters more than any single month. A run of "
                "months moving the same way is a trend; one month apart from the "
                "rest is usually an event."
            )
        if chart.kind == "bar":
            if not business:
                return (
                    "الفروق بين المجموعات تحدد أين يتركز النمط، لكنها لا تثبت أن اسم المجموعة هو سبب الفرق."
                    if arabic else
                    "Differences between groups show where the pattern concentrates, but do not prove that the group label caused it."
                )
            if arabic:
                return "تفوق مجموعة واحدة بفارق كبير يعني أن النشاط يعتمد عليها أكثر مما يبدو؛ وده مصدر قوة ومخاطرة في الوقت نفسه."
            return (
                "Where one group is far ahead of the others, the business depends "
                "on it more than it may realise. That is a strength and a risk at "
                "the same time."
            )
        if chart.kind == "histogram":
            if not business:
                return (
                    "شكل التوزيع يوضح ما هو معتاد وما إذا كانت القيم النادرة جزءًا مهمًا من الظاهرة وليست أخطاء تلقائيًا."
                    if arabic else
                    "The distribution separates typical observations from rare extremes; rare values may be important events, not automatic errors."
                )
            if arabic:
                return "الفرق بين القيمة المعتادة والمتوسط يوضح هل عدد قليل من السجلات الكبيرة هو الذي يرفع نتيجة النشاط كله."
            return (
                "The gap between the typical value and the average tells you "
                "whether a few large records are setting the tone for the whole "
                "business."
            )
        if chart.kind == "box":
            if arabic:
                return "قد يكون لمجموعتين نفس المتوسط لكن بسلوك مختلف تمامًا؛ التفاوت هو الذي يوضح أيهما أكثر استقرارًا."
            return (
                "Two groups with the same average can behave completely "
                "differently. The spread is what tells you which one is "
                "predictable."
            )
        if chart.kind == "heatmap":
            if not business:
                return (
                    "الخريطة تكشف العلاقات والتركيبات الأقوى، ويجب تفسيرها وفق طبيعة المجال وحجم العينة."
                    if arabic else
                    "The grid reveals the strongest relationships or combinations, which still need domain context and sample-size checks."
                )
            if arabic:
                return "الفرص تظهر في التركيبات: منتج قوي في منطقة ضعيفة قد يشير إلى مشكلة توزيع وليس ضعف طلب."
            return (
                "Combinations are where opportunity hides: a strong product in a "
                "weak region is usually a distribution problem, not a demand one."
            )
        if arabic:
            return "تحرك مقياسين معًا يستحق المتابعة، لكنه لا يثبت أن أحدهما تسبب في الآخر."
        return (
            "Two measures moving together is a lead worth following, but it is "
            "not proof that one causes the other."
        )

    @staticmethod
    def _action_for(chart: Chart, arabic: bool = False, business: bool = True) -> tuple[str, str]:
        if chart.kind == "line":
            return (
                "راجع ما حدث في أعلى وأقل فترة وحدد هل السبب قرار داخلي أم عامل خارجي."
                if arabic else
                "Check what happened in the peak and the trough month, and see whether it was something you did or something outside.",
                "medium",
            )
        if chart.kind == "bar":
            if not business:
                return (
                    "تحقق من حجم كل مجموعة ثم قارن المجموعة الأعلى بالأدنى على مقياس ثانٍ ذي صلة."
                    if arabic else
                    "Check each group size, then compare the highest and lowest groups on a second relevant measure.",
                    "medium",
                )
            return (
                "قرر هل تحافظ على المجموعة المتصدرة أم تستثمر في المجموعات المتأخرة؛ القرارين صحيحين لكن لكل واحد ميزانية مختلفة."
                if arabic else
                "Decide whether to defend the leading group or invest in the ones behind it - both are valid, but they need different budgets.",
                "high",
            )
        if chart.kind == "box":
            return (
                "ابدأ بالمجموعة ذات التفاوت الأكبر؛ علاج عدم الاستقرار غالبًا أسهل من رفع متوسط منخفض."
                if arabic else
                "Look at the group with the widest spread first: inconsistency is usually easier to fix than a low average.",
                "medium",
            )
        if chart.kind == "heatmap":
            if not business:
                return (
                    "تحقق من أقوى علاقة على البيانات الخام وقارنها بالفترات أو المجموعات المختلفة."
                    if arabic else "Validate the strongest relationship against the raw observations and across periods or groups.",
                    "medium",
                )
            return (
                "اختر أقوى تركيبة واختبر إمكانية تكرارها في مناطق أو منتجات أخرى."
                if arabic else "Pick the strongest combination and check whether it can be repeated elsewhere.",
                "medium",
            )
        if chart.kind == "scatter":
            return (
                "اختبر العلاقة على نطاق صغير قبل افتراض أنها ستستمر."
                if arabic else "Test the relationship on a small scale before assuming it will hold.",
                "low",
            )
        return (
            "قارن النتيجة بتوقعاتك؛ الفارق بينهما هو نقطة البداية للقرار."
            if arabic else "Compare this against what you expected. The gap is where the useful conversation is.",
            "medium",
        )

    # -- the owner's turn --------------------------------------------------

    def _challenge(self, state: PipelineState) -> Flow:
        """Give the owner the chance to overrule what the numbers suggest.

        This is the point of the whole product: we know what the data says, they
        know why. A correction here is saved permanently and changes how later
        analyses read the same pattern.
        """
        headline = "\n".join(
            f"- {insight.title}: {insight.result}" for insight in state.insights[:5]
        )
        ar = state.language.code == "ar"
        business = state.understanding.is_business

        decision = self.decide(
            topic="مراجعة الاستنتاجات" if ar else "Checking the conclusions",
            question=(
                "هل يوجد أي استنتاج يتعارض مع معرفتك بالبيانات أو المجال؟"
                if ar and not business else
                "هل يوجد أي استنتاج يتعارض مع معرفتك بنشاطك؟"
                if ar else
                "Does anything here contradict what you know about the data or its domain?"
                if not business else "Does anything here contradict what you know about your own business?"
            ),
            context=(
                ("دي أهم النتائج اللي الأرقام بتوضحها:\n\n" + f"{headline}\n\n"
                 "إحنا شايفين المسجل في الملف فقط، لكن إنت تعرف سياق المجال والظروف الحقيقية. لو فيه تفسير أو تعريف ناقص، قوله لنا علشان نصحح التقرير ونفتكره في التحليلات القادمة.")
                if ar else
                ("These are what the numbers say:\n\n"
                f"{headline}\n\n"
                "We can only see what was recorded in the file. You know the "
                "domain context, collection conditions and definitions. If a "
                "missing piece changes a conclusion, telling us now corrects "
                "the report and can become an approved lesson for later analyses.")
            ),
            suggestion=Option(
                label="النتائج متوافقة مع توقعاتي" if ar else "These match what I would expect",
                rationale=(
                    "ستدخل الاستنتاجات التقرير كما هي من غير تغيير درجات الثقة."
                    if ar else "The conclusions go into the report as they are, with their confidence levels unchanged."
                ),
                payload={"confirmed": True},
            ),
            alternatives=[
                Option(
                    label="اعتبر كل النتائج محتاجة مراجعة أدق" if ar else "Mark them all as needing a closer look",
                    rationale=(
                        "سيتم خفض الثقة في كل الاستنتاجات حتى لا يتم اتخاذ قرار قبل مراجعتها."
                        if ar else "Every conclusion is set to low confidence in the report, so nobody acts on them before they are checked."
                    ),
                    payload={"confirmed": False, "downgrade": True},
                ),
            ],
            custom_prompt=(
                ("اشرح الخطأ أو المعلومة الناقصة؛ مثلًا: الحد العلمي الصحيح مختلف، أو فترة الرصد كانت غير مكتملة."
                 if not business else "اشرح الخطأ أو المعلومة الناقصة؛ مثلًا: نوفمبر موسم الذروة بسبب معرض، أو المنطقة دي فيها فرع واحد فقط.")
                if ar else
                ("Tell us what is wrong or missing, for example: the accepted domain threshold is different, or the observation period was incomplete."
                 if not business else "Tell us what is wrong or what we are missing, for example: November is always the peak because of a trade fair, or that region has only one shop.")
            ),
            skip_effect=("ستدخل الاستنتاجات التقرير كما هي." if ar else "The conclusions go into the report exactly as written."),
        )
        answer = yield decision

        if answer.is_skip:
            return

        if answer.is_custom and answer.text:
            self.capture_custom(state, decision, answer, category="context")
            self._revise(state, answer.text)
            return

        if answer.payload.get("downgrade"):
            for insight in state.insights:
                insight.confidence = "low"
            self.note(
                state,
                "Every conclusion was marked low confidence at your request.",
            )

    def _revise(self, state: PipelineState, correction: str) -> None:
        """Fold the owner's correction back into the written conclusions."""
        current = "\n".join(
            f"{index + 1}. {insight.title}: {insight.result} "
            f"(confidence {insight.confidence})"
            for index, insight in enumerate(state.insights)
        )

        parsed = self.reason(
            "You wrote these conclusions from a user's dataset:\n\n"
            f"{current}\n\n"
            f'The owner has now told you: "{correction}"'
            f"{self.memory_block(state)}\n\n"
            "They know the dataset and its domain context, so their explanation wins "
            "over the pattern in the numbers. Return the conclusions their "
            "correction affects, with the interpretation rewritten to account for "
            "it, and the confidence lowered where their explanation means the "
            "pattern is not what it appeared to be. Leave out any conclusion the "
            "correction does not touch.",
            shape=(
                '[{"number": 1, "interpretation": "rewritten meaning", '
                '"confidence": "high|medium|low", "action": "revised action"}]'
            ),
            max_output_tokens=900,
        )

        if not isinstance(parsed, list):
            # Without a model we cannot rewrite prose, but the correction must
            # still change something visible, so the affected claims are flagged.
            for insight in state.insights:
                insight.interpretation += (
                    f" Note from the user: {correction}"
                )
            self.note(
                state,
                "Your correction was attached to the conclusions and saved to the "
                "project memory.",
            )
            return

        revised = 0
        for item in parsed:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("number", 0)) - 1
            except (TypeError, ValueError):
                continue
            if not 0 <= index < len(state.insights):
                continue

            insight = state.insights[index]
            if item.get("interpretation"):
                insight.interpretation = str(item["interpretation"]).strip()
            if item.get("action"):
                insight.action = str(item["action"]).strip()
            confidence = str(item.get("confidence", "")).casefold()
            if confidence in VALID_CONFIDENCE:
                insight.confidence = confidence
            revised += 1

        self.note(
            state,
            f"Your correction changed {revised} of the conclusions and was saved "
            "for future analyses.",
        )
