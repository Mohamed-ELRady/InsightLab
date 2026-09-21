"""Data Cleaning Agent: duplicates, outliers and columns.

Cleaning is where an automated tool most often destroys the thing that mattered.
A row that looks like a duplicate can be two genuine sales in the same minute;
a value that looks like an outlier can be the biggest order of the year. So each
of the three sub-flows shows the actual rows, explains what each option would do
to the totals, and then asks.

Nothing is changed silently. Every operation that runs is written to the
activity log with its effect on the row count.
"""

from __future__ import annotations

import pandas as pd

from ..analysis import cleaning
from ..analysis.profiling import ensure_profile
from ..core.decision import Option
from ..core.reasoning import AgentPersona
from ..core.state import PipelineState, Role
from .base import Agent, Flow

#: Never ask about more than this many outlier columns in one run. Beyond it the
#: user stops reading and starts clicking, which is worse than not asking.
MAX_OUTLIER_QUESTIONS = 3

#: Same limit for per-column questions.
MAX_COLUMN_QUESTIONS = 6


class DataCleaningAgent(Agent):
    stage = "clean"
    key = "cleaning"
    title = "Cleaning the data"

    persona = AgentPersona(
        role="Data quality consultant",
        goal=(
            "Get the data into a state where its analysis can be trusted, without "
            "throwing away any observation that is valid in its domain."
        ),
        backstory=(
            "You have learned that the row an algorithm calls an error may be the "
            "rare event that matters most. You never delete anything on your own "
            "judgement when the user is available to ask, and you always say what "
            "a change would do to the evidence before you "
            "make it."
        ),
    )

    def run(self, state: PipelineState) -> Flow:
        state.begin_stage(self.stage)

        if state.frame is None or state.frame.empty:
            state.skip_stage(self.stage, "There is no data to clean.")
            return

        rows_before = len(state.frame)
        columns_before = state.frame.shape[1]

        yield from self._duplicates_flow(state)
        yield from self._outliers_flow(state)
        yield from self._columns_flow(state)

        ensure_profile(state)
        rows_after = len(state.frame)
        columns_after = state.frame.shape[1]

        state.finish_stage(
            self.stage,
            f"Cleaning finished: {rows_before:,} rows became {rows_after:,}, "
            f"{columns_before} columns became {columns_after}.",
        )

    # -- duplicates --------------------------------------------------------

    def _duplicates_flow(self, state: PipelineState) -> Flow:
        report = cleaning.find_duplicates(state.frame)
        if not report.found:
            self.note(state, "No duplicate rows were found.")
            return

        share = report.total / len(state.frame) * 100
        preview = report.example.head(6).to_string(index=False, max_colwidth=16)
        ar = state.language.code == "ar"

        decision = self.decide(
            topic="الصفوف المكررة" if ar else "Duplicate rows",
            question=(
                f"يوجد {report.total:,} صفًا مطابقًا تمامًا لصف آخر. ماذا نفعل بها؟"
                if ar else
                f"{report.total:,} rows are exact copies of another row. What should we do with them?"
            ),
            context=(
                (f"تمثل {share:.1f}% من البيانات ضمن {report.groups:,} مجموعات مكررة. "
                 "غالبًا ينتج التكرار عن تصدير الملف مرتين أو دمج نظامين، وده قد يضاعف العدّ أو الإجماليات. "
                 "لكن لو كل صف يمثل ملاحظة حقيقية يمكن أن تتكرر بنفس القيم، فالحذف هيقلل عدد المشاهدات بشكل غير صحيح.\n\n"
                 f"نماذج من الصفوف:\n{preview}")
                if ar else
                (f"That is {share:.1f}% of your data, across {report.groups:,} "
                "repeated groups. Duplicates usually come from a file being "
                "exported twice or two systems being merged, and they inflate "
                "every total by the amount they repeat.\n\n"
                "But if each row is a real observation that can legitimately "
                "repeat with identical values, deleting it would undercount the "
                "observations.\n\n"
                f"Here are some of them:\n{preview}")
            ),
            suggestion=Option(
                label="احذف النسخ واحتفظ بصف واحد من كل مجموعة" if ar else "Delete the copies and keep one of each",
                rationale=(
                    (f"ستنخفض الإجماليات بقيمة {report.total:,} صفًا، وده الصحيح لو السبب تصدير مكرر.")
                    if ar else
                    (f"Totals drop by the value of {report.total:,} rows and become correct if these came from a double export, which is the usual cause.")
                ),
                payload={"action": "delete"},
            ),
            alternatives=[
                Option(
                    label="احتفظ بكل الصفوف كما هي" if ar else "Keep every row as it is",
                    rationale=(
                        "اختار ده لو ملاحظتان متطابقتان ممكن تحصلان فعلًا في مجال البيانات."
                        if ar else "Choose this if two identical observations can genuinely occur in this domain. Nothing changes."
                    ),
                    payload={"action": "keep"},
                ),
            ] + ([
                Option(
                    label="ادمجها في صف واحد واجمع القيم" if ar else "Merge them into one row and add the amounts up",
                    rationale=(
                        "سيقل عدد الصفوف مع بقاء إجماليات المقاييس القابلة للجمع كما هي. لا تستخدمه مع القياسات العلمية غير القابلة للجمع."
                        if ar else "The row count drops while additive totals stay the same. Do not use this for non-additive scientific measurements."
                    ),
                    payload={"action": "merge"},
                ),
            ] if state.understanding.is_business else []),
            custom_prompt=(
                "اشرح كيف يجب التعامل مع التكرار في مجال البيانات، مثل الأعمدة التي تحدد هوية الملاحظة فعلًا."
                if ar else "Explain how duplicates should be treated in this domain, for example which columns define one observation's identity."
            ),
            skip_effect=("ستظل الصفوف المكررة وستدخل في الإجماليات أكثر من مرة." if ar else "The duplicates stay in and every total includes them twice."),
            evidence={"total": report.total, "groups": report.groups},
        )
        answer = yield decision

        if answer.is_skip:
            self.note(state, f"{report.total:,} duplicate rows were left in place.")
            return

        if answer.is_custom:
            self.capture_custom(state, decision, answer, category="definition")
            action = self._interpret_duplicate_text(answer.text)
            self.note(
                state,
                f'Read your instruction as: {action} the duplicate rows.',
            )
        else:
            action = answer.payload.get("action", "delete")

        if action == "keep":
            self.note(state, f"{report.total:,} duplicate rows were kept deliberately.")
            state.remember(
                "Identical rows are genuine separate transactions in this "
                "business and must not be merged or deleted.",
                category="definition",
                stage=self.stage,
                topic=decision.topic,
            )
            return

        if action == "merge":
            operation = cleaning.merge_duplicates(state.frame, state.profile)
        else:
            operation = cleaning.drop_duplicates(state.frame)
        self.apply_operation(state, operation)

    @staticmethod
    def _interpret_duplicate_text(text: str) -> str:
        lowered = text.casefold()
        if any(word in lowered for word in ("keep", "leave", "don't", "do not", "real")):
            return "keep"
        if any(word in lowered for word in ("merge", "combine", "add up", "sum")):
            return "merge"
        return "delete"

    # -- outliers ----------------------------------------------------------

    def _outliers_flow(self, state: PipelineState) -> Flow:
        reports = cleaning.outlier_candidates(state.frame, state.profile)
        if not reports:
            self.note(state, "No unusual values needed a decision.")
            return

        for report in reports[:MAX_OUTLIER_QUESTIONS]:
            if report.column not in state.frame.columns:
                continue
            yield from self._one_outlier_column(state, report)

    def _one_outlier_column(self, state: PipelineState, report) -> Flow:
        ar = state.language.code == "ar"
        highs = ", ".join(f"{value:,.2f}" for value in report.high_values[:6])
        lows = ", ".join(f"{value:,.2f}" for value in report.low_values[:6])
        listed = []
        if highs:
            listed.append(f"unusually high: {highs}")
        if lows:
            listed.append(f"unusually low: {lows}")
        preview = report.rows.head(5).to_string(index=False, max_colwidth=16)

        decision = self.decide(
            topic=(f"قيم غير معتادة في {report.column}" if ar else f"Unusual values in {report.column}"),
            question=(
                (f"هناك {report.count:,} قيمة في {report.column} بعيدة جدًا عن النطاق المعتاد. هل تعتبرها طبيعية في هذا المجال؟")
                if ar else
                (f"{report.count:,} values in {report.column} are far outside the normal range. Are they plausible in this domain?")
            ),
            context=(
                (f"معظم قيم {report.column} تقع بين {report.lower_bound:,.2f} و{report.upper_bound:,.2f}. "
                 f"هناك {report.count:,} قيمة خارج النطاق ({report.share:.1%} من الصفوف). "
                 "القرار يعتمد على طبيعة البيانات: قيمة مكتوبة بالخطأ ستشوّه المتوسطات، لكن حذف حدث نادر حقيقي قد يخفي أهم ما في الظاهرة.\n\n"
                 f"نماذج من الصفوف:\n{preview}")
                if ar else
                (f"Most values in {report.column} sit between "
                f"{report.lower_bound:,.2f} and {report.upper_bound:,.2f}. "
                f"These {report.count:,} do not ({report.share:.1%} of the rows). "
                + "; ".join(listed)
                + ".\n\nThe answer depends on the dataset's domain, and it "
                "matters: leaving a mistyped value in will drag every average "
                "with it, while deleting a genuine rare event may hide the most "
                "important observation.\n\n"
                f"Here are some of the rows involved:\n{preview}")
            ),
            suggestion=Option(
                label="القيم غير معتادة لكنها حقيقية — احتفظ بها وعلّمها" if ar else "These are unusual but real - keep them and mark them",
                rationale=(
                    "لن نفقد أي بيانات، وسنعلّم الصفوف لمراجعتها منفصلة. ده الاختيار الآمن لو مش متأكد."
                    if ar else "Nothing is lost and the rows are tagged so you can look at them separately. This is the safe answer when you are not sure."
                ),
                payload={"action": "flag"},
            ),
            alternatives=[
                Option(
                    label="دي أخطاء إدخال — احذف الصفوف" if ar else "These are data-entry errors - remove those rows",
                    rationale=(
                        (f"سيتم حذف {report.count:,} صفًا. اختار ده فقط لو القيم مستحيل تكون حقيقية.")
                        if ar else (f"{report.count:,} rows are removed. Choose this only if the values could not possibly be real, such as a price typed with an extra zero.")
                    ),
                    payload={"action": "remove"},
                ),
                Option(
                    label="دي قيم نادرة لكنها طبيعية — احتفظ بها كما هي" if ar else "These are rare but plausible - keep them as they are",
                    rationale=(
                        "لن يتغير شيء. مناسب لو القيم حقيقية ولازم التحليل يعكس الظاهرة كاملة."
                        if ar else "Nothing changes. Correct when the observations are genuine and the analysis should preserve the full phenomenon."
                    ),
                    payload={"action": "keep"},
                ),
                Option(
                    label="تأثير موسمي أو استثنائي — حدّدها عند النطاق الطبيعي" if ar else "This is a seasonal or one-off effect - cap them at the normal range",
                    rationale=(
                        "الصفوف ستظل موجودة، لكن القيم المتطرفة ستتوقف عند حد النطاق الطبيعي حتى لا تسيطر على المتوسطات."
                        if ar else "The rows stay, but the extreme numbers are pulled back to the edge of normal so they stop dominating every average. Your totals will drop slightly."
                    ),
                    payload={"action": "cap"},
                ),
            ],
            custom_prompt=(
                "اشرح معنى القيم في مجال البيانات وإيه اللي نعمله فيها."
                if ar else "Explain what these values mean in this domain and what should happen to them."
            ),
            skip_effect=(
                (f"ستظل القيم غير المعتادة وعددها {report.count:,} كما هي وستؤثر على المتوسطات.")
                if ar else (f"The {report.count:,} unusual values stay exactly as they are and will pull the averages with them.")
            ),
            evidence={
                "column": report.column,
                "count": report.count,
                "share": report.share,
            },
        )
        answer = yield decision

        if answer.is_skip:
            self.note(
                state, f"Unusual values in {report.column} were left untouched."
            )
            return

        if answer.is_custom:
            self.capture_custom(state, decision, answer, category="context")
            action = self._interpret_outlier_text(answer.text)
            self.note(
                state,
                f"Read your explanation of {report.column} as: {action}.",
            )
        else:
            action = answer.payload.get("action", "flag")

        if action == "keep":
            state.remember(
                f"Very high values in {report.column} are normal for this "
                "dataset's domain and are not errors.",
                category="definition",
                stage=self.stage,
                topic=decision.topic,
            )
            self.note(state, f"Unusual values in {report.column} were confirmed as real.")
            return

        if action == "remove":
            operation = cleaning.remove_outliers(state.frame, report)
        elif action == "cap":
            operation = cleaning.cap_outliers(state.frame, report)
        else:
            operation = cleaning.flag_outliers(state.frame, report)
        self.apply_operation(state, operation)

    @staticmethod
    def _interpret_outlier_text(text: str) -> str:
        lowered = text.casefold()
        if any(word in lowered for word in ("error", "mistake", "typo", "wrong", "remove", "delete")):
            return "remove"
        if any(word in lowered for word in ("cap", "limit", "seasonal", "promotion", "one-off")):
            return "cap"
        if any(word in lowered for word in ("normal", "real", "genuine", "expensive", "bulk", "wholesale")):
            return "keep"
        return "flag"

    # -- columns -----------------------------------------------------------

    def _columns_flow(self, state: PipelineState) -> Flow:
        profile = ensure_profile(state)

        needs_attention = []
        for column in profile.columns:
            action, reason, payload = cleaning.suggest_column_action(
                column, profile.row_count
            )
            if action != "keep":
                needs_attention.append((column, action, reason, payload))

        if not needs_attention:
            self.note(state, "Every column looked usable as it is.")
            return

        # Deal with the worst first, so a user who stops answering has still
        # fixed the columns that mattered most.
        needs_attention.sort(key=lambda item: item[0].missing_rate, reverse=True)

        for column, action, reason, payload in needs_attention[:MAX_COLUMN_QUESTIONS]:
            if column.name not in state.frame.columns:
                continue
            yield from self._one_column(state, column, action, reason, payload)

    def _one_column(self, state: PipelineState, column, action, reason, payload) -> Flow:
        ar = state.language.code == "ar"
        samples = ", ".join(str(value) for value in column.sample_values[:4])
        alternatives = self._column_alternatives(column, action, ar)

        decision = self.decide(
            topic=(f"عمود {column.name}" if ar else f"The {column.name} column"),
            question=(f"ماذا نفعل بعمود {column.name}؟" if ar else f"What should we do with the {column.name} column?"),
            context=(
                (f"يحتوي على قيم من نوع {column.role.value} مثل {samples}. "
                 f"هناك {column.missing_count:,} صفوف فارغة من أصل {len(state.frame):,} ({column.missing_rate:.0%})، "
                 f"وفيه {column.unique_count:,} قيمة مختلفة.")
                if ar else
                (f"It holds {column.role.value} values such as {samples}. "
                f"{column.missing_count:,} of {len(state.frame):,} rows are empty "
                f"({column.missing_rate:.0%}), and there are "
                f"{column.unique_count:,} different values in it."
                + (f" {column.note.capitalize()}." if column.note else ""))
            ),
            suggestion=Option(
                label=self._action_label(action, column, payload, ar),
                rationale=("ده الإجراء الأنسب بناءً على نوع العمود ونسبة القيم الفارغة." if ar else reason.capitalize() + "."),
                payload=payload,
            ),
            alternatives=alternatives,
            custom_prompt=(
                (f"اشرح معنى {column.name} في مجال البيانات وطريقة التعامل معه. يمكنك طلب تغيير الاسم أو الحذف أو الإبقاء أو ملء الفراغات بصفر.")
                if ar else (f"Tell us what {column.name} means in this dataset and how it should be handled. You can also say \"rename it\", \"remove it\", \"keep it\" or \"fill blanks with 0\".")
            ),
            skip_effect=(
                (f"سيظل {column.name} كما هو، بما في ذلك القيم الفارغة.")
                if ar else (f"{column.name} is left exactly as it is, empty values included.")
            ),
            evidence={"column": column.to_dict()},
        )
        answer = yield decision

        if answer.is_skip:
            return

        if answer.is_custom:
            self.capture_custom(state, decision, answer, category="definition")
            instruction = self._interpret_column_text(answer.text)
            if instruction is None:
                # The note was worth keeping but did not name an operation, and
                # guessing at one would change the data on an assumption.
                self.note(
                    state,
                    f"Your note about {column.name} was saved. The column itself "
                    "was left unchanged, since the note did not ask for a "
                    "specific change.",
                )
                return
            payload = instruction
        else:
            payload = answer.payload

        operation = cleaning.apply_column_action(state.frame, column.name, payload)
        self.apply_operation(state, operation)

    @staticmethod
    def _interpret_column_text(text: str) -> dict | None:
        """Turn a free-text instruction into a column operation.

        Returns ``None`` when the text is a description rather than an
        instruction, so the caller can save it and change nothing.
        """
        lowered = text.casefold().strip()

        for marker in ("rename to ", "rename it to ", "call it ", "name it "):
            if marker in lowered:
                new_name = text[lowered.index(marker) + len(marker):].strip().strip("\"'.")
                if new_name:
                    return {"action": "rename", "new_name": new_name}

        if any(word in lowered for word in ("remove", "delete", "drop", "ignore", "not needed")):
            return {"action": "drop"}

        if "fill" in lowered or "replace" in lowered:
            if "zero" in lowered or " 0" in lowered:
                return {"action": "fill", "strategy": "zero"}
            if "average" in lowered or "mean" in lowered:
                return {"action": "fill", "strategy": "mean"}
            if "median" in lowered or "middle" in lowered:
                return {"action": "fill", "strategy": "median"}
            if "common" in lowered or "mode" in lowered:
                return {"action": "fill", "strategy": "mode"}

        if "as a date" in lowered or "is a date" in lowered:
            return {"action": "convert", "target": "date"}
        if "as a number" in lowered or "is a number" in lowered:
            return {"action": "convert", "target": "number"}

        if any(word in lowered for word in ("keep", "leave", "as it is", "don't change")):
            return {"action": "keep"}

        return None

    @staticmethod
    def _action_label(action: str, column, payload: dict, arabic: bool = False) -> str:
        if action == "drop":
            return f"احذف عمود {column.name}" if arabic else f"Remove the {column.name} column"
        if action == "fill":
            strategy = payload.get("strategy", "median")
            if strategy == "median":
                return "املأ القيم الفارغة بالقيمة الوسطى" if arabic else "Fill the empty values with the middle value"
            return (f'اجمع القيم الفارغة تحت "{strategy}"' if arabic else f'Group the empty values under "{strategy}"')
        if action == "convert":
            return (f"حوّل {column.name} إلى {payload.get('target', 'number')}" if arabic else f"Convert {column.name} to a {payload.get('target', 'number')}")
        return f"احتفظ بـ{column.name} كما هو" if arabic else f"Keep {column.name} as it is"

    @staticmethod
    def _column_alternatives(column, chosen: str, arabic: bool = False) -> list[Option]:
        options: list[Option] = []

        if chosen != "keep":
            options.append(
                Option(
                    label=(f"احتفظ بـ{column.name} كما هو" if arabic else f"Keep {column.name} exactly as it is"),
                    rationale=("لن يتغير شيء، بما في ذلك القيم الفارغة." if arabic else "Nothing changes, empty values included."),
                    payload={"action": "keep"},
                )
            )

        if chosen != "drop":
            options.append(
                Option(
                    label=(f"احذف عمود {column.name}" if arabic else f"Remove the {column.name} column"),
                    rationale=(
                        "اختار ده لو العمود غير مفيد لسؤال التحليل؛ سيختفي من كل الرسومات والتقارير."
                        if arabic else "Use this if the column is irrelevant to the analysis question. It disappears from every chart and report."
                    ),
                    payload={"action": "drop"},
                )
            )

        if column.missing_count:
            options.append(
                Option(
                    label=(f"احذف الصفوف التي يكون فيها {column.name} فارغًا" if arabic else f"Remove the rows where {column.name} is empty"),
                    rationale=(
                        (f"سيتم حذف {column.missing_count:,} صفًا بالكامل بكل قيمها.")
                        if arabic else (f"{column.missing_count:,} rows are removed entirely, including their values in every other column.")
                    ),
                    payload={"action": "drop_missing"},
                )
            )
            if column.role is Role.MEASURE and chosen != "fill":
                options.append(
                    Option(
                        label="املأ القيم الفارغة بصفر" if arabic else "Fill the empty values with zero",
                        rationale=(
                            "مناسب لما تكون الخانة الفارغة معناها إن مفيش شيء حدث، مش إن القيمة لم تُسجّل."
                            if arabic else "Correct when an empty cell genuinely means nothing happened, rather than that nobody recorded it."
                        ),
                        payload={"action": "fill", "strategy": "zero"},
                    )
                )

        if column.role is Role.TEXT:
            options.append(
                Option(
                    label=(f"اعتبر {column.name} مجموعات" if arabic else f"Treat {column.name} as a set of groups"),
                    rationale=(
                        "سيصبح متاحًا للتجميع والمقارنة في الرسومات."
                        if arabic else "Makes it available for grouping and comparison in the charts, which free text cannot be used for."
                    ),
                    payload={"action": "convert", "target": "category"},
                )
            )

        if column.role in (Role.TEXT, Role.CATEGORY) and column.name.casefold().endswith(
            ("date", "time")
        ):
            options.append(
                Option(
                    label=(f"اقرأ {column.name} كتاريخ" if arabic else f"Read {column.name} as a date"),
                    rationale=(
                        "يتيح المقارنة حسب الشهر والربع والموسم."
                        if arabic else "Unlocks month, quarter and season comparisons that text dates cannot support."
                    ),
                    payload={"action": "convert", "target": "date"},
                )
            )

        return options[:4]
