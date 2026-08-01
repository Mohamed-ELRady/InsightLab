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
from ..analysis.profiling import profile_dataset
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
            "Get the data into a state where the totals can be trusted, without "
            "throwing away anything that turns out to be real business."
        ),
        backstory=(
            "You have learned the hard way that the row an algorithm calls an "
            "error is often the most important sale of the year. You never delete "
            "anything on your own judgement when the owner is available to ask, "
            "and you always say what a change would do to the totals before you "
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

        state.profile = profile_dataset(state.frame)
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

        decision = self.decide(
            topic="Duplicate rows",
            question=(
                f"{report.total:,} rows are exact copies of another row. What "
                "should we do with them?"
            ),
            context=(
                f"That is {share:.1f}% of your data, across {report.groups:,} "
                "repeated groups. Duplicates usually come from a file being "
                "exported twice or two systems being merged, and they inflate "
                "every total by the amount they repeat.\n\n"
                "But if your business genuinely records two identical "
                "transactions - the same product, same price, same day - then "
                "these are real and deleting them would understate your sales.\n\n"
                f"Here are some of them:\n{preview}"
            ),
            suggestion=Option(
                label="Delete the copies and keep one of each",
                rationale=(
                    f"Totals drop by the value of {report.total:,} rows and become "
                    "correct if these came from a double export, which is the "
                    "usual cause."
                ),
                payload={"action": "delete"},
            ),
            alternatives=[
                Option(
                    label="Keep every row as it is",
                    rationale=(
                        "Choose this if two identical transactions really can "
                        "happen in your business. Nothing changes."
                    ),
                    payload={"action": "keep"},
                ),
                Option(
                    label="Merge them into one row and add the amounts up",
                    rationale=(
                        "The row count drops but the totals stay exactly the same. "
                        "Use this when the repeats are real sales that were "
                        "recorded as separate lines."
                    ),
                    payload={"action": "merge"},
                ),
            ],
            custom_prompt=(
                "Tell us how duplicates should be treated in your business, for "
                "example which columns make two rows genuinely the same."
            ),
            skip_effect="The duplicates stay in and every total includes them twice.",
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
        highs = ", ".join(f"{value:,.2f}" for value in report.high_values[:6])
        lows = ", ".join(f"{value:,.2f}" for value in report.low_values[:6])
        listed = []
        if highs:
            listed.append(f"unusually high: {highs}")
        if lows:
            listed.append(f"unusually low: {lows}")
        preview = report.rows.head(5).to_string(index=False, max_colwidth=16)

        decision = self.decide(
            topic=f"Unusual values in {report.column}",
            question=(
                f"{report.count:,} values in {report.column} are far outside the "
                "normal range. Do you consider these normal for your business?"
            ),
            context=(
                f"Most values in {report.column} sit between "
                f"{report.lower_bound:,.2f} and {report.upper_bound:,.2f}. "
                f"These {report.count:,} do not ({report.share:.1%} of the rows). "
                + "; ".join(listed)
                + ".\n\nThe answer depends entirely on your business, and it "
                "matters: leaving a mistyped value in will drag every average "
                "with it, while deleting a genuinely large sale hides your best "
                "customer.\n\n"
                f"Here are some of the rows involved:\n{preview}"
            ),
            suggestion=Option(
                label="These are unusual but real - keep them and mark them",
                rationale=(
                    "Nothing is lost and the rows are tagged so you can look at "
                    "them separately. This is the safe answer when you are not "
                    "sure."
                ),
                payload={"action": "flag"},
            ),
            alternatives=[
                Option(
                    label="These are data-entry errors - remove those rows",
                    rationale=(
                        f"{report.count:,} rows are removed. Choose this only if "
                        "the values could not possibly be real, such as a price "
                        "typed with an extra zero."
                    ),
                    payload={"action": "remove"},
                ),
                Option(
                    label="These are naturally high-priced items - keep them as they are",
                    rationale=(
                        "Nothing changes. Correct when you genuinely sell items at "
                        "these prices, and the averages should reflect that."
                    ),
                    payload={"action": "keep"},
                ),
                Option(
                    label="This is a seasonal or one-off effect - cap them at the normal range",
                    rationale=(
                        "The rows stay, but the extreme numbers are pulled back to "
                        "the edge of normal so they stop dominating every average. "
                        "Your totals will drop slightly."
                    ),
                    payload={"action": "cap"},
                ),
            ],
            custom_prompt=(
                "Tell us what these values are in your business - a bulk order, a "
                "particular customer, a promotion - and what should happen to them."
            ),
            skip_effect=(
                f"The {report.count:,} unusual values stay exactly as they are and "
                "will pull the averages with them."
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
                "business and are not errors.",
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
        profile = profile_dataset(state.frame)
        state.profile = profile

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
        samples = ", ".join(str(value) for value in column.sample_values[:4])
        alternatives = self._column_alternatives(column, action)

        decision = self.decide(
            topic=f"The {column.name} column",
            question=f"What should we do with the {column.name} column?",
            context=(
                f"It holds {column.role.value} values such as {samples}. "
                f"{column.missing_count:,} of {len(state.frame):,} rows are empty "
                f"({column.missing_rate:.0%}), and there are "
                f"{column.unique_count:,} different values in it."
                + (f" {column.note.capitalize()}." if column.note else "")
            ),
            suggestion=Option(
                label=self._action_label(action, column, payload),
                rationale=reason.capitalize() + ".",
                payload=payload,
            ),
            alternatives=alternatives,
            custom_prompt=(
                f"Tell us what {column.name} means in your business and how it "
                'should be handled. You can also say "rename to sales region", '
                '"remove it", "keep it" or "fill blanks with 0".'
            ),
            skip_effect=(
                f"{column.name} is left exactly as it is, empty values included."
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
    def _action_label(action: str, column, payload: dict) -> str:
        if action == "drop":
            return f"Remove the {column.name} column"
        if action == "fill":
            strategy = payload.get("strategy", "median")
            if strategy == "median":
                return "Fill the empty values with the middle value"
            return f'Group the empty values under "{strategy}"'
        if action == "convert":
            return f"Convert {column.name} to a {payload.get('target', 'number')}"
        return f"Keep {column.name} as it is"

    @staticmethod
    def _column_alternatives(column, chosen: str) -> list[Option]:
        options: list[Option] = []

        if chosen != "keep":
            options.append(
                Option(
                    label=f"Keep {column.name} exactly as it is",
                    rationale="Nothing changes, empty values included.",
                    payload={"action": "keep"},
                )
            )

        if chosen != "drop":
            options.append(
                Option(
                    label=f"Remove the {column.name} column",
                    rationale=(
                        "Use this if the column is not something your business "
                        "acts on. It disappears from every chart and report."
                    ),
                    payload={"action": "drop"},
                )
            )

        if column.missing_count:
            options.append(
                Option(
                    label=f"Remove the rows where {column.name} is empty",
                    rationale=(
                        f"{column.missing_count:,} rows are removed entirely, "
                        "including their values in every other column."
                    ),
                    payload={"action": "drop_missing"},
                )
            )
            if column.role is Role.MEASURE and chosen != "fill":
                options.append(
                    Option(
                        label="Fill the empty values with zero",
                        rationale=(
                            "Correct when an empty cell genuinely means nothing "
                            "happened, rather than that nobody recorded it."
                        ),
                        payload={"action": "fill", "strategy": "zero"},
                    )
                )

        if column.role is Role.TEXT:
            options.append(
                Option(
                    label=f"Treat {column.name} as a set of groups",
                    rationale=(
                        "Makes it available for grouping and comparison in the "
                        "charts, which free text cannot be used for."
                    ),
                    payload={"action": "convert", "target": "category"},
                )
            )

        if column.role in (Role.TEXT, Role.CATEGORY) and column.name.casefold().endswith(
            ("date", "time")
        ):
            options.append(
                Option(
                    label=f"Read {column.name} as a date",
                    rationale=(
                        "Unlocks month, quarter and season comparisons that text "
                        "dates cannot support."
                    ),
                    payload={"action": "convert", "target": "date"},
                )
            )

        return options[:4]
