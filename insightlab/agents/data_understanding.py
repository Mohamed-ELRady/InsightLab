"""Data Understanding Agent: the first plain-language read of the file.

This is where the collaboration actually starts. We can see the shape of the
data; only the owner can say what it represents and which rows count. What they
answer here is saved to the business memory and steers every later stage.
"""

from __future__ import annotations

from ..core.decision import Option
from ..core.reasoning import AgentPersona
from ..core.state import PipelineState, Role
from ..analysis.profiling import describe_shape, frame_digest, profile_dataset
from .base import Agent, Flow

#: Common shapes of business file, offered when we cannot tell which it is.
COMMON_SUBJECTS = (
    ("sales transactions", "one row per sale, order or invoice line"),
    ("customer records", "one row per customer or account"),
    ("inventory or stock", "one row per product, batch or stock movement"),
    ("financial entries", "one row per payment, expense or ledger entry"),
    ("operational events", "one row per delivery, ticket, visit or job"),
)


class DataUnderstandingAgent(Agent):
    stage = "understand"
    key = "understanding"
    title = "Understanding the data"

    persona = AgentPersona(
        role="Business data consultant",
        goal=(
            "Describe what is in the owner's file in language they would use "
            "themselves, and find out what the rows actually represent in their "
            "business before anyone starts calculating anything."
        ),
        backstory=(
            "You have sat across the table from hundreds of owners looking at "
            "their own data for the first time. You know that the column names "
            "rarely mean what an outsider assumes, and that one sentence from the "
            "owner about what a row represents saves an entire wrong analysis."
        ),
    )

    def run(self, state: PipelineState) -> Flow:
        state.begin_stage(self.stage)

        if state.frame is None or state.frame.empty:
            state.skip_stage(self.stage, "There is no data loaded to understand.")
            return

        frame = state.frame
        state.profile = profile_dataset(frame)
        profile = state.profile

        self.note(
            state,
            f"Profiled the data: {describe_shape(profile)}",
            duplicate_rows=profile.duplicate_rows,
        )

        summary = self._summarise(state)
        profile.summary = summary
        state.remember(
            f"The dataset is {state.source_name}: {describe_shape(profile)}",
            category="context",
            stage=self.stage,
            topic="Dataset shape",
        )

        decision = self._subject_decision(state, summary)
        answer = yield decision

        if answer.is_custom and answer.text:
            self.capture_custom(state, decision, answer, category="context")
        elif not answer.is_skip:
            option = answer.selected
            if option is not None:
                subject = option.payload.get("subject", "")
                if subject:
                    state.remember(
                        f"Each row in this data is {subject}.",
                        category="definition",
                        stage=self.stage,
                        topic=decision.topic,
                    )

        yield from self._confirm_exclusions(state)

        state.finish_stage(
            self.stage,
            f"Described the data: {describe_shape(profile)}",
        )

    # -- summary -----------------------------------------------------------

    def _summarise(self, state: PipelineState) -> str:
        """Ask the model to explain the file, and fall back to a written summary."""
        profile = state.profile
        digest = frame_digest(state.frame, profile, rows=5)

        summary = self.explain(
            "A business owner has just uploaded a file called "
            f"{state.source_name}. Here is what it contains:\n\n{digest}"
            f"{self.memory_block(state)}\n\n"
            "Write three or four sentences telling them what is in this file, in "
            "the language they would use about their own business. Say what one "
            "row appears to represent, which columns look most useful, and name "
            "anything that would stop a calculation being trusted - empty "
            "columns, a column where every row is identical, or dates stored as "
            "text. Do not list every column and do not use the words dataframe, "
            "dataset, null or dtype.",
            expected="Three or four plain sentences with no bullet points.",
        )
        if summary:
            return summary.strip()
        return self._written_summary(state)

    def _written_summary(self, state: PipelineState) -> str:
        """The summary we write ourselves when no model is available."""
        profile = state.profile
        measures = profile.names_with_role(Role.MEASURE)
        categories = profile.names_with_role(Role.CATEGORY)
        dates = profile.names_with_role(Role.DATETIME)

        parts = [
            f"This file holds {profile.row_count:,} rows across "
            f"{profile.column_count} columns."
        ]
        if measures:
            parts.append(
                f"There are {len(measures)} columns holding numbers you can total "
                f"or average, including {', '.join(measures[:3])}."
            )
        if categories:
            parts.append(
                f"You can group the data by {', '.join(categories[:3])}."
            )
        if dates:
            column = profile.column(dates[0])
            if column and column.stats.get("earliest"):
                parts.append(
                    f"It covers {column.stats['earliest']} to "
                    f"{column.stats['latest']}, so changes over time can be seen."
                )

        concerns = [column.note for column in profile.columns if column.note]
        if profile.duplicate_rows:
            concerns.insert(
                0, f"{profile.duplicate_rows:,} rows are exact copies of another row"
            )
        if concerns:
            parts.append("Worth knowing before we go further: " + "; ".join(concerns[:3]) + ".")
        return " ".join(parts)

    # -- decisions ---------------------------------------------------------

    def _subject_decision(self, state: PipelineState, summary: str):
        profile = state.profile
        guess, guess_reason = self._guess_subject(state)

        alternatives = [
            Option(
                label=f"Each row is {subject}",
                rationale=description.capitalize() + ".",
                payload={"subject": subject},
            )
            for subject, description in COMMON_SUBJECTS
            if subject != guess
        ]

        return self.decide(
            topic="What the data represents",
            question="Before we calculate anything: what does one row in this file represent?",
            context=(
                f"{summary}\n\nWe can see the shape of the file, but only you know "
                "what it is a record of. Getting this right changes what counts as "
                "a customer, an order and a total further down."
            ),
            suggestion=Option(
                label=f"Each row is {guess}",
                rationale=guess_reason,
                payload={"subject": guess},
            ),
            alternatives=alternatives,
            custom_prompt=(
                "Describe in one or two sentences what this data is and what one "
                "row represents. Anything you tell us here is remembered for the "
                "rest of the analysis."
            ),
            skip_effect=(
                "We will work from the column names alone, which is usually enough "
                "but occasionally leads to a wrong assumption about your business."
            ),
            evidence={"profile": profile.to_dict()},
        )

    def _guess_subject(self, state: PipelineState) -> tuple[str, str]:
        """Infer what the rows are from the column names present."""
        names = " ".join(column.name.casefold() for column in state.profile.columns)
        checks = (
            (("order", "invoice", "sale", "transaction"), "sales transactions",
             "the file has order, invoice or sale columns, which almost always "
             "means one row per transaction"),
            (("product", "sku", "stock", "inventory", "warehouse"), "inventory or stock",
             "the file is built around product and stock columns"),
            (("payment", "ledger", "account", "expense", "debit"), "financial entries",
             "the columns read like an accounting export"),
            (("ticket", "delivery", "shipment", "visit", "job"), "operational events",
             "the columns describe things that happen rather than things that sell"),
            (("customer", "client", "member"), "customer records",
             "customer columns dominate the file"),
        )
        for keywords, subject, reason in checks:
            if any(keyword in names for keyword in keywords):
                return subject, reason
        return (
            "one business record",
            "the column names do not point clearly at one kind of record, so this "
            "is the safest starting assumption",
        )

    def _confirm_exclusions(self, state: PipelineState) -> Flow:
        """Ask whether any rows should be left out before anything is totalled.

        Cancelled orders, test records and internal transfers sit in almost
        every export and quietly inflate every figure. This is the cheapest
        moment to remove them, and only the owner knows which they are.
        """
        profile = state.profile
        candidate = None
        for column in profile.columns:
            if column.role is not Role.CATEGORY:
                continue
            if not any(
                word in column.name.casefold()
                for word in ("status", "state", "type", "stage")
            ):
                continue
            values = list(column.stats.get("top_values", {}))
            suspicious = [
                value
                for value in values
                if any(
                    word in str(value).casefold()
                    for word in ("cancel", "void", "test", "refund", "return", "draft")
                )
            ]
            if suspicious:
                candidate = (column.name, values, suspicious)
                break

        if candidate is None:
            return

        column_name, values, suspicious = candidate
        listing = ", ".join(str(value) for value in values[:6])
        decision = self.decide(
            topic="Rows to leave out",
            question=(
                f"Your {column_name} column contains {listing}. Should any of "
                "these be left out of the analysis?"
            ),
            context=(
                "Cancelled, refunded and test records usually sit in the same "
                "export as real ones. If they stay in, every total and average "
                "will be higher than what actually happened. If they are genuine "
                "business you want counted, leave them in."
            ),
            suggestion=Option(
                label=f"Leave out rows where {column_name} is "
                + ", ".join(str(value) for value in suspicious),
                rationale=(
                    "These values normally mean the transaction did not complete, "
                    "so counting them would overstate the business."
                ),
                payload={"column": column_name, "exclude": suspicious},
            ),
            alternatives=[
                Option(
                    label="Keep every row",
                    rationale=(
                        "Use this if cancellations and returns are part of what "
                        "you want to see."
                    ),
                    payload={"column": column_name, "exclude": []},
                ),
            ]
            + [
                Option(
                    label=f"Leave out only {value}",
                    rationale=f"Removes rows where {column_name} is {value}.",
                    payload={"column": column_name, "exclude": [value]},
                )
                for value in suspicious[:3]
            ],
            custom_prompt=(
                f"List the {column_name} values you want left out, separated by "
                "commas."
            ),
            skip_effect="Every row stays in, including cancellations and returns.",
            evidence={"column": column_name, "values": values},
        )
        answer = yield decision

        if answer.is_skip:
            return

        if answer.is_custom:
            wanted = [part.strip() for part in answer.text.split(",") if part.strip()]
            self.capture_custom(state, decision, answer, category="exclusion")
        else:
            wanted = list(answer.payload.get("exclude", []))

        if not wanted:
            self.note(state, "Every row was kept, including cancellations and returns.")
            return

        from ..analysis.cleaning import filter_rows

        operation = filter_rows(state.frame, column_name, wanted)
        if operation.rows_removed:
            self.apply_operation(state, operation)
            state.remember(
                f"Rows where {column_name} is "
                f"{', '.join(str(value) for value in wanted)} are not part of the "
                "business and are excluded from every figure.",
                category="exclusion",
                stage=self.stage,
                topic=decision.topic,
            )
            state.profile = profile_dataset(state.frame)
