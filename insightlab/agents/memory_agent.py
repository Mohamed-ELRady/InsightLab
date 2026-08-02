"""Memory Agent: what we already know, checked against what just arrived.

Business memory used to be a passive list read into prompts. If the owner had
said "peak season starts in November" and the new file peaked in March, both
statements went into the prompt and nothing was said about the disagreement.

That moment is the most valuable conversation the product can have. The owner
knows something the file does not contain - a trade fair, a shop closure, a
supplier problem - or the rule has quietly stopped being true. Either way it is
worth one question, and it is a question no general-purpose assistant can ask,
because it has no memory of what the owner said last time.

The agent also does the quieter half of the job: turning statements into
testable claims, and applying the ones that describe a grouping so the owner
does not have to define their VIP tier again on every run.
"""

from __future__ import annotations

import pandas as pd

from ..analysis.features import apply_custom_rule
from ..analysis.profiling import profile_dataset
from ..core.claims import Claim, Test, read_claim, test_claim
from ..core.decision import Option
from ..core.reasoning import AgentPersona
from ..core.state import PipelineState
from .base import Agent, Flow

#: Contradictions raised in one run. Past this the user is being interrogated
#: rather than consulted.
MAX_CONTRADICTIONS = 3

CLAIM_SHAPE = """{
  "kind": "threshold|peak_period|exclusion|target|none",
  "column": "the column it is about, exactly as spelled in the list",
  "operator": ">|>=|<|<=|=|!=",
  "value": 5000,
  "label": "what records meeting the rule are called, e.g. VIP",
  "period": "a month or quarter name, only for peak_period"
}"""


class MemoryAgent(Agent):
    stage = "recall"
    key = "memory"
    title = "Checking what we already know"

    persona = AgentPersona(
        role="Account manager who remembers everything",
        goal=(
            "Bring what the owner told you before to bear on the file in front "
            "of you, and raise it when the two disagree."
        ),
        backstory=(
            "You keep notes on every client and you read them before every "
            "meeting. When the numbers disagree with what a client told you, you "
            "raise it once, politely, and you assume they know something you do "
            "not - because they usually do."
        ),
    )

    def run(self, state: PipelineState) -> Flow:
        state.begin_stage(self.stage)

        if state.frame is None or state.frame.empty:
            state.skip_stage(self.stage, "There is no data to check against.")
            return

        if not state.memory:
            state.finish_stage(
                self.stage,
                "Nothing has been recorded about your business yet, so there was "
                "nothing to check.",
            )
            return

        # This stage runs before the data is understood, so that what we
        # already know can inform those questions rather than arrive too late
        # to affect them. That means building the profile here if nobody has.
        if not state.profile.columns:
            state.profile = profile_dataset(state.frame)

        self._extract_claims(state)

        contradictions = self._find_contradictions(state)
        if not contradictions:
            checked = len(state.memory.testable())
            state.finish_stage(
                self.stage,
                f"Checked {checked} of the things you have told us against this "
                "file. Nothing disagrees."
                if checked
                else "Nothing you have told us could be checked against this file.",
            )
            return

        for fact, result in contradictions[:MAX_CONTRADICTIONS]:
            yield from self._raise(state, fact, result)

        applied = self._apply_classifications(state)
        state.finish_stage(
            self.stage,
            f"Checked what you have told us against this file: "
            f"{len(contradictions)} disagreement(s) raised"
            + (f", {applied} classification(s) applied automatically." if applied else "."),
        )

    # -- turning statements into claims ------------------------------------

    def _extract_claims(self, state: PipelineState) -> None:
        """Give structure to facts that do not have any yet."""
        columns = [str(name) for name in state.frame.columns]
        upgraded = 0

        for fact in state.memory:
            if fact.is_testable:
                continue
            claim = self._read(state, fact.statement, columns)
            if claim is None:
                continue
            fact.claim = claim
            upgraded += 1
            self.note(
                state,
                f'Understood "{fact.statement}" as: {claim.describe()}',
                fact_id=fact.id,
            )

        if upgraded:
            self.note(
                state,
                f"{upgraded} of the things you told us can now be checked against "
                "future files automatically.",
            )

    def _read(self, state: PipelineState, statement: str, columns: list[str]) -> Claim | None:
        """Read a claim out of a sentence, with the model or without it."""
        if self.reasoning.available:
            parsed = self.reason(
                f'A business owner stated a rule about their business: "{statement}"\n\n'
                f"These are the columns in their data: {', '.join(columns)}."
                "\n\nExpress the rule as something that can be tested against "
                "the data. Use a column name exactly as spelled above. If the "
                'statement is not a testable rule, set kind to "none" - a '
                "description of their business is not a rule.",
                shape=CLAIM_SHAPE,
            )
            if isinstance(parsed, dict) and parsed.get("kind") not in (None, "none"):
                claim = Claim.from_dict(parsed)
                if claim and self._is_usable(claim, state):
                    return claim

        return read_claim(statement, columns, state.frame)

    @staticmethod
    def _is_usable(claim: Claim, state: PipelineState) -> bool:
        """Reject a claim we cannot actually run, rather than storing a dud."""
        if claim.kind == "peak_period":
            return bool(claim.period)
        if not claim.column or claim.column not in state.frame.columns:
            return False
        if claim.kind == "threshold":
            try:
                float(claim.value)
            except (TypeError, ValueError):
                return False
            return pd.api.types.is_numeric_dtype(state.frame[claim.column])
        return claim.value is not None

    # -- disagreements -----------------------------------------------------

    def _find_contradictions(self, state: PipelineState) -> list[tuple]:
        results: list[tuple] = []
        for fact in state.memory.testable():
            if fact.disputed:
                # Already raised and settled once. Raising it every run would
                # be nagging, not diligence.
                continue
            result = test_claim(fact.claim, state.frame, state.profile)
            if result.contradicts:
                results.append((fact, result))
        return results

    def _raise(self, state: PipelineState, fact, result: Test) -> Flow:
        """Ask the owner which of the two is right."""
        decision = self.decide(
            topic="Something you told us does not match this file",
            question=(
                f'You told us: "{fact.statement}" — but this file does not '
                "agree. Which is right?"
            ),
            context=(
                f"{result.detail}\n\n"
                "There are three usual reasons for this. The rule has changed "
                "and the note is out of date. Or the rule still holds and this "
                "particular file is unusual - a short period, one branch, a bad "
                "month. Or the rule was always about a part of the business this "
                "file does not cover.\n\n"
                "You know which. We only see the file."
            ),
            suggestion=Option(
                label="This file is unusual — keep what I told you",
                rationale=(
                    "The note stays exactly as it is and keeps applying to future "
                    "analyses. We will not raise it again for this file."
                ),
                payload={"action": "keep"},
            ),
            alternatives=[
                Option(
                    label=f"The file is right — update it to: {result.observed}",
                    rationale=(
                        "The note is replaced with what this file shows, and the "
                        "new version is what future analyses will use."
                    ),
                    payload={"action": "update", "observed": result.observed},
                ),
                Option(
                    label="Forget that note entirely",
                    rationale=(
                        "It stops applying to this and every future analysis. Use "
                        "this when the rule no longer exists rather than having "
                        "changed."
                    ),
                    payload={"action": "forget"},
                ),
            ],
            custom_prompt=(
                "Explain what is going on — for example: that rule only applies "
                "to the retail side, or we changed the threshold last year."
            ),
            skip_effect=(
                "The note stays as it is and the disagreement is recorded in the "
                "run log without being resolved."
            ),
            evidence={
                "statement": fact.statement,
                "understood_as": fact.claim.describe(),
                "expected": result.expected,
                "observed": result.observed,
            },
        )
        answer = yield decision

        if answer.is_skip:
            fact.disputed = True
            self.warn(
                state,
                f'Unresolved disagreement: "{fact.statement}" — {result.detail}',
            )
            return

        if answer.is_custom:
            self.capture_custom(state, decision, answer, category=fact.category)
            fact.disputed = True
            self.note(
                state,
                f'Your explanation was saved and "{fact.statement}" was kept as '
                "it is.",
            )
            return

        action = answer.payload.get("action", "keep")
        if action == "forget":
            state.memory.forget(fact.id)
            self.note(state, f'Forgot: "{fact.statement}"')
        elif action == "update":
            replacement = self._restate(fact, result)
            state.memory.replace(fact.id, replacement, self._updated_claim(fact, result))
            self.note(state, f'Updated to: "{replacement}"')
        else:
            fact.disputed = True
            self.note(
                state,
                f'Kept "{fact.statement}" — this file was treated as unusual.',
            )

    @staticmethod
    def _restate(fact, result: Test) -> str:
        """Rewrite a fact to match what the data shows."""
        claim = fact.claim
        if claim.kind == "peak_period" and result.observed:
            return f"The strongest period is {result.observed}."
        if claim.kind == "exclusion":
            return (
                f"Records where {claim.column.replace('_', ' ')} is {claim.value} "
                "are part of the business and are counted."
            )
        return f"{fact.statement} (updated: {result.detail})"

    @staticmethod
    def _updated_claim(fact, result: Test) -> Claim | None:
        claim = fact.claim
        if claim.kind == "peak_period" and result.observed:
            return Claim(kind="peak_period", period=result.observed)
        if claim.kind == "exclusion":
            # The rule was that these rows should be gone. They are not, and the
            # owner accepted the file, so the exclusion no longer applies.
            return None
        return claim

    # -- applying what we know ---------------------------------------------

    def _apply_classifications(self, state: PipelineState) -> int:
        """Build the owner's own groupings without asking again.

        This is the payoff for structuring the memory: a VIP threshold stated
        once becomes a column on every future run.
        """
        applied = 0
        for fact in state.memory.testable():
            claim = fact.claim
            if claim.kind != "threshold" or not claim.label:
                continue
            if claim.column not in state.frame.columns:
                continue

            column_name = _column_name(claim.label)
            if column_name in state.frame.columns:
                continue

            above = claim.operator in (">", ">=")
            frame, created = apply_custom_rule(
                state.frame,
                column_name,
                {
                    "source": claim.column,
                    "bands": [{"upto": float(claim.value), "label": "Other"}],
                    "otherwise": claim.label,
                }
                if above
                else {
                    "source": claim.column,
                    "bands": [{"upto": float(claim.value), "label": claim.label}],
                    "otherwise": "Other",
                },
            )
            if not created:
                continue

            state.engineered_columns.append(created)
            state.set_frame(
                frame,
                self.stage,
                f"Applied your own rule without asking again: {claim.describe()} "
                f"Added the {created} column.",
            )
            applied += 1

        if applied:
            state.profile = profile_dataset(state.frame)
        return applied


def _column_name(label: str) -> str:
    """A safe column name from a user's label."""
    cleaned = "".join(
        character if character.isalnum() else "_" for character in label.casefold()
    )
    return "_".join(part for part in cleaned.split("_") if part) or "classification"
