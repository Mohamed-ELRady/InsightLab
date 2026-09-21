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
from ..analysis.profiling import ensure_profile
from ..core.claims import Claim, Test, evaluate_claim, read_claim
from ..core.business_memory import BusinessMemory
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
  "label": "what observations meeting the rule are called, e.g. severe",
  "period": "a month or quarter name, only for peak_period"
}"""


class MemoryAgent(Agent):
    stage = "recall"
    key = "memory"
    title = "Checking what we already know"

    persona = AgentPersona(
        role="Project memory steward",
        goal=(
            "Bring what the user established before to bear on the file in front "
            "of you, and raise it when the two disagree."
        ),
        backstory=(
            "You retain approved, versioned project notes and read them before "
            "each analysis. You keep domains isolated. When the file disagrees "
            "with a user-confirmed rule, you raise it once and preserve the audit trail."
        ),
    )

    def run(self, state: PipelineState) -> Flow:
        state.begin_stage(self.stage)

        if state.frame is None or state.frame.empty:
            state.skip_stage(self.stage, "There is no data to check against.")
            return

        # Project facts are available from the first stage. Dataset-scoped
        # facts join only after real column names are known, preventing a rule
        # for one export from leaking into another dataset in the same project.
        if state.project_id:
            from ..core.project_memory import ProjectMemoryStore

            stored = ProjectMemoryStore(state.workspace).load_memory(
                state.project_id, columns=[str(name) for name in state.frame.columns]
            )
            stored_ids = {fact.id for fact in stored}
            combined = [
                *stored, *(fact for fact in state.memory if fact.id not in stored_ids)
            ]
            state.memory = BusinessMemory(
                fact for fact in combined if self._relevant_to_domain(fact, state)
            )

        if not state.memory:
            state.finish_stage(
                self.stage,
                "Nothing has been recorded about this project yet, so there was "
                "nothing to check.",
            )
            return

        # Understanding normally builds the profile first. Keep this fallback
        # so the agent is still safe to run on its own in tests or integrations.
        ensure_profile(state)

        self._extract_claims(state)

        for fact, result in self._find_contradictions(state)[:MAX_CONTRADICTIONS]:
            yield from self._raise(state, fact, result)

        # Always, not only when something disagreed: applying what the owner
        # already told us is the point of storing it, and it has nothing to do
        # with whether this file happened to contradict anything.
        applied = self._apply_classifications(state)

        checked = len(state.memory.testable())
        raised = len(state.log.for_stage(self.stage))
        parts = []
        if checked:
            parts.append(f"Checked {checked} of the things you have told us")
        if applied:
            parts.append(f"applied {applied} of your own classifications")
        state.finish_stage(
            self.stage,
            ("; ".join(parts) + ".")
            if parts
            else "Nothing you have told us could be checked against this file.",
        )

    @staticmethod
    def _relevant_to_domain(fact, state: PipelineState) -> bool:
        """Prevent facts from an unrelated dataset domain leaking into this run."""
        family = state.understanding.domain_family
        topic = str(getattr(fact, "source_topic", ""))
        if topic.startswith("domain:"):
            tagged = topic.split("|", 1)[0].removeprefix("domain:")
            return tagged == family

        columns = {str(name).casefold() for name in state.frame.columns}
        claim = getattr(fact, "claim", None)
        if claim is not None and getattr(claim, "column", ""):
            return str(claim.column).casefold() in columns

        text = fact.statement.casefold()
        # Old automatically saved dataset summaries should follow their own
        # source file, not every file placed in the same project.
        if text.startswith("the dataset is "):
            return state.source_name.casefold() in text

        if family not in {"general", "business", "commerce", "finance"}:
            business_terms = (
                "revenue", "sales", "customer", "order_status", "invoice",
                "business record", "مبيعات", "إيراد", "عميل", "طلب", "سجل عمل",
            )
            domain_terms = {
                "earth_science": ("earthquake", "seismic", "زلزال"),
                "weather": ("weather", "temperature", "طقس", "حرارة"),
                "health": ("patient", "clinical", "health", "مريض", "صحي"),
                "education": ("student", "education", "grade", "طالب", "تعليم"),
                "technology": ("device", "system", "sensor", "جهاز", "نظام"),
            }.get(family, ())
            if any(term in text for term in business_terms) and not any(
                term in text for term in domain_terms
            ):
                return False
        return True

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
        deterministic = read_claim(statement, columns, state.frame)
        if deterministic is not None and self._is_usable(deterministic, state):
            return deterministic
        if self.reasoning.available:
            parsed = self.reason(
                f'A user stated a rule about their data or its domain: "{statement}"\n\n'
                f"These are the columns in their data: {', '.join(columns)}."
                "\n\nExpress the rule as something that can be tested against "
                "the data. Use a column name exactly as spelled above. If the "
                'statement is not a testable rule, set kind to "none" - a '
                "general description of the dataset is not a testable rule.",
                shape=CLAIM_SHAPE,
                max_output_tokens=450,
            )
            if isinstance(parsed, dict) and parsed.get("kind") not in (None, "none"):
                claim = Claim.from_dict(parsed)
                if claim and self._is_usable(claim, state):
                    return claim

        return None

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
            result = evaluate_claim(fact.claim, state.frame, state.profile)
            if result.contradicts:
                results.append((fact, result))
        return results

    def _raise(self, state: PipelineState, fact, result: Test) -> Flow:
        """Ask the owner which of the two is right."""
        ar = state.language.code == "ar"
        decision = self.decide(
            topic="معلومة سابقة لا تتفق مع الملف" if ar else "Something you told us does not match this file",
            question=(
                (f'أخبرتنا سابقًا: "{fact.statement}"، لكن الملف الحالي لا يتفق معها. أيهما الصحيح؟')
                if ar else (f'You told us: "{fact.statement}" — but this file does not agree. Which is right?')
            ),
            context=(
                (f"{result.detail}\n\nقد تكون القاعدة تغيرت، أو الملف الحالي حالة استثنائية، أو القاعدة تخص جزءًا من النشاط غير موجود هنا. إنت تعرف السبب وإحنا شايفين الملف فقط.")
                if ar else
                (f"{result.detail}\n\n"
                "There are three usual reasons for this. The rule has changed "
                "and the note is out of date. Or the rule still holds and this "
                "particular file is unusual - a short period, one branch, a bad "
                "period. Or the rule was always about a subset or context this "
                "file does not cover.\n\n"
                "You know which. We only see the file.")
            ),
            suggestion=Option(
                label="الملف الحالي استثنائي — احتفظ بالمعلومة السابقة" if ar else "This file is unusual — keep what I told you",
                rationale=(
                    "ستبقى المعلومة كما هي وتُطبق على التحليلات القادمة."
                    if ar else "The note stays exactly as it is and keeps applying to future analyses. We will not raise it again for this file."
                ),
                payload={"action": "keep"},
            ),
            alternatives=[
                Option(
                    label=(f"الملف هو الصحيح — حدّث المعلومة إلى: {result.observed}" if ar else f"The file is right — update it to: {result.observed}"),
                    rationale=(
                        "سيتم استبدال المعلومة بما يظهره الملف واستخدام النسخة الجديدة لاحقًا."
                        if ar else "The note is replaced with what this file shows, and the new version is what future analyses will use."
                    ),
                    payload={"action": "update", "observed": result.observed},
                ),
                Option(
                    label="احذف المعلومة نهائيًا" if ar else "Forget that note entirely",
                    rationale=(
                        "لن تُطبق على هذا التحليل أو أي تحليل قادم."
                        if ar else "It stops applying to this and every future analysis. Use this when the rule no longer exists rather than having changed."
                    ),
                    payload={"action": "forget"},
                ),
            ],
            custom_prompt=(
                "اشرح السبب؛ مثلًا: القاعدة تخص قطاع التجزئة فقط، أو تم تغيير الحد العام الماضي."
                if ar else "Explain what is going on — for example: that rule only applies to the retail side, or we changed the threshold last year."
            ),
            skip_effect=(
                "ستبقى المعلومة كما هي وسيتم تسجيل التعارض من غير حسم."
                if ar else "The note stays as it is and the disagreement is recorded in the run log without being resolved."
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
                "are valid observations and are counted."
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
            ensure_profile(state)
        return applied


def _column_name(label: str) -> str:
    """A safe column name from a user's label."""
    cleaned = "".join(
        character if character.isalnum() else "_" for character in label.casefold()
    )
    return "_".join(part for part in cleaned.split("_") if part) or "classification"
