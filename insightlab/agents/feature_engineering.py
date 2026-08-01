"""Feature Engineering Agent: build the measures the raw columns do not have.

A file of transactions cannot answer "which months are strongest" or "which
products actually make money" - not because the information is missing, but
because it is buried in a date column and split across revenue and cost. This
agent proposes the derived columns that make those questions answerable, and
asks the one question no amount of analysis can answer: what classifications
does this business already use internally?
"""

from __future__ import annotations

from ..analysis.features import apply_custom_rule, suggest_features
from ..analysis.profiling import profile_dataset
from ..core.decision import Option
from ..core.reasoning import AgentPersona
from ..core.state import PipelineState, Role
from .base import Agent, Flow


class FeatureEngineeringAgent(Agent):
    stage = "features"
    key = "features"
    title = "Building new measures"

    persona = AgentPersona(
        role="Business analyst",
        goal=(
            "Turn the raw columns into the measures the owner actually thinks in, "
            "and capture the classifications their business already uses."
        ),
        backstory=(
            "You know that every business has its own vocabulary - a VIP tier, an "
            "A/B/C product grade, a peak season - and that an analysis using "
            "generic groupings instead of theirs gets politely ignored. So you "
            "always ask."
        ),
    )

    def run(self, state: PipelineState) -> Flow:
        state.begin_stage(self.stage)

        if state.frame is None or state.frame.empty:
            state.skip_stage(self.stage, "There is no data to build measures from.")
            return

        suggestions = suggest_features(state.frame, state.profile)
        if suggestions:
            yield from self._offer_features(state, suggestions)
        else:
            self.note(
                state,
                "The existing columns already cover what the analysis needs, so "
                "nothing new was added.",
            )

        yield from self._ask_internal_classifications(state)

        state.profile = profile_dataset(state.frame)
        if state.engineered_columns:
            state.finish_stage(
                self.stage,
                f"Added {len(state.engineered_columns)} new measures: "
                f"{', '.join(state.engineered_columns)}.",
            )
        else:
            state.finish_stage(self.stage, "No new measures were added.")

    # -- standard features -------------------------------------------------

    def _offer_features(self, state: PipelineState, suggestions) -> Flow:
        listing = "\n".join(
            f"- {item.name}: {item.reason}" for item in suggestions
        )
        core = [item for item in suggestions if item.id in {"calendar", "profit"}]
        core_names = ", ".join(item.name for item in core) or suggestions[0].name

        decision = self.decide(
            topic="New measures to add",
            question=(
                f"We can build {len(suggestions)} new measures from the columns "
                "you already have. Which should we add?"
            ),
            context=(
                "None of these need extra data - they are all worked out from "
                "columns already in your file. Each one exists to answer a "
                "question the raw columns cannot:\n\n"
                f"{listing}\n\n"
                "Adding a measure costs nothing and never changes your original "
                "columns."
            ),
            suggestion=Option(
                label=f"Add all {len(suggestions)}",
                rationale=(
                    "Each one unlocks a question the raw columns cannot answer, "
                    "and unused ones simply do not appear in the charts."
                ),
                payload={"ids": [item.id for item in suggestions]},
            ),
            alternatives=[
                Option(
                    label=f"Add only the essentials ({core_names})",
                    rationale=(
                        "The smallest set that still allows time and profit "
                        "comparisons. Keeps the data close to what you recognise."
                    ),
                    payload={"ids": [item.id for item in core] or [suggestions[0].id]},
                ),
            ]
            + [
                Option(
                    label=f"Add only {item.name}",
                    rationale=item.question,
                    payload={"ids": [item.id]},
                )
                for item in suggestions[:3]
            ],
            custom_prompt=(
                "Describe a measure your business uses that is not listed, and "
                "which columns it comes from."
            ),
            skip_effect=(
                "No new measures are added. Time-based and profit comparisons "
                "will not be available later."
            ),
            evidence={"suggestions": [item.id for item in suggestions]},
        )
        answer = yield decision

        if answer.is_skip:
            self.note(state, "No new measures were added, at your request.")
            return

        if answer.is_custom:
            self.capture_custom(state, decision, answer, category="definition")
            wanted = {item.id for item in suggestions}
            self.note(
                state,
                "Your description was saved to the business memory, and the "
                "standard measures were added so the analysis still has "
                "something to work with.",
            )
        else:
            wanted = set(answer.payload.get("ids", []))

        frame = state.frame
        added: list[str] = []
        for item in suggestions:
            if item.id not in wanted:
                continue
            try:
                frame, created = item.apply(frame)
            except (ValueError, TypeError, KeyError) as error:
                self.warn(state, f"{item.name} could not be built: {error}")
                continue
            added.extend(created)

        if added:
            state.engineered_columns.extend(added)
            state.set_frame(
                frame,
                self.stage,
                f"Added {len(added)} new columns: {', '.join(added)}.",
            )

    # -- the business's own vocabulary -------------------------------------

    def _ask_internal_classifications(self, state: PipelineState) -> Flow:
        """Ask about the groupings the business already uses.

        This is the question that most often changes the whole analysis, because
        a company's own definition of a VIP or a slow product rarely matches the
        one a quartile split would produce.
        """
        profile = state.profile
        measures = profile.names_with_role(Role.MEASURE)
        example_measure = None
        for name in measures:
            if any(word in name.casefold() for word in ("revenue", "sales", "total", "amount")):
                example_measure = name
                break
        example_measure = example_measure or (measures[0] if measures else None)

        example = (
            f'For example: "customers who spend more than 5,000 in total are VIP" '
            f"would create a VIP group from your {example_measure} column."
            if example_measure
            else 'For example: "orders over 5,000 are large accounts".'
        )

        decision = self.decide(
            topic="Your own classifications",
            question=(
                "Does your company use any classifications of its own that we "
                "should build into the analysis?"
            ),
            context=(
                "Most businesses group their customers, products or orders in a "
                "way that is specific to them - a VIP tier, an A/B/C grade, a "
                "definition of a large account. If we do not know yours, the "
                "analysis will use generic groupings, which is usually the reason "
                "a report gets read once and never again.\n\n" + example
            ),
            suggestion=Option(
                label="We do not use any special classifications",
                rationale=(
                    "The analysis uses standard groupings based on size and "
                    "frequency, which works well when there is no internal system."
                ),
                payload={"rule": None},
            ),
            alternatives=[
                Option(
                    label="Use standard size bands (small, medium, large, very large)",
                    rationale=(
                        "Splits records into four equal-sized groups by value. A "
                        "reasonable default when there is no internal rule."
                    ),
                    payload={"rule": "quartiles"},
                ),
            ],
            custom_prompt=(
                "Describe your classification in plain words, including the "
                "column it is based on and the cut-off, for example: customers "
                "spending over 5,000 in total are VIP."
            ),
            skip_effect="Only generic groupings are used in the analysis.",
        )
        answer = yield decision

        if answer.is_skip or (not answer.is_custom and answer.payload.get("rule") is None):
            return

        if not answer.is_custom:
            return  # The quartile bands were already built by the size-band feature.

        self.capture_custom(state, decision, answer, category="classification")
        rule = self._rule_from_text(state, answer.text)
        if rule is None:
            self.note(
                state,
                "Your classification was saved to the business memory and every "
                "later stage will take it into account, but we could not work out "
                "an exact cut-off from it, so no new column was created.",
            )
            return

        frame, created = apply_custom_rule(state.frame, rule["column"], rule)
        if created:
            state.engineered_columns.append(created)
            state.set_frame(
                frame,
                self.stage,
                f"Built your own classification into a new column, {created}.",
            )

    def _rule_from_text(self, state: PipelineState, text: str) -> dict | None:
        """Turn a sentence like "VIPs spend over 5,000" into a banding rule."""
        available = [
            column.name
            for column in state.profile.columns
            if column.role is Role.MEASURE
        ] + [name for name in state.engineered_columns if name in state.frame.columns]
        if not available:
            return None

        parsed = self.reason(
            f'A business owner described one of their internal classifications: "{text}"\n\n'
            f"These numeric columns are available: {', '.join(available)}."
            f"{self.memory_block(state)}\n\n"
            "Work out which column the rule is about and where the cut-offs fall. "
            "Bands are listed from lowest to highest, each with the upper limit "
            "that still belongs to it. The label for everything above the last "
            'band goes in "otherwise". If the description does not contain a '
            "numeric cut-off, return null.",
            shape=(
                '{"column": "name of the new column", "source": "existing numeric column", '
                '"bands": [{"upto": 5000, "label": "Regular"}], "otherwise": "VIP"}'
            ),
        )

        if not isinstance(parsed, dict):
            return self._rule_from_numbers(text, available)

        source = parsed.get("source")
        bands = parsed.get("bands")
        if source not in available or not isinstance(bands, list) or not bands:
            return self._rule_from_numbers(text, available)

        return {
            "column": str(parsed.get("column") or "custom_classification"),
            "source": source,
            "bands": bands,
            "otherwise": str(parsed.get("otherwise") or "Other"),
        }

    @staticmethod
    def _rule_from_numbers(text: str, available: list[str]) -> dict | None:
        """Fallback: pull a single threshold out of the sentence ourselves.

        Handles the common shape - one number, one label - so the feature still
        works with no model available.
        """
        import re

        numbers = re.findall(r"\d[\d,]*(?:\.\d+)?", text)
        if not numbers:
            return None
        threshold = float(numbers[0].replace(",", ""))

        lowered = text.casefold()
        source = None
        for name in available:
            if name.casefold() in lowered:
                source = name
                break
        if source is None:
            for name in available:
                if any(
                    word in name.casefold()
                    for word in ("total", "revenue", "sales", "value", "amount")
                ):
                    source = name
                    break
        if source is None:
            source = available[0]

        # The word before "are" or "is" is usually the label being defined.
        match = re.search(r"\b(?:are|is|called|classed as)\s+([A-Za-z][\w \-]{0,24})", text)
        label = match.group(1).strip().rstrip(".") if match else "Top tier"

        return {
            "column": "custom_classification",
            "source": source,
            "bands": [{"upto": threshold, "label": "Other"}],
            "otherwise": label,
        }
