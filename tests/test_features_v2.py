"""The capabilities added after the first release.

Conversation, root-cause attribution, structured memory, cross-run comparison,
multi-file joining and the Arabic experience.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from insightlab.agents.supervisor import Supervisor
from insightlab.analysis import attribution, comparison, joining
from insightlab.analysis import query as query_module
from insightlab.analysis.profiling import profile_dataset
from insightlab.analysis.query import PlanError
from insightlab.core import language as language_module
from insightlab.core.business_memory import BusinessMemory
from insightlab.core.claims import Claim, read_claim, evaluate_claim
from insightlab.core.language import ARABIC, ENGLISH
from insightlab.core.state import PipelineState, RunMode

SAMPLE = Path(__file__).resolve().parents[1] / "data" / "samples" / "retail_sales.csv"


@pytest.fixture
def enriched(sample_frame):
    """The sample data as the query layer sees it, after feature engineering."""
    from insightlab.analysis.features import suggest_features

    frame = sample_frame.copy()
    frame["order_date"] = pd.to_datetime(frame["order_date"])
    profile = profile_dataset(frame)
    for suggestion in suggest_features(frame, profile):
        frame, _ = suggestion.apply(frame)
    return frame, profile_dataset(frame)


@pytest.fixture
def finished_state(enriched):
    frame, profile = enriched
    state = PipelineState()
    state.frame = frame
    state.raw_frame = frame
    state.profile = profile
    return state


# ---------------------------------------------------------------------------
# A1 — asking a question
# ---------------------------------------------------------------------------


class TestQueryPlans:
    def test_a_plan_becomes_the_calculation_it_describes(self, enriched):
        frame, profile = enriched
        plan = query_module.validate(
            {"measure": "revenue", "aggregation": "sum", "group_by": ["region"]},
            frame,
            profile,
        )
        result = query_module.run(plan, frame, profile)

        expected = frame.groupby("region")["revenue"].sum().max()
        assert result.table.iloc[0, 1] == pytest.approx(expected, rel=1e-6)

    def test_a_column_name_the_model_reshaped_still_resolves(self, enriched):
        # Models drop underscores and change case. Rejecting "Product Category"
        # for `product_category` would be pedantry, not safety.
        frame, profile = enriched
        plan = query_module.validate(
            {"measure": "revenue", "group_by": ["Product Category"]}, frame, profile
        )
        assert plan.group_by == ["product_category"]

    def test_a_column_that_does_not_exist_is_refused_with_the_real_names(
        self, enriched
    ):
        frame, profile = enriched
        with pytest.raises(PlanError) as error:
            query_module.validate({"measure": "profit_wizard"}, frame, profile)
        assert "profit_wizard" in str(error.value)
        assert "revenue" in str(error.value), "the refusal should list what exists"

    def test_totalling_a_text_column_is_refused(self, enriched):
        frame, profile = enriched
        with pytest.raises(PlanError):
            query_module.validate(
                {"measure": "region", "aggregation": "sum"}, frame, profile
            )

    def test_breakdowns_are_capped_at_two(self, enriched):
        # Three levels of grouping produces a table nobody can read.
        frame, profile = enriched
        plan = query_module.validate(
            {
                "measure": "revenue",
                "group_by": ["region", "product_category", "sales_channel"],
            },
            frame,
            profile,
        )
        assert len(plan.group_by) == 2

    def test_the_limit_is_bounded(self, enriched):
        frame, profile = enriched
        plan = query_module.validate({"measure": "revenue", "limit": 99999}, frame, profile)
        assert plan.limit == query_module.MAX_LIMIT

    def test_a_trend_headline_describes_the_whole_period_not_the_shown_rows(
        self, enriched
    ):
        # The table is truncated for display; describing the trend from the
        # truncated window would state a change across "the period" that is not
        # the change across the period.
        frame, profile = enriched
        plan = query_module.validate(
            {"measure": "revenue", "aggregation": "sum", "time_grain": "month", "limit": 5},
            frame,
            profile,
        )
        result = query_module.run(plan, frame, profile)

        monthly = frame.groupby(frame["order_date"].dt.to_period("M"))["revenue"].sum()
        change = (monthly.iloc[-1] / monthly.iloc[0] - 1) * 100
        assert f"{abs(change):.0f}%" in result.headline
        assert "Showing 5 of" in result.headline

    def test_a_count_is_not_shown_with_decimals(self, enriched):
        frame, profile = enriched
        plan = query_module.validate(
            {"measure": "customer_id", "aggregation": "nunique"}, frame, profile
        )
        result = query_module.run(plan, frame, profile)
        assert ".00" not in result.headline

    def test_filters_narrow_the_rows(self, enriched):
        frame, profile = enriched
        plan = query_module.validate(
            {
                "measure": "revenue",
                "group_by": ["product_category"],
                "filters": [{"column": "region", "operator": "=", "value": "Cairo"}],
            },
            frame,
            profile,
        )
        result = query_module.run(plan, frame, profile)
        expected = frame[frame["region"] == "Cairo"].groupby("product_category")[
            "revenue"
        ].sum().max()
        assert result.table.iloc[0, 1] == pytest.approx(expected, rel=1e-6)


class TestAsking:
    def test_a_question_it_cannot_answer_is_refused_rather_than_guessed(
        self, finished_state
    ):
        supervisor = Supervisor(finished_state, agents=[])
        answer = supervisor.ask("What is our staff turnover rate?")
        assert not answer.answered
        assert answer.refusal

    def test_an_answerable_question_returns_figures(self, finished_state):
        supervisor = Supervisor(finished_state, agents=[])
        answer = supervisor.ask("Which region brings in the most revenue?")
        assert answer.answered
        assert answer.understood_as
        assert not answer.table.empty

    def test_the_question_and_answer_reach_the_log(self, finished_state):
        supervisor = Supervisor(finished_state, agents=[])
        supervisor.ask("Which region brings in the most revenue?")
        messages = [event.message for event in finished_state.log.for_stage("ask")]
        assert any("Question asked" in message for message in messages)

    def test_every_suggested_question_is_one_that_actually_runs(self, finished_state):
        supervisor = Supervisor(finished_state, agents=[])
        for question in supervisor.suggested_questions():
            answer = supervisor.ask(question)
            assert answer.answered, f"suggested but unanswerable: {question}"


# ---------------------------------------------------------------------------
# E1 — root cause
# ---------------------------------------------------------------------------


class TestAttribution:
    def test_contributions_sum_exactly_to_the_movement(self, finished_state):
        result = attribution.explain(finished_state)
        assert result is not None
        for dimension in result.dimensions:
            total = sum(item.change for item in dimension.contributions)
            assert total == pytest.approx(result.change, abs=1.0), dimension.column

    def test_calendar_columns_never_explain_a_calendar_movement(self, finished_state):
        # "November was high because of November" is not an explanation.
        result = attribution.explain(finished_state)
        for dimension in result.dimensions:
            assert dimension.column not in attribution.CALENDAR_DERIVED

    def test_our_own_outlier_markers_never_explain_anything(self, finished_state):
        result = attribution.explain(finished_state)
        for dimension in result.dimensions:
            assert not dimension.column.endswith("_is_unusual")

    def test_a_group_gap_can_be_explained_too(self, finished_state):
        result = attribution.explain_group_gap(
            finished_state, "region", "revenue", "Cairo", "Aswan"
        )
        assert result is not None
        assert result.dimensions

    def test_a_file_with_no_dates_produces_nothing_rather_than_guessing(self):
        frame = pd.DataFrame({"g": list("abcd") * 20, "v": range(80)})
        state = PipelineState()
        state.frame = frame
        state.profile = profile_dataset(frame)
        assert attribution.explain(state) is None


# ---------------------------------------------------------------------------
# C2 + C1 — structured memory
# ---------------------------------------------------------------------------


class TestClaims:
    def test_a_threshold_is_read_out_of_a_sentence(self, sample_frame):
        claim = read_claim(
            "Large orders are anything above 10000 revenue",
            list(sample_frame.columns),
            sample_frame,
        )
        assert claim.kind == "threshold"
        assert claim.column == "revenue"
        assert claim.value == 10000

    def test_a_peak_period_is_read(self, sample_frame):
        claim = read_claim("Peak season starts in November", list(sample_frame.columns))
        assert claim.kind == "peak_period"
        assert claim.period == "November"

    def test_an_exclusion_resolves_to_the_spelling_in_the_file(self, sample_frame):
        # The user says "invoices", the file says order_status = "Cancelled".
        # Storing the user's wording would produce a rule matching nothing.
        claim = read_claim(
            "Ignore cancelled invoices", list(sample_frame.columns), sample_frame
        )
        assert claim.kind == "exclusion"
        assert claim.column == "order_status"
        assert claim.value == "Cancelled"

    def test_a_statement_that_is_not_a_rule_produces_nothing(self, sample_frame):
        assert read_claim("We prefer blue packaging", list(sample_frame.columns)) is None
        assert (
            read_claim("Exclude nonsense rows", list(sample_frame.columns), sample_frame)
            is None
        )

    def test_a_wrong_peak_is_detected_as_a_contradiction(self, sample_frame, sample_profile):
        frame = sample_frame.copy()
        frame["order_date"] = pd.to_datetime(frame["order_date"])
        profile = profile_dataset(frame)

        result = evaluate_claim(Claim(kind="peak_period", period="March"), frame, profile)
        assert result.contradicts
        assert result.observed == "November"

    def test_a_right_peak_holds(self, sample_frame):
        frame = sample_frame.copy()
        frame["order_date"] = pd.to_datetime(frame["order_date"])
        profile = profile_dataset(frame)

        result = evaluate_claim(Claim(kind="peak_period", period="November"), frame, profile)
        assert result.holds

    def test_a_rule_nothing_meets_is_a_contradiction(self, sample_frame, sample_profile):
        claim = Claim(kind="threshold", column="revenue", operator=">", value=10**12)
        result = evaluate_claim(claim, sample_frame, sample_profile)
        assert result.contradicts

    def test_a_rule_about_a_missing_column_is_not_a_contradiction(
        self, sample_frame, sample_profile
    ):
        # Out of scope for this file is not the same as wrong.
        claim = Claim(kind="threshold", column="not_here", value=5)
        result = evaluate_claim(claim, sample_frame, sample_profile)
        assert not result.testable
        assert not result.contradicts

    def test_a_fact_survives_a_round_trip_with_its_claim(self):
        memory = BusinessMemory()
        memory.remember(
            "VIPs spend over 5000",
            category="definition",
            claim=Claim(kind="threshold", column="revenue", value=5000, label="VIP"),
        )
        restored = BusinessMemory.from_list(memory.to_list())
        fact = restored.facts()[0]
        assert fact.claim is not None
        assert fact.claim.value == 5000
        assert fact.claim.label == "VIP"

    def test_restating_a_known_fact_with_structure_upgrades_it(self):
        memory = BusinessMemory()
        memory.remember("VIPs spend over 5000")
        assert memory.facts()[0].claim is None

        memory.remember(
            "VIPs spend over 5000", claim=Claim(kind="threshold", column="revenue", value=5000)
        )
        assert memory.facts()[0].claim is not None


class TestContradictionFlow:
    def test_a_disagreement_is_raised_and_the_fact_survives_being_kept(self, sample_path):
        memory = BusinessMemory()
        memory.remember("Peak season starts in March", category="seasonality")

        state = PipelineState(mode=RunMode.AUTONOMOUS)
        state.source_path = sample_path
        state.memory = memory
        Supervisor(state).run_to_completion()

        raised = [
            event
            for event in state.log.for_stage("recall")
            if "does not match" in event.message
        ]
        assert raised, "the disagreement should have been put to the user"

        fact = next(item for item in state.memory if "March" in item.statement)
        assert fact.disputed, "kept, but marked so it is not raised again"

    def test_a_threshold_with_a_label_becomes_a_column_without_asking(self, sample_path):
        memory = BusinessMemory()
        memory.remember(
            "VIP customers are the ones whose revenue exceeds 3000",
            category="definition",
        )

        state = PipelineState(mode=RunMode.AUTONOMOUS)
        state.source_path = sample_path
        state.memory = memory
        Supervisor(state).run_to_completion()

        assert "vip_customers" in state.frame.columns
        assert set(state.frame["vip_customers"].unique()) <= {"Vip Customers", "Other"}


# ---------------------------------------------------------------------------
# C3 — comparison across runs
# ---------------------------------------------------------------------------


class TestComparison:
    def test_the_fingerprint_ignores_the_row_count(self, sample_frame):
        assert comparison.fingerprint(sample_frame) == comparison.fingerprint(
            sample_frame.head(50)
        )

    def test_the_fingerprint_changes_when_the_columns_do(self, sample_frame):
        assert comparison.fingerprint(sample_frame) != comparison.fingerprint(
            sample_frame.drop(columns=["region"])
        )

    def test_two_runs_over_the_same_shape_are_compared(self, tmp_path, sample_frame):
        frame = sample_frame.copy()
        frame["order_date"] = pd.to_datetime(frame["order_date"])
        first = frame[frame["order_date"] < "2024-01-01"].copy()
        second = frame[frame["order_date"] >= "2024-01-01"].copy()
        for part, name in ((first, "one"), (second, "two")):
            part["order_date"] = part["order_date"].dt.strftime("%Y-%m-%d")
            part.to_csv(tmp_path / f"{name}.csv", index=False)

        results = []
        for name in ("one", "two"):
            state = PipelineState(mode=RunMode.AUTONOMOUS)
            state.source_path = tmp_path / f"{name}.csv"
            Supervisor(state).run_to_completion()
            results.append(state)

        assert results[0].comparison is None, "nothing to compare the first run with"
        assert results[1].comparison is not None
        assert results[1].comparison.material

    def test_a_change_finding_survives_grounding(self, tmp_path, sample_frame):
        # The previous run's figures are real facts. Without them every change
        # finding cites a number absent from this file and gets stripped.
        frame = sample_frame.copy()
        frame["order_date"] = pd.to_datetime(frame["order_date"])
        for name, part in (
            ("one", frame[frame["order_date"] < "2024-01-01"]),
            ("two", frame[frame["order_date"] >= "2024-01-01"]),
        ):
            copy = part.copy()
            copy["order_date"] = copy["order_date"].dt.strftime("%Y-%m-%d")
            copy.to_csv(tmp_path / f"{name}.csv", index=False)

        last = None
        for name in ("one", "two"):
            state = PipelineState(mode=RunMode.AUTONOMOUS)
            state.source_path = tmp_path / f"{name}.csv"
            Supervisor(state).run_to_completion()
            last = state

        change_findings = [item for item in last.insights if "since" in item.title]
        assert change_findings, "movements against the last run should be findings"

    def test_a_figure_whose_label_is_not_a_number_is_left_out(self):
        from insightlab.core.state import Kpi

        assert not comparison._is_numeric_measure(
            Kpi(name="Strongest month", value=824209.0, display_value="2023-11", formula="")
        )
        assert comparison._is_numeric_measure(
            Kpi(name="Total revenue", value=11.8e6, display_value="11.81M", formula="")
        )


# ---------------------------------------------------------------------------
# D1 — several files
# ---------------------------------------------------------------------------


@pytest.fixture
def three_files(tmp_path, sample_frame):
    sales = sample_frame[
        ["order_id", "order_date", "customer_id", "product_category", "quantity", "revenue"]
    ].copy()
    sales.to_csv(tmp_path / "sales.csv", index=False)

    generator = np.random.default_rng(3)
    products = pd.DataFrame({"product_category": sorted(sample_frame["product_category"].unique())})
    products["unit_cost"] = generator.uniform(30, 1800, len(products)).round(2)
    products.to_csv(tmp_path / "products.csv", index=False)

    customers = sample_frame[["customer_id", "customer_segment"]].drop_duplicates("customer_id")
    customers.to_csv(tmp_path / "customers.csv", index=False)
    return tmp_path


class TestJoining:
    def test_the_fact_table_becomes_the_base(self, three_files, sample_frame):
        tables = []
        for name in ("sales", "products", "customers"):
            frame = pd.read_csv(three_files / f"{name}.csv")
            tables.append(joining.Table(name, frame, profile_dataset(frame)))
        assert joining.choose_base(tables).name == "sales"

    def test_a_join_is_proposed_from_measured_overlap(self, three_files):
        sales = pd.read_csv(three_files / "sales.csv")
        products = pd.read_csv(three_files / "products.csv")
        base = joining.Table("sales", sales, profile_dataset(sales))
        other = joining.Table("products", products, profile_dataset(products))

        candidates = joining.find_candidates(base, other)
        assert candidates
        assert candidates[0].left_column == "product_category"
        assert candidates[0].overlap == pytest.approx(1.0)
        assert candidates[0].is_safe

    def test_a_lookup_covering_too_little_is_not_offered(self, three_files):
        # A supplier table naming two of your five categories is not the join
        # the user meant, however well those two match.
        sales = pd.read_csv(three_files / "sales.csv")
        partial = pd.DataFrame(
            {
                "product_category": ["Electronics", "Furniture"],
                "supplier": ["a", "b"],
            }
        )
        base = joining.Table("sales", sales, profile_dataset(sales))
        other = joining.Table("partial", partial, profile_dataset(partial))
        assert joining.find_candidates(base, other) == []

    def test_fan_out_is_computed_before_the_join_not_after(self, three_files):
        sales = pd.read_csv(three_files / "sales.csv")
        # Every category covered, two suppliers for one of them: the ordinary
        # shape of a lookup that silently multiplies rows.
        categories = sorted(sales["product_category"].unique())
        suppliers = pd.DataFrame(
            {
                "product_category": categories + [categories[0]],
                "supplier": [f"s{index}" for index in range(len(categories) + 1)],
            }
        )
        base = joining.Table("sales", sales, profile_dataset(sales))
        other = joining.Table("suppliers", suppliers, profile_dataset(suppliers))

        candidate = joining.find_candidates(base, other)[0]
        assert not candidate.right_is_unique
        assert candidate.expected_rows > candidate.left_rows
        assert "multiply" in candidate.describe() or "would turn" in candidate.describe()

        result = joining.apply_join(sales, other, candidate)
        assert result.rows_after == candidate.expected_rows, "the estimate must be exact"

    def test_three_files_become_one_analysis(self, three_files):
        state = PipelineState(mode=RunMode.AUTONOMOUS)
        state.source_path = three_files / "sales.csv"
        state.extra_paths = [three_files / "products.csv", three_files / "customers.csv"]
        Supervisor(state).run_to_completion()

        assert not state.errors
        assert "unit_cost" in state.frame.columns
        assert "customer_segment" in state.frame.columns

    def test_a_per_unit_cost_is_multiplied_out_before_profit(self, three_files):
        # unit_cost is a rate. Subtracting it from an order total would
        # overstate profit by the order size.
        state = PipelineState(mode=RunMode.AUTONOMOUS)
        state.source_path = three_files / "sales.csv"
        state.extra_paths = [three_files / "products.csv"]
        Supervisor(state).run_to_completion()

        frame = state.frame
        expected = frame["revenue"] - frame["unit_cost"] * frame["quantity"]
        assert (frame["profit"] - expected).abs().max() < 0.01


# ---------------------------------------------------------------------------
# F1 — Arabic
# ---------------------------------------------------------------------------


class TestLanguage:
    def test_a_missing_translation_falls_back_to_english(self):
        language_module.STRINGS["test.only_english"] = {"en": "Only English"}
        try:
            assert language_module.translate("test.only_english", ARABIC) == "Only English"
        finally:
            del language_module.STRINGS["test.only_english"]

    def test_an_unknown_key_returns_itself_loudly(self):
        assert language_module.translate("no.such.key", ENGLISH) == "no.such.key"

    def test_every_arabic_string_is_actually_arabic(self):
        from insightlab.reports.arabic import contains_arabic

        for key, entry in language_module.STRINGS.items():
            arabic_text = entry.get("ar")
            if arabic_text:
                assert contains_arabic(arabic_text), f"{key} is not translated"

    def test_the_run_carries_its_language_into_the_agents(self):
        state = PipelineState(language=ARABIC)
        supervisor = Supervisor(state, agents=[])
        assert supervisor.reasoning.language.code == "ar"

    def test_switching_language_drops_cached_agents(self):
        from insightlab.core.reasoning import ReasoningEngine

        engine = ReasoningEngine()
        engine._agents["stale"] = object()
        engine.set_language(ARABIC)
        assert not engine._agents, "a cached agent would keep the old language"

    def test_the_language_is_recorded_with_the_run(self, sample_frame):
        state = PipelineState(language=ARABIC)
        state.frame = sample_frame
        state.raw_frame = sample_frame
        state.profile = profile_dataset(sample_frame)
        assert state.summary_dict()["language"] == "ar"


class TestArabicRendering:
    def test_english_is_left_untouched(self):
        from insightlab.reports import arabic

        assert arabic.prepare("Total revenue 11.8M") == "Total revenue 11.8M"

    def test_arabic_is_reshaped_and_reordered(self):
        from insightlab.reports import arabic

        source = "إجمالي الإيرادات"
        prepared = arabic.prepare(source)
        assert prepared != source
        # Shaping substitutes contextual forms and may fuse a pair into one
        # ligature - lam followed by alef becomes a single glyph - so the
        # result can be shorter, but it must never grow.
        assert len(prepared) <= len(source)

    def test_a_font_without_digits_is_rejected(self):
        # macOS ships SFArabic: full Arabic coverage, no Latin digits. A product
        # about numbers cannot use it - every figure would be an empty box.
        from insightlab.reports import arabic

        candidate = Path("/System/Library/Fonts/SFArabic.ttf")
        if not candidate.exists():
            pytest.skip("that font is not on this machine")
        assert not arabic.covers_both(candidate)

    def test_an_arabic_pdf_contains_no_missing_glyphs(self, tmp_path):
        from insightlab.reports import arabic
        from insightlab.reports import write_pdf
        from insightlab.reports.content import ReportContent, Section

        if not arabic.available():
            pytest.skip("no Arabic-capable font on this machine")

        state = PipelineState(language=ARABIC)
        state.source_name = "ملف.csv"
        content = ReportContent(
            title="تحليل المبيعات",
            subtitle="أُعِدّ في 2 أغسطس 2026",
            sections=[
                Section(
                    title="الملخص",
                    paragraphs=["إجمالي الإيرادات 11.81 مليون بهامش 32.1 بالمئة."],
                    table=(["المنطقة", "الإيرادات"], [["القاهرة", "4,434,708"]]),
                )
            ],
        )
        target = tmp_path / "arabic.pdf"
        write_pdf(state, target, content)

        assert target.stat().st_size > 0
        pypdf = pytest.importorskip("pypdf")
        text = pypdf.PdfReader(str(target)).pages[0].extract_text()
        assert "\x00" not in text, "a missing glyph renders as an empty box"
        assert "4,434,708" in text, "digits must survive into the document"

    def test_a_run_in_arabic_completes(self, sample_path):
        state = PipelineState(mode=RunMode.AUTONOMOUS, language=ARABIC)
        state.source_path = sample_path
        Supervisor(state).run_to_completion()
        assert not state.errors
        assert state.insights
