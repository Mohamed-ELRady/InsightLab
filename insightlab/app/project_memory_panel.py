"""Project memory, feedback, and owner-controlled lesson approval UI."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from ..core.business_memory import CATEGORIES, BusinessMemory
from ..core.language import Language, axis_label, kpi_name
from ..core.project_memory import ProjectMemoryStore
from ..core.evaluation import evaluate_run
from ..core.state import Chart, Insight, Kpi, PipelineState


def _copy(language: Language, english: str, arabic: str) -> str:
    return arabic if language.code == "ar" else english


OPTION_LABELS_AR = {
    "context": "سياق العمل",
    "definition": "تعريف",
    "seasonality": "موسمية",
    "exclusion": "استبعاد",
    "classification": "تصنيف",
    "target": "هدف",
    "preference": "تفضيل",
    "other": "أخرى",
    "concise": "مختصر",
    "balanced": "متوازن",
    "detailed": "مفصّل",
    "guided": "بإرشاد خطوة بخطوة",
    "fewer questions": "أسئلة أقل",
    "business owner": "صاحب المشروع",
    "data owner": "صاحب البيانات",
    "team": "فريق العمل",
    "executives": "الإدارة التنفيذية",
    "clients": "العملاء",
    "project": "المشروع بالكامل",
    "dataset": "بيانات بأعمدة محددة",
    "active": "نشطة",
    "archived": "مؤرشفة",
    "expired": "منتهية",
    "draft": "مسودة",
    "validated": "تم اختبارها",
    "approved": "معتمدة",
    "rejected": "مرفوضة",
    "user": "المستخدم",
    "model": "النموذج",
    "created": "أُنشئت",
    "updated": "عُدّلت",
    "restored": "استُعيدت",
    "status": "تغيير الحالة",
}


def _option(language: Language, value: str) -> str:
    """Translate a displayed option while preserving its stable stored value."""
    return OPTION_LABELS_AR.get(value, value) if language.code == "ar" else value


def _store(state: PipelineState) -> ProjectMemoryStore:
    return ProjectMemoryStore(state.workspace)


def reload_state_memory(state: PipelineState, store: ProjectMemoryStore | None = None) -> None:
    if state.project_id:
        repository = store or _store(state)
        # Preserve facts learned earlier in this still-running analysis before
        # refreshing the sidebar from durable memory.
        repository.sync_memory(state.project_id, state.memory)
        columns = (
            [str(name) for name in state.frame.columns]
            if state.frame is not None else None
        )
        loaded = repository.load_memory(state.project_id, columns=columns)
        if state.frame is not None and state.understanding.domain_family != "general":
            from ..agents.memory_agent import MemoryAgent

            loaded = BusinessMemory(
                fact for fact in loaded if MemoryAgent._relevant_to_domain(fact, state)
            )
        state.memory = loaded


def project_picker(language: Language) -> str:
    """Choose a stable memory boundary before a run starts."""
    store = ProjectMemoryStore()
    projects = store.list_projects()
    names = [project.name for project in projects]
    create_value = "__create_new_project__"
    create_label = _copy(language, "＋ New project", "＋ مشروع جديد")
    options = [*names, create_value]
    default = 0 if names else len(options) - 1
    # Use a language-specific widget identity. Streamlit caches rendered option
    # labels by widget key, so reusing one key can leave the old language on
    # screen even when ``format_func`` already returns the new translation.
    choice_key = (
        "memory_project_choice" if language.code == "en"
        else f"memory_project_choice_{language.code}"
    )
    chosen = st.selectbox(
        _copy(language, "Memory project", "مشروع الذاكرة"),
        options,
        index=default,
        key=choice_key,
        format_func=lambda value: create_label if value == create_value else value,
        help=_copy(
            language,
            "Analyses in one project share approved domain rules and corrections. Projects never share memory.",
            "التحليلات داخل نفس المشروع بتشارك قواعد المجال والتصحيحات المعتمدة. الذاكرة لا تنتقل بين المشاريع.",
        ),
    )
    if chosen != create_value:
        project_name = chosen
    else:
        name_key = (
            "new_memory_project_name" if language.code == "en"
            else f"new_memory_project_name_{language.code}"
        )
        project_name = st.text_input(
            _copy(language, "New project name", "اسم المشروع الجديد"),
            value="My project" if language.code != "ar" else "مشروعي",
            max_chars=100,
            key=name_key,
        ).strip()

    legacy = sorted(
        store.root.glob("*/business_memory.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if legacy:
        with st.expander(_copy(language, "Import old run memory (one time)", "استورد ذاكرة تحليل قديم (مرة واحدة)")):
            chosen_legacy = st.selectbox(
                _copy(language, "Previous analysis", "التحليل السابق"),
                [""] + [str(path) for path in legacy[:20]],
                format_func=lambda value: _copy(language, "Do not import", "بدون استيراد")
                if not value else Path(value).parent.name,
                key="legacy_memory_import",
            )
            if chosen_legacy:
                st.caption(_copy(language, "Its facts will be copied into this project when the analysis starts.", "المعلومات هتتنقل للمشروع ده عند بدء التحليل."))
    return project_name


def persist_completed_run(state: PipelineState) -> None:
    """Idempotent final sync after the report stage has settled."""
    if not state.project_id:
        return
    store = _store(state)
    store.sync_memory(state.project_id, state.memory)
    store.record_run(
        state.run_id, state.project_id, state.source_name, state.started_at,
        completed=state.is_complete,
    )
    evaluation = evaluate_run(state)
    store.record_evaluation(
        state.run_id, state.project_id, score=evaluation.score,
        metrics=evaluation.metrics, cases=evaluation.cases,
    )


def render_sidebar(state: PipelineState, language: Language) -> None:
    store = _store(state)
    metrics = store.metrics(state.project_id)
    visible_memory = list(state.memory)
    if state.frame is not None and state.understanding.domain_family != "general":
        from ..agents.memory_agent import MemoryAgent

        visible_memory = [
            fact for fact in visible_memory
            if MemoryAgent._relevant_to_domain(fact, state)
        ]
    st.markdown(f"**{_copy(language, 'Project memory', 'ذاكرة المشروع')}**")
    st.caption(
        f"{state.project_name} · {len(visible_memory)} "
        + _copy(language, "relevant active memories", "معلومة نشطة مرتبطة بهذه البيانات")
    )
    if metrics["pending_lessons"]:
        st.warning(
            _copy(language, f"{metrics['pending_lessons']} lesson(s) need your review.",
                  f"{metrics['pending_lessons']} درس محتاج مراجعتك.")
        )
    for fact in visible_memory[-5:]:
        st.write(f"• {fact.statement}")
        st.caption(f"{_option(language, fact.category)} · {_copy(language, 'confirmed', 'مؤكدة')}")

    with st.expander(_copy(language, "Add a project rule", "أضف قاعدة عن المشروع")):
        text = st.text_area(
            _copy(language, "What should InsightLab remember?", "إيه اللي تحب InsightLab يفتكره؟"),
            key="project_memory_input", height=80,
            placeholder=_copy(language, "Example: treat readings above this threshold as high priority.", "مثال: اعتبر القراءات التي تتجاوز حدًا معينًا عالية الأولوية."),
        )
        category = st.selectbox(
            _copy(language, "Type", "النوع"), list(CATEGORIES),
            format_func=lambda value: _option(language, value),
            key="project_memory_category",
        )
        if st.button(_copy(language, "Remember this", "افتكر ده"), key="project_memory_add", width="stretch"):
            if text.strip():
                store.add_memory(
                    state.project_id, text, category=category, confidence="high",
                    source_type="user", source_stage="user", source_topic="Added from memory panel",
                )
                reload_state_memory(state, store)
                st.rerun()
            else:
                st.error(_copy(language, "Write the rule first.", "اكتب القاعدة الأول."))


def render_tab(state: PipelineState, language: Language) -> None:
    store = _store(state)
    st.subheader(_copy(language, "Project memory and learning", "ذاكرة المشروع والتعلّم"))
    st.caption(_copy(
        language,
        "Only active, approved memories affect future analyses. Model suggestions and corrections wait here until you approve them.",
        "المعلومات النشطة والمعتمدة فقط بتأثر على التحليلات القادمة. اقتراحات النموذج والتصحيحات بتستنى موافقتك هنا.",
    ))

    st.markdown(f"#### {_copy(language, 'Your preferences', 'تفضيلاتك')}")
    preferences = store.preferences(state.project_id)
    detail_options = ["concise", "balanced", "detailed"]
    question_options = ["guided", "fewer questions"]
    audience_options = ["data owner", "team", "executives", "clients"]
    saved_audience = preferences.get("report_audience", "data owner")
    if saved_audience == "business owner":
        saved_audience = "data owner"
    detail = st.selectbox(
        _copy(language, "Explanation detail", "تفاصيل الشرح"), detail_options,
        index=detail_options.index(preferences.get("explanation_detail", "balanced"))
        if preferences.get("explanation_detail", "balanced") in detail_options else 1,
        key="preference_detail", format_func=lambda value: _option(language, value),
    )
    question_style = st.selectbox(
        _copy(language, "Question style", "أسلوب الأسئلة"), question_options,
        index=question_options.index(preferences.get("question_style", "guided"))
        if preferences.get("question_style", "guided") in question_options else 0,
        key="preference_questions", format_func=lambda value: _option(language, value),
    )
    audience = st.selectbox(
        _copy(language, "Usual report audience", "الجمهور المعتاد للتقرير"), audience_options,
        index=audience_options.index(saved_audience)
        if saved_audience in audience_options else 0,
        key="preference_audience", format_func=lambda value: _option(language, value),
    )
    if st.button(_copy(language, "Save preferences", "احفظ التفضيلات"), key="save_project_preferences"):
        store.set_preferences(state.project_id, {
            "explanation_detail": detail,
            "question_style": question_style,
            "report_audience": audience,
        })
        state.project_preferences = store.preferences(state.project_id)
        st.success(_copy(language, "Preferences saved for future analyses.", "التفضيلات اتحفظت للتحليلات الجاية."))

    candidates = store.list_candidates(state.project_id, "pending")
    st.markdown(f"#### {_copy(language, 'Lessons waiting for approval', 'دروس في انتظار الموافقة')} ({len(candidates)})")
    if not candidates:
        st.info(_copy(language, "No pending lessons.", "مفيش دروس معلّقة."))
    for candidate in candidates:
        with st.container(border=True):
            st.caption(_copy(language, "Proposed from your correction", "مقترح من تصحيحك"))
            edited = st.text_area(
                _copy(language, "Rule to remember", "القاعدة المطلوب حفظها"),
                value=candidate.statement, key=f"candidate_text_{candidate.id}", height=80,
            )
            if candidate.reason:
                st.caption(f"{_copy(language, 'Why', 'السبب')}: {candidate.reason}")
            approve, reject = st.columns(2)
            if approve.button(_copy(language, "Approve and remember", "اعتمد واحفظ"), key=f"approve_{candidate.id}", width="stretch"):
                try:
                    store.review_candidate(candidate.id, approve=True, edited_statement=edited)
                    reload_state_memory(state, store)
                    st.rerun()
                except ValueError as error:
                    st.error(str(error))
            if reject.button(_copy(language, "Reject", "ارفض"), key=f"reject_{candidate.id}", width="stretch"):
                store.review_candidate(candidate.id, approve=False)
                st.rerun()

    st.markdown(f"#### {_copy(language, 'Memory records', 'سجل الذاكرة')}")
    records = store.list_memory(state.project_id)
    if not records:
        st.info(_copy(language, "This project has no saved memory yet.", "المشروع ده مفيهوش معلومات محفوظة لسه."))
        return
    for item in records:
        marker = "●" if item.status == "active" else "○"
        with st.expander(f"{marker} {item.statement[:90]}"):
            edited = st.text_area(
                _copy(language, "Statement", "المعلومة"), value=item.statement,
                key=f"memory_edit_{item.id}", height=80,
            )
            category = st.selectbox(
                _copy(language, "Category", "التصنيف"), list(CATEGORIES),
                index=list(CATEGORIES).index(item.category) if item.category in CATEGORIES else len(CATEGORIES) - 1,
                format_func=lambda value: _option(language, value),
                key=f"memory_category_{item.id}",
            )
            scope = st.selectbox(
                _copy(language, "Scope", "النطاق"), ["project", "dataset"],
                index=0 if item.scope == "project" else 1,
                format_func=lambda value: _option(language, value),
                key=f"memory_scope_{item.id}",
                help=_copy(language, "Dataset-scoped rules are reserved for matching columns; project rules apply throughout this project.", "قواعد البيانات ترتبط بالأعمدة المطابقة، وقواعد المشروع تنطبق على المشروع كله."),
            )
            valid_to = st.text_input(
                _copy(language, "Valid until (YYYY-MM-DD, optional)", "صالحة لحد (YYYY-MM-DD، اختياري)"),
                value=item.valid_to or "", key=f"memory_valid_to_{item.id}",
            )
            related_columns = st.text_input(
                _copy(language, "Related columns (comma separated)", "الأعمدة المرتبطة (افصل بفاصلة)"),
                value=", ".join(item.columns), key=f"memory_columns_{item.id}",
                disabled=scope != "dataset",
            )
            st.caption(
                f"{_copy(language, 'Status', 'الحالة')}: {_option(language, item.status)} · "
                f"{_copy(language, 'Source', 'المصدر')}: {_option(language, item.source_type)} · v{item.version}"
            )
            if item.disputed:
                st.warning(_copy(
                    language,
                    "This rule conflicted with a later dataset and is kept for review.",
                    "القاعدة دي تعارضت مع بيانات أحدث، ومحتفظين بيها للمراجعة.",
                ))
            save, toggle = st.columns(2)
            if save.button(_copy(language, "Save changes", "احفظ التعديل"), key=f"save_memory_{item.id}", width="stretch"):
                columns = [part.strip() for part in related_columns.split(",") if part.strip()]
                if scope == "dataset" and not columns:
                    st.error(_copy(
                        language,
                        "Add at least one related column for a dataset-scoped rule.",
                        "أضف اسم عمود واحد على الأقل للقاعدة المرتبطة بالبيانات.",
                    ))
                else:
                    try:
                        store.update_memory(
                            item.id, statement=edited, category=category, scope=scope,
                            valid_to=valid_to, columns=columns,
                        )
                        reload_state_memory(state, store)
                        st.rerun()
                    except ValueError:
                        st.error(_copy(language, "Use a valid date such as 2027-12-31.", "اكتب تاريخ صحيح زي 2027-12-31."))
            target = "archived" if item.status == "active" else "active"
            label = _copy(language, "Disable", "عطّل") if target == "archived" else _copy(language, "Reactivate", "فعّل تاني")
            if toggle.button(label, key=f"toggle_memory_{item.id}", width="stretch"):
                store.update_memory(
                    item.id, status=target,
                    valid_to=None if item.status == "expired" else ...,
                    reason="Status changed by owner",
                )
                reload_state_memory(state, store)
                st.rerun()
            history = store.history(item.id)
            if len(history) > 1:
                st.caption(_copy(language, f"{len(history)} saved versions; full history is retained.", f"محفوظ {len(history)} إصدار، والتاريخ الكامل موجود."))
                older = [entry for entry in history if entry["version"] != item.version]
                if older:
                    labels = {
                        entry["version"]: (
                            f"v{entry['version']} · {_option(language, entry['change_kind'])} · "
                            f"{entry['snapshot']['statement'][:55]}"
                        )
                        for entry in older
                    }
                    selected_version = st.selectbox(
                        _copy(language, "Previous version", "إصدار سابق"),
                        list(labels), format_func=lambda value: labels[value],
                        key=f"history_{item.id}",
                    )
                    if st.button(
                        _copy(language, "Restore selected version", "استرجع الإصدار المحدد"),
                        key=f"restore_{item.id}", width="stretch",
                    ):
                        store.restore_version(item.id, selected_version)
                        reload_state_memory(state, store)
                        st.rerun()

    metrics = store.metrics(state.project_id)
    st.markdown(f"#### {_copy(language, 'Learning summary', 'ملخص التعلّم')}")
    st.write(
        _copy(
            language,
            f"{metrics['useful']} useful ratings · {metrics['corrections']} corrections · {metrics['pending_lessons']} pending lessons",
            f"{metrics['useful']} تقييم مفيد · {metrics['corrections']} تصحيح · {metrics['pending_lessons']} درس معلّق",
        )
    )


def render_insight_feedback(
    state: PipelineState,
    insight: Insight,
    language: Language,
    item_index: int = 0,
) -> None:
    """Element-level feedback; only an explicit correction creates a lesson."""
    store = _store(state)
    item_key = store.item_key(str(item_index), insight.title, insight.evidence, insight.action)
    existing = store.feedback_for(state.run_id, "insight", item_key)
    if existing:
        labels = {
            "useful": _copy(language, "Marked useful", "تم تحديدها كمفيدة"),
            "not_useful": _copy(language, "Marked not useful", "تم تحديدها كغير مفيدة"),
            "incorrect": _copy(language, "Correction submitted for review", "التصحيح اتسجل للمراجعة"),
        }
        st.caption(f"✓ {labels.get(existing['rating'], existing['rating'])}")
        return

    st.caption(_copy(language, "Help the next analysis improve:", "ساعد التحليل الجاي يتحسن:"))
    useful, not_useful = st.columns(2)
    feedback_reason = st.text_input(
        _copy(language, "Optional note about usefulness", "ملاحظة اختيارية عن مدى الفائدة"),
        key=f"feedback_reason_{item_key}",
    )
    if useful.button(_copy(language, "Useful", "مفيدة"), key=f"useful_{item_key}", width="stretch"):
        store.record_feedback(
            run_id=state.run_id, project_id=state.project_id, item_type="insight",
            item_key=item_key, rating="useful", reason=feedback_reason,
            context={"axis": insight.axis, "confidence": insight.confidence},
        )
        st.rerun()
    if not_useful.button(_copy(language, "Not useful", "مش مفيدة"), key=f"not_useful_{item_key}", width="stretch"):
        store.record_feedback(
            run_id=state.run_id, project_id=state.project_id, item_type="insight",
            item_key=item_key, rating="not_useful", reason=feedback_reason,
            context={"axis": insight.axis, "confidence": insight.confidence},
        )
        st.rerun()
    with st.expander(_copy(language, "Incorrect — teach InsightLab", "غير صحيحة — صحّح لـInsightLab")):
        correction = st.text_area(
            _copy(language, "What should be true instead?", "إيه الصحيح اللي المفروض يفتكره؟"),
            key=f"correction_{item_key}", height=80,
            placeholder=_copy(language, "Write a reusable domain rule, not just 'this is wrong'.", "اكتب قاعدة واضحة عن المجال تتستخدم بعد كده، مش مجرد «ده غلط»."),
        )
        reason = st.text_input(
            _copy(language, "Why was it wrong? (optional)", "ليه كانت غلط؟ (اختياري)"),
            key=f"correction_reason_{item_key}",
        )
        if st.button(_copy(language, "Submit correction for review", "سجّل التصحيح للمراجعة"), key=f"submit_correction_{item_key}", width="stretch"):
            if not correction.strip():
                st.error(_copy(language, "Write the correction first.", "اكتب التصحيح الأول."))
            else:
                store.record_feedback(
                    run_id=state.run_id, project_id=state.project_id,
                    item_type="insight", item_key=item_key, rating="incorrect",
                    correction=correction, reason=reason,
                    context={"axis": insight.axis, "confidence": insight.confidence},
                )
                st.rerun()


def render_simple_feedback(
    state: PipelineState, item: Kpi | Chart, language: Language, *, item_type: str,
) -> None:
    """Small preference signal for KPIs and charts; never creates business facts."""
    store = _store(state)
    if item_type == "kpi":
        item_key = store.item_key(item.name, item.formula)
        context = {"kpi": item.name}
    else:
        item_key = store.item_key(item.id, item.title, item.kind)
        context = {"axis": item.axis, "chart_kind": item.kind}
    existing = store.feedback_for(state.run_id, item_type, item_key)
    if existing:
        st.caption(_copy(language, "✓ Preference saved", "✓ التفضيل اتحفظ"))
        return
    yes, no = st.columns(2)
    if yes.button(_copy(language, "Useful", "مفيد"), key=f"{item_type}_useful_{item_key}", width="stretch"):
        store.record_feedback(
            run_id=state.run_id, project_id=state.project_id, item_type=item_type,
            item_key=item_key, rating="useful", context=context,
        )
        st.rerun()
    if no.button(_copy(language, "Not useful", "مش مفيد"), key=f"{item_type}_not_useful_{item_key}", width="stretch"):
        store.record_feedback(
            run_id=state.run_id, project_id=state.project_id, item_type=item_type,
            item_key=item_key, rating="not_useful", context=context,
        )
        st.rerun()


def render_run_review(
    state: PipelineState, language: Language
) -> tuple[str, str] | None:
    """Evaluate the whole run and return ``(action, instruction)`` when needed."""
    store = _store(state)
    item_key = "overall_result"
    existing = store.feedback_for(state.run_id, "run", item_key)
    st.divider()
    with st.container(border=True):
        st.subheader(_copy(
            language,
            "Did this analysis understand your data?",
            "هل التحليل فهم بياناتك وحقق اللي كنت محتاجه؟",
        ))
        st.caption(_copy(
            language,
            "If something is wrong, describe the result you want. Apply it quickly to the current prepared data, or rebuild every stage from the original file.",
            "لو في حاجة غلط، اكتب النتيجة اللي كنت عايزها. تقدر تطبّقها بسرعة على البيانات المجهزة حاليًا، أو تعيد كل المراحل من الملف الأصلي.",
        ))
        if existing:
            st.success(_copy(language, "Your overall review was saved.", "تم حفظ تقييمك العام."))
            return None

        useful, needs_work = st.columns(2)
        if useful.button(
            _copy(language, "Yes, this is useful", "أيوه، النتيجة مفيدة"),
            key=f"run_review_useful_{state.run_id}", width="stretch",
        ):
            store.record_feedback(
                run_id=state.run_id, project_id=state.project_id,
                item_type="run", item_key=item_key, rating="useful",
                context={"domain_family": state.understanding.domain_family},
            )
            st.rerun()

        with needs_work:
            st.caption(_copy(language, "Needs a change? Tell us below.", "محتاج تعديل؟ اشرح تحت."))

        instruction = st.text_area(
            _copy(language, "What is wrong, or what should the result do instead?", "إيه المشكلة، أو عايز النتيجة تعمل إيه بدل كده؟"),
            key=f"run_review_instruction_{state.run_id}", height=110,
            placeholder=_copy(
                language,
                "Example: focus on earthquake frequency by region, keep rare extreme events, and explain the relationship between depth and magnitude.",
                "مثال: ركّز على تكرار الزلازل حسب المنطقة، احتفظ بالأحداث النادرة الشديدة، واشرح العلاقة بين العمق والقوة.",
            ),
        )
        st.caption(_copy(
            language,
            "Your request is applied now. A reusable lesson is also queued for your approval before it can affect later analyses in this project.",
            "طلبك هيتنفذ دلوقتي، وكمان هيتحوّل لدرس قابل لإعادة الاستخدام ينتظر موافقتك قبل ما يأثر على التحليلات القادمة داخل نفس المشروع.",
        ))
        revise, rebuild = st.columns(2)
        revise_clicked = revise.button(
            _copy(language, "Modify current results", "عدّل النتيجة الحالية"),
            key=f"run_review_revise_{state.run_id}", type="primary", width="stretch",
            help=_copy(
                language,
                "Keeps the understood and cleaned data, then regenerates analysis, measures, insights, dashboards and reports.",
                "يحافظ على فهم البيانات وتنظيفها، ثم يعيد التحليل والمؤشرات والرؤى ولوحات المعلومات والتقارير.",
            ),
        )
        rebuild_clicked = rebuild.button(
            _copy(language, "Rebuild from original file", "أعد البناء من الملف الأصلي"),
            key=f"run_review_rebuild_{state.run_id}", width="stretch",
            help=_copy(
                language,
                "Reloads the original file and repeats understanding, cleaning and every later stage.",
                "يعيد تحميل الملف الأصلي ويكرر الفهم والتنظيف وكل المراحل التالية.",
            ),
        )
        if revise_clicked or rebuild_clicked:
            if not instruction.strip():
                st.error(_copy(language, "Describe the change you need first.", "اكتب التعديل المطلوب الأول."))
                return None
            clean = instruction.strip()
            action = "rebuild" if rebuild_clicked else "revise"
            store.record_feedback(
                run_id=state.run_id, project_id=state.project_id,
                item_type="run", item_key=item_key, rating="incorrect",
                correction=clean,
                reason=(
                    _copy(language, "Requested a full analysis rebuild", "طلب إعادة بناء التحليل بالكامل")
                    if action == "rebuild" else
                    _copy(language, "Requested a focused result revision", "طلب تعديل النتائج الحالية")
                ),
                category="target",
                context={
                    "domain_family": state.understanding.domain_family,
                    "rebuild": str(action == "rebuild").lower(),
                    "review_action": action,
                },
            )
            return action, clean
    return None


def render_improvement_tab(state: PipelineState, language: Language) -> None:
    """Quality trend and owner-controlled policy promotion/rollback."""
    store = _store(state)
    evaluations = store.evaluations(state.project_id)
    profile = store.adaptive_profile(state.project_id)
    active = store.active_policy(state.project_id)

    st.subheader(_copy(language, "Controlled self-improvement", "التحسين الذاتي المتحكَّم فيه"))
    st.caption(_copy(
        language,
        "The platform learns in three safe layers: ratings automatically improve ordering; your correction changes the requested result immediately; reusable domain rules wait for your approval. Calculations and evidence are never rewritten by learning weights.",
        "المنصة بتتعلم في 3 طبقات آمنة: التقييمات بتحسّن ترتيب العرض تلقائيًا، وتصحيحك يعدّل النتيجة المطلوبة فورًا، والقواعد القابلة لإعادة الاستخدام تنتظر موافقتك. أوزان التعلّم لا تغيّر الحسابات أو الأدلة.",
    ))
    if evaluations:
        latest = evaluations[0]
        previous = evaluations[1]["score"] if len(evaluations) > 1 else None
        delta = None if previous is None else round(latest["score"] - previous, 1)
        st.metric(_copy(language, "Latest quality score", "آخر تقييم للجودة"), f"{latest['score']:.1f}/100", delta)
        metrics = latest["metrics"]
        st.caption(_copy(
            language,
            f"{metrics['insight_count']} insights · {metrics['chart_count']} charts · {metrics['kpi_count']} KPIs · {metrics['error_count']} errors",
            f"{metrics['insight_count']} استنتاج · {metrics['chart_count']} رسم · {metrics['kpi_count']} مؤشر · {metrics['error_count']} أخطاء",
        ))
        if "domain_understanding" in metrics:
            domain_score = round(float(metrics["domain_understanding"]) * 100)
            semantic_score = round(float(metrics.get("semantic_consistency", 0)) * 100)
            st.caption(_copy(
                language,
                f"Domain understanding {domain_score}% · semantic safety {semantic_score}% · unsafe aggregations {metrics.get('unsafe_aggregation_count', 0)}",
                f"فهم المجال {domain_score}٪ · السلامة الدلالية {semantic_score}٪ · تجميعات غير آمنة {metrics.get('unsafe_aggregation_count', 0)}",
            ))
    else:
        st.info(_copy(language, "Complete an analysis to create the first quality baseline.", "كمّل تحليل علشان يتسجل أول خط أساس للجودة."))

    st.markdown(f"#### {_copy(language, 'What the platform learned', 'المنصة اتعلمت إيه')}")
    st.write(_copy(
        language,
        f"Based on {profile['feedback_count']} explicit ratings. Active policy: v{active.version} — {active.name}.",
        f"بناءً على {profile['feedback_count']} تقييم صريح. السياسة النشطة: v{active.version} — {active.name}.",
    ))
    preferred_axes = sorted(profile["axes"].items(), key=lambda pair: -pair[1])[:5]
    preferred_kpis = sorted(profile["kpis"].items(), key=lambda pair: -pair[1])[:5]
    if preferred_axes:
        st.caption(_copy(language, "Preferred analysis areas: ", "محاور التحليل المفضلة: ") + ", ".join(axis_label(name, language) for name, score in preferred_axes if score > 0))
    if preferred_kpis:
        st.caption(_copy(language, "Preferred KPIs: ", "مؤشرات الأداء المفضلة: ") + ", ".join(kpi_name(name, language) for name, score in preferred_kpis if score > 0))

    st.markdown(f"#### {_copy(language, 'Create a safe policy candidate', 'أنشئ سياسة تحسين آمنة')}")
    with st.expander(_copy(language, "Tune learning strength", "اضبط قوة التعلّم")):
        name = st.text_input(_copy(language, "Policy name", "اسم السياسة"), value=_copy(language, "Adaptive ranking", "ترتيب متكيف"), key="policy_name")
        rules = {
            "axis_weight": st.slider(_copy(language, "Analysis-area weight", "وزن محور التحليل"), 0.0, 3.0, float(active.rules["axis_weight"]), 0.25, key="axis_weight"),
            "kpi_weight": st.slider(_copy(language, "KPI weight", "وزن مؤشرات الأداء"), 0.0, 3.0, float(active.rules["kpi_weight"]), 0.25, key="kpi_weight"),
            "chart_weight": st.slider(_copy(language, "Chart-type weight", "وزن نوع الرسم"), 0.0, 3.0, float(active.rules["chart_weight"]), 0.25, key="chart_weight"),
            "confidence_weight": st.slider(_copy(language, "Confidence preference weight", "وزن تفضيل الثقة"), 0.0, 3.0, float(active.rules["confidence_weight"]), 0.25, key="confidence_weight"),
        }
        if st.button(_copy(language, "Create draft", "أنشئ مسودة"), key="create_policy"):
            store.create_policy_candidate(state.project_id, name, rules)
            st.rerun()

    st.markdown(f"#### {_copy(language, 'Policy versions', 'إصدارات السياسات')}")
    for policy in store.list_policies(state.project_id):
        with st.container(border=True):
            st.write(f"v{policy.version} · {policy.name} · **{_option(language, policy.status)}**")
            st.caption(" · ".join(f"{key}={value:g}" for key, value in policy.rules.items()))
            if policy.evaluation:
                st.caption(_copy(
                    language,
                    f"Offline check: {policy.evaluation.get('historical_feedback', 0)} ratings · alignment {policy.evaluation.get('preference_alignment', 0)}% ({policy.evaluation.get('alignment_delta', 0):+g}) · no raw data",
                    f"اختبار تاريخي: {policy.evaluation.get('historical_feedback', 0)} تقييم · توافق {policy.evaluation.get('preference_alignment', 0)}% ({policy.evaluation.get('alignment_delta', 0):+g}) · بدون بيانات خام",
                ))
            if policy.status == "draft" and st.button(_copy(language, "Run offline check", "شغّل الاختبار التاريخي"), key=f"validate_policy_{policy.id}"):
                store.validate_policy(policy.id)
                st.rerun()
            if policy.status == "validated" and st.button(_copy(language, "Approve and activate", "وافق وفعّل"), key=f"activate_policy_{policy.id}"):
                store.activate_policy(policy.id)
                st.rerun()
    if active.parent_id and st.button(_copy(language, "Roll back active policy", "ارجع للسياسة السابقة"), key="rollback_policy"):
        store.rollback_policy(state.project_id)
        st.rerun()
