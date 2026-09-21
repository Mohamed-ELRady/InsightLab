"""Interactive chart builder shown with the completed analysis."""

from __future__ import annotations

import streamlit as st

from ..analysis.custom_charts import (
    AGGREGATIONS,
    CHART_KINDS,
    CustomChartError,
    build_custom_chart,
    numeric_columns,
)
from ..core.activity_log import EventKind
from ..core.language import Language
from ..core.state import PipelineState
from ..core.storage import save_run

NONE = "__none__"


def _copy(language: Language, english: str, arabic: str) -> str:
    return arabic if language.code == "ar" else english


def _next_id(state: PipelineState) -> str:
    used = {chart.id for chart in state.charts}
    number = 1
    while f"custom_{number}" in used:
        number += 1
    return f"custom_{number}"


def render(state: PipelineState, language: Language) -> None:
    """Let the user make one deliberate chart without changing the dataset."""
    if state.frame is None or state.frame.empty:
        return

    ar = language.code == "ar"
    columns = [str(column) for column in state.frame.columns]
    numbers = numeric_columns(state.frame)
    labels = {
        "bar": _copy(language, "Bar · compare groups", "أعمدة · مقارنة المجموعات"),
        "line": _copy(language, "Line · follow change", "خطي · تتبّع التغيّر"),
        "area": _copy(language, "Area · show volume over an axis", "مساحي · عرض الحجم عبر محور"),
        "scatter": _copy(language, "Scatter · relate two numbers", "مبعثر · علاقة رقمين"),
        "histogram": _copy(language, "Histogram · see a distribution", "توزيع تكراري · شكل القيم"),
        "box": _copy(language, "Box plot · compare spread", "صندوقي · مقارنة الانتشار"),
        "pie": _copy(language, "Donut · show shares", "دائري · عرض الحصص"),
        "heatmap": _copy(language, "Heatmap · cross two categories", "خريطة حرارية · تقاطع فئتين"),
    }
    kind_help = {
        "bar": _copy(language, "Choose a group column, then optionally a numeric value.", "اختر عمود المجموعات ثم قيمة رقمية اختيارية."),
        "line": _copy(language, "Choose an ordered/date column and a numeric value.", "اختر عمودًا مرتبًا أو تاريخيًا ثم قيمة رقمية."),
        "area": _copy(language, "Choose an ordered/date column and a numeric value.", "اختر عمودًا مرتبًا أو تاريخيًا ثم قيمة رقمية."),
        "scatter": _copy(language, "Choose two numeric columns; every dot is one row.", "اختر عمودين رقميين؛ كل نقطة تمثل صفًا."),
        "histogram": _copy(language, "Choose one numeric column.", "اختر عمودًا رقميًا واحدًا."),
        "box": _copy(language, "Choose a numeric value and optionally a grouping column.", "اختر قيمة رقمية وعمود تجميع اختياريًا."),
        "pie": _copy(language, "Choose categories, then optionally the value whose shares you want.", "اختر الفئات ثم القيمة التي تريد عرض حصصها اختياريًا."),
        "heatmap": _copy(language, "Choose two category columns to count their combinations.", "اختر عمودَي فئات لعدّ تقاطعاتهما."),
    }

    with st.expander(
        _copy(language, "＋ Create another chart", "＋ أضف رسمًا من اختيارك"),
        expanded=False,
    ):
        st.caption(_copy(
            language,
            "Pick the exact chart and columns. InsightLab validates the combination and keeps the chart with this analysis.",
            "اختر نوع الرسم والأعمدة بنفسك. المنصة تتحقق من صلاحية الاختيار وتحفظ الرسم مع هذا التحليل.",
        ))
        kind = st.selectbox(
            _copy(language, "Chart type", "نوع الرسم"),
            CHART_KINDS,
            format_func=lambda value: labels[value],
            key="custom_chart_kind",
        )
        st.caption(kind_help[kind])

        first_is_numeric = kind in {"scatter", "histogram", "box"}
        first_options = numbers if first_is_numeric else columns
        if not first_options:
            st.warning(_copy(
                language,
                "This dataset has no numeric column that can be used for this chart.",
                "البيانات لا تحتوي على عمود رقمي صالح لهذا النوع من الرسوم.",
            ))
            return

        first_label = {
            "histogram": _copy(language, "Numeric column", "العمود الرقمي"),
            "box": _copy(language, "Value column", "عمود القيم"),
            "scatter": _copy(language, "X axis", "المحور الأفقي X"),
            "heatmap": _copy(language, "Rows", "الصفوف"),
        }.get(kind, _copy(language, "X axis / groups", "المحور الأفقي / المجموعات"))
        x_column = st.selectbox(first_label, first_options, key="custom_chart_x")

        y_column: str | None = None
        optional_value = kind in {"bar", "pie"}
        optional_group = kind == "box"
        if kind in {"line", "area", "scatter"}:
            second_options = [column for column in numbers if column != x_column]
            if not second_options:
                st.warning(_copy(language, "Choose a dataset with a second numeric column for this chart.", "هذا الرسم يحتاج عمودًا رقميًا ثانيًا في البيانات."))
                return
            y_column = st.selectbox(
                _copy(language, "Y axis", "المحور الرأسي Y"),
                second_options,
                key="custom_chart_y_required",
            )
        elif kind == "heatmap":
            second_options = [column for column in columns if column != x_column]
            if not second_options:
                st.warning(_copy(language, "This chart needs a second column.", "هذا الرسم يحتاج عمودًا ثانيًا."))
                return
            y_column = st.selectbox(
                _copy(language, "Columns", "الأعمدة"),
                second_options,
                key="custom_chart_y_heatmap",
            )
        elif optional_value:
            selected = st.selectbox(
                _copy(language, "Value (optional)", "القيمة (اختيارية)"),
                [NONE, *numbers],
                format_func=lambda value: _copy(language, "Count rows", "عدّ الصفوف") if value == NONE else value,
                key="custom_chart_y_optional",
            )
            y_column = None if selected == NONE else selected
        elif optional_group:
            selected = st.selectbox(
                _copy(language, "Split into groups (optional)", "قسّم إلى مجموعات (اختياري)"),
                [NONE, *[column for column in columns if column != x_column]],
                format_func=lambda value: _copy(language, "No grouping", "بدون تجميع") if value == NONE else value,
                key="custom_chart_group_optional",
            )
            y_column = None if selected == NONE else selected

        aggregation = "sum"
        if kind in {"line", "area"} or (kind in {"bar", "pie"} and y_column):
            aggregation_labels = {
                "sum": _copy(language, "Total", "المجموع"),
                "mean": _copy(language, "Average", "المتوسط"),
                "median": _copy(language, "Median", "الوسيط"),
                "count": _copy(language, "Count non-empty values", "عدد القيم غير الفارغة"),
            }
            recommended = (
                state.understanding.aggregation_for(y_column, "mean")
                if y_column else "count"
            )
            aggregation = st.selectbox(
                _copy(language, "Calculation", "طريقة الحساب"),
                AGGREGATIONS,
                index=AGGREGATIONS.index(recommended) if recommended in AGGREGATIONS else 0,
                format_func=lambda value: aggregation_labels[value],
                key="custom_chart_aggregation",
            )

        title = st.text_input(
            _copy(language, "Chart title (optional)", "عنوان الرسم (اختياري)"),
            key="custom_chart_title",
            placeholder=_copy(language, "A clear title will be generated if left empty", "سيتم إنشاء عنوان واضح إذا تركته فارغًا"),
        )
        if st.button(
            _copy(language, "Add chart to this analysis", "أضف الرسم إلى هذا التحليل"),
            type="primary",
            key="add_custom_chart",
            width="stretch",
        ):
            try:
                chart = build_custom_chart(
                    state.frame,
                    chart_id=_next_id(state),
                    kind=kind,
                    x_column=x_column,
                    y_column=y_column,
                    aggregation=aggregation,
                    title=title,
                    language_code=language.code,
                )
            except CustomChartError as error:
                messages = {
                    "A scatter chart needs two numeric columns.": "الرسم المبعثر يحتاج عمودين رقميين.",
                    "This chart needs a numeric Y column.": "هذا الرسم يحتاج عمودًا رقميًا للمحور Y.",
                    "A heatmap needs two columns.": "الخريطة الحرارية تحتاج عمودين.",
                }
                st.error(messages.get(str(error), str(error)) if ar else str(error))
            else:
                state.charts.append(chart)
                state.log.record(
                    EventKind.ARTEFACT_CREATED,
                    "explore",
                    (f"أضاف المستخدم الرسم المخصص: {chart.title}." if ar else f"User added custom chart: {chart.title}."),
                    chart_id=chart.id,
                    chart_kind=chart.kind,
                    x_column=x_column,
                    y_column=y_column or "",
                    aggregation=chart.aggregation,
                )
                save_run(state)
                st.success(_copy(language, "Chart added and saved with this analysis.", "تمت إضافة الرسم وحفظه مع هذا التحليل."))
