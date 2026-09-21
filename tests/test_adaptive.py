from insightlab.core.adaptive import rank_objects, rank_values
from insightlab.core.state import Chart, Insight, Kpi


def test_axis_ranking_is_stable_and_uses_explicit_feedback():
    profile = {"axes": {"profits": 4.0, "sales": -1.0}}
    assert rank_values(["sales", "customers", "profits"], "axes", profile) == [
        "profits", "customers", "sales"
    ]


def test_object_ranking_can_learn_kpis_charts_and_insight_axes():
    profile = {
        "axes": {"customers": 3.0},
        "kpis": {"repeat rate": 5.0},
        "chart_kinds": {"line": 2.0},
    }
    policy = {"axis_weight": 1.0, "kpi_weight": 1.0, "chart_weight": 1.0}
    kpis = [
        Kpi("Revenue", 1, "1", "sum"),
        Kpi("Repeat rate", 1, "1", "repeat"),
    ]
    charts = [
        Chart("one", "Bar", "bar", "", axis="sales"),
        Chart("two", "Line", "line", "", axis="customers"),
    ]
    insights = [
        Insight("Sales", "", "e", "", axis="sales"),
        Insight("Customers", "", "e", "", axis="customers"),
    ]

    assert rank_objects(kpis, profile=profile, policy=policy, name_attr="name")[0].name == "Repeat rate"
    assert rank_objects(charts, profile=profile, policy=policy)[0].id == "two"
    assert rank_objects(insights, profile=profile, policy=policy)[0].title == "Customers"
