from django.shortcuts import render

from .analysis.runner import run_analysis


def index(request):
    r = run_analysis()
    html = lambda df: df.to_html(classes="table", border=0)
    return render(request, "forecasting/index.html", {
        "validation": html(r["validation"].round(3)),
        "q1": html(r["q1_summary"]),
        "q2": html(r["q2_table"]),
        "units": html(r["units"]),
        "plan": html(r["plan"].drop(columns=["period"]).round(1)),
        "comparison": html(r["comparison"].round(1)),
        "holdout": html(r["holdout"].round(0)),
        "grid_check": html(r["grid_check"].round(1)),
        "dense_check": html(r["dense_check"].round(1)),
        "forecast_eval": html(r["forecast_eval"].round(2)),
        "q2_oos": html(r["q2_oos"].round(3).to_frame("value")),
        "estimation_check": html(r["estimation_check"].round(2)),
        "sensitivity": html(r["sensitivity"].round(1)),
    })
