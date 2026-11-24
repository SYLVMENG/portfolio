# Bike dashboard (Django)

A small Django app that runs `forecasting/analysis/bike_demand_pricing.py` and shows the result tables (forecast validation, choice model, two-stage plan, sensitivity checks) on one page.

```bash
pip install -r requirements.txt
# put sales_transactions.csv (not included) in forecasting/analysis/
python manage.py migrate
python manage.py runserver
```

Then open http://127.0.0.1:8000/. The first load runs the full analysis and takes a few minutes.
