# Bus Demand Planner

Dynamic public transport demand forecasting and resource allocation using machine learning.

Forecasts hourly riders for New York City bus routes (MTA open data), compares the buses needed with the buses
normally run, and suggests where buses could be moved.

## Run locally
```
pip install -r requirements.txt
streamlit run app.py
```

## Files
- `app.py` – the Streamlit app
- `app_info.pkl`, `model_calendar.pkl`, `model_with_lag.pkl`, `recent_riders.pkl` – trained models and lookup data from the Colab notebook
- `requirements.txt` – Python packages

Data: MTA Bus Hourly Ridership (data.ny.gov). Forecasts are estimates, not exact counts.
