# AI-Driven Load Forecasting & Dynamic Feeder Rerouting

## Interface contract
- `data/forecast.csv` columns: `timestamp, transformer_id, predicted_kw, actual_kw`
- `grid/engine.py`: `evaluate_and_reroute(loads_kw: dict) -> dict`
  - returns: `status` ("rerouted" | "no_action" | "no_safe_option"), `action`, `before`, `after`, `log`

## Run
python grid/test_day1.py
streamlit run app/app.py