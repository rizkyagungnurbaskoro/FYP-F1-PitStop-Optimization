# Pitwall API (FastAPI)

## Run

```bash
cd pitwall_api
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Endpoints

- `GET /health`
- `GET /metrics/summary?mode=strict|standard`
- `GET /scenarios/showcase?dataset=my|ref`
- `GET /scenario/{scenario_id}?dataset=my|ref`
- `POST /whatif` with JSON: `{ "dataset": "my", "scenario_id": "..." }`

## Notes
- Trains an XGBoost pipeline on startup (cached) to match the Streamlit demo logic.
- Set `REFDATA_PATH` or `MYDATA_PATH` env vars to override dataset paths.
