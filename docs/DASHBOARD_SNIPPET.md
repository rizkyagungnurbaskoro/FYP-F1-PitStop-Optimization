# Dashboard Usage Snippet

## Run
```
streamlit run dashboard/streamlit_app.py
```

## Pages
- **Overview**: concise metrics + timing tower + holdout snapshot.
- **Demo**: “Showcase” fixed scenarios or “Explore” your own rows.

## Showcase vs Explore
- **Showcase**: curated, dataset‑backed scenarios (plus optional external demo).
- **Explore**: user selects dataset/season/race/driver/lap and runs what‑if.

## Strict vs Standard
- **Strict**: leakage-safe (prev-lap features) -> thesis claims.
- **Standard**: exploratory only.

## External scenarios
External FastF1 scenarios are **demo-only** and **not** part of S1-S4 evaluation.

## Example output (scenario run)
```json
{
  "scenario_name": "Auto-selected Leader Pit Event",
  "dataset": "RefData",
  "stage": "S2",
  "strict": true,
  "decision_lap": 28,
  "pit_probability": 0.62,
  "recommendation": "PIT",
  "impact_seconds": -3.4
}
```
