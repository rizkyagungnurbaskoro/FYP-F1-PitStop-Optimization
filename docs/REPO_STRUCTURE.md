# Repo Structure

```
FYP_FINAL/
  dashboard/              Streamlit UI and demo logic
  experiments/            Training, evaluation, diagnostics
  visualization/          Plot/diagram generators
  results/                Metrics, plots, and summaries
  reports/                Supervisor updates and thesis notes
  data/                   Reference datasets
  personal_datasets/      Custom datasets and weather data
  docs/                   Project documentation (this folder)
```

## Key files

- `experiments/exp_run_all.py` - Strict GroupKFold runs for all stages (S1-S4)
- `experiments/exp_run_stage34_holdout.py` - Holdout 70/30 for Stage 3/4
- `experiments/exp_models.py` - Model pipeline and evaluators
- `experiments/exp_plot_all.py` - Generates summary CSVs and plots
- `dashboard/streamlit_app.py` - Pitwall dashboard

## Results folders

- `results/replication/` - Stage 1 metrics (RefTech on RefData)
- `results/refdata_mymethod/` - Stage 2 metrics (MyMethod on RefData)
- `results/mydata_reftech_weather/` - Stage 3 metrics (RefTech on MyData+W)
- `results/mydata_mymethod_weather/` - Stage 4 metrics (MyMethod on MyData+W)
- `results/holdout70_mydata_reftech_weather/` - Stage 3 holdout metrics
- `results/holdout70_mydata_mymethod_weather/` - Stage 4 holdout metrics
- `results/summary_plots/` - CSV summaries + thesis figures
