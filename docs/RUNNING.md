# Running the pipeline

## 1) Strict GroupKFold (S1-S4)
```
python -m experiments.exp_run_all
```

## 2) Stage 3/4 holdout (70/30)
```
python -m experiments.exp_run_stage34_holdout
```

## 3) Generate plots and summaries
```
python -m experiments.exp_plot_all
```

## 4) Run the dashboard
```
streamlit run dashboard/streamlit_app.py
```
