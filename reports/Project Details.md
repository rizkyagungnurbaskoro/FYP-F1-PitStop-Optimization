# Project Details

## Overview
- Goal: Predict pit-stop decisions and present results in an F1-style pitwall dashboard.
- Model: XGBoost classifier (scikit-learn pipeline).
- Evaluation: GroupKFold by race to prevent leakage across events.
- Metrics: F1, F2, Precision, Recall, PR-AUC (mean and std across folds).

## CRISP-DM Mapping
- Business Understanding: Define decision-support goals for pit-stop timing and compare against a reference method.
- Data Understanding: Audit race, lap, and weather signals; check class balance and leakage risks.
- Data Preparation: Clean CSVs and engineer pace, pit-window, and weather features.
- Modeling: Train XGBoost models with calibrated probabilities across the four stages.
- Evaluation: Use GroupKFold by race and report F1/F2/PR-AUC/Recall with fold-level summaries.
- Deployment: Deliver an F1-style Streamlit dashboard and a strategy-impact backtest panel.

## Method (4 Stages)
- S1: RefTech on RefData (baseline replication).
- S2: MyMethod on RefData (same data, improved method).
- S3: RefTech on MyData+W (own data + weather).
- S4: MyMethod on MyData+W (own data + weather, improved method).

## Datasets
- Ref/standard dataset: `data/strategy_weather_dataset.csv`
- Strict stage 3/4 dataset: `personal_datasets/fastf1_strategy_dataset.csv`
- Strict uses previous-lap features (`*_prev`) to reduce leakage risk.

## Framework and Pipeline
- Python stack: pandas, numpy, scikit-learn, xgboost, matplotlib, streamlit.
- Feature engineering and CSV utilities: `experiments/exp_utils.py`
- Model training and evaluation: `experiments/exp_models.py`
- End-to-end run: `experiments/exp_run_all.py`
- Plot generation: `experiments/exp_plot_all.py`

## Results (Interpretation)
- Use **Strict** results for thesis claims (leakage-safe).
- Standard results are exploratory and may include same-lap features.
- Current strict results show modest improvements for S2 vs S1, and small gains for S4 vs S3, but absolute performance on MyData+W remains low.
- Practical framing: decision-support tool, not a fully automated race strategy system.

## Dashboard (F1 Pitwall Theme)
- App: `dashboard/streamlit_app.py`
- Uses summary CSVs to show metrics, deltas, and plots.
- Demo: driver/circuit/weather/lap selection and a visual pitwall decision (PIT/STANDBY/STAY OUT).
- Demo logic is model-driven and adjusted by pit-window and race context.

## Diagnostics
- Summary: `results/summary_plots/diagnostic_summary.json`
- Class balance by group: `results/summary_plots/diagnostic_class_balance_by_group.csv`
- Key finding: feature mismatch between RefData and MyData+W is large (few shared features).

## How to Run
- Experiments: `python -m experiments.exp_run_all`
- Plots: `python -m experiments.exp_plot_all`
- Dashboard: `streamlit run dashboard/streamlit_app.py`

## Limitations
- Low overlap of features between RefData and MyData+W reduces transfer.
- Small effect sizes and race-specific variance limit headline gains.
- Strict evaluation reduces leakage but can lower absolute metrics.

## Future Work
- Probability calibration and reliability reporting.
- Scenario-aware thresholds (SC/VSC, pit window, tire wear).
- Monotonic constraints on key features (e.g., tire age).
- Larger dataset and per-circuit validation.
- Live inference pipeline for in-race decision support.
