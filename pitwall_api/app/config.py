from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

REFDATA_PATH = Path(os.getenv("REFDATA_PATH", BASE_DIR / "data" / "strategy_weather_dataset.csv"))
MYDATA_PATH = Path(os.getenv("MYDATA_PATH", BASE_DIR / "personal_datasets" / "fastf1_strategy_dataset.csv"))

SUMMARY_STRICT_PATH = Path(
    os.getenv("SUMMARY_STRICT_PATH", BASE_DIR / "results" / "summary_plots" / "stage_summary_strict.csv")
)
SUMMARY_STD_PATH = Path(
    os.getenv("SUMMARY_STD_PATH", BASE_DIR / "results" / "summary_plots" / "stage_summary.csv")
)
SUMMARY_HOLDOUT_PATH = Path(
    os.getenv("SUMMARY_HOLDOUT_PATH", BASE_DIR / "results" / "summary_plots" / "stage34_holdout_summary.csv")
)

MODEL_PATH = os.getenv("MODEL_PATH", "")
