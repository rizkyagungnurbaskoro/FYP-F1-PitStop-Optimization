from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Paths:
    root: Path

    # datasets
    ref_csv: Path
    my_csv: Path
    my_weather_csv: Path

    # outputs
    results_dir: Path
    out_replication: Path
    out_refdata_mymethod: Path
    out_mydata_reftech_weather: Path
    out_mydata_mymethod_weather: Path
    out_mydata_reftech_weather_holdout: Path
    out_mydata_mymethod_weather_holdout: Path
    out_summary_plots: Path


@dataclass(frozen=True)
class ColumnConfig:
    ref_target: str
    ref_group: str

    my_target: str
    my_group: str

    myw_target: str
    myw_group: str


def get_paths(project_root: str | Path | None = None) -> Paths:
    root = Path(project_root) if project_root else Path(__file__).resolve().parents[1]

    ref_csv = root / "data" / "hettmann_replica_dataset.csv"
    my_csv = root / "personal_datasets" / "fastf1_strategy_dataset.csv"

    # Stage 3/4 must use leakage-safe lagged data (prefer personal_datasets).
    my_weather_csv_primary = root / "personal_datasets" / "fastf1_strategy_dataset.csv"
    my_weather_csv_secondary = root / "personal_datasets" / "fastf1_strategy_weather_dataset.csv"
    my_weather_csv_fallback = root / "data" / "strategy_weather_dataset.csv"
    if my_weather_csv_primary.exists():
        my_weather_csv = my_weather_csv_primary
    elif my_weather_csv_secondary.exists():
        my_weather_csv = my_weather_csv_secondary
    elif my_weather_csv_fallback.exists():
        my_weather_csv = my_weather_csv_fallback
    else:
        raise FileNotFoundError(
            "Leakage-safe dataset missing for Stage 3/4. "
            f"Expected one of: {my_weather_csv_primary}, {my_weather_csv_secondary}, "
            f"or {my_weather_csv_fallback}"
        )

    results_dir = root / "results"
    return Paths(
        root=root,
        ref_csv=ref_csv,
        my_csv=my_csv,
        my_weather_csv=my_weather_csv,
        results_dir=results_dir,
        out_replication=results_dir / "replication",
        out_refdata_mymethod=results_dir / "refdata_mymethod",
        out_mydata_reftech_weather=results_dir / "mydata_reftech_weather",
        out_mydata_mymethod_weather=results_dir / "mydata_mymethod_weather",
        out_mydata_reftech_weather_holdout=results_dir / "holdout70_mydata_reftech_weather",
        out_mydata_mymethod_weather_holdout=results_dir / "holdout70_mydata_mymethod_weather",
        out_summary_plots=results_dir / "summary_plots",
    )


def get_column_config() -> ColumnConfig:
    return ColumnConfig(
        # Reference dataset (Hettmann replica)
        ref_target="decide_pitstop",
        ref_group="race_id",

        # My dataset (if you ever run Stage 2 on mydata without weather)
        my_target="decide_pitstop",
        my_group="race_id",

        # My dataset + weather (Stage 3 & Stage 4)
        myw_target="decide_pitstop",
        myw_group="race_id",
    )
