from __future__ import annotations
import pandas as pd

from experiments.exp_config import get_paths
from experiments.exp_utils import build_feature_list, ensure_dir, load_csv, save_json, safe_numeric
from experiments.exp_models import ref_tech_spec, my_method_spec, run_stage_strict

TARGET_COL = "decide_pitstop"
GROUP_COL = "race_id"


def _require_columns(df: pd.DataFrame, required: list[str], label: str) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {', '.join(missing)}")


def _leakage_guard_stage34(df: pd.DataFrame, dataset_path: str) -> None:
    cols_preview = ", ".join(list(df.columns[:20]))

    prev_cols = [c for c in df.columns if c.endswith("_prev")]
    if not prev_cols:
        msg = (
            "Stage 3/4 dataset is not leakage-safe: no *_prev columns found.\n"
            f"Dataset: {dataset_path}\n"
            f"Columns (first 20): {cols_preview}"
        )
        print(msg)
        raise ValueError(msg)

    same_lap_cols = [
        "tireage",
        "stint_laps",
        "pitstops_so_far",
        "pitstops_remaining",
        "lapno",
        "relative_pace",
        "delta_best_race",
        "delta_interval",
        "in_pit_window",
        "tyre_wear_pct",
        "compound_rank",
        "position",
        "race_progress",
    ]
    suspicious = [c for c in same_lap_cols if c in df.columns]
    if suspicious:
        msg = (
            "Stage 3/4 dataset contains same-lap columns (leakage risk): "
            + ", ".join(sorted(suspicious))
            + "\n"
            f"Dataset: {dataset_path}\n"
            f"Columns (first 20): {cols_preview}"
        )
        print(msg)
        raise ValueError(msg)


def main() -> None:
    paths = get_paths()

    for d in [
        paths.results_dir,
        paths.out_replication,
        paths.out_refdata_mymethod,
        paths.out_mydata_reftech_weather,
        paths.out_mydata_mymethod_weather,
        paths.out_summary_plots,
    ]:
        ensure_dir(d)

    df_ref = safe_numeric(load_csv(paths.ref_csv))
    df_my = safe_numeric(load_csv(paths.my_csv))
    df_my_w = safe_numeric(load_csv(paths.my_weather_csv))

    _require_columns(df_ref, [TARGET_COL, GROUP_COL], "Reference dataset")
    _require_columns(df_my, [TARGET_COL, GROUP_COL], "Personal dataset")
    _require_columns(df_my_w, [TARGET_COL, GROUP_COL], "Stage 3/4 dataset")
    print(f"[INFO] Stage 3/4 dataset: {paths.my_weather_csv}")
    _leakage_guard_stage34(df_my_w, str(paths.my_weather_csv))

    tgt_ref = TARGET_COL
    grp_ref = GROUP_COL

    tgt_my = TARGET_COL
    grp_my = GROUP_COL

    tgt_my_w = TARGET_COL
    grp_my_w = GROUP_COL

    feats_ref = build_feature_list(df_ref, tgt_ref, grp_ref)
    feats_my_w = build_feature_list(df_my_w, tgt_my_w, grp_my_w)

    threshold = 0.5
    n_splits = 5

    # Stage 1
    m1 = run_stage_strict(
        df=df_ref, features=feats_ref, target_col=tgt_ref, group_col=grp_ref,
        spec=ref_tech_spec(), n_splits=n_splits, threshold=threshold, use_scale_pos_weight=True
    )
    save_json(paths.out_replication / "metrics.json", m1)

    # Stage 2
    m2 = run_stage_strict(
        df=df_ref, features=feats_ref, target_col=tgt_ref, group_col=grp_ref,
        spec=my_method_spec(), n_splits=n_splits, threshold=threshold, use_scale_pos_weight=True
    )
    save_json(paths.out_refdata_mymethod / "metrics.json", m2)

    # Stage 3
    m3 = run_stage_strict(
        df=df_my_w, features=feats_my_w, target_col=tgt_my_w, group_col=grp_my_w,
        spec=ref_tech_spec(), n_splits=n_splits, threshold=threshold, use_scale_pos_weight=True
    )
    save_json(paths.out_mydata_reftech_weather / "metrics.json", m3)

    # Stage 4
    m4 = run_stage_strict(
        df=df_my_w, features=feats_my_w, target_col=tgt_my_w, group_col=grp_my_w,
        spec=my_method_spec(), n_splits=n_splits, threshold=threshold, use_scale_pos_weight=True
    )
    save_json(paths.out_mydata_mymethod_weather / "metrics.json", m4)

    print("[OK] Saved metrics.json for all stages.")
    print("[NEXT] python -m experiments.exp_plot_all")


if __name__ == "__main__":
    main()
