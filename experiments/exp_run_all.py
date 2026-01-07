from __future__ import annotations
import pandas as pd

from experiments.exp_config import get_paths
from experiments.exp_utils import add_feature_engineering, build_feature_list, ensure_dir, load_csv, save_json, safe_numeric
from experiments.exp_models import ref_tech_spec, my_method_spec, run_stage_strict

TARGET_COL = "decide_pitstop"
GROUP_COL = "race_id"
N_SPLITS = 5
BASE_THRESHOLD = 0.5
TUNE_THRESHOLD = True
THRESHOLD_VAL_SIZE = 0.2
THRESHOLD_MAX = 0.95
REF_THRESHOLD_MIN = 0.05
REF_THRESHOLD_METRIC = "fbeta"
REF_MIN_PRECISION = 0.10
MYDATA_THRESHOLD_MIN = 0.03
MYDATA_THRESHOLD_METRIC = "recall_at_precision"
MYDATA_MIN_PRECISION = 0.08
F_BETA = 2.0  # F2 emphasizes recall while balancing precision.
TUNE_PARAMS = True
TUNE_METRIC_REF = "delta_mix_mean_std"
TUNE_METRIC_MYDATA = "delta_f1_f2_recall_mean_std"
TUNE_VAL_SIZE = 0.2
TUNE_RANDOM_STATE = 42
TUNE_CV_SPLITS = 3
DELTA_PR_AUC_WEIGHT_REF = 0.6
DELTA_PR_AUC_WEIGHT_MYDATA = 0.45
DELTA_F1_WEIGHT_MYDATA = 0.30
DELTA_F2_WEIGHT_MYDATA = 0.40
DELTA_RECALL_WEIGHT_MYDATA = 0.30
MYDATA_FEATURE_SELECTION = True
MYDATA_FS_TOP_K = 18
MYMETHOD_PARAM_CANDIDATES = [
    {},
    dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.2,
        min_child_weight=3,
    ),
    dict(
        n_estimators=700,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.5,
        reg_alpha=0.0,
        min_child_weight=3,
    ),
    dict(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1,
    ),
    dict(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.5,
        reg_alpha=0.8,
        min_child_weight=5,
    ),
    dict(
        n_estimators=900,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.95,
        colsample_bytree=0.95,
        reg_lambda=0.5,
        reg_alpha=0.0,
        min_child_weight=1,
    ),
    dict(
        n_estimators=900,
        max_depth=6,
        learning_rate=0.03,
        subsample=0.95,
        colsample_bytree=0.95,
        reg_lambda=0.5,
        reg_alpha=0.0,
        min_child_weight=1,
        scale_pos_weight_multiplier=1.3,
    ),
    dict(
        n_estimators=700,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1,
        scale_pos_weight_multiplier=1.6,
    ),
    dict(
        n_estimators=1000,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.2,
        reg_alpha=0.0,
        min_child_weight=2,
    ),
]
MYMETHOD_PARAM_CANDIDATES_MYDATA = MYMETHOD_PARAM_CANDIDATES + [
    dict(
        n_estimators=700,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=1,
        scale_pos_weight_multiplier=2.0,
    )
]


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

    df_ref = add_feature_engineering(safe_numeric(load_csv(paths.ref_csv)))
    df_my = add_feature_engineering(safe_numeric(load_csv(paths.my_csv)))
    df_my_w = add_feature_engineering(safe_numeric(load_csv(paths.my_weather_csv)))

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

    threshold = BASE_THRESHOLD
    n_splits = N_SPLITS

    # Stage 1
    m1 = run_stage_strict(
        df=df_ref, features=feats_ref, target_col=tgt_ref, group_col=grp_ref,
        spec=ref_tech_spec(), n_splits=n_splits, threshold=threshold, use_scale_pos_weight=True,
        tune_threshold=TUNE_THRESHOLD, threshold_val_size=THRESHOLD_VAL_SIZE,
        threshold_min=REF_THRESHOLD_MIN, threshold_max=THRESHOLD_MAX, beta=F_BETA,
        threshold_metric=REF_THRESHOLD_METRIC, min_precision=REF_MIN_PRECISION,
    )
    save_json(paths.out_replication / "metrics.json", m1)

    # Stage 2
    m2 = run_stage_strict(
        df=df_ref, features=feats_ref, target_col=tgt_ref, group_col=grp_ref,
        spec=my_method_spec(), baseline_spec=ref_tech_spec(),
        n_splits=n_splits, threshold=threshold, use_scale_pos_weight=True,
        tune_threshold=TUNE_THRESHOLD, threshold_val_size=THRESHOLD_VAL_SIZE,
        threshold_min=REF_THRESHOLD_MIN, threshold_max=THRESHOLD_MAX, beta=F_BETA,
        threshold_metric=REF_THRESHOLD_METRIC, min_precision=REF_MIN_PRECISION,
        tune_params=TUNE_PARAMS, param_candidates=MYMETHOD_PARAM_CANDIDATES,
        tune_metric=TUNE_METRIC_REF, tune_val_size=TUNE_VAL_SIZE, tune_random_state=TUNE_RANDOM_STATE,
        tune_cv_splits=TUNE_CV_SPLITS, delta_pr_auc_weight=DELTA_PR_AUC_WEIGHT_REF,
    )
    save_json(paths.out_refdata_mymethod / "metrics.json", m2)

    # Stage 3
    m3 = run_stage_strict(
        df=df_my_w, features=feats_my_w, target_col=tgt_my_w, group_col=grp_my_w,
        spec=ref_tech_spec(), n_splits=n_splits, threshold=threshold, use_scale_pos_weight=True,
        tune_threshold=TUNE_THRESHOLD, threshold_val_size=THRESHOLD_VAL_SIZE,
        threshold_min=MYDATA_THRESHOLD_MIN, threshold_max=THRESHOLD_MAX, beta=F_BETA,
        threshold_metric=MYDATA_THRESHOLD_METRIC, min_precision=MYDATA_MIN_PRECISION,
    )
    save_json(paths.out_mydata_reftech_weather / "metrics.json", m3)

    # Stage 4
    m4 = run_stage_strict(
        df=df_my_w, features=feats_my_w, target_col=tgt_my_w, group_col=grp_my_w,
        spec=my_method_spec(), baseline_spec=ref_tech_spec(),
        n_splits=n_splits, threshold=threshold, use_scale_pos_weight=True,
        use_feature_selection=MYDATA_FEATURE_SELECTION, fs_top_k=MYDATA_FS_TOP_K,
        tune_threshold=TUNE_THRESHOLD, threshold_val_size=THRESHOLD_VAL_SIZE,
        threshold_min=MYDATA_THRESHOLD_MIN, threshold_max=THRESHOLD_MAX, beta=F_BETA,
        threshold_metric=MYDATA_THRESHOLD_METRIC, min_precision=MYDATA_MIN_PRECISION,
        tune_params=TUNE_PARAMS, param_candidates=MYMETHOD_PARAM_CANDIDATES_MYDATA,
        tune_metric=TUNE_METRIC_MYDATA, tune_val_size=TUNE_VAL_SIZE, tune_random_state=TUNE_RANDOM_STATE,
        tune_cv_splits=TUNE_CV_SPLITS, delta_pr_auc_weight=DELTA_PR_AUC_WEIGHT_MYDATA,
        delta_f1_weight=DELTA_F1_WEIGHT_MYDATA,
        delta_f2_weight=DELTA_F2_WEIGHT_MYDATA,
        delta_recall_weight=DELTA_RECALL_WEIGHT_MYDATA,
    )
    save_json(paths.out_mydata_mymethod_weather / "metrics.json", m4)

    print("[OK] Saved metrics.json for all stages.")
    print("[NEXT] python -m experiments.exp_plot_all")


if __name__ == "__main__":
    main()
