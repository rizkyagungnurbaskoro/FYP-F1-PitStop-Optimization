from __future__ import annotations

import numpy as np
import pandas as pd

from experiments.exp_config import get_paths
from experiments.exp_utils import (
    add_feature_engineering,
    build_feature_list,
    ensure_dir,
    load_csv,
    read_json,
    safe_numeric,
    save_json,
    select_canonical_features,
)
from experiments.exp_models import ModelSpec, ref_tech_spec, my_method_spec, run_stage_strict

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
MYDATA_THRESHOLD_MIN = 0.02
MYDATA_THRESHOLD_METRIC = "f1"
MYDATA_MIN_PRECISION = 0.10
F_BETA = 2.0  # F2 emphasizes recall while balancing precision.
TUNE_PARAMS = True
TUNE_METRIC_REF = "delta_mix_all_guard"
TUNE_METRIC_MYDATA = "delta_mix_all_guard"
TUNE_VAL_SIZE = 0.2
TUNE_RANDOM_STATE = 42
TUNE_CV_SPLITS = 2
DELTA_PR_AUC_WEIGHT_REF = 0.30
DELTA_F1_WEIGHT_REF = 0.30
DELTA_F2_WEIGHT_REF = 0.25
DELTA_RECALL_WEIGHT_REF = 0.15
DELTA_PR_AUC_WEIGHT_MYDATA = 0.30
DELTA_F1_WEIGHT_MYDATA = 0.30
DELTA_F2_WEIGHT_MYDATA = 0.25
DELTA_RECALL_WEIGHT_MYDATA = 0.15
MYMETHOD_THRESHOLD_MIN = 0.02
MYMETHOD_THRESHOLD_METRIC = "f1"
MYMETHOD_FEATURE_SELECTION = True
MYMETHOD_FS_TOP_K = 30
MYMETHOD_TARGET_ENCODING = True
MYMETHOD_TARGET_ENCODING_SPECS = [
    (("Driver",), "te_driver", 30.0),
    (("Driver", "season"), "te_driver_season", 50.0),
    (("Driver", "track_deg_category"), "te_driver_track", 50.0),
    (("track_deg_category",), "te_track_deg", 20.0),
]
MYMETHOD_TARGET_ENCODING_DROP_COLS = False
MYMETHOD_CALIBRATION_METHOD = "sigmoid"
MYMETHOD_CALIBRATION_VAL_SIZE = 0.2
MYMETHOD_CALIBRATION_RANDOM_STATE = 42
MYMETHOD_CALIBRATE = False
MYMETHOD_ENSEMBLE_SEEDS = [42]
MYMETHOD_BALANCED_SUBSAMPLE = True
# Keep the training distribution closer to reality while still boosting positives.
MYMETHOD_NEG_POS_RATIO = 10.0
MYMETHOD_USE_SCALE_POS_WEIGHT = False
MYMETHOD_SUBSAMPLE_RANDOM_STATE = 42
RANDOM_PARAM_SEED = 1337
RANDOM_CANDIDATES_REF = 8
RANDOM_CANDIDATES_MYDATA = 12


def _random_param_candidates(
    count: int,
    seed: int,
    spw_min: float,
    spw_max: float,
) -> list[dict[str, float | int]]:
    rng = np.random.default_rng(seed)
    out: list[dict[str, float | int]] = []
    for _ in range(int(count)):
        out.append(
            dict(
                n_estimators=int(rng.integers(400, 1600)),
                max_depth=int(rng.integers(3, 8)),
                learning_rate=float(rng.uniform(0.015, 0.12)),
                subsample=float(rng.uniform(0.75, 1.0)),
                colsample_bytree=float(rng.uniform(0.75, 1.0)),
                reg_lambda=float(rng.uniform(0.5, 3.0)),
                reg_alpha=float(rng.uniform(0.0, 0.8)),
                min_child_weight=int(rng.integers(1, 8)),
                gamma=float(rng.uniform(0.0, 0.3)),
                max_delta_step=int(rng.integers(0, 6)),
                scale_pos_weight_multiplier=float(rng.uniform(spw_min, spw_max)),
                random_state=int(rng.integers(1, 10_000)),
            )
        )
    return out


BASE_MYMETHOD_PARAM_CANDIDATES = [
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
MYMETHOD_PARAM_CANDIDATES = BASE_MYMETHOD_PARAM_CANDIDATES + _random_param_candidates(
    RANDOM_CANDIDATES_REF, RANDOM_PARAM_SEED, 0.6, 2.2
)
EXTRA_MYDATA_PARAM_CANDIDATES = [
    dict(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.8,
        reg_alpha=0.2,
        min_child_weight=4,
        gamma=0.0,
        max_delta_step=1,
        scale_pos_weight_multiplier=0.8,
    ),
    dict(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        reg_alpha=0.0,
        min_child_weight=6,
        gamma=0.1,
        max_delta_step=1,
        scale_pos_weight_multiplier=1.0,
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
        scale_pos_weight_multiplier=2.0,
    ),
    dict(
        n_estimators=800,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,
        reg_alpha=0.4,
        min_child_weight=8,
        gamma=0.1,
        scale_pos_weight_multiplier=1.4,
    ),
    dict(
        n_estimators=1200,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        reg_alpha=0.0,
        min_child_weight=4,
        scale_pos_weight_multiplier=1.2,
    ),
    dict(
        n_estimators=900,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=1.2,
        reg_alpha=0.0,
        min_child_weight=3,
        gamma=0.0,
        scale_pos_weight_multiplier=2.5,
    ),
    dict(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=2,
        gamma=0.0,
        max_delta_step=1,
        scale_pos_weight_multiplier=2.0,
    ),
    dict(
        n_estimators=1000,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        reg_alpha=0.3,
        min_child_weight=6,
        gamma=0.1,
        max_delta_step=1,
        scale_pos_weight_multiplier=1.5,
    ),
    dict(
        n_estimators=600,
        max_depth=4,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.5,
        reg_alpha=0.6,
        min_child_weight=8,
        gamma=0.1,
        max_delta_step=1,
        scale_pos_weight_multiplier=0.8,
    ),
    dict(
        n_estimators=800,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        reg_alpha=0.0,
        min_child_weight=2,
        gamma=0.0,
        max_delta_step=1,
        scale_pos_weight_multiplier=3.0,
    ),
    dict(
        n_estimators=700,
        max_depth=3,
        learning_rate=0.04,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.5,
        reg_alpha=0.2,
        min_child_weight=4,
        gamma=0.0,
        max_delta_step=1,
        scale_pos_weight_multiplier=3.5,
    ),
]
MYMETHOD_PARAM_CANDIDATES_SHARED = (
    MYMETHOD_PARAM_CANDIDATES
    + EXTRA_MYDATA_PARAM_CANDIDATES
    + _random_param_candidates(RANDOM_CANDIDATES_MYDATA, RANDOM_PARAM_SEED + 1, 0.6, 3.5)
)

ALIGN_FEATURES_STAGE34 = False
CANONICAL_FEATURES = [
    "season",
    "lapno",
    "race_progress",
    "pitstops_so_far",
    "position",
    "gap_to_leader",
    "gap_to_front",
    "sc_active",
    "vsc_active",
    "SCAny",
]
CANONICAL_MAP_REF = {
    "gap": "gap_to_leader",
    "interval": "gap_to_front",
}
CANONICAL_MAP_MY_W = {
    "gap_to_leader_prev": "gap_to_leader",
    "gap_to_front_prev": "gap_to_front",
    "lapno_prev": "lapno",
    "race_progress_prev": "race_progress",
    "pitstops_so_far_prev": "pitstops_so_far",
    "Position_prev": "position",
    "sc_active_prev": "sc_active",
    "vsc_active_prev": "vsc_active",
    "SCAny_prev": "SCAny",
}

MYMETHOD_FIXED_PARAMS_FILENAME = "stage4_best_params.json"
MYMETHOD_USE_FIXED_PARAMS = True
MYMETHOD_SHARED_FEATURES = [
    "season",
    "lapno",
    "race_progress",
    "pitstops_so_far",
    "position",
    "gap",
    "interval",
    "sc_active",
    "vsc_active",
    "SCAny",
    "GapOverInterval",
    "tireage",
]


def _load_fixed_params(path: "Path") -> dict[str, float | int] | None:
    if not path.exists():
        return None
    data = read_json(path)
    params = data.get("params") if isinstance(data, dict) else None
    if not isinstance(params, dict):
        return None
    return params


def _apply_feature_allowlist(
    features: list[str],
    allowlist: list[str],
    label: str,
) -> list[str]:
    allow = [f for f in allowlist if f in features]
    missing = [f for f in allowlist if f not in features]
    if missing:
        print(f"[WARN] {label} missing shared features: {missing}")
    if not allow:
        raise ValueError(f"{label} has no shared features available.")
    return allow


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
    alias_from_prev = {
        "lapno": "lapno_prev",
        "pitstops_so_far": "pitstops_so_far_prev",
        "position": "Position_prev",
        "race_progress": "race_progress_prev",
        "tireage": "stint_laps_prev",
        "sc_active": "sc_active_prev",
        "vsc_active": "vsc_active_prev",
        "SCAny": "SCAny_prev",
    }
    suspicious = []
    for col in same_lap_cols:
        if col not in df.columns:
            continue
        src = alias_from_prev.get(col)
        if src and src in df.columns:
            continue
        suspicious.append(col)
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

    if ALIGN_FEATURES_STAGE34:
        df_my_w, feats_my_w = select_canonical_features(
            df_my_w,
            feats_my_w,
            tgt_my_w,
            grp_my_w,
            CANONICAL_MAP_MY_W,
            CANONICAL_FEATURES,
        )
        print(f"[INFO] Aligned my_w features: {len(feats_my_w)} -> {feats_my_w}")

    feats_ref_shared = _apply_feature_allowlist(
        feats_ref,
        MYMETHOD_SHARED_FEATURES,
        "RefData",
    )
    feats_my_w_shared = _apply_feature_allowlist(
        feats_my_w,
        MYMETHOD_SHARED_FEATURES,
        "MyData+W",
    )

    fixed_params = None
    if MYMETHOD_USE_FIXED_PARAMS:
        fixed_params = _load_fixed_params(paths.out_summary_plots / MYMETHOD_FIXED_PARAMS_FILENAME)
        if fixed_params:
            print(f"[INFO] Using fixed MyMethod params from {MYMETHOD_FIXED_PARAMS_FILENAME}")

    mymethod_spec = my_method_spec()
    if fixed_params:
        mymethod_spec = ModelSpec(
            name=mymethod_spec.name,
            params={**mymethod_spec.params, **fixed_params},
        )

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
        df=df_ref, features=feats_ref_shared, target_col=tgt_ref, group_col=grp_ref,
        spec=mymethod_spec, baseline_spec=ref_tech_spec(),
        n_splits=n_splits, threshold=threshold, use_scale_pos_weight=MYMETHOD_USE_SCALE_POS_WEIGHT,
        use_feature_selection=MYMETHOD_FEATURE_SELECTION, fs_top_k=MYMETHOD_FS_TOP_K,
        use_target_encoding=MYMETHOD_TARGET_ENCODING,
        target_encoding_specs=MYMETHOD_TARGET_ENCODING_SPECS,
        drop_target_encoded_cols=MYMETHOD_TARGET_ENCODING_DROP_COLS,
        use_balanced_subsample=MYMETHOD_BALANCED_SUBSAMPLE,
        neg_pos_ratio=MYMETHOD_NEG_POS_RATIO,
        subsample_random_state=MYMETHOD_SUBSAMPLE_RANDOM_STATE,
        tune_threshold=TUNE_THRESHOLD, threshold_val_size=THRESHOLD_VAL_SIZE,
        threshold_min=MYMETHOD_THRESHOLD_MIN, threshold_max=THRESHOLD_MAX, beta=F_BETA,
        threshold_metric=MYMETHOD_THRESHOLD_METRIC, min_precision=REF_MIN_PRECISION,
        calibrate_proba=MYMETHOD_CALIBRATE,
        calibration_method=MYMETHOD_CALIBRATION_METHOD,
        calibration_val_size=MYMETHOD_CALIBRATION_VAL_SIZE,
        calibration_random_state=MYMETHOD_CALIBRATION_RANDOM_STATE,
        ensemble_seeds=MYMETHOD_ENSEMBLE_SEEDS,
        tune_params=TUNE_PARAMS, param_candidates=MYMETHOD_PARAM_CANDIDATES_SHARED,
        tune_metric=TUNE_METRIC_REF, tune_val_size=TUNE_VAL_SIZE, tune_random_state=TUNE_RANDOM_STATE,
        tune_cv_splits=TUNE_CV_SPLITS, delta_pr_auc_weight=DELTA_PR_AUC_WEIGHT_REF,
        delta_f1_weight=DELTA_F1_WEIGHT_REF,
        delta_f2_weight=DELTA_F2_WEIGHT_REF,
        delta_recall_weight=DELTA_RECALL_WEIGHT_REF,
    )
    save_json(paths.out_refdata_mymethod / "metrics.json", m2)

    # Stage 3
    m3 = run_stage_strict(
        df=df_my_w, features=feats_my_w_shared, target_col=tgt_my_w, group_col=grp_my_w,
        spec=ref_tech_spec(), n_splits=n_splits, threshold=threshold, use_scale_pos_weight=True,
        tune_threshold=TUNE_THRESHOLD, threshold_val_size=THRESHOLD_VAL_SIZE,
        threshold_min=MYDATA_THRESHOLD_MIN, threshold_max=THRESHOLD_MAX, beta=F_BETA,
        threshold_metric=MYDATA_THRESHOLD_METRIC, min_precision=MYDATA_MIN_PRECISION,
    )
    save_json(paths.out_mydata_reftech_weather / "metrics.json", m3)

    # Stage 4
    m4 = run_stage_strict(
        df=df_my_w, features=feats_my_w_shared, target_col=tgt_my_w, group_col=grp_my_w,
        spec=mymethod_spec, baseline_spec=ref_tech_spec(),
        n_splits=n_splits, threshold=threshold, use_scale_pos_weight=MYMETHOD_USE_SCALE_POS_WEIGHT,
        use_feature_selection=MYMETHOD_FEATURE_SELECTION, fs_top_k=MYMETHOD_FS_TOP_K,
        use_target_encoding=MYMETHOD_TARGET_ENCODING,
        target_encoding_specs=MYMETHOD_TARGET_ENCODING_SPECS,
        drop_target_encoded_cols=MYMETHOD_TARGET_ENCODING_DROP_COLS,
        use_balanced_subsample=MYMETHOD_BALANCED_SUBSAMPLE,
        neg_pos_ratio=MYMETHOD_NEG_POS_RATIO,
        subsample_random_state=MYMETHOD_SUBSAMPLE_RANDOM_STATE,
        tune_threshold=TUNE_THRESHOLD, threshold_val_size=THRESHOLD_VAL_SIZE,
        threshold_min=MYMETHOD_THRESHOLD_MIN, threshold_max=THRESHOLD_MAX, beta=F_BETA,
        threshold_metric=MYMETHOD_THRESHOLD_METRIC, min_precision=MYDATA_MIN_PRECISION,
        calibrate_proba=MYMETHOD_CALIBRATE,
        calibration_method=MYMETHOD_CALIBRATION_METHOD,
        calibration_val_size=MYMETHOD_CALIBRATION_VAL_SIZE,
        calibration_random_state=MYMETHOD_CALIBRATION_RANDOM_STATE,
        ensemble_seeds=MYMETHOD_ENSEMBLE_SEEDS,
        tune_params=TUNE_PARAMS, param_candidates=MYMETHOD_PARAM_CANDIDATES_SHARED,
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
