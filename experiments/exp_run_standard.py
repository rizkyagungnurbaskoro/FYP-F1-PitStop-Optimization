from __future__ import annotations

from pathlib import Path

from experiments.exp_config import get_paths
from experiments.exp_utils import add_feature_engineering, build_feature_list, ensure_dir, load_csv, safe_numeric
from experiments.exp_models import ref_tech_spec, my_method_spec, run_stage_strict

TARGET_COL = "decide_pitstop"
GROUP_COL = "race_id"
N_SPLITS = 5
BASE_THRESHOLD = 0.5
TUNE_THRESHOLD = True
THRESHOLD_VAL_SIZE = 0.2
THRESHOLD_MAX = 0.95
F_BETA = 2.0
TUNE_PARAMS = True
TUNE_VAL_SIZE = 0.2
TUNE_RANDOM_STATE = 42
TUNE_CV_SPLITS = 5

STANDARD_WEATHER_CSV = "data/strategy_weather_dataset.csv"
STANDARD_THRESHOLD_MIN = 0.02
STANDARD_THRESHOLD_METRIC = "f1"
STANDARD_MIN_PRECISION = 0.10
STANDARD_TUNE_METRIC = "delta_pr_auc_f1_guard"
STANDARD_DELTA_PR_AUC_WEIGHT = 0.65
STANDARD_FEATURE_SELECTION = True
STANDARD_FS_TOP_K = 28
STANDARD_BALANCED_SUBSAMPLE = False
STANDARD_NEG_POS_RATIO = 4.0
STANDARD_CALIBRATE = True
STANDARD_CALIBRATION_METHOD = "sigmoid"
STANDARD_CALIBRATION_VAL_SIZE = 0.2
STANDARD_CALIBRATION_RANDOM_STATE = 42

MYMETHOD_TARGET_ENCODING = True
MYMETHOD_TARGET_ENCODING_SPECS = [
    (("Driver",), "te_driver", 30.0),
    (("Driver", "season"), "te_driver_season", 50.0),
    (("Driver", "track_deg_category"), "te_driver_track", 50.0),
    (("track_deg_category",), "te_track_deg", 20.0),
]
MYMETHOD_TARGET_ENCODING_DROP_COLS = False
MYMETHOD_PARAM_CANDIDATES_MYDATA = [
    {},
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
]


def main() -> None:
    paths = get_paths()
    std_csv = Path(paths.root) / STANDARD_WEATHER_CSV
    if not std_csv.exists():
        raise FileNotFoundError(f"Standard dataset missing: {std_csv}")

    df_std = add_feature_engineering(safe_numeric(load_csv(std_csv)))
    feats_std = build_feature_list(df_std, TARGET_COL, GROUP_COL)

    out_std_ref = paths.results_dir / "standard_mydata_reftech"
    out_std_my = paths.results_dir / "standard_mydata_mymethod"
    ensure_dir(out_std_ref)
    ensure_dir(out_std_my)

    m3_std = run_stage_strict(
        df=df_std, features=feats_std, target_col=TARGET_COL, group_col=GROUP_COL,
        spec=ref_tech_spec(), n_splits=N_SPLITS, threshold=BASE_THRESHOLD, use_scale_pos_weight=True,
        tune_threshold=TUNE_THRESHOLD, threshold_val_size=THRESHOLD_VAL_SIZE,
        threshold_min=STANDARD_THRESHOLD_MIN, threshold_max=THRESHOLD_MAX, beta=F_BETA,
        threshold_metric=STANDARD_THRESHOLD_METRIC, min_precision=STANDARD_MIN_PRECISION,
    )
    out_std_ref.joinpath("metrics.json").write_text(
        __import__("json").dumps(m3_std, indent=2),
        encoding="utf-8",
    )

    m4_std = run_stage_strict(
        df=df_std, features=feats_std, target_col=TARGET_COL, group_col=GROUP_COL,
        spec=my_method_spec(), baseline_spec=ref_tech_spec(),
        n_splits=N_SPLITS, threshold=BASE_THRESHOLD, use_scale_pos_weight=True,
        use_feature_selection=STANDARD_FEATURE_SELECTION, fs_top_k=STANDARD_FS_TOP_K,
        use_target_encoding=MYMETHOD_TARGET_ENCODING,
        target_encoding_specs=MYMETHOD_TARGET_ENCODING_SPECS,
        drop_target_encoded_cols=MYMETHOD_TARGET_ENCODING_DROP_COLS,
        use_balanced_subsample=STANDARD_BALANCED_SUBSAMPLE,
        neg_pos_ratio=STANDARD_NEG_POS_RATIO,
        subsample_random_state=STANDARD_CALIBRATION_RANDOM_STATE,
        tune_threshold=TUNE_THRESHOLD, threshold_val_size=THRESHOLD_VAL_SIZE,
        threshold_min=STANDARD_THRESHOLD_MIN, threshold_max=THRESHOLD_MAX, beta=F_BETA,
        threshold_metric=STANDARD_THRESHOLD_METRIC, min_precision=STANDARD_MIN_PRECISION,
        calibrate_proba=STANDARD_CALIBRATE,
        calibration_method=STANDARD_CALIBRATION_METHOD,
        calibration_val_size=STANDARD_CALIBRATION_VAL_SIZE,
        calibration_random_state=STANDARD_CALIBRATION_RANDOM_STATE,
        tune_params=TUNE_PARAMS, param_candidates=MYMETHOD_PARAM_CANDIDATES_MYDATA,
        tune_metric=STANDARD_TUNE_METRIC, tune_val_size=TUNE_VAL_SIZE, tune_random_state=TUNE_RANDOM_STATE,
        tune_cv_splits=TUNE_CV_SPLITS, delta_pr_auc_weight=STANDARD_DELTA_PR_AUC_WEIGHT,
        delta_f1_weight=0.30,
        delta_f2_weight=0.40,
        delta_recall_weight=0.30,
    )
    out_std_my.joinpath("metrics.json").write_text(
        __import__("json").dumps(m4_std, indent=2),
        encoding="utf-8",
    )

    print("[OK] Saved standard metrics.json for Stage 3/4.")


if __name__ == "__main__":
    main()
