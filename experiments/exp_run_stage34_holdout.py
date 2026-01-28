from __future__ import annotations

import numpy as np
import pandas as pd

from experiments import exp_run_all as cfg
from experiments.exp_config import get_paths
from experiments.exp_utils import (
    add_feature_engineering,
    build_feature_list,
    ensure_dir,
    load_csv,
    safe_numeric,
    save_json,
    select_canonical_features,
)
from experiments.exp_models import ModelSpec, my_method_spec, ref_tech_spec, run_stage_holdout

TARGET_COL = cfg.TARGET_COL
GROUP_COL = cfg.GROUP_COL
HOLDOUT_TEST_SIZE = 0.3
HOLDOUT_RANDOM_STATE = 42


def _fold_mean_std(folds: list[dict], key: str) -> tuple[float, float]:
    vals = [f[key] for f in folds if key in f]
    if not vals:
        return 0.0, 0.0
    mean = float(np.mean(vals))
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return mean, std


def _write_stage34_summary(outpath, m3, m4) -> None:
    rows = []
    for name, m in [
        ("Stage 3: RefTech on MyData+W", m3),
        ("Stage 4: MyMethod on MyData+W", m4),
    ]:
        folds = m.get("folds", [])
        prec_mu, prec_sd = _fold_mean_std(folds, "precision")
        rec_mu, rec_sd = _fold_mean_std(folds, "recall")
        pr_mu, pr_sd = _fold_mean_std(folds, "pr_auc")
        fbeta_mu = m.get("mean_fbeta", 0.0)
        fbeta_sd = m.get("std_fbeta", 0.0)
        threshold_mean = m.get("mean_threshold", m.get("threshold", 0.5))
        threshold_std = m.get("std_threshold", 0.0)
        rows.append(
            dict(
                stage=name,
                mean_f1=m.get("mean_f1", 0.0),
                std_f1=m.get("std_f1", 0.0),
                mean_fbeta=fbeta_mu,
                std_fbeta=fbeta_sd,
                mean_precision=prec_mu,
                std_precision=prec_sd,
                mean_recall=rec_mu,
                std_recall=rec_sd,
                mean_pr_auc=pr_mu,
                std_pr_auc=pr_sd,
                threshold=threshold_mean,
                std_threshold=threshold_std,
                n_splits=m.get("n_splits", 1),
                beta=m.get("beta", 1.0),
                split_kind=m.get("split_kind", ""),
                holdout_test_size=m.get("holdout_test_size", HOLDOUT_TEST_SIZE),
            )
        )
    pd.DataFrame(rows).to_csv(outpath, index=False)


def main() -> None:
    paths = get_paths()
    for d in [
        paths.results_dir,
        paths.out_mydata_reftech_weather_holdout,
        paths.out_mydata_mymethod_weather_holdout,
        paths.out_summary_plots,
    ]:
        ensure_dir(d)

    df_my_w = add_feature_engineering(safe_numeric(load_csv(paths.my_weather_csv)))
    cfg._require_columns(df_my_w, [TARGET_COL, GROUP_COL], "Stage 3/4 dataset")
    print(f"[INFO] Stage 3/4 dataset: {paths.my_weather_csv}")
    cfg._leakage_guard_stage34(df_my_w, str(paths.my_weather_csv))

    feats_my_w = build_feature_list(df_my_w, TARGET_COL, GROUP_COL)
    if cfg.ALIGN_FEATURES_STAGE34:
        df_my_w, feats_my_w = select_canonical_features(
            df_my_w,
            feats_my_w,
            TARGET_COL,
            GROUP_COL,
            cfg.CANONICAL_MAP_MY_W,
            cfg.CANONICAL_FEATURES,
        )
        print(f"[INFO] Aligned my_w features: {len(feats_my_w)} -> {feats_my_w}")

    feats_my_w_shared = cfg._apply_feature_allowlist(
        feats_my_w,
        cfg.MYMETHOD_SHARED_FEATURES,
        "MyData+W",
    )

    fixed_params = None
    if cfg.MYMETHOD_USE_FIXED_PARAMS:
        fixed_params = cfg._load_fixed_params(
            paths.out_summary_plots / cfg.MYMETHOD_FIXED_PARAMS_FILENAME
        )
        if fixed_params:
            print(f"[INFO] Using fixed MyMethod params from {cfg.MYMETHOD_FIXED_PARAMS_FILENAME}")

    mymethod_spec = my_method_spec()
    if fixed_params:
        mymethod_spec = ModelSpec(
            name=mymethod_spec.name,
            params={**mymethod_spec.params, **fixed_params},
        )

    # Stage 3 (RefTech on MyData+W) - Group holdout 70/30
    m3 = run_stage_holdout(
        df=df_my_w,
        features=feats_my_w_shared,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        spec=ref_tech_spec(),
        test_size=HOLDOUT_TEST_SIZE,
        random_state=HOLDOUT_RANDOM_STATE,
        threshold=cfg.BASE_THRESHOLD,
        use_scale_pos_weight=True,
        tune_threshold=cfg.TUNE_THRESHOLD,
        threshold_val_size=cfg.THRESHOLD_VAL_SIZE,
        threshold_min=cfg.MYDATA_THRESHOLD_MIN,
        threshold_max=cfg.THRESHOLD_MAX,
        beta=cfg.F_BETA,
        threshold_metric=cfg.MYDATA_THRESHOLD_METRIC,
        min_precision=cfg.MYDATA_MIN_PRECISION,
    )
    save_json(paths.out_mydata_reftech_weather_holdout / "metrics.json", m3)

    # Stage 4 (MyMethod on MyData+W) - Group holdout 70/30
    m4 = run_stage_holdout(
        df=df_my_w,
        features=feats_my_w_shared,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        spec=mymethod_spec,
        baseline_spec=ref_tech_spec(),
        test_size=HOLDOUT_TEST_SIZE,
        random_state=HOLDOUT_RANDOM_STATE,
        threshold=cfg.BASE_THRESHOLD,
        use_scale_pos_weight=cfg.MYMETHOD_USE_SCALE_POS_WEIGHT,
        use_feature_selection=cfg.MYMETHOD_FEATURE_SELECTION,
        fs_top_k=cfg.MYMETHOD_FS_TOP_K,
        use_target_encoding=cfg.MYMETHOD_TARGET_ENCODING,
        target_encoding_specs=cfg.MYMETHOD_TARGET_ENCODING_SPECS,
        drop_target_encoded_cols=cfg.MYMETHOD_TARGET_ENCODING_DROP_COLS,
        use_balanced_subsample=cfg.MYMETHOD_BALANCED_SUBSAMPLE,
        neg_pos_ratio=cfg.MYMETHOD_NEG_POS_RATIO,
        subsample_random_state=cfg.MYMETHOD_SUBSAMPLE_RANDOM_STATE,
        tune_threshold=cfg.TUNE_THRESHOLD,
        threshold_val_size=cfg.THRESHOLD_VAL_SIZE,
        threshold_min=cfg.MYMETHOD_THRESHOLD_MIN,
        threshold_max=cfg.THRESHOLD_MAX,
        beta=cfg.F_BETA,
        threshold_metric=cfg.MYMETHOD_THRESHOLD_METRIC,
        min_precision=cfg.MYDATA_MIN_PRECISION,
        calibrate_proba=cfg.MYMETHOD_CALIBRATE,
        calibration_method=cfg.MYMETHOD_CALIBRATION_METHOD,
        calibration_val_size=cfg.MYMETHOD_CALIBRATION_VAL_SIZE,
        calibration_random_state=cfg.MYMETHOD_CALIBRATION_RANDOM_STATE,
        ensemble_seeds=cfg.MYMETHOD_ENSEMBLE_SEEDS,
        tune_params=cfg.TUNE_PARAMS,
        param_candidates=cfg.MYMETHOD_PARAM_CANDIDATES_SHARED,
        tune_metric=cfg.TUNE_METRIC_MYDATA,
        tune_val_size=cfg.TUNE_VAL_SIZE,
        tune_random_state=cfg.TUNE_RANDOM_STATE,
        tune_cv_splits=cfg.TUNE_CV_SPLITS,
        delta_pr_auc_weight=cfg.DELTA_PR_AUC_WEIGHT_MYDATA,
        delta_f1_weight=cfg.DELTA_F1_WEIGHT_MYDATA,
        delta_f2_weight=cfg.DELTA_F2_WEIGHT_MYDATA,
        delta_recall_weight=cfg.DELTA_RECALL_WEIGHT_MYDATA,
    )
    save_json(paths.out_mydata_mymethod_weather_holdout / "metrics.json", m4)

    summary_path = paths.out_summary_plots / "stage34_holdout_summary.csv"
    _write_stage34_summary(summary_path, m3, m4)

    print("[OK] Saved Stage 3/4 holdout metrics + summary.")
    print("[OUT]", paths.out_mydata_reftech_weather_holdout)
    print("[OUT]", paths.out_mydata_mymethod_weather_holdout)
    print("[OUT]", summary_path)


if __name__ == "__main__":
    # run from project root:
    #   python -m experiments.exp_run_stage34_holdout
    main()
