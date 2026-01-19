from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiments.exp_config import get_paths
from experiments.exp_utils import add_feature_engineering, build_feature_list, ensure_dir, load_csv, safe_numeric
from experiments.exp_models import my_method_spec, ref_tech_spec, run_stage_strict

TARGET_COL = "decide_pitstop"
GROUP_COL = "race_id"
N_SPLITS = 2
BASE_THRESHOLD = 0.5
THRESHOLD_VAL_SIZE = 0.2
THRESHOLD_MIN = 0.02
THRESHOLD_MAX = 0.95
F_BETA = 2.0

FS_TOP_K = 30
MYMETHOD_TARGET_ENCODING_SPECS = [
    (("Driver",), "te_driver", 30.0),
    (("Driver", "season"), "te_driver_season", 50.0),
    (("Driver", "track_deg_category"), "te_driver_track", 50.0),
    (("track_deg_category",), "te_track_deg", 20.0),
]

CALIBRATION_METHOD = "sigmoid"
CALIBRATION_VAL_SIZE = 0.2
CALIBRATION_RANDOM_STATE = 42
SUBSAMPLE_RANDOM_STATE = 42
ENSEMBLE_SEEDS = [42]


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
        raise ValueError(msg)


def _mean_std(values: list[float]) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return mean, std


def _summarize(result: dict, config: dict) -> dict:
    folds = result.get("folds", [])
    precision_vals = [float(f["precision"]) for f in folds]
    recall_vals = [float(f["recall"]) for f in folds]
    f1_vals = [float(f["f1"]) for f in folds]
    fbeta_vals = [float(f["fbeta"]) for f in folds]
    pr_auc_vals = [float(f["pr_auc"]) for f in folds]
    thresholds = [float(f["threshold"]) for f in folds]
    feat_counts = [int(f.get("feature_count", 0)) for f in folds]

    mean_precision, std_precision = _mean_std(precision_vals)
    mean_recall, std_recall = _mean_std(recall_vals)
    mean_f1, std_f1 = _mean_std(f1_vals)
    mean_fbeta, std_fbeta = _mean_std(fbeta_vals)
    mean_pr_auc, std_pr_auc = _mean_std(pr_auc_vals)
    mean_threshold, std_threshold = _mean_std(thresholds)
    mean_features, _ = _mean_std(feat_counts)

    return {
        "config": config["name"],
        "feature_selection": bool(config["feature_selection"]),
        "calibrate_proba": bool(config["calibrate_proba"]),
        "target_encoding": bool(config["target_encoding"]),
        "drop_target_encoded_cols": bool(config["drop_target_encoded_cols"]),
        "balanced_subsample": bool(config["balanced_subsample"]),
        "neg_pos_ratio": float(config["neg_pos_ratio"]),
        "threshold_metric": config["threshold_metric"],
        "min_precision": float(config["min_precision"]),
        "ensemble_size": int(len(ENSEMBLE_SEEDS)),
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "mean_fbeta": mean_fbeta,
        "std_fbeta": std_fbeta,
        "mean_precision": mean_precision,
        "std_precision": std_precision,
        "mean_recall": mean_recall,
        "std_recall": std_recall,
        "mean_pr_auc": mean_pr_auc,
        "std_pr_auc": std_pr_auc,
        "mean_threshold": mean_threshold,
        "std_threshold": std_threshold,
        "mean_feature_count": float(mean_features),
    }


def main() -> None:
    paths = get_paths()
    df_my_w = add_feature_engineering(safe_numeric(load_csv(paths.my_weather_csv)))

    _require_columns(df_my_w, [TARGET_COL, GROUP_COL], "Stage 3/4 dataset")
    _leakage_guard_stage34(df_my_w, str(paths.my_weather_csv))

    feats = build_feature_list(df_my_w, TARGET_COL, GROUP_COL)

    configs = [
        {
            "name": "base",
            "feature_selection": True,
            "calibrate_proba": True,
            "target_encoding": True,
            "drop_target_encoded_cols": False,
            "balanced_subsample": False,
            "neg_pos_ratio": 4.0,
            "threshold_metric": "f1",
            "min_precision": 0.10,
        },
        {
            "name": "no_calibration",
            "feature_selection": True,
            "calibrate_proba": False,
            "target_encoding": True,
            "drop_target_encoded_cols": False,
            "balanced_subsample": False,
            "neg_pos_ratio": 4.0,
            "threshold_metric": "f1",
            "min_precision": 0.10,
        },
        {
            "name": "no_target_encoding",
            "feature_selection": True,
            "calibrate_proba": True,
            "target_encoding": False,
            "drop_target_encoded_cols": False,
            "balanced_subsample": False,
            "neg_pos_ratio": 4.0,
            "threshold_metric": "f1",
            "min_precision": 0.10,
        },
        {
            "name": "te_drop_cols",
            "feature_selection": True,
            "calibrate_proba": True,
            "target_encoding": True,
            "drop_target_encoded_cols": True,
            "balanced_subsample": False,
            "neg_pos_ratio": 4.0,
            "threshold_metric": "f1",
            "min_precision": 0.10,
        },
        {
            "name": "no_feature_selection",
            "feature_selection": False,
            "calibrate_proba": True,
            "target_encoding": True,
            "drop_target_encoded_cols": False,
            "balanced_subsample": False,
            "neg_pos_ratio": 4.0,
            "threshold_metric": "f1",
            "min_precision": 0.10,
        },
        {
            "name": "balanced_ratio_3",
            "feature_selection": True,
            "calibrate_proba": True,
            "target_encoding": True,
            "drop_target_encoded_cols": False,
            "balanced_subsample": True,
            "neg_pos_ratio": 3.0,
            "threshold_metric": "f1",
            "min_precision": 0.10,
        },
        {
            "name": "threshold_fbeta",
            "feature_selection": True,
            "calibrate_proba": True,
            "target_encoding": True,
            "drop_target_encoded_cols": False,
            "balanced_subsample": False,
            "neg_pos_ratio": 4.0,
            "threshold_metric": "fbeta",
            "min_precision": 0.10,
        },
        {
            "name": "threshold_recall_at_precision",
            "feature_selection": True,
            "calibrate_proba": True,
            "target_encoding": True,
            "drop_target_encoded_cols": False,
            "balanced_subsample": False,
            "neg_pos_ratio": 4.0,
            "threshold_metric": "recall_at_precision",
            "min_precision": 0.12,
        },
    ]

    rows = []
    for config in configs:
        result = run_stage_strict(
            df=df_my_w,
            features=feats,
            target_col=TARGET_COL,
            group_col=GROUP_COL,
            spec=my_method_spec(),
            baseline_spec=ref_tech_spec(),
            n_splits=N_SPLITS,
            threshold=BASE_THRESHOLD,
            use_scale_pos_weight=True,
            use_feature_selection=config["feature_selection"],
            fs_top_k=FS_TOP_K,
            use_target_encoding=config["target_encoding"],
            target_encoding_specs=MYMETHOD_TARGET_ENCODING_SPECS,
            drop_target_encoded_cols=config["drop_target_encoded_cols"],
            use_balanced_subsample=config["balanced_subsample"],
            neg_pos_ratio=config["neg_pos_ratio"],
            subsample_random_state=SUBSAMPLE_RANDOM_STATE,
            tune_threshold=True,
            threshold_val_size=THRESHOLD_VAL_SIZE,
            threshold_min=THRESHOLD_MIN,
            threshold_max=THRESHOLD_MAX,
            beta=F_BETA,
            threshold_metric=config["threshold_metric"],
            min_precision=config["min_precision"],
            calibrate_proba=config["calibrate_proba"],
            calibration_method=CALIBRATION_METHOD,
            calibration_val_size=CALIBRATION_VAL_SIZE,
            calibration_random_state=CALIBRATION_RANDOM_STATE,
            ensemble_seeds=ENSEMBLE_SEEDS,
            tune_params=False,
            param_candidates=[],
        )
        rows.append(_summarize(result, config))

    out_dir = Path(paths.out_summary_plots)
    ensure_dir(out_dir)
    df_out = pd.DataFrame(rows)
    df_out = df_out.sort_values(["mean_f1", "mean_pr_auc"], ascending=[False, False])
    out_csv = out_dir / "stage4_ablation.csv"
    df_out.to_csv(out_csv, index=False)

    best = df_out.iloc[0].to_dict() if not df_out.empty else {}
    best_path = out_dir / "stage4_ablation_best.json"
    best_path.write_text(pd.Series(best).to_json(indent=2), encoding="utf-8")

    print(f"[OK] Saved {out_csv}")
    if best:
        print("[BEST] config=", best.get("config"))
        print("[BEST] mean_f1=", best.get("mean_f1"))
        print("[BEST] mean_pr_auc=", best.get("mean_pr_auc"))


if __name__ == "__main__":
    main()
