from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.exp_config import get_paths
from experiments.exp_models import ModelSpec, my_method_spec, ref_tech_spec, run_stage_strict
from experiments.exp_utils import (
    add_feature_engineering,
    build_feature_list,
    ensure_dir,
    load_csv,
    safe_numeric,
    select_canonical_features,
)
from experiments.exp_run_all import (
    ALIGN_FEATURES_STAGE34,
    CANONICAL_FEATURES,
    CANONICAL_MAP_MY_W,
    MYDATA_MIN_PRECISION,
    MYMETHOD_BALANCED_SUBSAMPLE,
    MYMETHOD_FS_TOP_K,
    MYMETHOD_NEG_POS_RATIO,
    MYMETHOD_SUBSAMPLE_RANDOM_STATE,
    MYMETHOD_TARGET_ENCODING,
    MYMETHOD_TARGET_ENCODING_DROP_COLS,
    MYMETHOD_TARGET_ENCODING_SPECS,
    MYMETHOD_THRESHOLD_METRIC,
    MYMETHOD_THRESHOLD_MIN,
    MYMETHOD_FEATURE_SELECTION,
    MYMETHOD_CALIBRATE,
    MYMETHOD_CALIBRATION_METHOD,
    MYMETHOD_CALIBRATION_VAL_SIZE,
    MYMETHOD_CALIBRATION_RANDOM_STATE,
    MYMETHOD_ENSEMBLE_SEEDS,
    MYMETHOD_PARAM_CANDIDATES_SHARED,
)

TARGET_COL = "decide_pitstop"
GROUP_COL = "race_id"
N_SPLITS = 3
THRESHOLD_VAL_SIZE = 0.2
THRESHOLD_MAX = 0.95
F_BETA = 2.0
TOP_K = 25


def _summarize(result: dict) -> dict[str, Any]:
    folds = result.get("folds", [])
    f1_vals = [float(f["f1"]) for f in folds]
    pr_vals = [float(f["pr_auc"]) for f in folds]
    mean_f1 = float(sum(f1_vals) / len(f1_vals)) if f1_vals else 0.0
    mean_pr = float(sum(pr_vals) / len(pr_vals)) if pr_vals else 0.0
    return {"mean_f1": mean_f1, "mean_pr_auc": mean_pr}


def _evaluate(
    df: pd.DataFrame,
    features: list[str],
    spec: ModelSpec,
) -> dict[str, Any]:
    result = run_stage_strict(
        df=df,
        features=features,
        target_col=TARGET_COL,
        group_col=GROUP_COL,
        spec=spec,
        baseline_spec=ref_tech_spec(),
        n_splits=N_SPLITS,
        threshold=0.5,
        use_scale_pos_weight=True,
        use_feature_selection=MYMETHOD_FEATURE_SELECTION,
        fs_top_k=MYMETHOD_FS_TOP_K,
        use_target_encoding=MYMETHOD_TARGET_ENCODING,
        target_encoding_specs=MYMETHOD_TARGET_ENCODING_SPECS,
        drop_target_encoded_cols=MYMETHOD_TARGET_ENCODING_DROP_COLS,
        use_balanced_subsample=MYMETHOD_BALANCED_SUBSAMPLE,
        neg_pos_ratio=MYMETHOD_NEG_POS_RATIO,
        subsample_random_state=MYMETHOD_SUBSAMPLE_RANDOM_STATE,
        tune_threshold=True,
        threshold_val_size=THRESHOLD_VAL_SIZE,
        threshold_min=MYMETHOD_THRESHOLD_MIN,
        threshold_max=THRESHOLD_MAX,
        beta=F_BETA,
        threshold_metric=MYMETHOD_THRESHOLD_METRIC,
        min_precision=MYDATA_MIN_PRECISION,
        calibrate_proba=MYMETHOD_CALIBRATE,
        calibration_method=MYMETHOD_CALIBRATION_METHOD,
        calibration_val_size=MYMETHOD_CALIBRATION_VAL_SIZE,
        calibration_random_state=MYMETHOD_CALIBRATION_RANDOM_STATE,
        ensemble_seeds=MYMETHOD_ENSEMBLE_SEEDS,
        tune_params=False,
        param_candidates=[],
    )
    return _summarize(result)


def main() -> None:
    paths = get_paths()
    out_dir = Path(paths.out_summary_plots)
    ensure_dir(out_dir)

    df_my_w = add_feature_engineering(safe_numeric(load_csv(paths.my_weather_csv)))
    feats_my_w = build_feature_list(df_my_w, TARGET_COL, GROUP_COL)

    if ALIGN_FEATURES_STAGE34:
        df_my_w, feats_my_w = select_canonical_features(
            df_my_w,
            feats_my_w,
            TARGET_COL,
            GROUP_COL,
            CANONICAL_MAP_MY_W,
            CANONICAL_FEATURES,
        )

    base_spec = my_method_spec()
    baseline_spec = ref_tech_spec()
    baseline_metrics = _evaluate(df_my_w, feats_my_w, baseline_spec)

    rows: list[dict[str, Any]] = []
    for cand in MYMETHOD_PARAM_CANDIDATES_SHARED[:TOP_K]:
        params = {**base_spec.params, **(cand or {})}
        spec = ModelSpec(name=base_spec.name, params=params)
        metrics = _evaluate(df_my_w, feats_my_w, spec)
        mean_f1 = metrics["mean_f1"]
        mean_pr = metrics["mean_pr_auc"]
        score = 0.5 * mean_f1 + 0.5 * mean_pr
        rows.append(
            {
                "score": score,
                "mean_f1": mean_f1,
                "mean_pr_auc": mean_pr,
                "delta_f1": mean_f1 - baseline_metrics["mean_f1"],
                "delta_pr_auc": mean_pr - baseline_metrics["mean_pr_auc"],
                "params": json.dumps(cand or {}, sort_keys=True),
            }
        )

    df_out = pd.DataFrame(rows).sort_values(
        ["score", "mean_f1", "mean_pr_auc"],
        ascending=[False, False, False],
    )
    out_csv = out_dir / "stage4_param_search.csv"
    df_out.to_csv(out_csv, index=False)

    best_row = df_out.iloc[0].to_dict() if not df_out.empty else {}
    best_params = json.loads(best_row.get("params", "{}")) if best_row else {}
    best_json = {
        "params": best_params,
        "score": best_row.get("score", 0.0),
        "mean_f1": best_row.get("mean_f1", 0.0),
        "mean_pr_auc": best_row.get("mean_pr_auc", 0.0),
        "delta_f1": best_row.get("delta_f1", 0.0),
        "delta_pr_auc": best_row.get("delta_pr_auc", 0.0),
        "baseline_mean_f1": baseline_metrics["mean_f1"],
        "baseline_mean_pr_auc": baseline_metrics["mean_pr_auc"],
    }
    (out_dir / "stage4_best_params.json").write_text(
        json.dumps(best_json, indent=2),
        encoding="utf-8",
    )

    print(f"[OK] Saved {out_csv}")
    print("[BEST] score=", best_json["score"])
    print("[BEST] mean_f1=", best_json["mean_f1"])
    print("[BEST] mean_pr_auc=", best_json["mean_pr_auc"])


if __name__ == "__main__":
    main()
