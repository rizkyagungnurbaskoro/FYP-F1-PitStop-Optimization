from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from xgboost import XGBClassifier


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: Dict[str, Any]


def _split_cols(df: pd.DataFrame, features: List[str]) -> Tuple[List[str], List[str]]:
    cat_cols = [c for c in features if df[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]
    return num_cols, cat_cols


def _make_pipeline(df: pd.DataFrame, features: List[str], spec: ModelSpec) -> Pipeline:
    num_cols, cat_cols = _split_cols(df, features)

    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )

    clf = XGBClassifier(**spec.params)

    return Pipeline([("pre", pre), ("clf", clf)])


def _fbeta(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    return 0.0 if denom == 0.0 else (1.0 + b2) * precision * recall / denom


def _apply_scale_pos_weight(params: Dict[str, Any], ytr: np.ndarray) -> Dict[str, Any]:
    out = dict(params)
    mult = float(out.pop("scale_pos_weight_multiplier", 1.0))
    pos = int((ytr == 1).sum())
    neg = int((ytr == 0).sum())
    spw = float(neg / pos) if pos > 0 else 1.0
    out["scale_pos_weight"] = spw * mult
    return out


def _predict_proba_model(
    dtr_fit: pd.DataFrame,
    dtr_cal: pd.DataFrame | None,
    dte: pd.DataFrame,
    features: List[str],
    target_col: str,
    spec: ModelSpec,
    calibrate_proba: bool,
    calibration_method: str,
) -> np.ndarray:
    pipe = _make_pipeline(dtr_fit, features, spec)
    pipe.fit(dtr_fit[features], dtr_fit[target_col].astype(int).values)

    if calibrate_proba and dtr_cal is not None and dtr_cal[target_col].astype(int).nunique() >= 2:
        try:
            cal = CalibratedClassifierCV(pipe, method=calibration_method, cv="prefit")
            cal.fit(dtr_cal[features], dtr_cal[target_col].astype(int).values)
            return cal.predict_proba(dte[features])[:, 1]
        except Exception:
            cal = CalibratedClassifierCV(pipe, method=calibration_method, cv=3)
            cal.fit(dtr_fit[features], dtr_fit[target_col].astype(int).values)
            return cal.predict_proba(dte[features])[:, 1]

    return pipe.predict_proba(dte[features])[:, 1]


def _predict_proba_ensemble(
    dtr_fit: pd.DataFrame,
    dtr_cal: pd.DataFrame | None,
    dte: pd.DataFrame,
    features: List[str],
    target_col: str,
    spec: ModelSpec,
    calibrate_proba: bool,
    calibration_method: str,
    ensemble_seeds: List[int] | None,
) -> np.ndarray:
    if not ensemble_seeds:
        return _predict_proba_model(
            dtr_fit,
            dtr_cal,
            dte,
            features,
            target_col,
            spec,
            calibrate_proba,
            calibration_method,
        )

    probs = []
    for seed in ensemble_seeds:
        params = dict(spec.params)
        params["random_state"] = int(seed)
        spec_seed = ModelSpec(name=spec.name, params=params)
        probs.append(
            _predict_proba_model(
                dtr_fit,
                dtr_cal,
                dte,
                features,
                target_col,
                spec_seed,
                calibrate_proba,
                calibration_method,
            )
        )
    return np.mean(np.vstack(probs), axis=0)


def _select_features_mi(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    top_k: int,
    random_state: int,
) -> List[str]:
    num_cols = [c for c in features if df[c].dtype != "object"]
    cat_cols = [c for c in features if c not in num_cols]

    if not num_cols:
        return features

    X = df[num_cols].apply(pd.to_numeric, errors="coerce")
    if X.isna().any().any():
        X = X.fillna(X.median(numeric_only=True))

    y = df[target_col].astype(int).values
    mi = mutual_info_classif(X, y, random_state=random_state)
    k = min(max(1, int(top_k)), len(num_cols))
    top_idx = np.argsort(mi)[::-1][:k]
    sel_num = [num_cols[i] for i in top_idx]
    return sel_num + cat_cols


def _add_target_encodings(
    dtr: pd.DataFrame,
    dte: pd.DataFrame,
    target_col: str,
    specs: List[Tuple],
    default_smoothing: float = 20.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if not specs:
        return dtr, dte, []

    dtr_out = dtr.copy()
    dte_out = dte.copy()
    new_cols: List[str] = []
    global_mean = float(dtr_out[target_col].astype(float).mean())

    for spec in specs:
        if len(spec) < 2:
            continue
        cols = spec[0]
        out_col = spec[1]
        smoothing = default_smoothing if len(spec) < 3 else float(spec[2])
        if out_col in dtr_out.columns:
            continue
        if any(c not in dtr_out.columns for c in cols):
            continue
        grouped = (
            dtr_out.groupby(list(cols), dropna=False)[target_col]
            .agg(["sum", "count"])
            .reset_index()
        )
        grouped[out_col] = (grouped["sum"] + global_mean * smoothing) / (
            grouped["count"] + smoothing
        )
        enc = grouped[list(cols) + [out_col]]
        dtr_out = dtr_out.merge(enc, on=list(cols), how="left", sort=False)
        dte_out = dte_out.merge(enc, on=list(cols), how="left", sort=False)
        dtr_out[out_col] = dtr_out[out_col].fillna(global_mean)
        dte_out[out_col] = dte_out[out_col].fillna(global_mean)
        new_cols.append(out_col)

    return dtr_out, dte_out, new_cols


def _apply_target_encoding(
    dtr: pd.DataFrame,
    dte: pd.DataFrame,
    target_col: str,
    features: List[str],
    specs: List[Tuple],
    drop_original_cols: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    if not specs:
        return dtr, dte, list(features)

    dtr_out, dte_out, new_cols = _add_target_encodings(dtr, dte, target_col, specs)
    features_out = list(features)
    if drop_original_cols:
        encoded_cols: set[str] = set()
        for spec in specs:
            if len(spec) >= 1:
                encoded_cols.update(list(spec[0]))
        features_out = [f for f in features_out if f not in encoded_cols]
    for col in new_cols:
        if col not in features_out:
            features_out.append(col)
    return dtr_out, dte_out, features_out


def _balanced_subsample(
    df: pd.DataFrame,
    target_col: str,
    neg_pos_ratio: float,
    random_state: int,
) -> pd.DataFrame:
    y = df[target_col].astype(int).values
    pos_idx = df.index[y == 1]
    neg_idx = df.index[y == 0]
    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return df

    n_neg = int(min(len(neg_idx), max(1, int(len(pos_idx) * neg_pos_ratio))))
    rng = np.random.RandomState(int(random_state))
    neg_sample = rng.choice(neg_idx, size=n_neg, replace=False)
    keep = np.concatenate([pos_idx, neg_sample])
    return df.loc[keep]


def _split_calibration_indices(
    df: pd.DataFrame,
    target_col: str,
    group_col: str,
    val_size: float,
    random_state: int,
) -> Tuple[np.ndarray | None, np.ndarray | None]:
    y = df[target_col].astype(int).values
    groups = df[group_col].values
    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    try:
        tr_idx, cal_idx = next(splitter.split(df, y, groups))
    except ValueError:
        return None, None
    if len(tr_idx) == 0 or len(cal_idx) == 0:
        return None, None
    return tr_idx, cal_idx


def _best_threshold_from_pr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    default_threshold: float,
    min_threshold: float,
    max_threshold: float,
    beta: float,
) -> float:
    if len(np.unique(y_true)) < 2:
        return default_threshold

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if thresholds.size == 0:
        return default_threshold

    precision = precision[:-1]
    recall = recall[:-1]
    b2 = beta * beta
    denom = b2 * precision + recall
    fbeta_vals = np.where(denom > 0, (1.0 + b2) * precision * recall / denom, 0.0)

    mask = (thresholds >= min_threshold) & (thresholds <= max_threshold)
    if mask.any():
        thresholds = thresholds[mask]
        fbeta_vals = fbeta_vals[mask]

    if thresholds.size == 0:
        return default_threshold

    best_idx = int(np.nanargmax(fbeta_vals))
    return float(thresholds[best_idx])


def _best_threshold_recall_at_precision(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    default_threshold: float,
    min_threshold: float,
    max_threshold: float,
    min_precision: float,
) -> float:
    if len(np.unique(y_true)) < 2:
        return default_threshold

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if thresholds.size == 0:
        return default_threshold

    precision = precision[:-1]
    recall = recall[:-1]

    mask = (thresholds >= min_threshold) & (thresholds <= max_threshold) & (precision >= min_precision)
    if not mask.any():
        return default_threshold

    idxs = np.where(mask)[0]
    best_recall = float(np.max(recall[idxs]))
    best_idxs = idxs[recall[idxs] == best_recall]
    best_idx = best_idxs[np.argmax(precision[best_idxs])]
    return float(thresholds[best_idx])


def _max_recall_at_precision(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_threshold: float,
    max_threshold: float,
    min_precision: float,
) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if thresholds.size == 0:
        return 0.0

    precision = precision[:-1]
    recall = recall[:-1]

    mask = (thresholds >= min_threshold) & (thresholds <= max_threshold) & (precision >= min_precision)
    if not mask.any():
        return 0.0

    return float(np.max(recall[mask]))


def _max_fbeta_from_pr(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    beta: float,
    min_threshold: float,
    max_threshold: float,
) -> float:
    if len(np.unique(y_true)) < 2:
        return 0.0

    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    if thresholds.size == 0:
        return 0.0

    precision = precision[:-1]
    recall = recall[:-1]
    b2 = beta * beta
    denom = b2 * precision + recall
    fbeta_vals = np.where(denom > 0, (1.0 + b2) * precision * recall / denom, 0.0)

    mask = (thresholds >= min_threshold) & (thresholds <= max_threshold)
    if mask.any():
        fbeta_vals = fbeta_vals[mask]

    if fbeta_vals.size == 0:
        return 0.0

    return float(np.nanmax(fbeta_vals))


def _tune_threshold(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    group_col: str,
    spec: ModelSpec,
    default_threshold: float,
    val_size: float,
    min_threshold: float,
    max_threshold: float,
    random_state: int,
    beta: float,
    threshold_metric: str,
    min_precision: float,
    use_target_encoding: bool = False,
    target_encoding_specs: List[Tuple] | None = None,
    drop_target_encoded_cols: bool = False,
    use_balanced_subsample: bool = False,
    neg_pos_ratio: float = 4.0,
    subsample_random_state: int = 42,
    calibrate_proba: bool = False,
    calibration_method: str = "sigmoid",
    calibration_val_size: float = 0.2,
    calibration_random_state: int = 42,
    ensemble_seeds: List[int] | None = None,
) -> float:
    y = df[target_col].astype(int).values
    groups = df[group_col].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    try:
        tr_idx, val_idx = next(splitter.split(df, y, groups))
    except ValueError:
        return default_threshold

    if len(tr_idx) == 0 or len(val_idx) == 0:
        return default_threshold

    dtr = df.iloc[tr_idx].copy()
    dval = df.iloc[val_idx].copy()
    if dval[target_col].astype(int).nunique() < 2:
        return default_threshold

    features_te = list(features)
    dtr_te = dtr
    dval_te = dval
    if use_target_encoding:
        dtr_te, dval_te, features_te = _apply_target_encoding(
            dtr, dval, target_col, features, target_encoding_specs or [], drop_target_encoded_cols
        )

    dtr_train = dtr_te
    if use_balanced_subsample:
        dtr_train = _balanced_subsample(
            dtr_te, target_col, neg_pos_ratio, subsample_random_state
        )

    dtr_fit = dtr_train
    dtr_cal = None
    if calibrate_proba:
        tr_idx, cal_idx = _split_calibration_indices(
            dtr_train,
            target_col=target_col,
            group_col=group_col,
            val_size=calibration_val_size,
            random_state=calibration_random_state,
        )
        if tr_idx is not None and cal_idx is not None:
            dtr_fit = dtr_train.iloc[tr_idx].copy()
            dtr_cal = dtr_train.iloc[cal_idx].copy()

    proba = _predict_proba_ensemble(
        dtr_fit,
        dtr_cal,
        dval_te,
        features_te,
        target_col,
        spec,
        calibrate_proba,
        calibration_method,
        ensemble_seeds,
    )

    if threshold_metric == "fbeta":
        return _best_threshold_from_pr(
            dval[target_col].astype(int).values,
            proba,
            default_threshold=default_threshold,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            beta=beta,
        )
    if threshold_metric == "f1":
        return _best_threshold_from_pr(
            dval[target_col].astype(int).values,
            proba,
            default_threshold=default_threshold,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            beta=1.0,
        )
    if threshold_metric == "recall_at_precision":
        return _best_threshold_recall_at_precision(
            dval[target_col].astype(int).values,
            proba,
            default_threshold=default_threshold,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            min_precision=min_precision,
        )
    raise ValueError(f"Unsupported threshold_metric: {threshold_metric}")


def _select_best_params(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    group_col: str,
    base_spec: ModelSpec,
    baseline_spec: ModelSpec | None,
    param_candidates: List[Dict[str, Any]],
    use_scale_pos_weight: bool,
    use_target_encoding: bool,
    target_encoding_specs: List[Tuple] | None,
    drop_target_encoded_cols: bool,
    use_balanced_subsample: bool,
    neg_pos_ratio: float,
    subsample_random_state: int,
    tune_metric: str,
    beta: float,
    val_size: float,
    random_state: int,
    threshold_min: float,
    threshold_max: float,
    min_precision: float,
    tune_cv_splits: int,
    delta_pr_auc_weight: float,
    delta_f1_weight: float,
    delta_f2_weight: float,
    delta_recall_weight: float,
    ensemble_seeds: List[int] | None = None,
    baseline_ensemble_seeds: List[int] | None = None,
) -> Dict[str, Any]:
    if not param_candidates:
        return {}

    te_specs = target_encoding_specs or []

    if tune_metric in {"delta_pr_auc_mean_std", "delta_fbeta_mean_std", "delta_mix_mean_std"}:
        if baseline_spec is None:
            raise ValueError("baseline_spec is required for delta tuning.")

        X = df[features]
        y = df[target_col].astype(int).values
        groups = df[group_col].values
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            return {}

        n_splits = min(max(2, tune_cv_splits), n_groups)
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X, y, groups))

        base_scores = []
        fold_data = []
        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            dtr = df.iloc[tr_idx].copy()
            dval = df.iloc[val_idx].copy()
            if use_target_encoding:
                dtr, dval, features_te = _apply_target_encoding(
                    dtr, dval, target_col, features, te_specs, drop_target_encoded_cols
                )
            else:
                features_te = list(features)

            dtr_train = dtr
            if use_balanced_subsample:
                dtr_train = _balanced_subsample(
                    dtr, target_col, neg_pos_ratio, subsample_random_state + fold_idx
                )
            fold_data.append((dtr_train, dval, features_te))
            params = dict(baseline_spec.params)
            if use_scale_pos_weight:
                ytr = dtr_train[target_col].astype(int).values
                params = _apply_scale_pos_weight(params, ytr)
            base_spec_fold = ModelSpec(name=baseline_spec.name, params=params)
            proba = _predict_proba_ensemble(
                dtr_train,
                None,
                dval,
                features_te,
                target_col,
                base_spec_fold,
                False,
                "sigmoid",
                baseline_ensemble_seeds,
            )
            y_val = dval[target_col].astype(int).values
            base_pr_auc = average_precision_score(y_val, proba) if int((y_val == 1).sum()) > 0 else 0.0
            base_fbeta = _max_fbeta_from_pr(
                y_val,
                proba,
                beta=beta,
                min_threshold=threshold_min,
                max_threshold=threshold_max,
            )
            base_scores.append((base_pr_auc, base_fbeta))

        w = float(np.clip(delta_pr_auc_weight, 0.0, 1.0))
        best_params: Dict[str, Any] = {}
        best_score = float("-inf")
        best_tie = float("-inf")

        for cand in param_candidates:
            cand = cand or {}
            fold_scores = []
            for (dtr_train, dval, features_te), (base_pr_auc, base_fbeta) in zip(fold_data, base_scores):
                params = {**base_spec.params, **cand}
                if use_scale_pos_weight:
                    ytr = dtr_train[target_col].astype(int).values
                    params = _apply_scale_pos_weight(params, ytr)
                spec_fold = ModelSpec(name=base_spec.name, params=params)
                proba = _predict_proba_ensemble(
                    dtr_train,
                    None,
                    dval,
                    features_te,
                    target_col,
                    spec_fold,
                    False,
                    "sigmoid",
                    ensemble_seeds,
                )
                y_val = dval[target_col].astype(int).values
                cand_pr_auc = average_precision_score(y_val, proba) if int((y_val == 1).sum()) > 0 else 0.0
                cand_fbeta = _max_fbeta_from_pr(
                    y_val,
                    proba,
                    beta=beta,
                    min_threshold=threshold_min,
                    max_threshold=threshold_max,
                )
                if tune_metric == "delta_pr_auc_mean_std":
                    score = cand_pr_auc - base_pr_auc
                elif tune_metric == "delta_fbeta_mean_std":
                    score = cand_fbeta - base_fbeta
                else:
                    score = w * (cand_pr_auc - base_pr_auc) + (1.0 - w) * (cand_fbeta - base_fbeta)
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
            std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
            score = mean_score - std_score
            if score > best_score or (score == best_score and mean_score > best_tie):
                best_score = score
                best_tie = mean_score
                best_params = cand

        return best_params

    if tune_metric == "delta_pr_auc_f2_guard":
        if baseline_spec is None:
            raise ValueError("baseline_spec is required for delta tuning.")

        X = df[features]
        y = df[target_col].astype(int).values
        groups = df[group_col].values
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            return {}

        n_splits = min(max(2, tune_cv_splits), n_groups)
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X, y, groups))

        base_scores = []
        fold_data = []
        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            dtr = df.iloc[tr_idx].copy()
            dval = df.iloc[val_idx].copy()
            base_features = list(features)
            base_dtr = dtr
            base_dval = dval
            params = dict(baseline_spec.params)
            if use_scale_pos_weight:
                ytr = base_dtr[target_col].astype(int).values
                params = _apply_scale_pos_weight(params, ytr)
            base_spec_fold = ModelSpec(name=baseline_spec.name, params=params)
            proba = _predict_proba_ensemble(
                base_dtr,
                None,
                base_dval,
                base_features,
                target_col,
                base_spec_fold,
                False,
                "sigmoid",
                baseline_ensemble_seeds,
            )
            y_val = base_dval[target_col].astype(int).values
            base_pr_auc = average_precision_score(y_val, proba) if int((y_val == 1).sum()) > 0 else 0.0
            base_f2 = _max_fbeta_from_pr(
                y_val,
                proba,
                beta=2.0,
                min_threshold=threshold_min,
                max_threshold=threshold_max,
            )
            base_f1 = _max_fbeta_from_pr(
                y_val,
                proba,
                beta=1.0,
                min_threshold=threshold_min,
                max_threshold=threshold_max,
            )
            base_scores.append((base_pr_auc, base_f2, base_f1))

            cand_dtr = dtr
            cand_dval = dval
            if use_target_encoding:
                cand_dtr, cand_dval, features_te = _apply_target_encoding(
                    dtr, dval, target_col, features, te_specs, drop_target_encoded_cols
                )
            else:
                features_te = list(features)

            cand_train = cand_dtr
            if use_balanced_subsample:
                cand_train = _balanced_subsample(
                    cand_dtr, target_col, neg_pos_ratio, subsample_random_state + fold_idx
                )
            fold_data.append((cand_train, cand_dval, features_te))

        w = float(np.clip(delta_pr_auc_weight, 0.0, 1.0))
        best_params: Dict[str, Any] = {}
        best_score = float("-inf")
        best_tie = float("-inf")
        found_candidate = False

        for cand in param_candidates:
            cand = cand or {}
            fold_scores = []
            fold_f1_deltas = []
            fold_pr_deltas = []
            fold_f2_deltas = []
            for (dtr_train, dval, features_te), (base_pr_auc, base_f2, base_f1) in zip(fold_data, base_scores):
                params = {**base_spec.params, **cand}
                if use_scale_pos_weight:
                    ytr = dtr_train[target_col].astype(int).values
                    params = _apply_scale_pos_weight(params, ytr)

                spec_fold = ModelSpec(name=base_spec.name, params=params)
                proba = _predict_proba_ensemble(
                    dtr_train,
                    None,
                    dval,
                    features_te,
                    target_col,
                    spec_fold,
                    False,
                    "sigmoid",
                    ensemble_seeds,
                )
                y_val = dval[target_col].astype(int).values
                cand_pr_auc = average_precision_score(y_val, proba) if int((y_val == 1).sum()) > 0 else 0.0
                cand_f2 = _max_fbeta_from_pr(
                    y_val,
                    proba,
                    beta=2.0,
                    min_threshold=threshold_min,
                    max_threshold=threshold_max,
                )
                cand_f1 = _max_fbeta_from_pr(
                    y_val,
                    proba,
                    beta=1.0,
                    min_threshold=threshold_min,
                    max_threshold=threshold_max,
                )
                delta_pr = cand_pr_auc - base_pr_auc
                delta_f2 = cand_f2 - base_f2
                delta_f1 = cand_f1 - base_f1
                fold_scores.append(w * delta_pr + (1.0 - w) * delta_f2)
                fold_f1_deltas.append(delta_f1)
                fold_pr_deltas.append(delta_pr)
                fold_f2_deltas.append(delta_f2)

            mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
            std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
            score = mean_score - std_score
            mean_f1_delta = float(np.mean(fold_f1_deltas)) if fold_f1_deltas else 0.0
            mean_pr_delta = float(np.mean(fold_pr_deltas)) if fold_pr_deltas else 0.0
            mean_f2_delta = float(np.mean(fold_f2_deltas)) if fold_f2_deltas else 0.0
            if mean_f1_delta < -0.001 or mean_pr_delta < -0.0005 or mean_f2_delta < -0.0005:
                continue

            found_candidate = True
            if score > best_score or (score == best_score and mean_score > best_tie):
                best_score = score
                best_tie = mean_score
                best_params = cand

        return best_params if found_candidate else {}

    if tune_metric == "delta_pr_auc_f1_guard":
        if baseline_spec is None:
            raise ValueError("baseline_spec is required for delta tuning.")

        X = df[features]
        y = df[target_col].astype(int).values
        groups = df[group_col].values
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            return {}

        n_splits = min(max(2, tune_cv_splits), n_groups)
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X, y, groups))

        base_scores = []
        fold_data = []
        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            dtr = df.iloc[tr_idx].copy()
            dval = df.iloc[val_idx].copy()
            base_features = list(features)
            base_dtr = dtr
            base_dval = dval
            params = dict(baseline_spec.params)
            if use_scale_pos_weight:
                ytr = base_dtr[target_col].astype(int).values
                params = _apply_scale_pos_weight(params, ytr)
            base_spec_fold = ModelSpec(name=baseline_spec.name, params=params)
            proba = _predict_proba_ensemble(
                base_dtr,
                None,
                base_dval,
                base_features,
                target_col,
                base_spec_fold,
                False,
                "sigmoid",
                baseline_ensemble_seeds,
            )
            y_val = base_dval[target_col].astype(int).values
            base_pr_auc = average_precision_score(y_val, proba) if int((y_val == 1).sum()) > 0 else 0.0
            base_f1 = _max_fbeta_from_pr(
                y_val,
                proba,
                beta=1.0,
                min_threshold=threshold_min,
                max_threshold=threshold_max,
            )
            base_scores.append((base_pr_auc, base_f1))

            cand_dtr = dtr
            cand_dval = dval
            if use_target_encoding:
                cand_dtr, cand_dval, features_te = _apply_target_encoding(
                    dtr, dval, target_col, features, te_specs, drop_target_encoded_cols
                )
            else:
                features_te = list(features)

            cand_train = cand_dtr
            if use_balanced_subsample:
                cand_train = _balanced_subsample(
                    cand_dtr, target_col, neg_pos_ratio, subsample_random_state + fold_idx
                )
            fold_data.append((cand_train, cand_dval, features_te))

        w = float(np.clip(delta_pr_auc_weight, 0.0, 1.0))
        best_params: Dict[str, Any] = {}
        best_score = float("-inf")
        best_tie = float("-inf")
        found_candidate = False

        for cand in param_candidates:
            cand = cand or {}
            fold_scores = []
            fold_f1_deltas = []
            fold_pr_deltas = []
            for (dtr_train, dval, features_te), (base_pr_auc, base_f1) in zip(fold_data, base_scores):
                params = {**base_spec.params, **cand}
                if use_scale_pos_weight:
                    ytr = dtr_train[target_col].astype(int).values
                    params = _apply_scale_pos_weight(params, ytr)

                spec_fold = ModelSpec(name=base_spec.name, params=params)
                proba = _predict_proba_ensemble(
                    dtr_train,
                    None,
                    dval,
                    features_te,
                    target_col,
                    spec_fold,
                    False,
                    "sigmoid",
                    ensemble_seeds,
                )
                y_val = dval[target_col].astype(int).values
                cand_pr_auc = average_precision_score(y_val, proba) if int((y_val == 1).sum()) > 0 else 0.0
                cand_f1 = _max_fbeta_from_pr(
                    y_val,
                    proba,
                    beta=1.0,
                    min_threshold=threshold_min,
                    max_threshold=threshold_max,
                )
                delta_pr = cand_pr_auc - base_pr_auc
                delta_f1 = cand_f1 - base_f1
                fold_scores.append(w * delta_pr + (1.0 - w) * delta_f1)
                fold_f1_deltas.append(delta_f1)
                fold_pr_deltas.append(delta_pr)

            mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
            std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
            score = mean_score - std_score
            mean_f1_delta = float(np.mean(fold_f1_deltas)) if fold_f1_deltas else 0.0
            mean_pr_delta = float(np.mean(fold_pr_deltas)) if fold_pr_deltas else 0.0
            if mean_f1_delta < -0.001 or mean_pr_delta < -0.0005:
                continue

            found_candidate = True
            if score > best_score or (score == best_score and mean_score > best_tie):
                best_score = score
                best_tie = mean_score
                best_params = cand

        return best_params if found_candidate else {}

    if tune_metric == "delta_mix_all_guard":
        if baseline_spec is None:
            raise ValueError("baseline_spec is required for delta tuning.")

        X = df[features]
        y = df[target_col].astype(int).values
        groups = df[group_col].values
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            return {}

        n_splits = min(max(2, tune_cv_splits), n_groups)
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X, y, groups))

        base_scores = []
        fold_data = []
        for fold_idx, (tr_idx, val_idx) in enumerate(splits):
            dtr = df.iloc[tr_idx].copy()
            dval = df.iloc[val_idx].copy()
            base_features = list(features)
            base_dtr = dtr
            base_dval = dval
            params = dict(baseline_spec.params)
            if use_scale_pos_weight:
                ytr = base_dtr[target_col].astype(int).values
                params = _apply_scale_pos_weight(params, ytr)
            base_spec_fold = ModelSpec(name=baseline_spec.name, params=params)
            proba = _predict_proba_ensemble(
                base_dtr,
                None,
                base_dval,
                base_features,
                target_col,
                base_spec_fold,
                False,
                "sigmoid",
                baseline_ensemble_seeds,
            )
            y_val = base_dval[target_col].astype(int).values
            base_pr_auc = average_precision_score(y_val, proba) if int((y_val == 1).sum()) > 0 else 0.0
            base_f1 = _max_fbeta_from_pr(
                y_val,
                proba,
                beta=1.0,
                min_threshold=threshold_min,
                max_threshold=threshold_max,
            )
            base_f2 = _max_fbeta_from_pr(
                y_val,
                proba,
                beta=2.0,
                min_threshold=threshold_min,
                max_threshold=threshold_max,
            )
            base_recall = _max_recall_at_precision(
                y_val,
                proba,
                min_threshold=threshold_min,
                max_threshold=threshold_max,
                min_precision=min_precision,
            )
            base_scores.append((base_pr_auc, base_f1, base_f2, base_recall))

            cand_dtr = dtr
            cand_dval = dval
            if use_target_encoding:
                cand_dtr, cand_dval, features_te = _apply_target_encoding(
                    dtr, dval, target_col, features, te_specs, drop_target_encoded_cols
                )
            else:
                features_te = list(features)

            cand_train = cand_dtr
            if use_balanced_subsample:
                cand_train = _balanced_subsample(
                    cand_dtr, target_col, neg_pos_ratio, subsample_random_state + fold_idx
                )
            fold_data.append((cand_train, cand_dval, features_te))

        weights = np.array(
            [delta_pr_auc_weight, delta_f1_weight, delta_f2_weight, delta_recall_weight],
            dtype=float,
        )
        if weights.sum() <= 0:
            weights = np.array([0.35, 0.25, 0.25, 0.15], dtype=float)
        weights = weights / weights.sum()

        best_params: Dict[str, Any] = {}
        best_score = float("-inf")
        best_tie = float("-inf")
        best_any_params: Dict[str, Any] = {}
        best_any_score = float("-inf")
        best_any_tie = float("-inf")
        found_guard = False
        tol = -0.0005

        for cand in param_candidates:
            cand = cand or {}
            fold_scores = []
            fold_deltas = []
            for (dtr_train, dval, features_te), (base_pr_auc, base_f1, base_f2, base_rec) in zip(
                fold_data, base_scores
            ):
                params = {**base_spec.params, **cand}
                if use_scale_pos_weight:
                    ytr = dtr_train[target_col].astype(int).values
                    params = _apply_scale_pos_weight(params, ytr)

                spec_fold = ModelSpec(name=base_spec.name, params=params)
                proba = _predict_proba_ensemble(
                    dtr_train,
                    None,
                    dval,
                    features_te,
                    target_col,
                    spec_fold,
                    False,
                    "sigmoid",
                    ensemble_seeds,
                )
                y_val = dval[target_col].astype(int).values
                cand_pr_auc = average_precision_score(y_val, proba) if int((y_val == 1).sum()) > 0 else 0.0
                cand_f1 = _max_fbeta_from_pr(
                    y_val,
                    proba,
                    beta=1.0,
                    min_threshold=threshold_min,
                    max_threshold=threshold_max,
                )
                cand_f2 = _max_fbeta_from_pr(
                    y_val,
                    proba,
                    beta=2.0,
                    min_threshold=threshold_min,
                    max_threshold=threshold_max,
                )
                cand_rec = _max_recall_at_precision(
                    y_val,
                    proba,
                    min_threshold=threshold_min,
                    max_threshold=threshold_max,
                    min_precision=min_precision,
                )
                deltas = np.array(
                    [cand_pr_auc - base_pr_auc, cand_f1 - base_f1, cand_f2 - base_f2, cand_rec - base_rec],
                    dtype=float,
                )
                fold_scores.append(float(np.dot(weights, deltas)))
                fold_deltas.append(deltas)

            mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
            std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
            score = mean_score - std_score
            mean_deltas = np.mean(np.stack(fold_deltas), axis=0) if fold_deltas else np.zeros(4)

            if score > best_any_score or (score == best_any_score and mean_score > best_any_tie):
                best_any_score = score
                best_any_tie = mean_score
                best_any_params = cand

            if (mean_deltas < tol).any():
                continue

            found_guard = True
            if score > best_score or (score == best_score and mean_score > best_tie):
                best_score = score
                best_tie = mean_score
                best_params = cand

        return best_params if found_guard else best_any_params

    if tune_metric == "delta_f1_f2_recall_mean_std":
        if baseline_spec is None:
            raise ValueError("baseline_spec is required for delta tuning.")

        X = df[features]
        y = df[target_col].astype(int).values
        groups = df[group_col].values
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            return {}

        n_splits = min(max(2, tune_cv_splits), n_groups)
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(X, y, groups))

        base_scores = []
        for tr_idx, val_idx in splits:
            dtr = df.iloc[tr_idx].copy()
            dval = df.iloc[val_idx].copy()
            params = dict(baseline_spec.params)
            if use_scale_pos_weight:
                ytr = dtr[target_col].astype(int).values
                params = _apply_scale_pos_weight(params, ytr)
            base_spec_fold = ModelSpec(name=baseline_spec.name, params=params)
            proba = _predict_proba_ensemble(
                dtr,
                None,
                dval,
                features,
                target_col,
                base_spec_fold,
                False,
                "sigmoid",
                baseline_ensemble_seeds,
            )
            y_val = dval[target_col].astype(int).values
            base_f1 = _max_fbeta_from_pr(
                y_val,
                proba,
                beta=1.0,
                min_threshold=threshold_min,
                max_threshold=threshold_max,
            )
            base_f2 = _max_fbeta_from_pr(
                y_val,
                proba,
                beta=2.0,
                min_threshold=threshold_min,
                max_threshold=threshold_max,
            )
            base_recall = _max_recall_at_precision(
                y_val,
                proba,
                min_threshold=threshold_min,
                max_threshold=threshold_max,
                min_precision=min_precision,
            )
            base_scores.append((base_f1, base_f2, base_recall))

        weights = np.array(
            [delta_f1_weight, delta_f2_weight, delta_recall_weight], dtype=float
        )
        if weights.sum() <= 0:
            weights = np.array([0.3, 0.4, 0.3], dtype=float)
        weights = weights / weights.sum()

        best_params: Dict[str, Any] = {}
        best_score = float("-inf")
        best_tie = float("-inf")

        for cand in param_candidates:
            cand = cand or {}
            fold_scores = []
            for (tr_idx, val_idx), (base_f1, base_f2, base_rec) in zip(splits, base_scores):
                dtr = df.iloc[tr_idx].copy()
                dval = df.iloc[val_idx].copy()
                params = {**base_spec.params, **cand}
                if use_scale_pos_weight:
                    ytr = dtr[target_col].astype(int).values
                    params = _apply_scale_pos_weight(params, ytr)

                spec_fold = ModelSpec(name=base_spec.name, params=params)
                proba = _predict_proba_ensemble(
                    dtr,
                    None,
                    dval,
                    features,
                    target_col,
                    spec_fold,
                    False,
                    "sigmoid",
                    ensemble_seeds,
                )
                y_val = dval[target_col].astype(int).values
                cand_f1 = _max_fbeta_from_pr(
                    y_val,
                    proba,
                    beta=1.0,
                    min_threshold=threshold_min,
                    max_threshold=threshold_max,
                )
                cand_f2 = _max_fbeta_from_pr(
                    y_val,
                    proba,
                    beta=2.0,
                    min_threshold=threshold_min,
                    max_threshold=threshold_max,
                )
                cand_rec = _max_recall_at_precision(
                    y_val,
                    proba,
                    min_threshold=threshold_min,
                    max_threshold=threshold_max,
                    min_precision=min_precision,
                )
                deltas = np.array([cand_f1 - base_f1, cand_f2 - base_f2, cand_rec - base_rec])
                fold_scores.append(float(np.dot(weights, deltas)))

            mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
            std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
            score = mean_score - std_score
            if score > best_score or (score == best_score and mean_score > best_tie):
                best_score = score
                best_tie = mean_score
                best_params = cand

        return best_params

    if tune_metric == "pr_auc_mean_std":
        X = df[features]
        y = df[target_col].astype(int).values
        groups = df[group_col].values
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            return {}

        n_splits = min(max(2, tune_cv_splits), n_groups)
        gkf = GroupKFold(n_splits=n_splits)
        best_params: Dict[str, Any] = {}
        best_score = float("-inf")
        best_tie = float("-inf")

        for cand in param_candidates:
            cand = cand or {}
            fold_scores = []
            for tr_idx, val_idx in gkf.split(X, y, groups):
                dtr = df.iloc[tr_idx].copy()
                dval = df.iloc[val_idx].copy()
                params = {**base_spec.params, **cand}
                if use_scale_pos_weight:
                    ytr = dtr[target_col].astype(int).values
                    params = _apply_scale_pos_weight(params, ytr)

                spec_fold = ModelSpec(name=base_spec.name, params=params)
                proba = _predict_proba_ensemble(
                    dtr,
                    None,
                    dval,
                    features,
                    target_col,
                    spec_fold,
                    False,
                    "sigmoid",
                    ensemble_seeds,
                )
                y_val = dval[target_col].astype(int).values
                score = average_precision_score(y_val, proba) if int((y_val == 1).sum()) > 0 else 0.0
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
            std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
            score = mean_score - std_score
            if score > best_score or (score == best_score and mean_score > best_tie):
                best_score = score
                best_tie = mean_score
                best_params = cand

        return best_params

    if tune_metric == "recall_at_precision_mean_std":
        X = df[features]
        y = df[target_col].astype(int).values
        groups = df[group_col].values
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            return {}

        n_splits = min(max(2, tune_cv_splits), n_groups)
        gkf = GroupKFold(n_splits=n_splits)
        best_params: Dict[str, Any] = {}
        best_score = float("-inf")
        best_tie = float("-inf")

        for cand in param_candidates:
            cand = cand or {}
            fold_scores = []
            for tr_idx, val_idx in gkf.split(X, y, groups):
                dtr = df.iloc[tr_idx].copy()
                dval = df.iloc[val_idx].copy()
                params = {**base_spec.params, **cand}
                if use_scale_pos_weight:
                    ytr = dtr[target_col].astype(int).values
                    params = _apply_scale_pos_weight(params, ytr)

                spec_fold = ModelSpec(name=base_spec.name, params=params)
                proba = _predict_proba_ensemble(
                    dtr,
                    None,
                    dval,
                    features,
                    target_col,
                    spec_fold,
                    False,
                    "sigmoid",
                    ensemble_seeds,
                )
                y_val = dval[target_col].astype(int).values
                score = _max_recall_at_precision(
                    y_val,
                    proba,
                    min_threshold=threshold_min,
                    max_threshold=threshold_max,
                    min_precision=min_precision,
                )
                fold_scores.append(score)

            mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
            std_score = float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0
            score = mean_score - std_score
            if score > best_score or (score == best_score and mean_score > best_tie):
                best_score = score
                best_tie = mean_score
                best_params = cand

        return best_params

    y = df[target_col].astype(int).values
    groups = df[group_col].values

    splitter = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
    try:
        tr_idx, val_idx = next(splitter.split(df, y, groups))
    except ValueError:
        return {}

    if len(tr_idx) == 0 or len(val_idx) == 0:
        return {}

    dtr = df.iloc[tr_idx].copy()
    dval = df.iloc[val_idx].copy()
    y_val = dval[target_col].astype(int).values

    best_params: Dict[str, Any] = {}
    best_score = float("-inf")
    best_tie = float("-inf")

    for cand in param_candidates:
        cand = cand or {}
        params = {**base_spec.params, **cand}
        if use_scale_pos_weight:
            ytr = dtr[target_col].astype(int).values
            params = _apply_scale_pos_weight(params, ytr)

        spec_fold = ModelSpec(name=base_spec.name, params=params)
        proba = _predict_proba_ensemble(
            dtr,
            None,
            dval,
            features,
            target_col,
            spec_fold,
            False,
            "sigmoid",
            ensemble_seeds,
        )

        if tune_metric == "pr_auc":
            score = average_precision_score(y_val, proba) if int((y_val == 1).sum()) > 0 else 0.0
        elif tune_metric == "fbeta":
            score = _max_fbeta_from_pr(
                y_val,
                proba,
                beta=beta,
                min_threshold=threshold_min,
                max_threshold=threshold_max,
            )
        elif tune_metric == "recall_at_precision":
            score = _max_recall_at_precision(
                y_val,
                proba,
                min_threshold=threshold_min,
                max_threshold=threshold_max,
                min_precision=min_precision,
            )
        else:
            raise ValueError(f"Unsupported tune_metric: {tune_metric}")

        tie = average_precision_score(y_val, proba) if int((y_val == 1).sum()) > 0 else 0.0
        if score > best_score or (score == best_score and tie > best_tie):
            best_score = score
            best_tie = tie
            best_params = cand

    return best_params


def _eval_groupkfold(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    group_col: str,
    spec: ModelSpec,
    baseline_spec: ModelSpec | None = None,
    n_splits: int = 5,
    threshold: float = 0.5,
    use_scale_pos_weight: bool = False,
    use_feature_selection: bool = False,
    use_target_encoding: bool = False,
    target_encoding_specs: List[Tuple] | None = None,
    drop_target_encoded_cols: bool = False,
    use_balanced_subsample: bool = False,
    neg_pos_ratio: float = 4.0,
    subsample_random_state: int = 42,
    fs_top_k: int = 15,
    fs_random_state: int = 42,
    tune_threshold: bool = False,
    threshold_val_size: float = 0.2,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_random_state: int = 42,
    threshold_metric: str = "fbeta",
    min_precision: float = 0.0,
    beta: float = 1.0,
    calibrate_proba: bool = False,
    calibration_method: str = "sigmoid",
    calibration_val_size: float = 0.2,
    calibration_random_state: int = 42,
    tune_params: bool = False,
    param_candidates: List[Dict[str, Any]] | None = None,
    tune_metric: str = "fbeta",
    tune_val_size: float = 0.2,
    tune_random_state: int = 42,
    tune_cv_splits: int = 3,
    delta_pr_auc_weight: float = 0.5,
    delta_f1_weight: float = 0.0,
    delta_f2_weight: float = 0.0,
    delta_recall_weight: float = 0.0,
    ensemble_seeds: List[int] | None = None,
) -> Dict[str, Any]:
    X = df[features]
    y = df[target_col].astype(int).values
    groups = df[group_col].values

    gkf = GroupKFold(n_splits=n_splits)

    folds: List[Dict[str, Any]] = []
    for k, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        dtr = df.iloc[tr].copy()
        dte = df.iloc[te].copy()

        tuned_params: Dict[str, Any] = {}
        spec_fold = spec
        if tune_params:
            tuned_params = _select_best_params(
                df=dtr,
                features=features,
                target_col=target_col,
                group_col=group_col,
                base_spec=spec,
                baseline_spec=baseline_spec,
                param_candidates=param_candidates or [],
                use_scale_pos_weight=use_scale_pos_weight,
                use_target_encoding=use_target_encoding,
                target_encoding_specs=target_encoding_specs,
                drop_target_encoded_cols=drop_target_encoded_cols,
                use_balanced_subsample=use_balanced_subsample,
                neg_pos_ratio=neg_pos_ratio,
                subsample_random_state=subsample_random_state,
                tune_metric=tune_metric,
                beta=beta,
                val_size=tune_val_size,
                random_state=tune_random_state,
                threshold_min=threshold_min,
                threshold_max=threshold_max,
                min_precision=min_precision,
                tune_cv_splits=tune_cv_splits,
                delta_pr_auc_weight=delta_pr_auc_weight,
                delta_f1_weight=delta_f1_weight,
                delta_f2_weight=delta_f2_weight,
                delta_recall_weight=delta_recall_weight,
                ensemble_seeds=ensemble_seeds,
            )
            if tuned_params:
                spec_fold = ModelSpec(name=spec.name, params={**spec.params, **tuned_params})

        features_fold = list(features)
        dtr_fold = dtr
        dte_fold = dte
        if use_target_encoding:
            dtr_fold, dte_fold, features_fold = _apply_target_encoding(
                dtr, dte, target_col, features_fold, target_encoding_specs or [], drop_target_encoded_cols
            )

        if use_balanced_subsample:
            dtr_fold = _balanced_subsample(
                dtr_fold, target_col, neg_pos_ratio, subsample_random_state + k
            )

        if use_feature_selection:
            features_fold = _select_features_mi(
                dtr_fold,
                features_fold,
                target_col=target_col,
                top_k=fs_top_k,
                random_state=fs_random_state,
            )

        if use_scale_pos_weight:
            ytr = dtr_fold[target_col].astype(int).values
            spec_fold = ModelSpec(
                name=spec_fold.name,
                params=_apply_scale_pos_weight(spec_fold.params, ytr),
            )

        tuned_threshold = threshold
        if tune_threshold:
            tuned_threshold = _tune_threshold(
                df=dtr,
                features=features_fold,
                target_col=target_col,
                group_col=group_col,
                spec=spec_fold,
                default_threshold=threshold,
                val_size=threshold_val_size,
                min_threshold=threshold_min,
                max_threshold=threshold_max,
                random_state=threshold_random_state,
                beta=beta,
                threshold_metric=threshold_metric,
                min_precision=min_precision,
                use_target_encoding=use_target_encoding,
                target_encoding_specs=target_encoding_specs,
                drop_target_encoded_cols=drop_target_encoded_cols,
                use_balanced_subsample=use_balanced_subsample,
                neg_pos_ratio=neg_pos_ratio,
                subsample_random_state=subsample_random_state + k,
                calibrate_proba=calibrate_proba,
                calibration_method=calibration_method,
                calibration_val_size=calibration_val_size,
                calibration_random_state=calibration_random_state + k,
                ensemble_seeds=ensemble_seeds,
            )

        dtr_fit = dtr_fold
        dtr_cal = None
        if calibrate_proba:
            tr_idx, cal_idx = _split_calibration_indices(
                dtr_fold,
                target_col=target_col,
                group_col=group_col,
                val_size=calibration_val_size,
                random_state=calibration_random_state + k,
            )
            if tr_idx is not None and cal_idx is not None:
                dtr_fit = dtr_fold.iloc[tr_idx].copy()
                dtr_cal = dtr_fold.iloc[cal_idx].copy()

        # probabilities -> threshold (optionally ensembled + calibrated)
        proba = _predict_proba_ensemble(
            dtr_fit,
            dtr_cal,
            dte_fold,
            features_fold,
            target_col,
            spec_fold,
            calibrate_proba,
            calibration_method,
            ensemble_seeds,
        )
        pred = (proba >= tuned_threshold).astype(int)
        yt = dte_fold[target_col].astype(int).values

        f1 = f1_score(yt, pred, zero_division=0)
        prec = precision_score(yt, pred, zero_division=0)
        rec = recall_score(yt, pred, zero_division=0)
        fbeta = _fbeta(float(prec), float(rec), beta)
        pr_auc = average_precision_score(yt, proba) if int((yt == 1).sum()) > 0 else 0.0
        cm = confusion_matrix(yt, pred).tolist()

        folds.append(
            {
                "fold": k,
                "f1": float(f1),
                "fbeta": float(fbeta),
                "precision": float(prec),
                "recall": float(rec),
                "pr_auc": float(pr_auc),
                "support_pos": int((yt == 1).sum()),
                "support_neg": int((yt == 0).sum()),
                "confusion_matrix": cm,
                "threshold": float(tuned_threshold),
                "tuned_params": tuned_params,
                "feature_count": int(len(features_fold)),
            }
        )

    mean_f1 = float(np.mean([f["f1"] for f in folds]))
    std_f1 = float(np.std([f["f1"] for f in folds], ddof=1)) if len(folds) > 1 else 0.0
    mean_fbeta = float(np.mean([f["fbeta"] for f in folds]))
    std_fbeta = float(np.std([f["fbeta"] for f in folds], ddof=1)) if len(folds) > 1 else 0.0
    mean_pr_auc = float(np.mean([f["pr_auc"] for f in folds]))
    std_pr_auc = float(np.std([f["pr_auc"] for f in folds], ddof=1)) if len(folds) > 1 else 0.0
    mean_threshold = float(np.mean([f["threshold"] for f in folds])) if folds else float(threshold)
    std_threshold = float(np.std([f["threshold"] for f in folds], ddof=1)) if len(folds) > 1 else 0.0

    return {
        "model_kind": spec.name,
        "n_splits": n_splits,
        "threshold": float(threshold),
        "threshold_tuned": bool(tune_threshold),
        "threshold_val_size": float(threshold_val_size),
        "threshold_min": float(threshold_min),
        "threshold_max": float(threshold_max),
        "threshold_metric": threshold_metric,
        "min_precision": float(min_precision),
        "mean_threshold": mean_threshold,
        "std_threshold": std_threshold,
        "beta": float(beta),
        "feature_selection": bool(use_feature_selection),
        "feature_selection_top_k": int(fs_top_k),
        "target_encoding": bool(use_target_encoding),
        "drop_target_encoded_cols": bool(drop_target_encoded_cols),
        "balanced_subsample": bool(use_balanced_subsample),
        "neg_pos_ratio": float(neg_pos_ratio),
        "calibrate_proba": bool(calibrate_proba),
        "calibration_method": calibration_method,
        "calibration_val_size": float(calibration_val_size),
        "ensemble_size": int(len(ensemble_seeds)) if ensemble_seeds else 1,
        "tune_params": bool(tune_params),
        "tune_metric": tune_metric,
        "tune_val_size": float(tune_val_size),
        "tune_random_state": int(tune_random_state),
        "tune_cv_splits": int(tune_cv_splits),
        "delta_pr_auc_weight": float(delta_pr_auc_weight),
        "delta_f1_weight": float(delta_f1_weight),
        "delta_f2_weight": float(delta_f2_weight),
        "delta_recall_weight": float(delta_recall_weight),
        "param_candidates": int(len(param_candidates or [])),
        "mean_f1": mean_f1,
        "std_f1": std_f1,
        "mean_fbeta": mean_fbeta,
        "std_fbeta": std_fbeta,
        "mean_pr_auc": mean_pr_auc,
        "std_pr_auc": std_pr_auc,
        "folds": folds,
    }


def ref_tech_spec() -> ModelSpec:
    # Reference XGBoost replication (from the paper's reported best params).
    return ModelSpec(
        name="reftech",
        params=dict(
            n_estimators=295,
            max_depth=6,
            learning_rate=0.225,
            subsample=0.9236,
            colsample_bytree=0.7744,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        ),
    )


def my_method_spec() -> ModelSpec:
    # Your enhanced XGB: slightly more regularization + class imbalance weight
    # IMPORTANT: scale_pos_weight should be computed per dataset; we do that in exp_run_all.
    return ModelSpec(
        name="mymethod",
        params=dict(
            n_estimators=600,
            max_depth=5,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            reg_alpha=0.5,
            min_child_weight=5,
            gamma=0.0,
            random_state=42,
            n_jobs=-1,
            eval_metric="logloss",
        ),
    )


def run_stage_strict(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    group_col: str,
    spec: ModelSpec,
    baseline_spec: ModelSpec | None = None,
    n_splits: int = 5,
    threshold: float = 0.5,
    use_scale_pos_weight: bool = False,
    use_feature_selection: bool = False,
    use_target_encoding: bool = False,
    target_encoding_specs: List[Tuple] | None = None,
    drop_target_encoded_cols: bool = False,
    use_balanced_subsample: bool = False,
    neg_pos_ratio: float = 4.0,
    subsample_random_state: int = 42,
    fs_top_k: int = 15,
    fs_random_state: int = 42,
    tune_threshold: bool = False,
    threshold_val_size: float = 0.2,
    threshold_min: float = 0.05,
    threshold_max: float = 0.95,
    threshold_random_state: int = 42,
    threshold_metric: str = "fbeta",
    min_precision: float = 0.0,
    beta: float = 1.0,
    calibrate_proba: bool = False,
    calibration_method: str = "sigmoid",
    calibration_val_size: float = 0.2,
    calibration_random_state: int = 42,
    ensemble_seeds: List[int] | None = None,
    tune_params: bool = False,
    param_candidates: List[Dict[str, Any]] | None = None,
    tune_metric: str = "fbeta",
    tune_val_size: float = 0.2,
    tune_random_state: int = 42,
    tune_cv_splits: int = 3,
    delta_pr_auc_weight: float = 0.5,
    delta_f1_weight: float = 0.0,
    delta_f2_weight: float = 0.0,
    delta_recall_weight: float = 0.0,
) -> Dict[str, Any]:
    return _eval_groupkfold(
        df=df,
        features=features,
        target_col=target_col,
        group_col=group_col,
        spec=spec,
        baseline_spec=baseline_spec,
        n_splits=n_splits,
        threshold=threshold,
        use_scale_pos_weight=use_scale_pos_weight,
        use_feature_selection=use_feature_selection,
        use_target_encoding=use_target_encoding,
        target_encoding_specs=target_encoding_specs,
        drop_target_encoded_cols=drop_target_encoded_cols,
        use_balanced_subsample=use_balanced_subsample,
        neg_pos_ratio=neg_pos_ratio,
        subsample_random_state=subsample_random_state,
        fs_top_k=fs_top_k,
        fs_random_state=fs_random_state,
        tune_threshold=tune_threshold,
        threshold_val_size=threshold_val_size,
        threshold_min=threshold_min,
        threshold_max=threshold_max,
        threshold_random_state=threshold_random_state,
        threshold_metric=threshold_metric,
        min_precision=min_precision,
        beta=beta,
        calibrate_proba=calibrate_proba,
        calibration_method=calibration_method,
        calibration_val_size=calibration_val_size,
        calibration_random_state=calibration_random_state,
        ensemble_seeds=ensemble_seeds,
        tune_params=tune_params,
        param_candidates=param_candidates,
        tune_metric=tune_metric,
        tune_val_size=tune_val_size,
        tune_random_state=tune_random_state,
        tune_cv_splits=tune_cv_splits,
        delta_pr_auc_weight=delta_pr_auc_weight,
        delta_f1_weight=delta_f1_weight,
        delta_f2_weight=delta_f2_weight,
        delta_recall_weight=delta_recall_weight,
    )
