from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

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

    pipe = _make_pipeline(dtr, features, spec)
    pipe.fit(dtr[features], dtr[target_col].astype(int).values)
    proba = pipe.predict_proba(dval[features])[:, 1]

    if threshold_metric == "fbeta":
        return _best_threshold_from_pr(
            dval[target_col].astype(int).values,
            proba,
            default_threshold=default_threshold,
            min_threshold=min_threshold,
            max_threshold=max_threshold,
            beta=beta,
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
) -> Dict[str, Any]:
    if not param_candidates:
        return {}

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
        for tr_idx, val_idx in splits:
            dtr = df.iloc[tr_idx].copy()
            dval = df.iloc[val_idx].copy()
            params = dict(baseline_spec.params)
            if use_scale_pos_weight:
                ytr = dtr[target_col].astype(int).values
                params = _apply_scale_pos_weight(params, ytr)
            base_pipe = _make_pipeline(dtr, features, ModelSpec(name=baseline_spec.name, params=params))
            base_pipe.fit(dtr[features], dtr[target_col].astype(int).values)
            proba = base_pipe.predict_proba(dval[features])[:, 1]
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
            for (tr_idx, val_idx), (base_pr_auc, base_fbeta) in zip(splits, base_scores):
                dtr = df.iloc[tr_idx].copy()
                dval = df.iloc[val_idx].copy()
                params = {**base_spec.params, **cand}
                if use_scale_pos_weight:
                    ytr = dtr[target_col].astype(int).values
                    params = _apply_scale_pos_weight(params, ytr)

                pipe = _make_pipeline(dtr, features, ModelSpec(name=base_spec.name, params=params))
                pipe.fit(dtr[features], dtr[target_col].astype(int).values)
                proba = pipe.predict_proba(dval[features])[:, 1]
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
            base_pipe = _make_pipeline(dtr, features, ModelSpec(name=baseline_spec.name, params=params))
            base_pipe.fit(dtr[features], dtr[target_col].astype(int).values)
            proba = base_pipe.predict_proba(dval[features])[:, 1]
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

                pipe = _make_pipeline(dtr, features, ModelSpec(name=base_spec.name, params=params))
                pipe.fit(dtr[features], dtr[target_col].astype(int).values)
                proba = pipe.predict_proba(dval[features])[:, 1]
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

                pipe = _make_pipeline(dtr, features, ModelSpec(name=base_spec.name, params=params))
                pipe.fit(dtr[features], dtr[target_col].astype(int).values)
                proba = pipe.predict_proba(dval[features])[:, 1]
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

                pipe = _make_pipeline(dtr, features, ModelSpec(name=base_spec.name, params=params))
                pipe.fit(dtr[features], dtr[target_col].astype(int).values)
                proba = pipe.predict_proba(dval[features])[:, 1]
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

        pipe = _make_pipeline(dtr, features, ModelSpec(name=base_spec.name, params=params))
        pipe.fit(dtr[features], dtr[target_col].astype(int).values)
        proba = pipe.predict_proba(dval[features])[:, 1]

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
            )
            if tuned_params:
                spec_fold = ModelSpec(name=spec.name, params={**spec.params, **tuned_params})

        if use_scale_pos_weight:
            ytr = dtr[target_col].astype(int).values
            spec_fold = ModelSpec(
                name=spec_fold.name,
                params=_apply_scale_pos_weight(spec_fold.params, ytr),
            )

        features_fold = features
        if use_feature_selection:
            features_fold = _select_features_mi(
                dtr,
                features,
                target_col=target_col,
                top_k=fs_top_k,
                random_state=fs_random_state,
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
            )

        pipe = _make_pipeline(dtr, features_fold, spec_fold)
        pipe.fit(dtr[features_fold], dtr[target_col].astype(int).values)

        # probabilities -> threshold
        proba = pipe.predict_proba(dte[features_fold])[:, 1]
        pred = (proba >= tuned_threshold).astype(int)
        yt = dte[target_col].astype(int).values

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
    # “Anchor-like” XGB: keep it simpler
    return ModelSpec(
        name="reftech",
        params=dict(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
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
