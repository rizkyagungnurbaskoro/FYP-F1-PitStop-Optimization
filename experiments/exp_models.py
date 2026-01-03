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
    precision_score,
    recall_score,
)
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
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


def _eval_groupkfold(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    group_col: str,
    spec: ModelSpec,
    n_splits: int = 5,
    threshold: float = 0.5,
    use_scale_pos_weight: bool = False,
) -> Dict[str, Any]:
    X = df[features]
    y = df[target_col].astype(int).values
    groups = df[group_col].values

    gkf = GroupKFold(n_splits=n_splits)

    folds: List[Dict[str, Any]] = []
    for k, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
        dtr = df.iloc[tr].copy()
        dte = df.iloc[te].copy()

        spec_fold = spec
        if use_scale_pos_weight:
            ytr = dtr[target_col].astype(int).values
            pos = int((ytr == 1).sum())
            neg = int((ytr == 0).sum())
            spw = float(neg / pos) if pos > 0 else 1.0
            spec_fold = ModelSpec(name=spec.name, params={**spec.params, "scale_pos_weight": spw})

        pipe = _make_pipeline(dtr, features, spec_fold)
        pipe.fit(dtr[features], dtr[target_col].astype(int).values)

        # probabilities -> threshold
        proba = pipe.predict_proba(dte[features])[:, 1]
        pred = (proba >= threshold).astype(int)
        yt = dte[target_col].astype(int).values

        f1 = f1_score(yt, pred, zero_division=0)
        prec = precision_score(yt, pred, zero_division=0)
        rec = recall_score(yt, pred, zero_division=0)
        pr_auc = average_precision_score(yt, proba) if int((yt == 1).sum()) > 0 else 0.0
        cm = confusion_matrix(yt, pred).tolist()

        folds.append(
            {
                "fold": k,
                "f1": float(f1),
                "precision": float(prec),
                "recall": float(rec),
                "pr_auc": float(pr_auc),
                "support_pos": int((yt == 1).sum()),
                "support_neg": int((yt == 0).sum()),
                "confusion_matrix": cm,
            }
        )

    mean_f1 = float(np.mean([f["f1"] for f in folds]))
    std_f1 = float(np.std([f["f1"] for f in folds], ddof=1)) if len(folds) > 1 else 0.0
    mean_pr_auc = float(np.mean([f["pr_auc"] for f in folds]))
    std_pr_auc = float(np.std([f["pr_auc"] for f in folds], ddof=1)) if len(folds) > 1 else 0.0

    return {
        "model_kind": spec.name,
        "n_splits": n_splits,
        "threshold": float(threshold),
        "mean_f1": mean_f1,
        "std_f1": std_f1,
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
    n_splits: int = 5,
    threshold: float = 0.5,
    use_scale_pos_weight: bool = False,
) -> Dict[str, Any]:
    return _eval_groupkfold(
        df=df,
        features=features,
        target_col=target_col,
        group_col=group_col,
        spec=spec,
        n_splits=n_splits,
        threshold=threshold,
        use_scale_pos_weight=use_scale_pos_weight,
    )
