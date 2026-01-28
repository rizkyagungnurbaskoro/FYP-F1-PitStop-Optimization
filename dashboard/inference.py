from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.exp_utils import add_feature_engineering, build_feature_list, load_csv, safe_numeric

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.compose import ColumnTransformer
    from sklearn.metrics import f1_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from xgboost import XGBClassifier

    _ML_OK = True
except Exception:
    _ML_OK = False

try:
    import joblib
except Exception:
    joblib = None


ROOT = Path(__file__).resolve().parents[1]


@dataclass
class InferenceBundle:
    model: object | None
    features: list[str]
    threshold: float
    strict_ok: bool
    strict_warn: list[str]


@dataclass
class PredictionResult:
    probability: float
    recommendation: str
    missing_features: list[str]
    decision_source: str


def _pick_column(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    lower_map = {c.lower(): c for c in df.columns}
    for c in candidates:
        key = c.lower()
        if key in lower_map:
            return lower_map[key]
    return None


def _infer_target_col(df: pd.DataFrame) -> str:
    if "decide_pitstop" in df.columns:
        return "decide_pitstop"
    raise ValueError("Target column 'decide_pitstop' not found.")


def _infer_group_col(df: pd.DataFrame) -> str | None:
    return _pick_column(df, ["race_id", "RaceID", "race", "event_id"])


def _detect_same_lap_cols(features: list[str]) -> list[str]:
    suspects = {"lapno", "pitstops_so_far", "position", "race_progress", "tireage"}
    return [c for c in features if c in suspects and not c.endswith("_prev")]


def _strict_filter(features: list[str]) -> tuple[list[str], bool, list[str]]:
    prev = [f for f in features if f.endswith("_prev")]
    strict_ok = len(prev) > 0
    warn = _detect_same_lap_cols(features)
    return prev, strict_ok, warn


def _split_features(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    cat_cols = [c for c in features if df[c].dtype == "object"]
    num_cols = [c for c in features if c not in cat_cols]
    return num_cols, cat_cols


def _train_calibrated_model(df: pd.DataFrame, features: list[str], target_col: str) -> object | None:
    if not _ML_OK or df.empty or not features:
        return None
    X = df[features]
    y = df[target_col].astype(int)
    num_cols, cat_cols = _split_features(df, features)
    pre = ColumnTransformer(
        [
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ],
        remainder="drop",
    )
    base = XGBClassifier(
        n_estimators=140,
        max_depth=5,
        learning_rate=0.08,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        random_state=42,
        n_jobs=2,
    )
    pipe = Pipeline([("prep", pre), ("model", base)])
    try:
        cal = CalibratedClassifierCV(pipe, cv=3, method="sigmoid")
        cal.fit(X, y)
        return cal
    except Exception:
        pipe.fit(X, y)
        return pipe


def _select_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    if y_true.size == 0:
        return 0.5
    best_t = 0.5
    best_f1 = -1.0
    for t in np.linspace(0.05, 0.95, 19):
        pred = probs >= t
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_t = float(t)
    return best_t


def load_dataset(path: Path) -> pd.DataFrame:
    df = add_feature_engineering(safe_numeric(load_csv(path)))
    return df


def _try_load_artifact(stage_label: str) -> tuple[object | None, list[str]]:
    if joblib is None:
        return None, []
    stage_num = stage_label.lower().replace("s", "")
    candidates = list((ROOT / "results").rglob(f"*stage{stage_num}*model*.joblib"))
    candidates += list((ROOT / "results").rglob(f"*stage{stage_num}*model*.pkl"))
    if not candidates:
        return None, []
    for path in candidates:
        try:
            model = joblib.load(path)
            features = getattr(model, "feature_names_in_", []) or []
            return model, list(features)
        except Exception:
            continue
    return None, []


def build_bundle(df: pd.DataFrame, strict: bool, stage_label: str | None = None) -> InferenceBundle:
    target_col = _infer_target_col(df)
    group_col = _infer_group_col(df) or "race_id"
    features = build_feature_list(df, target_col, group_col)
    strict_ok = True
    strict_warn: list[str] = []
    if strict:
        features, strict_ok, strict_warn = _strict_filter(features)
    model = None
    if stage_label:
        model, model_features = _try_load_artifact(stage_label)
        if model is not None and model_features:
            features = list(model_features)
    if model is None:
        model = _train_calibrated_model(df, features, target_col)
    threshold = 0.5
    if model is not None:
        try:
            probs = model.predict_proba(df[features])[:, 1]
            threshold = _select_threshold(df[target_col].astype(int).values, probs)
        except Exception:
            threshold = 0.5
    return InferenceBundle(
        model=model,
        features=features,
        threshold=threshold,
        strict_ok=strict_ok,
        strict_warn=strict_warn,
    )


def _align_row(df_row: pd.DataFrame, features: list[str]) -> tuple[pd.DataFrame, list[str]]:
    missing = [c for c in features if c not in df_row.columns]
    aligned = df_row.copy()
    for c in missing:
        aligned[c] = 0.0
    return aligned[features], missing


def predict_row(bundle: InferenceBundle, row: pd.Series) -> PredictionResult:
    if bundle.model is None or not bundle.features:
        return PredictionResult(0.0, "STAY OUT", bundle.features, "model")
    df_row = row.to_frame().T
    aligned, missing = _align_row(df_row, bundle.features)
    try:
        prob = float(bundle.model.predict_proba(aligned)[:, 1][0])
    except Exception:
        prob = 0.0
    recommendation = "STAY OUT"
    if prob >= bundle.threshold:
        recommendation = "PIT"
    elif prob >= bundle.threshold * 0.6:
        recommendation = "STANDBY"
    return PredictionResult(prob, recommendation, missing, "model")


def estimate_pit_loss_seconds(row: pd.Series) -> tuple[float, dict[str, float]]:
    sc = float(row.get("sc_active_prev", row.get("sc_active", 0)) or 0)
    vsc = float(row.get("vsc_active_prev", row.get("vsc_active", 0)) or 0)
    rain = float(row.get("Rainfall_prev", row.get("rain", 0)) or 0)
    track_temp = row.get("TrackTemp_prev", row.get("TrackTemp", row.get("track_temp", None)))
    pit_window = row.get("in_pit_window_prev", row.get("in_pit_window", row.get("pit_window", 0)))
    gap_behind = row.get("gap_to_behind_prev", row.get("gap_behind", row.get("gap", None)))
    race_progress = row.get("race_progress_prev", row.get("race_progress", None))
    lapno = row.get("lapno_prev", row.get("lapno", None))
    nolaps = row.get("nolaps_prev", row.get("nolaps", None))
    wear = row.get("tyre_wear_pct_prev", row.get("tyre_wear_pct", None))

    if sc == 1:
        base = 18.0
    elif vsc == 1:
        base = 13.0
    else:
        base = 21.5

    adjustments: dict[str, float] = {}

    # Rain makes pit delta more uncertain (often slightly higher loss)
    if rain and float(rain) > 0:
        adjustments["rain"] = 1.0

    # Track temperature (cool track slows warm-up -> slightly higher loss)
    if track_temp is not None:
        try:
            temp = float(track_temp)
            if temp < 25:
                adjustments["cool_track"] = 0.6
            elif temp > 38:
                adjustments["hot_track"] = 0.3
        except Exception:
            pass

    # Pit window open -> slightly lower cost
    if pit_window == 1:
        adjustments["pit_window"] = -0.8

    # Traffic proxy
    if gap_behind is not None:
        try:
            gap = float(gap_behind)
            if gap < 1.5:
                adjustments["traffic_close"] = 1.2
            elif gap > 4.0:
                adjustments["traffic_clear"] = -0.4
        except Exception:
            pass

    # Race progress
    prog = None
    if race_progress is not None:
        try:
            prog = float(race_progress)
        except Exception:
            prog = None
    if prog is None and lapno is not None and nolaps is not None:
        try:
            prog = float(lapno) / max(1.0, float(nolaps))
        except Exception:
            prog = None
    if prog is not None:
        if prog < 0.33:
            adjustments["early_race"] = 0.7
        elif prog > 0.8:
            adjustments["late_race"] = 0.9
        else:
            adjustments["mid_race"] = -0.3

    # Tyre wear reduces effective pit loss (pitting becomes more valuable)
    if wear is not None:
        try:
            wear_val = float(wear)
            wear_pct = wear_val / 100.0 if wear_val > 1 else wear_val
            if wear_pct >= 0.7:
                adjustments["high_wear"] = -1.2
            elif wear_pct <= 0.2:
                adjustments["low_wear"] = 0.4
        except Exception:
            pass

    pit_time = row.get("pit_time", None)
    if pit_time is not None:
        try:
            pit_time = float(pit_time)
            adjustments["queue_penalty"] = max(0.0, pit_time - 2.6)
        except Exception:
            pass

    total = base + sum(adjustments.values())
    return total, adjustments
