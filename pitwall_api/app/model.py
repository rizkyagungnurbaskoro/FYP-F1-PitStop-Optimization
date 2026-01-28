from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import math
import sys

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from .data import detect_decision_col, get_col, load_dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from experiments.exp_utils import build_feature_list  # type: ignore
except Exception:  # pragma: no cover
    build_feature_list = None


@dataclass
class ModelResult:
    prob: float
    recommendation: str
    threshold: float
    source: str


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _get_bool(row: pd.Series, candidates: list[str]) -> bool:
    for name in candidates:
        if name in row:
            try:
                return bool(int(float(row[name])))
            except Exception:
                return bool(row[name])
    return False


def _get_float(row: pd.Series, candidates: list[str]) -> float | None:
    for name in candidates:
        if name in row:
            try:
                val = float(row[name])
                if math.isfinite(val):
                    return val
            except Exception:
                continue
    return None


def _get_val(row: pd.Series, candidates: list[str]) -> float | None:
    for key in candidates:
        if key in row and pd.notna(row[key]):
            try:
                return float(row[key])
            except Exception:
                continue
    return None


def detect_lap_col(df: pd.DataFrame) -> str | None:
    for cand in ("lapno", "lapno_prev", "Lap", "lap"):
        if cand in df.columns:
            return cand
    return None


def pit_window_bounds(
    df: pd.DataFrame,
    lap_col: str | None,
    total_laps_hint: int | None = None,
    current_lap: int | None = None,
    compound: str | None = None
) -> tuple[int, int] | None:
    if not lap_col or lap_col not in df.columns or df.empty:
        total_laps = total_laps_hint or 70
        return _compound_fallback(compound, total_laps)

    # STRATEGIC REFINEMENT: Anchor window around the NEXT historical pit stop
    from .data import detect_decision_col
    decision_col = detect_decision_col(df)
    if decision_col and decision_col in df.columns:
        # Filter pits that happen AFTER or AT the current lap (to find the next/current stop)
        start_search = current_lap or 1
        pits = df[
            (pd.to_numeric(df[decision_col], errors="coerce").fillna(0) > 0) &
            (pd.to_numeric(df[lap_col], errors="coerce") >= start_search)
        ]

        if not pits.empty:
            laps = pd.to_numeric(pits[lap_col], errors="coerce").dropna()
            if not laps.empty:
                # Use the closest upcoming pit stop as the anchor
                next_pit = int(laps.min())
                margin = 8
                return int(max(1, next_pit - margin)), int(next_pit + margin)
        else:
            # If we already pitted and there are no more pits, the window is closed/NA
            if current_lap and current_lap > df[lap_col].max() * 0.8:
                return None

    # FALLBACK to compound-aware logic if no pits found or flags are missing
    total_laps = total_laps_hint or (int(df[lap_col].max()) if not df[lap_col].empty else 70)

    # Try to find 'in_pit_window' flags in the dataset as a secondary source
    window_col = None
    for cand in ("in_pit_window_prev", "in_pit_window", "pit_window_prev", "pit_window"):
        if cand in df.columns:
            window_col = cand
            break

    if window_col is not None:
        mask = pd.to_numeric(df[window_col], errors="coerce").fillna(0) > 0
        # Again, only look at windows ahead of us
        if current_lap:
            mask = mask & (pd.to_numeric(df[lap_col], errors="coerce") >= current_lap - 5)

        if mask.any():
            laps = pd.to_numeric(df.loc[mask, lap_col], errors="coerce").dropna()
            if not laps.empty:
                w_start, w_end = int(laps.min()), int(laps.max())
                # Cap overly broad windows
                if (w_end - w_start) > (total_laps * 0.4):
                    mid = (w_start + w_end) // 2
                    half_width = int(total_laps * 0.15)
                    return max(w_start, mid - half_width), min(w_end, mid + half_width)
                return w_start, w_end

    return _compound_fallback(compound, total_laps)


def _compound_fallback(compound: str | None, total_laps: int) -> tuple[int, int]:
    """Provides a realistic fallback pit window based on tire compound."""
    c = str(compound).upper() if compound else "MEDIUM"
    if "SOFT" in c:
        return int(total_laps * 0.15), int(total_laps * 0.28)
    if "HARD" in c:
        return int(total_laps * 0.40), int(total_laps * 0.55)
    # Default to Medium
    return int(total_laps * 0.25), int(total_laps * 0.42)


def detect_crossover_state(row: pd.Series, df_context: pd.DataFrame | None = None, lap_value: int | None = None, lap_col: str | None = None) -> str:
    """Detects if the track is in a crossover state (Drying)."""
    rain_now = _get_bool(row, ["Rainfall_prev", "rain", "Rain"])
    if rain_now:
        return "STABLE" # It's just Wet
        
    # Heuristic 1: Trend based (if we have context)
    if df_context is not None and not df_context.empty and lap_value is not None:
        l_col = lap_col or "lapno_prev"
        if l_col in df_context.columns:
            # Look back at the last 12 laps
            lookback = 12
            history = df_context[
                (pd.to_numeric(df_context[l_col], errors="coerce") < lap_value) &
                (pd.to_numeric(df_context[l_col], errors="coerce") >= lap_value - lookback)
            ]
            if not history.empty:
                # If it was raining in the last 12 laps but not now -> Crossover
                was_raining = any(_get_bool(r, ["Rainfall_prev", "rain", "Rain"]) for _, r in history.iterrows())
                if was_raining:
                    return "CROSSOVER"

    # Heuristic 2: Environmental fallback
    humidity = _get_float(row, ["Humidity_prev"]) or 50.0
    track_temp = _get_float(row, ["TrackTemp_prev"]) or 25.0
    if humidity > 68 and track_temp < 27:
        return "CROSSOVER"
        
    return "STABLE"


def smooth_prob_by_lap(
    df_context: pd.DataFrame,
    lap_col: str | None,
    proba_col: str,
    lap_value: int | None,
    window: int = 3,
) -> float | None:
    if lap_col is None or lap_col not in df_context.columns or proba_col not in df_context.columns:
        return None
    df_laps = df_context[[lap_col, proba_col]].copy()
    df_laps[lap_col] = pd.to_numeric(df_laps[lap_col], errors="coerce")
    df_laps = df_laps.dropna(subset=[lap_col, proba_col])
    if df_laps.empty:
        return None
    df_laps = df_laps.groupby(lap_col, as_index=False)[proba_col].mean().sort_values(lap_col)
    df_laps["p_smooth"] = (
        df_laps[proba_col].rolling(window=window, min_periods=1, center=True).mean()
    )
    if lap_value is not None:
        match = df_laps.loc[df_laps[lap_col] == lap_value, "p_smooth"]
        if not match.empty:
            return float(match.iloc[-1])
    return float(df_laps["p_smooth"].iloc[-1])


def model_proba(df: pd.DataFrame, bundle: dict[str, Any]) -> np.ndarray:
    features = bundle["features"]
    pipe = bundle["pipe"]
    calibrator = bundle["calibrator"]
    probs_raw = pipe.predict_proba(df[features])[:, 1]
    if calibrator is not None:
        try:
            probs = calibrator.predict_proba(probs_raw.reshape(-1, 1))[:, 1]
        except Exception:
            probs = probs_raw
    else:
        probs = probs_raw
    return probs


DEMO_SHARED_FEATURES = [
    "season",
    "lapno",
    "lapno_prev",
    "race_progress",
    "race_progress_prev",
    "pitstops_so_far",
    "pitstops_so_far_prev",
    "position",
    "Position_prev",
    "gap",
    "interval",
    "gap_to_leader_prev",
    "gap_to_front_prev",
    "gap_to_behind_prev",
    "gap_after_pit_vs_behind_prev",
    "undercut_potential_prev",
    "sc_active",
    "sc_active_prev",
    "vsc_active",
    "vsc_active_prev",
    "SCAny",
    "GapOverInterval",
    "tireage",
    "stint_laps_prev",
    "tyre_wear_pct_prev",
    "relative_pace_prev",
    "delta_best_so_far_prev",
    "delta_interval_prev",
    "AirTemp_prev",
    "TrackTemp_prev",
    "Humidity_prev",
    "Pressure_prev",
    "WindSpeed_prev",
    "WindDirection_prev",
    "Rainfall_prev",
]


def _heuristic_prob(row: pd.Series) -> float:
    score = 0.05
    sc = _get_bool(row, ["sc_active", "sc_active_prev", "SC"])
    vsc = _get_bool(row, ["vsc_active", "vsc_active_prev", "VSC"])
    rain = _get_bool(row, ["rain", "Rain", "Rainfall_prev"])
    pit_window = _get_bool(row, ["pit_window", "pit_window_prev", "in_pit_window", "in_pit_window_prev"])
    tire_age = _get_float(row, ["tireage", "stint_laps_prev", "tyre_age", "tyre_age_prev"]) or 0.0
    position = _get_float(row, ["position", "Position_prev", "pos", "grid_pos"]) or 99.0

    if sc:
        score += 0.18
    if vsc:
        score += 0.12
    if rain:
        score += 0.12
    if pit_window:
        score += 0.25

    score += min(0.4, (tire_age / 40.0) * 0.4)
    if position <= 5:
        score += 0.05

    prob = max(0.02, min(0.95, score))
    return float(prob)


def _recommend(prob: float, threshold: float) -> str:
    margin = 0.05
    if prob >= threshold + margin:
        return "BOX BOX"
    if prob >= threshold - margin:
        return "STANDBY"
    return "STAY OUT"


def demo_policy_decision(
    row: pd.Series,
    proba: float,
    threshold: float,
    lap_value: int | None,
    lap_col: str | None,
    tire_max: float,
    lookahead_laps: int = 4,
    decision_margin: float = 0.05,
    window_start: int | None = None,
    window_end: int | None = None,
    df_context: pd.DataFrame | None = None,
) -> dict[str, Any]:
    race_progress = _get_val(row, ["race_progress", "race_progress_prev"])
    if race_progress is None and "nolaps_prev" in row and lap_col and lap_col in row:
        try:
            race_progress = float(row[lap_col]) / float(row["nolaps_prev"])
        except Exception:
            race_progress = None

    pit_window_val = _get_val(row, ["in_pit_window", "in_pit_window_prev", "pit_window", "pit_window_prev"])
    pit_window_text = "OPEN" if pit_window_val is not None and pit_window_val > 0 else "CLOSED"
    if pit_window_val is None:
        if window_start is not None and window_end is not None and lap_value is not None:
            pit_window_text = "OPEN" if window_start <= lap_value <= window_end else "CLOSED"
        else:
            pit_window_text = "N/A"

    sc_flag = _get_val(row, ["sc_active", "sc_active_prev"])
    vsc_flag = _get_val(row, ["vsc_active", "vsc_active_prev"])
    sc_text = "SC" if sc_flag and sc_flag > 0 else "VSC" if vsc_flag and vsc_flag > 0 else "CLEAR"

    weather_val = _get_val(row, ["Rainfall_prev", "rain_flag"])
    is_wet = weather_val and weather_val > 0

    # Weather Status for UI Labeling
    if is_wet:
        weather_status = "Wet"
    elif detect_crossover_state(row, df_context=df_context, lap_value=lap_value, lap_col=lap_col) == "CROSSOVER":
        weather_status = "Crossover"
    else:
        weather_status = "Dry"

    tire_age = _get_val(row, ["tireage", "tireage_prev", "stint_laps_prev"])
    tire_wear_pct = None
    if "tyre_wear_pct_prev" in row and pd.notna(row["tyre_wear_pct_prev"]):
        val = float(row["tyre_wear_pct_prev"])
        tire_wear_pct = val / 100.0 if val > 1.0 else val
    elif "tyre_wear_pct" in row and pd.notna(row["tyre_wear_pct"]):
        val = float(row["tyre_wear_pct"])
        tire_wear_pct = val / 100.0 if val > 1.0 else val
    elif tire_age is not None:
        tire_wear_pct = min(1.0, float(tire_age) / float(max(1.0, tire_max)))
    if tire_wear_pct is not None and tire_age is not None and tire_age <= 2:
        tire_wear_pct = min(float(tire_wear_pct), 0.1)

    gap_val = _get_val(row, ["gap", "gap_to_leader_prev", "interval", "gap_to_front_prev"])
    gap_delta = _get_val(row, ["delta_interval_prev", "delta_best_so_far_prev", "relative_pace_prev"])
    gap_trend_text = "STEADY"
    if gap_delta is not None:
        if gap_delta < -0.1:
            gap_trend_text = "GAIN"
        elif gap_delta > 0.1:
            gap_trend_text = "LOSS"
    elif gap_val is not None:
        if gap_val <= 1.0:
            gap_trend_text = "ATTACK"
        elif gap_val >= 3.0:
            gap_trend_text = "SAFE"

    decision_reasons: list[str] = []
    high_wear = tire_wear_pct is not None and tire_wear_pct >= 0.65
    urgent_wear = tire_wear_pct is not None and tire_wear_pct >= 0.8
    critical_wear = tire_wear_pct is not None and tire_wear_pct >= 0.9
    window_soon = (
        window_start is not None
        and lap_value is not None
        and lap_value < window_start
        and (window_start - lap_value) <= 2
    )
    if pit_window_text == "OPEN":
        decision_reasons.append("WINDOW")
    if sc_text in ("SC", "VSC"):
        decision_reasons.append(sc_text)
    if tire_wear_pct is not None and tire_wear_pct >= 0.7:
        decision_reasons.append("WEAR")
    if urgent_wear:
        decision_reasons.append("WEAR-URGENT")
    if critical_wear:
        decision_reasons.append("WEAR-CRIT")
    if window_soon:
        decision_reasons.append("WINDOW-SOON")
    if race_progress is not None and race_progress >= 0.75:
        decision_reasons.append("LATE")
    if not decision_reasons:
        decision_reasons.append("CORE")

    lock_decision = False
    used_threshold = float(threshold)
    if pit_window_text == "OPEN":
        used_threshold = max(0.05, used_threshold - 0.08)
    if sc_text in ("SC", "VSC"):
        used_threshold = max(0.05, used_threshold - 0.12)
    if tire_wear_pct is not None and tire_wear_pct >= 0.7:
        used_threshold = max(0.05, used_threshold - 0.06)
    if urgent_wear:
        used_threshold = max(0.05, used_threshold - 0.10)
    if high_wear:
        used_threshold = max(0.05, used_threshold - 0.04)
    if race_progress is not None and race_progress >= 0.75:
        used_threshold = max(0.05, used_threshold - 0.03)
    precision_floor = max(0.12, threshold - 0.03)
    used_threshold = max(used_threshold, precision_floor)

    margin = float(np.clip(decision_margin, 0.02, 0.12))
    if pit_window_text == "OPEN":
        if critical_wear:
            decision = "BOX BOX"
            lock_decision = True
            decision_reasons.append("LOW-POS")
        elif urgent_wear:
            decision = "BOX BOX"
            decision_reasons.append("LOW-POS")
        elif high_wear and proba >= used_threshold - margin:
            decision = "BOX BOX"
        elif proba >= used_threshold + margin:
            decision = "BOX BOX"
        elif proba >= used_threshold - margin:
            decision = "STANDBY"
        else:
            decision = "STANDBY" if high_wear else "STAY OUT"
        if sc_text in ("SC", "VSC") and decision == "BOX BOX":
            lock_decision = True
    else:
        if critical_wear:
            decision = "BOX BOX"
            decision_reasons.append("CRITICAL-WEAR-EMERGENCY")
        elif urgent_wear:
            decision = "STANDBY"
            decision_reasons.append("NOWINDOW-HIGHWEAR")
        elif window_soon and high_wear:
            decision = "STANDBY"
        else:
            decision = "STAY OUT"
            if proba >= used_threshold + margin:
                decision_reasons.append("NOWINDOW-CONSERVATIVE")

    cooldown_laps = 2
    if tire_age is not None and tire_age <= cooldown_laps:
        if decision != "STAY OUT":
            decision = "STAY OUT"
            decision_reasons.append("COOLDOWN")

    pit_loss_sec = 20.0
    if sc_text in ("SC", "VSC"):
        pit_loss_sec = 12.0
    if gap_val is not None:
        pit_loss_sec = max(8.0, pit_loss_sec - min(6.0, gap_val / 5.0))

    remaining_laps = None
    if lap_value is not None:
        total_laps = _get_val(row, ["nolaps_prev", "nolaps", "n_laps", "laps"])
        if total_laps is not None:
            remaining_laps = max(1, int(total_laps) - int(lap_value))
    if remaining_laps is None and race_progress is not None:
        remaining_laps = max(1, int((1.0 - race_progress) * 50))
    horizon = int(max(1, lookahead_laps))
    if remaining_laps is not None:
        horizon = max(horizon, min(remaining_laps, 12))

    wear_factor = 0.3 if tire_wear_pct is None else float(np.clip(tire_wear_pct, 0.0, 1.0))
    gain_per_lap = 0.4 + 1.2 * wear_factor
    pace_factor = 1.05 if gap_trend_text == "GAIN" else 0.85 if gap_trend_text == "LOSS" else 0.95
    expected_gain_sec = gain_per_lap * float(horizon) * pace_factor
    net_gain_sec = expected_gain_sec - pit_loss_sec
    if net_gain_sec < -8.0 and decision in ("BOX BOX", "PIT NOW"):
        if pit_window_text == "OPEN" and proba >= used_threshold + 0.1:
            decision_reasons.append("COST-HOLD")
        else:
            decision = "STANDBY" if pit_window_text == "OPEN" else "STAY OUT"
            decision_reasons.append("COST-")
    elif net_gain_sec > 0.0:
        decision_reasons.append("COST+")

    # Final Alignment: If the policy forces a higher decision, the confidence should reflect the urgency
    final_proba = proba
    if decision == "BOX BOX":
        final_proba = max(proba, used_threshold + margin + 0.1)
    elif decision == "STANDBY":
        final_proba = max(proba, used_threshold - margin + 0.05)
    
    decision_source = "MODEL"
    if lock_decision:
        decision_source = "POLICY"
    elif decision in ("BOX BOX", "PIT NOW") and proba < used_threshold:
        decision_source = "POLICY"
    elif decision in ("STAY OUT", "STANDBY") and proba >= used_threshold + margin:
        decision_source = "POLICY"
    if any(tag in decision_reasons for tag in ("COOLDOWN", "COST-", "NOWINDOW")):
        decision_source = "POLICY"

    urgency = float(final_proba)
    if race_progress is not None:
        rp = float(np.clip(race_progress, 0.0, 1.0))
        urgency = float(np.clip(0.5 * final_proba + 0.5 * rp, 0.0, 1.0))
    else:
        rp = None

    pit_target_text = "N/A"
    if pit_window_text == "OPEN":
        pit_target_text = "NOW"
    elif pit_window_text == "CLOSED":
        pit_target_text = "SOON" if urgent_wear else "HOLD"

    return {
        "decision": decision,
        "decision_source": decision_source,
        "proba": final_proba, # Return the adjusted probability to the UI
        "used_threshold": float(used_threshold),
        "net_gain_sec": float(net_gain_sec),
        "pit_window_text": pit_window_text,
        "pit_target_text": pit_target_text,
        "tire_wear_pct": tire_wear_pct,
        "gap_trend_text": gap_trend_text,
        "race_progress": race_progress,
        "urgency": urgency,
        "reason_text": "+".join(decision_reasons),
        "window_start": window_start,
        "window_end": window_end,
        "weather_status": weather_status,
    }


def _make_pipeline(df: pd.DataFrame, features: list[str], params: dict) -> Pipeline:
    num_cols = [c for c in features if df[c].dtype != "object"]
    cat_cols = [c for c in features if c not in num_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    clf = XGBClassifier(**params)
    return Pipeline([("pre", pre), ("clf", clf)])


def _apply_scale_pos_weight(params: dict, y: np.ndarray) -> dict:
    out = dict(params)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    spw = float(neg / pos) if pos > 0 else 1.0
    out["scale_pos_weight"] = spw
    return out


def _select_threshold(y_true: np.ndarray, probs: np.ndarray, beta: float = 1.0) -> float:
    if y_true.size == 0:
        return 0.5
    beta2 = beta * beta
    best_t = 0.5
    best_score = -1.0
    for t in np.linspace(0.05, 0.95, 50):
        pred = probs >= t
        tp = int(np.logical_and(pred, y_true == 1).sum())
        fp = int(np.logical_and(pred, y_true == 0).sum())
        fn = int(np.logical_and(~pred, y_true == 1).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        denom = beta2 * precision + recall
        score = (1 + beta2) * precision * recall / denom if denom > 0 else 0.0
        if score > best_score or (abs(score - best_score) < 1e-6 and t < best_t):
            best_t = float(t)
            best_score = float(score)
    return best_t


def _threshold_for_precision(y_true: np.ndarray, probs: np.ndarray, min_precision: float) -> float | None:
    if y_true.size == 0:
        return None
    best_t = None
    for t in np.linspace(0.05, 0.95, 50):
        pred = probs >= t
        tp = int(np.logical_and(pred, y_true == 1).sum())
        fp = int(np.logical_and(pred, y_true == 0).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        if precision >= min_precision:
            best_t = float(t)
            break
    return best_t


def _split_calibration(df: pd.DataFrame, y: np.ndarray, group_col: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    if group_col and group_col in df.columns and df[group_col].nunique() > 1:
        groups = df[group_col]
        n_splits = max(2, min(5, groups.nunique()))
        try:
            gkf = GroupKFold(n_splits=n_splits)
            return next(gkf.split(df, y, groups=groups))
        except Exception:
            return None
    try:
        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        return next(sss.split(df, y))
    except Exception:
        return None


def _apply_feature_allowlist(features: list[str]) -> list[str]:
    allow = [f for f in DEMO_SHARED_FEATURES if f in features]
    return allow if allow else features


def _align_features(features: list[str], df: pd.DataFrame) -> list[str]:
    aligned = []
    for feat in features:
        if feat not in df.columns:
            continue
        if df[feat].dropna().empty:
            continue
        aligned.append(feat)
    return aligned


def _calc_tire_max(df: pd.DataFrame) -> float:
    if "tireage" in df.columns:
        tire_vals = pd.to_numeric(df["tireage"], errors="coerce").dropna()
    elif "stint_laps_prev" in df.columns:
        tire_vals = pd.to_numeric(df["stint_laps_prev"], errors="coerce").dropna()
    else:
        return 35.0
    if tire_vals.empty:
        return 35.0
    return float(np.nanpercentile(tire_vals, 90))


def _load_best_params() -> dict:
    root = Path(__file__).resolve().parents[2]
    best_path = root / "results" / "summary_plots" / "stage4_ablation_best.json"
    if not best_path.exists():
        return {}
    try:
        import json

        data = json.loads(best_path.read_text(encoding="utf-8"))
        params = data.get("params") if isinstance(data, dict) else None
        if isinstance(params, dict):
            return params
    except Exception:
        return {}
    return {}


@lru_cache(maxsize=4)
def _train_bundle(dataset_key: str) -> dict[str, Any]:
    df = load_dataset(dataset_key)
    if df.empty:
        raise ValueError("Dataset empty")

    target_col = detect_decision_col(df) or "decide_pitstop"
    group_col = "race_id" if "race_id" in df.columns else None

    if build_feature_list is None:
        features = [c for c in df.columns if c not in {target_col, group_col}]
    else:
        features = build_feature_list(df, target_col, group_col or "race_id")

    features = _apply_feature_allowlist(features)
    features = _align_features(features, df)
    if not features:
        raise ValueError("No features for training")

    base_params = {
        "n_estimators": 400,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "reg_lambda": 1.0,
        "reg_alpha": 0.2,
        "min_child_weight": 3,
        "eval_metric": "logloss",
        "random_state": 42,
        "n_jobs": -1,
    }
    params = {**base_params, **_load_best_params()}

    y = df[target_col].astype(int).values
    params = _apply_scale_pos_weight(params, y)

    split = _split_calibration(df, y, group_col)
    if split is None:
        train_idx = np.arange(len(df))
        cal_idx = np.array([], dtype=int)
    else:
        train_idx, cal_idx = split
        if train_idx.size == 0:
            train_idx = np.arange(len(df))
        if cal_idx.size == 0:
            cal_idx = np.array([], dtype=int)

    pipe = _make_pipeline(df, features, params)
    pipe.fit(df.iloc[train_idx][features], y[train_idx])

    calibrator = None
    cal_threshold = None
    if cal_idx.size > 0:
        cal_df = df.iloc[cal_idx]
        if len(cal_df) > 20000:
            cal_df = cal_df.sample(n=20000, random_state=42)
        y_cal = cal_df[target_col].astype(int).values
        if np.unique(y_cal).size > 1:
            p_cal_raw = pipe.predict_proba(cal_df[features])[:, 1]
            try:
                cal = LogisticRegression(solver="lbfgs")
                cal.fit(p_cal_raw.reshape(-1, 1), y_cal)
                calibrator = cal
                p_cal = calibrator.predict_proba(p_cal_raw.reshape(-1, 1))[:, 1]
            except Exception:
                calibrator = None
                p_cal = p_cal_raw
            cal_threshold = _select_threshold(y_cal, p_cal, beta=1.0)

    train_probs_raw = pipe.predict_proba(df.iloc[train_idx][features])[:, 1]
    if calibrator is not None:
        try:
            train_probs = calibrator.predict_proba(train_probs_raw.reshape(-1, 1))[:, 1]
        except Exception:
            train_probs = train_probs_raw
    else:
        train_probs = train_probs_raw

    decision_threshold = cal_threshold if cal_threshold is not None else _select_threshold(y[train_idx], train_probs, beta=1.0)
    prec_thresh = _threshold_for_precision(y[train_idx], train_probs, min_precision=0.6)
    if prec_thresh is not None:
        decision_threshold = max(decision_threshold, prec_thresh + 0.03)
    decision_threshold = float(np.clip(decision_threshold, 0.05, 0.95))

    return {
        "pipe": pipe,
        "calibrator": calibrator,
        "threshold": decision_threshold,
        "features": features,
        "tire_max": _calc_tire_max(df),
    }


def predict_row(row: pd.Series, dataset_key: str) -> ModelResult:
    try:
        bundle = _train_bundle(dataset_key)
    except Exception:
        prob = _heuristic_prob(row)
        return ModelResult(prob=prob, recommendation=_recommend(prob, 0.5), threshold=0.5, source="heuristic")

    features = bundle["features"]
    pipe = bundle["pipe"]
    calibrator = bundle["calibrator"]
    threshold = bundle["threshold"]

    row_df = pd.DataFrame([row])
    prob_raw = float(pipe.predict_proba(row_df[features])[:, 1][0])
    if calibrator is not None:
        try:
            prob = float(calibrator.predict_proba(np.array([[prob_raw]]))[:, 1][0])
        except Exception:
            prob = prob_raw
    else:
        prob = prob_raw

    recommendation = _recommend(prob, threshold)
    return ModelResult(prob=prob, recommendation=recommendation, threshold=threshold, source="xgboost")


def get_bundle(dataset_key: str) -> dict[str, Any]:
    return _train_bundle(dataset_key)


def estimate_pit_loss_seconds(row: pd.Series, pit_time_median: float | None = None) -> float:
    sc = _get_bool(row, ["sc_active", "sc_active_prev"])
    vsc = _get_bool(row, ["vsc_active", "vsc_active_prev"])
    rain = _get_bool(row, ["rain", "Rain", "Rainfall_prev"])
    pit_window = _get_bool(row, ["pit_window", "pit_window_prev", "in_pit_window", "in_pit_window_prev"])

    baseline = 22.0
    if sc:
        baseline = 20.0
    elif vsc:
        baseline = 14.0

    if rain:
        baseline += 1.5

    track_temp = _get_float(row, ["TrackTemp_prev", "TrackTemp", "track_temp"])
    if track_temp is not None and track_temp < 25.0:
        baseline += 0.5

    if pit_window:
        baseline -= 0.5

    gap = _get_float(row, ["gap_behind", "gap", "delta_interval_prev"]) or 99.0
    if gap < 2.0:
        baseline += 0.5

    if pit_time_median is not None:
        pit_time = _get_float(row, ["pit_time", "pitstop_time", "pit_duration"])
        if pit_time is not None:
            queue_penalty = max(0.0, pit_time - pit_time_median)
            baseline += queue_penalty

    return float(baseline)


def estimate_impact_seconds(row: pd.Series, recommendation: str, pit_time_median: float | None = None) -> float:
    pit_loss = estimate_pit_loss_seconds(row, pit_time_median)
    if recommendation == "PIT":
        return -pit_loss
    return pit_loss


def estimate_net_gain_seconds(row: pd.Series, tire_max: float, lookahead_laps: int = 4) -> float:
    race_progress = _get_val(row, ["race_progress", "race_progress_prev"])
    if race_progress is None and "nolaps_prev" in row and "lapno_prev" in row:
        try:
            race_progress = float(row["lapno_prev"]) / float(row["nolaps_prev"])
        except Exception:
            race_progress = None

    tire_age = _get_val(row, ["tireage", "tireage_prev", "stint_laps_prev"])
    tire_wear_pct = None
    if "tyre_wear_pct_prev" in row and pd.notna(row["tyre_wear_pct_prev"]):
        val = float(row["tyre_wear_pct_prev"])
        tire_wear_pct = val / 100.0 if val > 1.0 else val
    elif tire_age is not None:
        tire_wear_pct = min(1.0, float(tire_age) / float(max(1.0, tire_max)))

    gap_val = _get_val(row, ["gap", "gap_to_leader_prev", "interval", "gap_to_front_prev"])
    gap_delta = _get_val(row, ["delta_interval_prev", "delta_best_so_far_prev", "relative_pace_prev"])
    gap_trend = "STEADY"
    if gap_delta is not None:
        if gap_delta < -0.1:
            gap_trend = "GAIN"
        elif gap_delta > 0.1:
            gap_trend = "LOSS"
    elif gap_val is not None:
        if gap_val <= 1.0:
            gap_trend = "ATTACK"
        elif gap_val >= 3.0:
            gap_trend = "SAFE"

    sc_flag = _get_val(row, ["sc_active", "sc_active_prev"])
    vsc_flag = _get_val(row, ["vsc_active", "vsc_active_prev"])
    pit_loss_sec = 20.0
    if sc_flag is not None and sc_flag > 0:
        pit_loss_sec = 12.0
    elif vsc_flag is not None and vsc_flag > 0:
        pit_loss_sec = 12.0
    if gap_val is not None:
        pit_loss_sec = max(8.0, pit_loss_sec - min(6.0, gap_val / 5.0))

    remaining_laps = None
    if race_progress is not None:
        remaining_laps = max(1, int((1.0 - race_progress) * 50))
    horizon = int(max(1, lookahead_laps))
    if remaining_laps is not None:
        horizon = max(horizon, min(remaining_laps, 12))

    wear_factor = 0.3 if tire_wear_pct is None else float(np.clip(tire_wear_pct, 0.0, 1.0))
    gain_per_lap = 0.4 + 1.2 * wear_factor
    pace_factor = 1.05 if gap_trend == "GAIN" else 0.85 if gap_trend == "LOSS" else 0.95
    expected_gain_sec = gain_per_lap * float(horizon) * pace_factor
    net_gain_sec = expected_gain_sec - pit_loss_sec
    return float(net_gain_sec)


def data_quality_badge(row: pd.Series, required_cols: list[str]) -> str:
    missing = 0
    for col in required_cols:
        if col not in row or pd.isna(row[col]):
            missing += 1
    ratio = missing / max(1, len(required_cols))
    if ratio <= 0.1:
        return "HIGH"
    if ratio <= 0.3:
        return "MED"
    return "LOW"


def pit_time_median(df: pd.DataFrame) -> float | None:
    col = get_col(df, ["pit_time", "pitstop_time", "pit_duration"])
    if col is None:
        return None
    vals = pd.to_numeric(df[col], errors="coerce").dropna()
    if vals.empty:
        return None
    return float(np.median(vals))
