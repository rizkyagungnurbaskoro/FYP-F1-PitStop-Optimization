from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    return pd.read_csv(path, low_memory=False)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def safe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = pd.to_numeric(out[c], errors="ignore")
    return out


def add_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    new_cols: List[str] = []

    def _num(col: str) -> pd.Series | None:
        if col not in out.columns:
            return None
        return pd.to_numeric(out[col], errors="coerce")

    def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-6) -> pd.Series:
        b_vals = b.astype(float).to_numpy()
        b_vals = np.where(np.abs(b_vals) < eps, eps, b_vals)
        return pd.Series(a.astype(float).to_numpy() / b_vals, index=a.index)

    def _add(col: str, series: pd.Series | None) -> None:
        if series is None:
            return
        if col in out.columns:
            return
        out[col] = series
        new_cols.append(col)

    # Weather and track features (prev)
    air = _num("AirTemp_prev")
    track = _num("TrackTemp_prev")
    if air is not None and track is not None:
        _add("TempDelta_prev", track - air)

    humidity = _num("Humidity_prev")
    rainfall = _num("Rainfall_prev")
    if humidity is not None and rainfall is not None:
        _add("HumidityRain_prev", humidity * (rainfall > 0).astype(float))

    wind_dir = _num("WindDirection_prev")
    if wind_dir is not None:
        rad = np.deg2rad(wind_dir.astype(float))
        _add("WindDirSin_prev", pd.Series(np.sin(rad), index=wind_dir.index))
        _add("WindDirCos_prev", pd.Series(np.cos(rad), index=wind_dir.index))

    # Stint and wear features (prev)
    stint_laps = _num("stint_laps_prev")
    tyre_wear = _num("tyre_wear_pct_prev")
    if stint_laps is not None and tyre_wear is not None:
        _add("WearPerLap_prev", _safe_div(tyre_wear, stint_laps))

    nolaps = _num("nolaps_prev")
    lapno = _num("lapno_prev")
    if nolaps is not None and lapno is not None:
        _add("LapsRemaining_prev", nolaps - lapno)

    if stint_laps is not None and nolaps is not None:
        _add("StintProgress_prev", _safe_div(stint_laps, nolaps))

    # Gap dynamics (prev)
    gap_leader = _num("gap_to_leader_prev")
    gap_front = _num("gap_to_front_prev")
    gap_behind = _num("gap_to_behind_prev")
    if gap_front is not None and gap_behind is not None:
        _add("GapFrontOverBehind_prev", _safe_div(gap_front, gap_behind))
        _add("GapDelta_prev", gap_front - gap_behind)
    if gap_leader is not None and gap_front is not None:
        _add("GapLeaderOverFront_prev", _safe_div(gap_leader, gap_front))

    gap_after_pit = _num("gap_after_pit_vs_behind_prev")
    if gap_after_pit is not None and gap_behind is not None:
        _add("GapAfterPitMargin_prev", gap_after_pit - gap_behind)

    undercut = _num("undercut_potential_prev")
    pit_window = _num("in_pit_window_prev")
    if undercut is not None and pit_window is not None:
        _add("UndercutPressure_prev", undercut * pit_window)

    # Pace deltas (prev)
    delta_best = _num("delta_best_so_far_prev")
    delta_interval = _num("delta_interval_prev")
    if delta_best is not None and delta_interval is not None:
        _add("DeltaBestOverInterval_prev", _safe_div(delta_best, delta_interval))

    rel_pace = _num("relative_pace_prev")
    if rel_pace is not None and gap_front is not None:
        _add("PaceGap_prev", rel_pace * gap_front)

    # Safety car flags (prev)
    sc = _num("sc_active_prev")
    vsc = _num("vsc_active_prev")
    if sc is not None and vsc is not None:
        _add("SCAny_prev", ((sc > 0) | (vsc > 0)).astype(float))

    # Reference dataset extras (non-prev)
    gap = _num("gap")
    interval = _num("interval")
    if gap is not None and interval is not None:
        _add("GapOverInterval", _safe_div(gap, interval))

    pit_so_far = _num("pitstops_so_far")
    pit_rem = _num("pitstops_remaining")
    if pit_so_far is not None and pit_rem is not None:
        _add("PitstopBalance", pit_rem - pit_so_far)

    sc_now = _num("sc_active")
    vsc_now = _num("vsc_active")
    if sc_now is not None and vsc_now is not None:
        _add("SCAny", ((sc_now > 0) | (vsc_now > 0)).astype(float))

    for c in new_cols:
        if out[c].isna().all():
            out = out.drop(columns=[c])

    return out


def _normalize(s: str) -> str:
    return "".join(ch.lower() for ch in s if ch.isalnum())


def guess_target_column(df: pd.DataFrame) -> str:
    candidates = [
        "decide_pitstop",
        "target",
        "y",
        "label",
        "pit",
        "pitstop",
        "is_pit",
        "is_pitstop",
        "pit_stop",
        "pitstop_label",
        "will_pit",
        "PitStop",
        "Pit_Stop",
        "IsPitStop",
    ]

    for c in candidates:
        if c in df.columns:
            return c

    norm_map = {_normalize(c): c for c in df.columns}
    for c in candidates:
        key = _normalize(c)
        if key in norm_map:
            return norm_map[key]

    for col in df.columns:
        n = _normalize(col)
        if "pit" in n and ("label" in n or "target" in n or "decide" in n):
            return col

    preview = ", ".join(list(df.columns[:30]))
    more = "" if len(df.columns) <= 30 else f" ... (+{len(df.columns)-30} more)"
    raise ValueError(
        "Cannot find target column.\n"
        "Fix: set the correct target column name in experiments/exp_config.py -> get_column_config().\n\n"
        f"Columns preview: {preview}{more}"
    )


def guess_group_column(df: pd.DataFrame) -> str:
    candidates = ["race_id", "RaceID", "race", "event_id", "EventID", "Round", "round"]
    for c in candidates:
        if c in df.columns:
            return c

    norm_map = {_normalize(c): c for c in df.columns}
    for c in candidates:
        key = _normalize(c)
        if key in norm_map:
            return norm_map[key]

    preview = ", ".join(list(df.columns[:30]))
    more = "" if len(df.columns) <= 30 else f" ... (+{len(df.columns)-30} more)"
    raise ValueError(
        "Cannot find group column (race/event id).\n"
        "Fix: set the correct group column name in experiments/exp_config.py -> get_column_config().\n\n"
        f"Columns preview: {preview}{more}"
    )


def build_feature_list(df: pd.DataFrame, target_col: str, group_col: str) -> List[str]:
    drop = {target_col, group_col}

    leakage_like = {
        "in_pit", "InPit", "pit_now", "pit_this_lap", "pitted_this_lap",
        "pitstop_this_lap", "is_pit_this_lap", "is_pitstop_this_lap",
    }
    drop |= {c for c in df.columns if c in leakage_like}

    feats = [c for c in df.columns if c not in drop]
    if not feats:
        raise ValueError("No features left after dropping target/group/leakage columns.")
    return feats
