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

    def _alias_missing(base_col: str, src_col: str) -> None:
        if base_col in out.columns or src_col not in out.columns:
            return
        _add(base_col, _num(src_col))

    # Backfill ref-style columns from *_prev where available (Stage 3/4 alignment).
    _alias_missing("lapno", "lapno_prev")
    _alias_missing("race_progress", "race_progress_prev")
    _alias_missing("pitstops_so_far", "pitstops_so_far_prev")
    _alias_missing("sc_active", "sc_active_prev")
    _alias_missing("vsc_active", "vsc_active_prev")
    _alias_missing("gap", "gap_to_leader_prev")
    _alias_missing("interval", "gap_to_front_prev")
    _alias_missing("position", "Position_prev")
    _alias_missing("tireage", "stint_laps_prev")

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
        lap_progress = _safe_div(lapno, nolaps)
        _add("LapProgress_prev", lap_progress)
        _add("LapProgressRemaining_prev", 1.0 - lap_progress)

    if stint_laps is not None and nolaps is not None:
        _add("StintProgress_prev", _safe_div(stint_laps, nolaps))

    race_prog = _num("race_progress_prev")
    if race_prog is not None:
        _add("RaceProgressRemaining_prev", 1.0 - race_prog.astype(float))
    if tyre_wear is not None and race_prog is not None:
        _add("WearProgress_prev", tyre_wear * race_prog)
        _add("WearToProgress_prev", _safe_div(tyre_wear, race_prog))

    pit_window = _num("in_pit_window_prev")
    if pit_window is not None and race_prog is not None:
        _add("PitWindowProgress_prev", pit_window * race_prog)
    if pit_window is not None and tyre_wear is not None:
        _add("PitWindowWear_prev", pit_window * tyre_wear)

    pitstops_so_far = _num("pitstops_so_far_prev")
    if pitstops_so_far is not None and lapno is not None:
        _add("PitstopRate_prev", _safe_div(pitstops_so_far, lapno))

    # Gap dynamics (prev)
    gap_leader = _num("gap_to_leader_prev")
    gap_front = _num("gap_to_front_prev")
    gap_behind = _num("gap_to_behind_prev")
    if gap_front is not None and gap_behind is not None:
        _add("GapFrontOverBehind_prev", _safe_div(gap_front, gap_behind))
        _add("GapDelta_prev", gap_front - gap_behind)
    if gap_front is not None and lapno is not None:
        _add("GapFrontPerLap_prev", _safe_div(gap_front, lapno))
    if gap_behind is not None and lapno is not None:
        _add("GapBehindPerLap_prev", _safe_div(gap_behind, lapno))
    if gap_leader is not None and gap_front is not None:
        _add("GapLeaderOverFront_prev", _safe_div(gap_leader, gap_front))
    if gap_leader is not None and lapno is not None:
        _add("LeaderGapPerLap_prev", _safe_div(gap_leader, lapno))

    gap_after_pit = _num("gap_after_pit_vs_behind_prev")
    if gap_after_pit is not None and gap_behind is not None:
        _add("GapAfterPitMargin_prev", gap_after_pit - gap_behind)

    undercut = _num("undercut_potential_prev")
    if undercut is not None and pit_window is not None:
        _add("UndercutPressure_prev", undercut * pit_window)
    if undercut is not None and gap_behind is not None:
        _add("UndercutVsGap_prev", _safe_div(undercut, gap_behind))

    # Pace deltas (prev)
    delta_best = _num("delta_best_so_far_prev")
    delta_interval = _num("delta_interval_prev")
    if delta_best is not None and delta_interval is not None:
        _add("DeltaBestOverInterval_prev", _safe_div(delta_best, delta_interval))
        _add("DeltaBestMinusInterval_prev", delta_best - delta_interval)

    rel_pace = _num("relative_pace_prev")
    if rel_pace is not None and gap_front is not None:
        _add("PaceGap_prev", rel_pace * gap_front)
        _add("PaceToGap_prev", _safe_div(rel_pace, gap_front))

    # Weather extras (prev)
    pressure = _num("Pressure_prev")
    if pressure is not None and track is not None:
        _add("PressureToTrackTemp_prev", _safe_div(pressure, track))
    wind_speed = _num("WindSpeed_prev")
    if wind_speed is not None and rainfall is not None:
        _add("WindRain_prev", wind_speed * rainfall)
    if rainfall is not None:
        _add("RainFlag_prev", (rainfall > 0).astype(float))

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


def select_canonical_features(
    df: pd.DataFrame,
    features: List[str],
    target_col: str,
    group_col: str,
    mapping: Dict[str, str],
    canonical_features: List[str],
) -> tuple[pd.DataFrame, List[str]]:
    selected: Dict[str, str] = {}
    for col in features:
        canonical = mapping.get(col, col)
        if canonical in canonical_features and canonical not in selected:
            selected[canonical] = col

    if not selected:
        raise ValueError("No canonical features found for alignment.")

    ordered = [c for c in canonical_features if c in selected]
    cols = [selected[c] for c in ordered]
    df_out = df[[target_col, group_col] + cols].copy()
    rename_map = {selected[c]: c for c in ordered}
    df_out = df_out.rename(columns=rename_map)
    return df_out, ordered
