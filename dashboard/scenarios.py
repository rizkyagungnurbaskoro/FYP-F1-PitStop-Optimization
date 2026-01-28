from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from experiments.exp_utils import add_feature_engineering, load_csv, safe_numeric


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Scenario:
    name: str
    dataset_label: str
    dataset_path: Path
    row_index: int
    source_note: str
    is_external: bool = False


def get_dataset_paths() -> tuple[Path, Path]:
    ref = ROOT / "data" / "strategy_weather_dataset.csv"
    my = ROOT / "personal_datasets" / "fastf1_strategy_dataset.csv"
    return ref, my


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


def _ensure_df(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = add_feature_engineering(safe_numeric(load_csv(path)))
    return df


def _find_event_row(
    df: pd.DataFrame,
    season: int,
    name_key: str,
    driver_candidates: list[str] | None = None,
) -> int | None:
    season_col = _pick_column(df, ["season", "Season", "year", "Year"])
    race_col = _pick_column(df, ["race_id", "race", "event", "EventName", "event_name"])
    driver_col = _pick_column(df, ["Driver", "driver", "driver_name"])
    pos_col = _pick_column(df, ["Position_prev", "position", "Position"])
    sc_col = _pick_column(df, ["sc_active_prev", "sc_active", "SC", "sc"])
    vsc_col = _pick_column(df, ["vsc_active_prev", "vsc_active", "VSC", "vsc"])
    rain_col = _pick_column(df, ["Rainfall_prev", "rain", "Rainfall"])
    if season_col is None or race_col is None:
        return None
    subset = df[
        (df[season_col].astype(str) == str(season))
        & df[race_col].astype(str).str.contains(name_key, case=False, na=False)
    ]
    if subset.empty:
        return None
    if driver_candidates and driver_col is not None:
        subset = subset[subset[driver_col].isin(driver_candidates)]
        if subset.empty:
            return None
    if "decide_pitstop" in subset.columns:
        subset = subset[subset["decide_pitstop"] == 1]
        if subset.empty:
            return None
    if driver_candidates is None:
        if pos_col is not None:
            subset = subset[subset[pos_col] == 1]
            if subset.empty:
                return None
        if sc_col is not None or vsc_col is not None or rain_col is not None:
            flags = []
            if sc_col is not None:
                flags.append(subset[sc_col] == 1)
            if vsc_col is not None:
                flags.append(subset[vsc_col] == 1)
            if rain_col is not None:
                flags.append(subset[rain_col] == 1)
            if flags:
                mask = flags[0]
                for f in flags[1:]:
                    mask = mask | f
                subset = subset[mask]
                if subset.empty:
                    return None
    return int(subset.index[0])


def _auto_select_leader_event(df: pd.DataFrame) -> tuple[int | None, str]:
    if df.empty:
        return None, "No data"
    pos_col = _pick_column(df, ["Position_prev", "position", "Position"])
    sc_col = _pick_column(df, ["sc_active_prev", "sc_active", "SC", "sc"])
    vsc_col = _pick_column(df, ["vsc_active_prev", "vsc_active", "VSC", "vsc"])
    rain_col = _pick_column(df, ["Rainfall_prev", "rain", "Rainfall", "rain_flag"])
    lap_col = _pick_column(df, ["lapno_prev", "lapno", "lap", "Lap"])
    gap_col = _pick_column(df, ["gap_to_behind_prev", "gap_behind", "gap", "Gap"])

    filt = df.copy()
    if "decide_pitstop" in filt.columns:
        filt = filt[filt["decide_pitstop"] == 1]
    if pos_col:
        filt = filt[filt[pos_col] == 1]
    if sc_col:
        filt = filt[filt[sc_col] == 1]
    if filt.empty and vsc_col:
        filt = df.copy()
        if "decide_pitstop" in filt.columns:
            filt = filt[filt["decide_pitstop"] == 1]
        if pos_col:
            filt = filt[filt[pos_col] == 1]
        filt = filt[filt[vsc_col] == 1]
    if filt.empty and rain_col:
        filt = df.copy()
        if "decide_pitstop" in filt.columns:
            filt = filt[filt["decide_pitstop"] == 1]
        if pos_col:
            filt = filt[filt[pos_col] == 1]
        filt = filt[filt[rain_col] == 1]
    if filt.empty:
        filt = df.copy()
        if "decide_pitstop" in filt.columns:
            filt = filt[filt["decide_pitstop"] == 1]

    if filt.empty:
        return None, "No pit events found"

    lap_mid = None
    if lap_col:
        lap_vals = pd.to_numeric(filt[lap_col], errors="coerce").dropna()
        if not lap_vals.empty:
            lap_mid = float(lap_vals.max() * 0.5)

    sort_cols = []
    if sc_col:
        sort_cols.append(sc_col)
    if vsc_col:
        sort_cols.append(vsc_col)
    if rain_col:
        sort_cols.append(rain_col)
    if lap_col and lap_mid is not None:
        filt["_lap_mid_dist"] = (pd.to_numeric(filt[lap_col], errors="coerce") - lap_mid).abs()
        sort_cols.append("_lap_mid_dist")
    if gap_col:
        filt["_gap_val"] = pd.to_numeric(filt[gap_col], errors="coerce")
        sort_cols.append("_gap_val")

    if sort_cols:
        ascending = [False] * min(3, len(sort_cols))
        if len(sort_cols) > 3:
            ascending += [True] * (len(sort_cols) - 3)
        filt = filt.sort_values(sort_cols, ascending=ascending, na_position="last")

    return int(filt.index[0]), "Auto-selected Leader Pit Event"


def build_showcase_scenarios() -> list[Scenario]:
    ref_path, my_path = get_dataset_paths()
    scenarios: list[Scenario] = []

    df_ref = _ensure_df(ref_path)
    idx_ref = _find_event_row(df_ref, 2015, "Monte")
    if idx_ref is None:
        idx_ref = _find_event_row(df_ref, 2015, "Monaco")
    if idx_ref is None:
        idx_ref, label = _auto_select_leader_event(df_ref)
        name = label
    else:
        name = "Monaco 2015 - Leader pits under SC"
    if idx_ref is not None:
        scenarios.append(
            Scenario(
                name=name,
                dataset_label="RefData",
                dataset_path=ref_path,
                row_index=idx_ref,
                source_note="Dataset-backed (RefData)",
                is_external=False,
            )
        )

    df_my = _ensure_df(my_path)
    ferrari_codes = ["LEC", "SAI"]
    idx_my = _find_event_row(df_my, 2022, "Monte", ferrari_codes)
    if idx_my is None:
        idx_my = _find_event_row(df_my, 2022, "Monaco", ferrari_codes)
    if idx_my is None:
        idx_my, label = _auto_select_leader_event(df_my)
        name = f"{label} (External)"
    else:
        name = "Monaco 2022 - Ferrari pit event"

    if idx_my is not None:
        scenarios.append(
            Scenario(
                name=name,
                dataset_label="MyData+W",
                dataset_path=my_path,
                row_index=idx_my,
                source_note="External (FastF1) - Demo only",
                is_external=True,
            )
        )

    return scenarios


def load_scenario_row(scenario: Scenario) -> tuple[pd.DataFrame, pd.Series]:
    df = _ensure_df(scenario.dataset_path)
    if df.empty or scenario.row_index not in df.index:
        return df, pd.Series(dtype=float)
    row = df.loc[scenario.row_index]
    return df, row
