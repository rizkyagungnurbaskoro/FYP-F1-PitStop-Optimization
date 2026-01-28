from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .data import detect_decision_col, get_col


@dataclass
class Scenario:
    scenario_id: str
    title: str
    dataset: str
    row_index: int
    note: str


def _as_str(val: Any) -> str:
    return str(val) if val is not None else ""


def _find_monaco_double_stack(df: pd.DataFrame) -> list[int]:
    if df.empty:
        return []
    season_col = get_col(df, ["season", "Year"])
    race_col = get_col(df, ["race_id", "race", "Race"])
    driver_col = get_col(df, ["Driver", "driver", "driver_id"])
    lap_col = get_col(df, ["lapno", "Lap", "lap", "lapno_prev"])
    if season_col is None or race_col is None or driver_col is None:
        return []

    df_f = df.copy()
    try:
        df_f = df_f[df_f[season_col].astype(int) == 2022]
    except Exception:
        pass

    df_f = df_f[df_f[race_col].astype(str).str.contains("Monaco", case=False, na=False)]
    if df_f.empty:
        return []

    ferrari_drivers = ["LEC", "SAI"]
    df_f = df_f[df_f[driver_col].astype(str).isin(ferrari_drivers)]
    if df_f.empty:
        return []

    indices = []
    target_laps = [21, 22]
    if lap_col is None:
        return []

    for drv in ferrari_drivers:
        df_d = df_f[df_f[driver_col].astype(str) == drv]
        if df_d.empty:
            continue
        laps = pd.to_numeric(df_d[lap_col], errors="coerce")
        exact = df_d[laps.isin(target_laps)]
        if not exact.empty:
            exact = exact.assign(_lap_val=pd.to_numeric(exact[lap_col], errors="coerce"))
            exact = exact.sort_values("_lap_val")
            indices.append(int(exact.index[0]))
            continue
        laps = laps.fillna(target_laps[0])
        best_idx = None
        best_dist = None
        for target in target_laps:
            dist = (laps - target).abs()
            idx = dist.idxmin()
            dist_val = float(dist.loc[idx])
            if best_dist is None or dist_val < best_dist:
                best_dist = dist_val
                best_idx = int(idx)
        if best_idx is not None:
            indices.append(best_idx)

    return indices


def _auto_leader_event(df: pd.DataFrame) -> int | None:
    if df.empty:
        return None
    position_col = get_col(df, ["position", "Position_prev", "pos"])
    sc_col = get_col(df, ["sc_active", "sc_active_prev"])
    vsc_col = get_col(df, ["vsc_active", "vsc_active_prev"])
    rain_col = get_col(df, ["rain", "Rain", "Rainfall_prev"])
    lap_col = get_col(df, ["lapno", "lap", "lapno_prev"])
    decision_col = detect_decision_col(df)

    df_f = df.copy()
    if position_col is not None:
        df_f = df_f[pd.to_numeric(df_f[position_col], errors="coerce").fillna(99) <= 1]
    if decision_col is not None:
        df_f = df_f[pd.to_numeric(df_f[decision_col], errors="coerce").fillna(0) == 1]

    if df_f.empty:
        return None

    def score(row: pd.Series) -> float:
        sc = 1 if sc_col and row.get(sc_col, 0) == 1 else 0
        vsc = 1 if vsc_col and row.get(vsc_col, 0) == 1 else 0
        rain = 1 if rain_col and row.get(rain_col, 0) else 0
        lap = row.get(lap_col, 0) if lap_col else 0
        try:
            lap_val = float(lap)
        except Exception:
            lap_val = 0.0
        return sc * 3 + vsc * 2 + rain * 1 + abs(lap_val - 30) * -0.01

    best_idx = df_f.apply(score, axis=1).idxmax()
    return int(best_idx)


def build_showcase_scenarios(df: pd.DataFrame, dataset: str) -> list[Scenario]:
    scenarios: list[Scenario] = []
    indices = _find_monaco_double_stack(df)
    lap_col = get_col(df, ["lapno", "Lap", "lap", "lapno_prev"])
    labels = ["Monaco 2022 - Ferrari double-stack (LEC)", "Monaco 2022 - Ferrari double-stack (SAI)"]
    for idx, label in zip(indices, labels):
        lap_val = None
        if lap_col and idx in df.index:
            try:
                lap_val = int(pd.to_numeric(df.loc[idx, lap_col], errors="coerce"))
            except Exception:
                lap_val = None
        lap_note = f"Lap {lap_val}" if lap_val is not None else "Lap N/A"
        base_note = "External demo" if dataset != "ref" else "Dataset-backed"
        scenarios.append(
            Scenario(
                scenario_id=f"monaco_2022_{idx}",
                title=label,
                dataset=dataset,
                row_index=idx,
                note=f"{base_note} (Lap 21-22 locked; {lap_note})",
            )
        )
    if not scenarios:
        auto_idx = _auto_leader_event(df)
        if auto_idx is not None:
            scenarios.append(
                Scenario(
                    scenario_id=f"auto_leader_{auto_idx}",
                    title="Auto-selected Leader Pit Event",
                    dataset=dataset,
                    row_index=auto_idx,
                    note="Auto-selected fallback",
                )
            )

    return scenarios


def resolve_scenario(df: pd.DataFrame, scenario_id: str) -> pd.Series | None:
    if df.empty:
        return None
    if "_" in scenario_id:
        try:
            idx = int(scenario_id.split("_")[-1])
            if idx in df.index:
                return df.loc[idx]
        except Exception:
            pass
    return None
