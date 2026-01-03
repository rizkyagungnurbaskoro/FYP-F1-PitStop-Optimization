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
