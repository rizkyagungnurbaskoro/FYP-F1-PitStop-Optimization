from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

import sys
import pandas as pd

from .config import MYDATA_PATH, REFDATA_PATH

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from experiments.exp_utils import add_feature_engineering, safe_numeric  # type: ignore
except Exception:  # pragma: no cover
    add_feature_engineering = None
    safe_numeric = None


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path, low_memory=False)
    except Exception:
        return pd.DataFrame()

    # Drop completely empty columns to avoid dtype issues
    df = df.dropna(axis=1, how='all')
    
    # Force numeric conversion for known problematic columns
    for col in ['LapTime_valid', 'TrackTemp_valid', 'Gap_valid']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    if safe_numeric is not None:
        try:
            df = safe_numeric(df)
        except Exception:
            pass
    if add_feature_engineering is not None:
        try:
            df = add_feature_engineering(df)
        except Exception:
            pass
    return df


@lru_cache(maxsize=4)
def load_dataset(kind: str) -> pd.DataFrame:
    kind = kind.lower().strip()
    if kind in {"my", "mydata", "mydata+w", "fastf1"}:
        demo_path = MYDATA_PATH.parent / "fastf1_demo_dataset.csv"
        if demo_path.exists():
            print(f"[INFO] Using Demo Dataset: {demo_path}")
            return _safe_read_csv(demo_path)
        return _safe_read_csv(MYDATA_PATH)
    if kind in {"ref", "refdata", "strategy"}:
        return _safe_read_csv(REFDATA_PATH)
    return pd.DataFrame()


def get_col(df: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    return None


def detect_decision_col(df: pd.DataFrame) -> str | None:
    for name in ("decide_pitstop", "pitstop", "pit_decision", "pit"): 
        if name in df.columns:
            return name
    return None
