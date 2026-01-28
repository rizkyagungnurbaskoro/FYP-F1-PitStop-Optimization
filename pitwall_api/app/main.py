from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

from .config import SUMMARY_STD_PATH, SUMMARY_STRICT_PATH, SUMMARY_HOLDOUT_PATH
from .data import detect_decision_col, load_dataset
from .demo import run_demo_state
from .routers import demo as demo_router
from .model import (
    data_quality_badge,
    demo_policy_decision,
    predict_row,
    get_bundle,
    detect_lap_col,
    pit_window_bounds,
    smooth_prob_by_lap,
    model_proba,
)
from .scenarios import build_showcase_scenarios, resolve_scenario

app = FastAPI(title="Pitwall API", version="0.1.3") # Trigger reload 2
app.include_router(demo_router.router)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)




def _summary_to_records(path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        df = pd.read_csv(path)
    except Exception:
        return []
    return df.to_dict(orient="records")


def _build_context(row: pd.Series) -> dict[str, Any]:
    ctx = {
        "sc_active": row.get("sc_active", row.get("sc_active_prev", 0)),
        "vsc_active": row.get("vsc_active", row.get("vsc_active_prev", 0)),
        "rain": row.get("rain", row.get("Rain", row.get("Rainfall_prev", 0))),
        "track_temp": row.get("TrackTemp_prev", row.get("TrackTemp")),
        "tyre_age": row.get("tireage", row.get("stint_laps_prev")),
        "position": row.get("position", row.get("Position_prev")),
        "gap": row.get("gap_behind", row.get("gap", row.get("delta_interval_prev"))),
    }
    return ctx


def _lap_value(row: pd.Series, lap_col: str | None = None) -> int | None:
    if lap_col and lap_col in row and pd.notna(row[lap_col]):
        try:
            return int(float(row[lap_col]))
        except Exception:
            pass
    for key in ("lapno_prev", "lapno", "Lap", "lap"):
        if key in row and pd.notna(row[key]):
            try:
                return int(float(row[key]))
            except Exception:
                continue
    return None


def _normalize_value(val: Any) -> Any:
    if isinstance(val, (np.generic,)):
        return val.item()
    if isinstance(val, (np.ndarray,)):
        return val.tolist()
    if isinstance(val, (pd.Timestamp, pd.Timedelta)):
        return str(val)
    try:
        if pd.isna(val):
            return None
    except Exception:
        pass
    return val


def _context_df(df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    ctx = df
    race_col = None
    for cand in ("race_id", "race", "Race"):
        if cand in df.columns:
            race_col = cand
            break
    driver_col = None
    for cand in ("Driver", "driver", "driver_id"):
        if cand in df.columns:
            driver_col = cand
            break
    if race_col and race_col in row and pd.notna(row[race_col]):
        ctx = ctx[ctx[race_col] == row[race_col]]
    if driver_col and driver_col in row and pd.notna(row[driver_col]):
        ctx_driver = ctx[ctx[driver_col] == row[driver_col]]
        if not ctx_driver.empty:
            ctx = ctx_driver
    if ctx.empty:
        ctx = df
    return ctx


def _normalize_obj(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {_normalize_value(k): _normalize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return [_normalize_obj(v) for v in obj]
    return _normalize_value(obj)


def _normalize_row(row: pd.Series) -> dict[str, Any]:
    data = row.to_dict()
    return {k: _normalize_value(v) for k, v in data.items()}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok"}


@app.get("/metrics/summary")
def metrics_summary(mode: str = "strict") -> dict[str, Any]:
    path = SUMMARY_STRICT_PATH if mode.lower() == "strict" else SUMMARY_STD_PATH
    return {"mode": mode, "rows": _summary_to_records(path)}


@app.get("/metrics/holdout")
def metrics_holdout() -> dict[str, Any]:
    return {"rows": _summary_to_records(SUMMARY_HOLDOUT_PATH)}


@app.get("/scenarios/showcase")
def showcase_scenarios(dataset: str = "my") -> dict[str, Any]:
    df = load_dataset(dataset)
    scenarios = build_showcase_scenarios(df, dataset)
    return {
        "dataset": dataset,
        "count": len(scenarios),
        "scenarios": [s.__dict__ for s in scenarios],
    }


@app.get("/scenario/{scenario_id}")
def get_scenario(scenario_id: str, dataset: str = "my") -> dict[str, Any]:
    df = load_dataset(dataset)
    if df.empty:
        raise HTTPException(status_code=404, detail="Dataset not found or empty")

    row = resolve_scenario(df, scenario_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Scenario not found")

    decision_col = detect_decision_col(df)
    if decision_col is None:
        raise HTTPException(status_code=400, detail="Decision column missing")

    model = predict_row(row, dataset)
    bundle = get_bundle(dataset)
    tire_max = float(bundle.get("tire_max", 35.0))
    lap_col = detect_lap_col(df)
    lap_val = _lap_value(row, lap_col)

    prob_used = model.prob
    win_start, win_end = None, None
    is_monaco = scenario_id.startswith("monaco_2022")
    if not is_monaco:
        context_df = _context_df(df, row)
        compound = str(row.get("compound", row.get("Compound", "MEDIUM"))).upper()
        window = pit_window_bounds(context_df, lap_col, current_lap=lap_val, compound=compound)
        win_start, win_end = window if window else (None, None)
        try:
            context_probs = model_proba(context_df, bundle)
            context_df = context_df.copy()
            context_df["proba"] = context_probs
            prob_smooth = smooth_prob_by_lap(context_df, lap_col, "proba", lap_val, window=3)
            if prob_smooth is not None:
                prob_used = float(prob_smooth)
        except Exception:
            prob_used = model.prob

    demo = demo_policy_decision(
        row,
        prob_used,
        model.threshold,
        lap_val,
        lap_col,
        tire_max,
        lookahead_laps=4,
        window_start=win_start,
        window_end=win_end,
        df_context=context_df,
    )
    model.recommendation = demo["decision"]
    model.threshold = float(demo["used_threshold"])
    model.source = demo["decision_source"]
    model.prob = float(prob_used)

    required = ["lapno", "race_id", "Driver", "position"]
    badge = data_quality_badge(row, required)

    hist_call = "PIT" if int(float(row[decision_col])) == 1 else "STAY OUT"
    model_call = str(model.recommendation)
    net_gain = float(demo.get("net_gain_sec", 0.0))
    impact_hist = net_gain if hist_call == "PIT" else 0.0
    model_is_pit = model_call.upper().startswith("BOX") or model_call.upper().startswith("PIT")
    impact_model = net_gain if model_is_pit else 0.0
    impact_delta = impact_model - impact_hist

    result = {
        "scenario_id": scenario_id,
        "dataset": dataset,
        "row": _normalize_row(row),
        "model": model.__dict__,
        "historical_call": hist_call,
        "impact_seconds": impact_delta,
        "impact_net_gain": net_gain,
        "impact_hist": impact_hist,
        "impact_model": impact_model,
        "impact_delta": impact_delta,
        "lap": lap_val,
        "context": _build_context(row),
        "data_quality": badge,
        "demo": demo,
    }
    return jsonable_encoder(_normalize_obj(result))


@app.post("/whatif")
def whatif(payload: dict[str, Any]) -> dict[str, Any]:
    dataset = payload.get("dataset", "my")
    scenario_id = payload.get("scenario_id")
    if not scenario_id:
        raise HTTPException(status_code=400, detail="scenario_id required")

    df = load_dataset(dataset)
    if df.empty:
        raise HTTPException(status_code=404, detail="Dataset not found or empty")

    row = resolve_scenario(df, scenario_id)
    if row is None:
        raise HTTPException(status_code=404, detail="Scenario not found")

    decision_col = detect_decision_col(df)
    if decision_col is None:
        raise HTTPException(status_code=400, detail="Decision column missing")

    model = predict_row(row, dataset)
    bundle = get_bundle(dataset)
    tire_max = float(bundle.get("tire_max", 35.0))
    lap_col = detect_lap_col(df)
    lap_val = _lap_value(row, lap_col)

    prob_used = model.prob
    win_start, win_end = None, None
    is_monaco = scenario_id.startswith("monaco_2022")
    if not is_monaco:
        context_df = _context_df(df, row)
        compound = str(row.get("compound", row.get("Compound", "MEDIUM"))).upper()
        window = pit_window_bounds(context_df, lap_col, current_lap=lap_val, compound=compound)
        win_start, win_end = window if window else (None, None)
        try:
            context_probs = model_proba(context_df, bundle)
            context_df = context_df.copy()
            context_df["proba"] = context_probs
            prob_smooth = smooth_prob_by_lap(context_df, lap_col, "proba", lap_val, window=3)
            if prob_smooth is not None:
                prob_used = float(prob_smooth)
        except Exception:
            prob_used = model.prob

    demo = demo_policy_decision(
        row,
        prob_used,
        model.threshold,
        lap_val,
        lap_col,
        tire_max,
        lookahead_laps=4,
        window_start=win_start,
        window_end=win_end,
        df_context=context_df,
    )
    model.recommendation = demo["decision"]
    model.threshold = float(demo["used_threshold"])
    model.source = demo["decision_source"]
    model.prob = float(prob_used)

    actual_call = "PIT" if int(float(row[decision_col])) == 1 else "STAY OUT"
    whatif_call = "PIT" if actual_call == "STAY OUT" else "STAY OUT"

    net_gain = float(demo.get("net_gain_sec", 0.0))
    impact_hist = net_gain if actual_call == "PIT" else 0.0
    model_call = str(model.recommendation)
    model_is_pit = model_call.upper().startswith("BOX") or model_call.upper().startswith("PIT")
    impact_model = net_gain if model_is_pit else 0.0
    impact_delta = impact_model - impact_hist

    payload_out = {
        "scenario_id": scenario_id,
        "dataset": dataset,
        "actual": {
            "call": actual_call,
            "pit_probability": model.prob,
            "impact_seconds": impact_hist,
        },
        "model": {
            "call": model.recommendation,
            "pit_probability": model.prob,
            "impact_seconds": impact_model,
        },
        "whatif": {
            "call": whatif_call,
            "impact_seconds": -impact_delta,
        },
        "delta": {
            "call": f"{actual_call} -> {model.recommendation}",
            "impact_seconds": impact_delta,
        },
    }
    return jsonable_encoder(_normalize_obj(payload_out))
