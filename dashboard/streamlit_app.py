from __future__ import annotations

import json
import sys
from math import comb
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from experiments.exp_utils import add_feature_engineering, build_feature_list, load_csv, safe_numeric

try:
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import GroupKFold, StratifiedShuffleSplit
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from xgboost import XGBClassifier

    _XGB_OK = True
except Exception:
    _XGB_OK = False

SUMMARY_STRICT = ROOT / "results" / "summary_plots" / "stage_summary_strict.csv"
SUMMARY_STD = ROOT / "results" / "summary_plots" / "stage_summary.csv"
FOLDS_STRICT = ROOT / "results" / "summary_plots" / "stage_folds_strict.csv"
FOLDS_STANDARD = ROOT / "results" / "summary_plots" / "stage_folds_standard.csv"
STAGE4_BEST_PARAMS = ROOT / "results" / "summary_plots" / "stage4_best_params.json"
PROJECT_DETAILS = ROOT / "reports" / "Project Details.md"

METRICS = {
    "F1": ("mean_f1", "std_f1"),
    "F2": ("mean_fbeta", "std_fbeta"),
    "Precision": ("mean_precision", "std_precision"),
    "Recall": ("mean_recall", "std_recall"),
    "PR-AUC": ("mean_pr_auc", "std_pr_auc"),
}

DEMO_SHARED_FEATURES = [
    "season",
    "lapno",
    "race_progress",
    "pitstops_so_far",
    "position",
    "gap",
    "interval",
    "sc_active",
    "vsc_active",
    "SCAny",
    "GapOverInterval",
    "tireage",
]

CIRCUIT_COL_CANDIDATES = [
    "track_deg_category",
    "circuit",
    "circuit_name",
    "track",
    "event",
    "EventName",
]


def _inject_css() -> None:
    st.markdown(
        """
<style>
:root {
  --bg1: #0b0e14;
  --bg2: #151a23;
  --panel: #12161f;
  --panel-2: #0f131b;
  --ink: #eef2f6;
  --muted: #a6adbb;
  --accent: #ff2b2b;
  --accent2: #28c1d6;
  --border: #232a36;
}
@import url('https://fonts.googleapis.com/css2 | family=Teko:wght@400;600;700&family=Rajdhani:wght@400;600&display=swap');
html, body, [class*="css"]  {
  font-family: "Rajdhani", "Segoe UI", sans-serif;
  color: var(--ink);
}
.stApp {
  background:
    radial-gradient(1200px 600px at 80% -10%, #1b2332 0%, rgba(0,0,0,0) 70%),
    linear-gradient(135deg, var(--bg1), var(--bg2));
}
section[data-testid="stSidebar"] {
  background: #0d1118;
  border-right: 1px solid var(--border);
}
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 18px;
  border: 1px solid var(--border);
  border-radius: 14px;
  background: linear-gradient(180deg, #141a24, #0c1017);
  box-shadow: 0 12px 28px rgba(0, 0, 0, 0.35);
  margin-bottom: 14px;
  position: relative;
  overflow: hidden;
}
.topbar:before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  height: 3px;
  width: 100%;
  background: linear-gradient(90deg, var(--accent), #ff9d2b 60%, #ffd02b);
}
.topbar-left, .topbar-center, .topbar-right {
  display: flex;
  align-items: center;
  gap: 10px;
}
.logo-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 34px;
  height: 34px;
  border-radius: 10px;
  background: var(--accent);
  color: #fff;
  font-family: "Teko", "Rajdhani", sans-serif;
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: 0.04em;
}
.top-title {
  font-family: "Teko", "Rajdhani", sans-serif;
  font-size: 1.4rem;
  font-weight: 700;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.top-sub {
  color: var(--muted);
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid #2b3342;
  background: #0b0f16;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  font-size: 0.75rem;
}
.pill strong {
  color: var(--accent2);
  font-size: 0.85rem;
}
.card {
  background: linear-gradient(160deg, var(--panel), var(--panel-2));
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 16px 18px;
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
  animation: fadein 0.6s ease-out;
}
.card-title {
  font-size: 0.95rem;
  color: var(--muted);
  margin-bottom: 6px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.card-value {
  font-family: "Teko", "Rajdhani", sans-serif;
  font-size: 2.1rem;
  font-weight: 700;
  margin-bottom: 2px;
  letter-spacing: 0.02em;
}
.card-sub {
  font-size: 0.85rem;
  color: var(--muted);
}
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(255, 43, 43, 0.15);
  color: var(--accent);
  font-weight: 600;
  font-size: 0.8rem;
  border: 1px solid rgba(255, 43, 43, 0.35);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.section-title {
  font-family: "Teko", "Rajdhani", sans-serif;
  font-size: 1.5rem;
  font-weight: 700;
  margin: 6px 0 12px 0;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.callout {
  background: linear-gradient(160deg, rgba(255,43,43,0.08), rgba(18,22,31,0.9));
  border: 1px solid rgba(255,43,43,0.3);
  border-radius: 12px;
  padding: 12px 14px;
  color: #e9edf5;
  font-size: 0.95rem;
}
.callout strong {
  color: var(--accent);
}
.example-card {
  background: linear-gradient(160deg, rgba(40,193,214,0.08), rgba(18,22,31,0.9));
  border: 1px solid rgba(40,193,214,0.3);
  border-radius: 12px;
  padding: 12px 14px;
  color: #e9edf5;
  font-size: 0.95rem;
}
.example-card strong {
  color: var(--accent2);
}
.decision-card {
  background: linear-gradient(160deg, rgba(255,157,43,0.08), rgba(18,22,31,0.9));
  border: 1px solid rgba(255,157,43,0.3);
  border-radius: 12px;
  padding: 12px 14px;
  color: #e9edf5;
  font-size: 0.95rem;
}
.decision-pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.04em;
}
.decision-pit {
  background: rgba(255,43,43,0.18);
  color: #ff6b6b;
  border: 1px solid rgba(255,43,43,0.35);
}
.decision-stay {
  background: rgba(40,193,214,0.18);
  color: #7be7f3;
  border: 1px solid rgba(40,193,214,0.35);
}
.decision-wait {
  background: rgba(255,157,43,0.18);
  color: #ffb25c;
  border: 1px solid rgba(255,157,43,0.45);
}
.summary-card {
  background: linear-gradient(160deg, rgba(40,193,214,0.06), rgba(18,22,31,0.95));
  border: 1px solid rgba(40,193,214,0.35);
}
.summary-diff {
  color: #ffb25c;
  font-weight: 700;
  letter-spacing: 0.03em;
}
.timeline-wrap {
  margin-top: 8px;
}
.timeline-title {
  color: var(--muted);
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.timeline {
  position: relative;
  height: 10px;
  border-radius: 999px;
  background: #1c2330;
  border: 1px solid #2a3342;
  margin-top: 6px;
}
.timeline-marker {
  position: absolute;
  top: -4px;
  width: 8px;
  height: 18px;
  border-radius: 4px;
}
.timeline-current {
  background: #28c1d6;
}
.timeline-window {
  background: #ff9d2b;
}
.timeline-rec {
  background: #ff2b2b;
}
.timeline-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.7rem;
  color: var(--muted);
  margin-top: 4px;
}
.strategy-card {
  background: linear-gradient(160deg, rgba(255,43,43,0.08), rgba(18,22,31,0.95));
  border: 1px solid rgba(255,43,43,0.35);
  border-radius: 14px;
  padding: 14px 16px;
  color: #e9edf5;
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
}
.strategy-title {
  font-family: "Teko", "Rajdhani", sans-serif;
  font-size: 1.1rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.strategy-sub {
  color: var(--muted);
  font-size: 0.85rem;
  margin-top: 4px;
}
.track-card {
  background: linear-gradient(160deg, rgba(40,193,214,0.08), rgba(18,22,31,0.95));
  border: 1px solid rgba(40,193,214,0.35);
  border-radius: 16px;
  padding: 14px 16px;
  color: #e9edf5;
  box-shadow: 0 10px 28px rgba(0, 0, 0, 0.35);
}
.track-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.track-title {
  font-family: "Teko", "Rajdhani", sans-serif;
  font-size: 1.1rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.track-sub {
  color: var(--muted);
  font-size: 0.8rem;
  letter-spacing: 0.04em;
}
.track-pills {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  align-items: center;
}
.metric-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid #2b3342;
  background: #0b0f16;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  font-size: 0.7rem;
}
.track-svg {
  width: 100%;
  height: 180px;
  margin-top: 10px;
  display: block;
}
.track-legend {
  display: flex;
  gap: 12px;
  align-items: center;
  font-size: 0.8rem;
  color: var(--muted);
  margin-top: 6px;
}
.track-meter {
  height: 6px;
  border-radius: 999px;
  background: #1c2330;
  border: 1px solid #2a3342;
  overflow: hidden;
  margin-top: 8px;
}
.track-meter-fill {
  height: 100%;
  background: linear-gradient(90deg, #28c1d6, #ff2b2b);
}
.signal-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 10px;
}
.signal-chip {
  background: #0b0f16;
  border: 1px solid #2b3342;
  border-radius: 10px;
  padding: 6px 10px;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--muted);
}
.signal-chip strong {
  color: #f2f6fb;
  font-weight: 700;
  margin-left: 6px;
}
.demo-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin: 8px 0 6px 0;
}
.demo-chip {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid #2b3342;
  background: #0b0f16;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
}
.demo-chip strong {
  color: #ff9d2b;
  font-weight: 700;
}
.tire-gauge {
  height: 8px;
  border-radius: 999px;
  background: #1c2330;
  border: 1px solid #2a3342;
  overflow: hidden;
  margin-top: 6px;
}
.tire-fill {
  height: 100%;
  background: linear-gradient(90deg, #28c1d6, #ff9d2b, #ff2b2b);
}
.telemetry-section {
  margin-top: 10px;
  padding-top: 8px;
  border-top: 1px solid #1e2531;
}
.telemetry-title {
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin-bottom: 6px;
}
.telemetry-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 6px 14px;
}
.telemetry-item {
  display: flex;
  align-items: center;
  justify-content: space-between;
  font-size: 0.75rem;
  color: #9aa4b3;
}
.telemetry-item span {
  color: #e6ebf2;
  font-weight: 600;
}
.helper-card {
  background: linear-gradient(160deg, rgba(40, 193, 214, 0.08), rgba(18, 22, 31, 0.9));
  border: 1px solid #203040;
  border-radius: 12px;
  padding: 12px 14px;
  margin: 10px 0 12px;
}
.helper-title {
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin-bottom: 8px;
}
.helper-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px 16px;
}
.helper-pill {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid #2b3342;
  background: #0b0f16;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: var(--muted);
}
.helper-pill strong {
  color: #e6ebf2;
}
.helper-note {
  font-size: 0.78rem;
  color: #c8d1dd;
  margin-top: 6px;
}
.car-attack {
  animation: car-attack 1.4s ease-in-out infinite alternate;
}
.car-press {
  animation: car-press 1.4s ease-in-out infinite alternate;
}
@keyframes car-attack {
  from { transform: translateX(0); }
  to { transform: translateX(14px); }
}
@keyframes car-press {
  from { transform: translateX(0); }
  to { transform: translateX(-14px); }
}
.strategy-signals {
  display: grid;
  grid-template-columns: repeat(2, minmax(0, 1fr));
  gap: 10px;
  margin-top: 10px;
}
.signal {
  background: rgba(15,19,27,0.9);
  border: 1px solid var(--border);
  border-radius: 10px;
  padding: 8px 10px;
  font-size: 0.9rem;
}
.signal strong {
  color: #fff3d6;
}
.urgency {
  margin-top: 10px;
}
.urgency-bar {
  height: 10px;
  border-radius: 999px;
  background: #1c2330;
  border: 1px solid #2a3342;
  overflow: hidden;
}
.urgency-fill {
  height: 100%;
  background: linear-gradient(90deg, #ff9d2b, #ff2b2b);
}
.radio-call {
  margin-top: 10px;
  font-style: italic;
  color: #f0c67b;
  font-size: 0.9rem;
}
.delta-up {
  color: #38d996;
  font-weight: 700;
}
.delta-down {
  color: #ff5c5c;
  font-weight: 700;
}
.delta-flat {
  color: #f1c232;
  font-weight: 700;
}
.legend {
  display: flex;
  gap: 12px;
  align-items: center;
  font-size: 0.85rem;
  color: var(--muted);
}
.legend span {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}
.legend .swatch {
  width: 10px;
  height: 10px;
  border-radius: 4px;
  display: inline-block;
}
.swatch-ref { background: #2c3545; }
.swatch-my { background: #ff2b2b; }
@keyframes fadein {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
""",
        unsafe_allow_html=True,
    )


@st.cache_data
def _load_summary(path: Path, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    stage_match = df["stage"].astype(str).str.extract(r"Stage\\s+(\\d+)")
    df["stage_id"] = pd.to_numeric(stage_match[0], errors="coerce")
    if df["stage_id"].isna().any():
        fallback = pd.Series(np.arange(1, len(df) + 1), index=df.index)
        df["stage_id"] = df["stage_id"].fillna(fallback)
    df["stage_id"] = df["stage_id"].astype(int)
    df["stage_short"] = "S" + df["stage_id"].astype(str)
    df["method"] = np.where(df["stage"].str.contains("MyMethod", case=False), "MyMethod", "RefTech")
    df["dataset"] = df["stage"].str.split(" on ").str[-1]
    return df.sort_values("stage_id").reset_index(drop=True)


@st.cache_data
def _load_folds(path: Path, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    stage_match = df["stage"].astype(str).str.extract(r"Stage\\s+(\\d+)")
    df["stage_id"] = pd.to_numeric(stage_match[0], errors="coerce")
    if df["stage_id"].isna().any():
        fallback = pd.Series(np.arange(1, len(df) + 1), index=df.index)
        df["stage_id"] = df["stage_id"].fillna(fallback)
    df["stage_id"] = df["stage_id"].astype(int)
    df["stage_short"] = "S" + df["stage_id"].astype(str)
    df["method"] = np.where(df["stage"].str.contains("MyMethod", case=False), "MyMethod", "RefTech")
    return df.sort_values(["stage_id", "fold"]).reset_index(drop=True)


def _fmt(x: float) -> str:
    return f"{x:.3f}"


def _binom_cdf(k: int, n: int) -> float:
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    total = 0
    for i in range(0, k + 1):
        total += comb(n, i)
    return total / (2**n)


def _sign_test_pvalue(diffs: np.ndarray) -> float:
    diffs = np.asarray(diffs)
    diffs = diffs[~np.isclose(diffs, 0.0)]
    n = int(diffs.size)
    if n == 0:
        return 1.0
    k = int((diffs > 0).sum())
    lower = _binom_cdf(k, n)
    upper = 1.0 - _binom_cdf(k - 1, n)
    return float(min(1.0, 2.0 * min(lower, upper)))


def _paired_pvalue(diffs: np.ndarray) -> tuple[str, float]:
    diffs = np.asarray(diffs)
    diffs = diffs[~np.isclose(diffs, 0.0)]
    if diffs.size == 0:
        return "sign", 1.0
    try:
        from scipy.stats import wilcoxon

        stat, p = wilcoxon(diffs)
        _ = stat
        return "wilcoxon", float(p)
    except Exception:
        return "sign", _sign_test_pvalue(diffs)


def _metric_delta_table(summary: pd.DataFrame, metric_cols: dict) -> pd.DataFrame:
    comparisons = [(1, 2, "S2-S1"), (3, 4, "S4-S3")]
    rows = []
    for base_stage, new_stage, label in comparisons:
        base = summary.loc[summary["stage_id"] == base_stage]
        new = summary.loc[summary["stage_id"] == new_stage]
        if base.empty or new.empty:
            continue
        base_row = base.iloc[0]
        new_row = new.iloc[0]
        for metric_name, (mean_col, _std_col) in metric_cols.items():
            if mean_col not in summary.columns:
                continue
            delta = float(new_row[mean_col] - base_row[mean_col])
            rows.append(
                {
                    "comparison": label,
                    "metric": metric_name,
                    "baseline": float(base_row[mean_col]),
                    "new": float(new_row[mean_col]),
                    "delta": delta,
                }
            )
    return pd.DataFrame(rows)


def _fold_delta_stats(folds: pd.DataFrame, metric_cols: dict) -> pd.DataFrame:
    if folds is None or folds.empty:
        return pd.DataFrame()
    comparisons = [(1, 2, "S2-S1"), (3, 4, "S4-S3")]
    rows = []
    for base_stage, new_stage, label in comparisons:
        base = folds.loc[folds["stage_id"] == base_stage]
        new = folds.loc[folds["stage_id"] == new_stage]
        if base.empty or new.empty:
            continue
        for metric_name, (mean_col, _std_col) in metric_cols.items():
            fold_col = mean_col.replace("mean_", "")
            if fold_col not in folds.columns:
                continue
            base_vals = base.set_index("fold")[fold_col]
            new_vals = new.set_index("fold")[fold_col]
            common = base_vals.index.intersection(new_vals.index)
            if common.empty:
                continue
            diffs = (new_vals.loc[common] - base_vals.loc[common]).to_numpy()
            test_name, p_val = _paired_pvalue(diffs)
            rows.append(
                {
                    "comparison": label,
                    "metric": metric_name,
                    "mean_delta": float(np.mean(diffs)),
                    "std_delta": float(np.std(diffs, ddof=1)) if diffs.size > 1 else 0.0,
                    "p_value": float(p_val),
                    "test": test_name,
                    "n_folds": int(diffs.size),
                }
            )
    return pd.DataFrame(rows)


def _build_thesis_summary(
    summary: pd.DataFrame,
    metric_cols: dict,
    fold_stats: pd.DataFrame,
) -> str:
    deltas = _metric_delta_table(summary, metric_cols)
    lines = []
    lines.append("Thesis Results Summary")
    lines.append("")
    lines.append("Evaluation setup:")
    lines.append("- GroupKFold split by race to prevent leakage across events.")
    lines.append("- Stages: S1 RefTech on RefData, S2 MyMethod on RefData, S3 RefTech on MyData+W, S4 MyMethod on MyData+W.")
    lines.append("- Metrics reported as mean +/- std across folds.")
    lines.append("")
    lines.append("Key improvements (mean deltas):")
    for _, row in deltas.iterrows():
        lines.append(
            f"- {row['comparison']} {row['metric']}: {row['delta']:+.6f} "
            f"(baseline {row['baseline']:.6f} -> {row['new']:.6f})"
        )
    if not fold_stats.empty:
        lines.append("")
        lines.append("Fold-level paired tests:")
        for _, row in fold_stats.iterrows():
            lines.append(
                f"- {row['comparison']} {row['metric']}: mean delta {row['mean_delta']:+.6f} "
                f"(std {row['std_delta']:.6f}), p={row['p_value']:.4f} ({row['test']}, n={row['n_folds']})"
            )
    lines.append("")
    lines.append("Interpretation:")
    lines.append("- MyMethod shows consistent but modest gains over RefTech across most metrics.")
    lines.append("- Improvements are small; report them as incremental performance gains with leakage-safe validation.")
    lines.append("")
    lines.append("Limitations:")
    lines.append("- Effect sizes are small; results are sensitive to race-specific variance.")
    lines.append("- More data or additional signals may be required for larger gains.")
    return "\n".join(lines)


def _delta_badge(delta: float) -> str:
    if delta > 0.001:
        return f"<span class='delta-up'>UP {delta:+.3f}</span>"
    if delta < -0.001:
        return f"<span class='delta-down'>DOWN {delta:+.3f}</span>"
    return f"<span class='delta-flat'>FLAT {delta:+.3f}</span>"


def _metric_guide() -> dict[str, str]:
    return {
        "F1": "Balance of precision and recall (higher is better).",
        "F2": "Recall-weighted F1 (prioritizes catching pit-stops).",
        "Precision": "How many predicted pit-stops were correct.",
        "Recall": "How many true pit-stops were detected.",
        "PR-AUC": "Overall precision-recall tradeoff across thresholds.",
    }


def _pick_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def _derive_weather_label(df: pd.DataFrame) -> pd.Series:
    if "RainFlag_prev" in df.columns:
        return np.where(df["RainFlag_prev"] > 0, "Wet", "Dry")
    if "Rainfall_prev" in df.columns:
        return np.where(df["Rainfall_prev"] > 0, "Wet", "Dry")
    if "HumidityRain_prev" in df.columns:
        return np.where(df["HumidityRain_prev"] > 0, "Wet", "Dry")
    return pd.Series(["Unknown"] * len(df), index=df.index)


def _format_seconds(val: float | None) -> str:
    if val is None or np.isnan(val):
        return "N/A"
    return f"{val:.1f}s"


def _first_valid(row: pd.Series, keys: list[str]) -> float | str | None:
    for key in keys:
        if key in row and pd.notna(row[key]):
            return row[key]
    return None


def _fmt_num(val: float | str | None, decimals: int = 1) -> str:
    if val is None:
        return "N/A"
    try:
        out = float(val)
        if not np.isfinite(out):
            return "N/A"
        return f"{out:.{decimals}f}"
    except Exception:
        return "N/A"


def _fmt_int(val: float | str | None) -> str:
    if val is None:
        return "N/A"
    try:
        out = int(round(float(val)))
        return str(out)
    except Exception:
        return "N/A"


def _fmt_pct(val: float | str | None) -> str:
    if val is None:
        return "N/A"
    try:
        out = float(val)
        if not np.isfinite(out):
            return "N/A"
        if out <= 1.5:
            out *= 100.0
        return f"{out:.0f}%"
    except Exception:
        return "N/A"


def _fmt_flag(val: float | str | None) -> str:
    if val is None:
        return "N/A"
    try:
        return "ON" if float(val) > 0 else "OFF"
    except Exception:
        return "N/A"


def _telemetry_sections(row: pd.Series, payload: dict) -> str:
    def _item(label: str, value: str) -> str:
        return f"<div class='telemetry-item'>{label}<span>{value}</span></div>"

    def _section(title: str, items: list[tuple[str, str]]) -> str:
        if not items:
            return ""
        html_items = "".join(_item(label, value) for label, value in items)
        return (
            "<div class='telemetry-section'>"
            f"<div class='telemetry-title'>{title}</div>"
            f"<div class='telemetry-grid'>{html_items}</div>"
            "</div>"
        )

    race_items: list[tuple[str, str]] = []
    race_items.append(("Race progress", payload.get("progress_text", "N/A")))
    race_items.append(("Position", _fmt_int(_first_valid(row, ["Position_prev", "position"]))))
    race_items.append(("Track deg", _fmt_int(_first_valid(row, ["track_deg_category"]))))
    race_items.append(("SC", _fmt_flag(_first_valid(row, ["sc_active_prev", "sc_active"]))))
    race_items.append(("VSC", _fmt_flag(_first_valid(row, ["vsc_active_prev", "vsc_active"]))))

    strategy_items: list[tuple[str, str]] = []
    strategy_items.append(("Pit window", payload.get("pit_window_text", "N/A")))
    strategy_items.append(("Pit stops", _fmt_int(_first_valid(row, ["pitstops_so_far_prev", "pitstops_so_far"]))))
    strategy_items.append(("Pit remaining", _fmt_int(_first_valid(row, ["pitstops_remaining"]))))
    strategy_items.append(("Undercut", _fmt_num(_first_valid(row, ["undercut_potential_prev"])) ))
    strategy_items.append(("Gap after pit", _format_seconds(_first_valid(row, ["gap_after_pit_vs_behind_prev"]))))

    tyre_items: list[tuple[str, str]] = []
    tyre_items.append(("Tyre age", payload.get("tire_text", "N/A")))
    tyre_items.append(("Tyre wear", _fmt_pct(payload.get("tire_wear_pct"))))
    tyre_items.append(("Stint laps", _fmt_int(_first_valid(row, ["stint_laps_prev", "stint_laps"]))))
    compound_text = payload.get("compound_text")
    if compound_text:
        tyre_items.append(("Compound", str(compound_text)))

    pace_items: list[tuple[str, str]] = []
    pace_items.append(("Relative pace", _fmt_num(_first_valid(row, ["relative_pace_prev", "relative_pace"])) ))
    pace_items.append(("Delta best", _format_seconds(_first_valid(row, ["delta_best_so_far_prev", "delta_best_race"]))))
    pace_items.append(("Delta interval", _format_seconds(_first_valid(row, ["delta_interval_prev", "delta_interval"]))))
    pace_items.append(("Gap leader", _format_seconds(_first_valid(row, ["gap_to_leader_prev", "gap"])) ))
    pace_items.append(("Gap front", _format_seconds(_first_valid(row, ["gap_to_front_prev", "interval"])) ))
    pace_items.append(("Gap behind", _format_seconds(_first_valid(row, ["gap_to_behind_prev", "gap_to_behind"])) ))

    weather_items: list[tuple[str, str]] = []
    weather_items.append(("Air temp", _fmt_num(_first_valid(row, ["AirTemp_prev", "AirTemp"])) ))
    weather_items.append(("Track temp", _fmt_num(_first_valid(row, ["TrackTemp_prev", "TrackTemp"])) ))
    weather_items.append(("Humidity", _fmt_pct(_first_valid(row, ["Humidity_prev", "Humidity"])) ))
    weather_items.append(("Pressure", _fmt_num(_first_valid(row, ["Pressure_prev", "Pressure"])) ))
    weather_items.append(("Wind speed", _fmt_num(_first_valid(row, ["WindSpeed_prev", "WindSpeed"])) ))
    weather_items.append(("Wind dir", _fmt_int(_first_valid(row, ["WindDirection_prev", "WindDirection"])) ))
    weather_items.append(("Rainfall", _fmt_num(_first_valid(row, ["Rainfall_prev", "Rainfall"])) ))

    sections = [
        _section("Race", [(l, v) for l, v in race_items if v != "N/A"]),
        _section("Strategy", [(l, v) for l, v in strategy_items if v != "N/A"]),
        _section("Tyre", [(l, v) for l, v in tyre_items if v != "N/A"]),
        _section("Pace", [(l, v) for l, v in pace_items if v != "N/A"]),
        _section("Weather", [(l, v) for l, v in weather_items if v != "N/A"]),
    ]
    return "".join([s for s in sections if s])


def _pit_window_series(df: pd.DataFrame) -> pd.Series:
    for cand in ("in_pit_window_prev", "in_pit_window", "pit_window_prev", "pit_window"):
        if cand in df.columns:
            return pd.to_numeric(df[cand], errors="coerce").fillna(0) > 0
    return pd.Series([False] * len(df), index=df.index)


def _tire_wear_series(df: pd.DataFrame, tire_max: float) -> pd.Series:
    if "tyre_wear_pct_prev" in df.columns:
        wear = pd.to_numeric(df["tyre_wear_pct_prev"], errors="coerce")
        wear = wear.where(wear <= 1.5, wear / 100.0)
        return wear
    if "tyre_wear_pct" in df.columns:
        wear = pd.to_numeric(df["tyre_wear_pct"], errors="coerce")
        wear = wear.where(wear <= 1.5, wear / 100.0)
        return wear
    if "tireage" in df.columns:
        age = pd.to_numeric(df["tireage"], errors="coerce")
        return age / float(max(1.0, tire_max))
    if "stint_laps_prev" in df.columns:
        age = pd.to_numeric(df["stint_laps_prev"], errors="coerce")
        return age / float(max(1.0, tire_max))
    return pd.Series([np.nan] * len(df), index=df.index)


def _apply_cooldown(
    df: pd.DataFrame,
    action_col: str,
    lap_col: str | None,
    group_cols: list[str],
    cooldown_laps: int,
) -> pd.DataFrame:
    if lap_col is None or lap_col not in df.columns:
        return df

    def _filter(group: pd.DataFrame) -> pd.DataFrame:
        if lap_col not in group.columns:
            return group
        group = group.sort_values(lap_col)
        last_pit = None
        actions: list[bool] = []
        for _, row in group.iterrows():
            lap_val = row.get(lap_col)
            action = bool(row.get(action_col, False))
            if pd.isna(lap_val):
                actions.append(False)
                continue
            lap_val = int(lap_val)
            if action and last_pit is not None and (lap_val - last_pit) <= cooldown_laps:
                action = False
            if action:
                last_pit = lap_val
            actions.append(action)
        group[action_col] = actions
        return group

    if group_cols:
        return df.groupby(group_cols, group_keys=False).apply(_filter)
    return _filter(df)


def _strategy_impact(
    df: pd.DataFrame,
    features: list[str],
    model: Pipeline,
    calibrator: LogisticRegression | None,
    threshold: float,
    lap_col: str | None,
    tire_max: float,
    lookahead_laps: int,
    group_cols: list[str],
    sample_limit: int = 12000,
) -> tuple[dict, pd.DataFrame] | None:
    if df.empty or not features:
        return None
    df_bt = df.copy()
    if len(df_bt) > sample_limit:
        df_bt = df_bt.sample(n=sample_limit, random_state=42)

    probs_raw = model.predict_proba(df_bt[features])[:, 1]
    if calibrator is not None:
        try:
            probs = calibrator.predict_proba(probs_raw.reshape(-1, 1))[:, 1]
        except Exception:
            probs = probs_raw
    else:
        probs = probs_raw

    pit_open = _pit_window_series(df_bt)
    wear = _tire_wear_series(df_bt, tire_max)
    model_action = pit_open & (probs >= threshold)
    baseline_action = pit_open & (wear >= 0.7)

    df_bt = df_bt.assign(
        prob=probs,
        action_model=model_action,
        action_base=baseline_action,
    )
    df_bt = _apply_cooldown(df_bt, "action_model", lap_col, group_cols, cooldown_laps=4)
    df_bt = _apply_cooldown(df_bt, "action_base", lap_col, group_cols, cooldown_laps=4)

    def _net_gain(row: pd.Series) -> float:
        lap_val = None
        if lap_col and lap_col in row and pd.notna(row[lap_col]):
            try:
                lap_val = int(row[lap_col])
            except Exception:
                lap_val = None
        payload = _demo_decision(
            row,
            0.5,
            0.5,
            lap_val,
            lap_col,
            tire_max,
            lookahead_laps,
        )
        return float(payload["net_gain_sec"])

    df_bt["net_gain_sec"] = df_bt.apply(_net_gain, axis=1)
    df_bt["impact_model"] = df_bt["net_gain_sec"] * df_bt["action_model"].astype(float)
    df_bt["impact_base"] = df_bt["net_gain_sec"] * df_bt["action_base"].astype(float)

    if not group_cols:
        df_bt["__group__"] = 0
        group_cols = ["__group__"]

    grouped = df_bt.groupby(group_cols)[["impact_model", "impact_base"]].sum()
    grouped["delta"] = grouped["impact_model"] - grouped["impact_base"]
    grouped = grouped.reset_index()

    delta = grouped["delta"]
    summary = {
        "avg_delta": float(delta.mean()) if not delta.empty else 0.0,
        "median_delta": float(delta.median()) if not delta.empty else 0.0,
        "improve_rate": float((delta > 0).mean()) if not delta.empty else 0.0,
        "groups": int(len(grouped)),
        "rows": int(len(df_bt)),
    }
    return summary, grouped


def _fbeta_score(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom == 0.0:
        return 0.0
    return (1.0 + b2) * precision * recall / denom


def _eval_probs(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> dict:
    preds = probs >= float(threshold)
    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    f2 = _fbeta_score(precision, recall, beta=2.0)
    pr_auc = average_precision_score(y_true, probs) if y_true.size else 0.0
    return {
        "F1": float(f1),
        "F2": float(f2),
        "Precision": float(precision),
        "Recall": float(recall),
        "PR-AUC": float(pr_auc),
    }


def _make_sklearn_pipeline(
    df: pd.DataFrame, features: list[str], estimator: object
) -> Pipeline:
    num_cols = [c for c in features if df[c].dtype != "object"]
    cat_cols = [c for c in features if c not in num_cols]
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ],
        remainder="drop",
    )
    return Pipeline([("pre", pre), ("clf", estimator)])


def _sign_test_pvalue(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    k = min(wins, losses)
    p = sum(comb(n, i) for i in range(0, k + 1)) / (2**n)
    return float(min(1.0, 2 * p))


def _group_f1(
    y_true: np.ndarray, probs: np.ndarray, threshold: float, groups: np.ndarray
) -> pd.Series:
    preds = probs >= float(threshold)
    df_tmp = pd.DataFrame({"y": y_true, "pred": preds, "group": groups})
    scores = {}
    for g, sub in df_tmp.groupby("group"):
        tp = int(((sub["pred"] == 1) & (sub["y"] == 1)).sum())
        fp = int(((sub["pred"] == 1) & (sub["y"] == 0)).sum())
        fn = int(((sub["pred"] == 0) & (sub["y"] == 1)).sum())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        scores[g] = _fbeta_score(precision, recall, beta=1.0)
    return pd.Series(scores)


def _baseline_compare(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    model: Pipeline,
    calibrator: LogisticRegression | None,
    threshold: float,
    group_col: str | None,
    sample_limit: int = 20000,
) -> tuple[pd.DataFrame, dict[str, float | None]] | None:
    if train_df.empty or test_df.empty or not features:
        return None
    tr = train_df.copy()
    te = test_df.copy()
    if len(tr) > sample_limit:
        tr = tr.sample(n=sample_limit, random_state=42)
    if len(te) > sample_limit:
        te = te.sample(n=sample_limit, random_state=42)

    y_tr = tr["decide_pitstop"].astype(int).values
    y_te = te["decide_pitstop"].astype(int).values
    if np.unique(y_tr).size < 2 or np.unique(y_te).size < 2:
        return None

    xgb_raw = model.predict_proba(te[features])[:, 1]
    if calibrator is not None:
        try:
            xgb_probs = calibrator.predict_proba(xgb_raw.reshape(-1, 1))[:, 1]
        except Exception:
            xgb_probs = xgb_raw
    else:
        xgb_probs = xgb_raw

    rows = []
    rows.append(("XGBoost", _eval_probs(y_te, xgb_probs, threshold)))

    def _fit_baseline(estimator: object, name: str) -> tuple[np.ndarray, float]:
        pipe = _make_sklearn_pipeline(tr, features, estimator)
        pipe.fit(tr[features], y_tr)
        tr_probs = pipe.predict_proba(tr[features])[:, 1]
        th = _select_threshold(y_tr, tr_probs, beta=1.0)
        te_probs = pipe.predict_proba(te[features])[:, 1]
        rows.append((name, _eval_probs(y_te, te_probs, th)))
        return te_probs, th

    lr_probs, lr_th = _fit_baseline(
        LogisticRegression(max_iter=200, class_weight="balanced"),
        "LogReg",
    )
    rf_probs, rf_th = _fit_baseline(
        RandomForestClassifier(
            n_estimators=200,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
        "RandomForest",
    )

    metrics_df = pd.DataFrame(
        [
            {"Model": name, **metrics}
            for name, metrics in rows
        ]
    )

    sign_stats: dict[str, float | None] = {}
    if group_col and group_col in te.columns:
        groups = te[group_col].astype(str).values
        f1_xgb = _group_f1(y_te, xgb_probs, threshold, groups)
        f1_lr = _group_f1(y_te, lr_probs, lr_th, groups)
        f1_rf = _group_f1(y_te, rf_probs, rf_th, groups)

        def _wins(a: pd.Series, b: pd.Series) -> tuple[int, int]:
            aligned = a.align(b, join="inner")
            diff = aligned[0] - aligned[1]
            wins = int((diff > 0).sum())
            losses = int((diff < 0).sum())
            return wins, losses

        wins_lr, losses_lr = _wins(f1_xgb, f1_lr)
        wins_rf, losses_rf = _wins(f1_xgb, f1_rf)
        sign_stats = {
            "xgb_vs_lr_win_rate": float(wins_lr / max(1, wins_lr + losses_lr)),
            "xgb_vs_lr_p": _sign_test_pvalue(wins_lr, losses_lr),
            "xgb_vs_rf_win_rate": float(wins_rf / max(1, wins_rf + losses_rf)),
            "xgb_vs_rf_p": _sign_test_pvalue(wins_rf, losses_rf),
        }

    return metrics_df, sign_stats


def _decision_strength(prob: float, threshold: float) -> tuple[str, float]:
    gap = float(abs(prob - threshold))
    if gap >= 0.15:
        return "Strong", gap
    if gap >= 0.07:
        return "Medium", gap
    return "Weak", gap


def _reliability_label(rows: int) -> str:
    if rows >= 5000:
        return "High"
    if rows >= 1500:
        return "Medium"
    return "Low"


def _reason_phrase(reason_text: str) -> str:
    mapping = {
        "WINDOW": "window open",
        "WEAR": "high tyre wear",
        "SC": "safety car",
        "VSC": "virtual safety car",
        "LATE": "late race",
        "NOWINDOW": "window closed",
        "COST-": "costly stop",
        "COST+": "time gain",
        "COST-HOLD": "cost warning",
        "CORE": "core signals",
    }
    reasons = []
    for token in reason_text.split("+"):
        token = token.strip()
        if not token:
            continue
        reasons.append(mapping.get(token, token.lower()))
    if not reasons:
        return "core signals"
    return ", ".join(reasons)


def _decision_sentence(payload: dict, prob: float, threshold: float) -> str:
    window = str(payload.get("pit_window_text", "N/A"))
    decision = str(payload.get("decision", "STAY OUT"))
    reasons = _reason_phrase(str(payload.get("reason_text", "")))
    if window == "OPEN":
        if prob >= threshold:
            return f"Window open and confidence above threshold: BOX (signals: {reasons})."
        return f"Window open but confidence below threshold: STANDBY (signals: {reasons})."
    if prob >= threshold:
        return f"Window closed, so stay out for now (signals: {reasons})."
    return f"Window closed and confidence low: STAY OUT (signals: {reasons})."


def _key_takeaways(summary: pd.DataFrame, metric_col: str) -> str:
    def _get_stage(stage_id: int) -> float | None:
        row = summary.loc[summary["stage_id"] == stage_id]
        if row.empty:
            return None
        return float(row.iloc[0][metric_col])

    s1 = _get_stage(1)
    s2 = _get_stage(2)
    s3 = _get_stage(3)
    s4 = _get_stage(4)

    def _line(a: float | None, b: float | None, label: str) -> str:
        if a is None or b is None:
            return f"{label}: data unavailable."
        delta = b - a
        trend = "improves" if delta > 0.001 else "drops" if delta < -0.001 else "matches"
        return f"{label}: {trend} by {delta:+.3f}."

    line1 = _line(s1, s2, "S2 vs S1 (RefData)")
    line2 = _line(s3, s4, "S4 vs S3 (MyData+W)")
    return f"{line1} {line2}"


def _example_explainer(summary: pd.DataFrame, metric_col: str, std_col: str) -> str:
    if summary.empty:
        return "Example: no data loaded."
    row = summary.iloc[0]
    metric_name = metric_col.replace("mean_", "").upper()
    return (
        f"{row['stage_short']} {metric_name} "
        f"{row[metric_col]:.3f} +/- {row[std_col]:.3f} (mean/std)."
    )


def _decision_example(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "Example output: Driver needs to pit (illustrative)."

    prefer = summary.loc[summary["stage"].str.contains("MyMethod", case=False, na=False)]
    row = prefer.iloc[-1] if not prefer.empty else summary.iloc[-1]

    if "mean_threshold" in summary.columns:
        threshold = float(row["mean_threshold"])
    elif "threshold" in summary.columns:
        threshold = float(row["threshold"])
    else:
        threshold = 0.5

    example_prob = min(0.95, max(0.05, threshold + 0.12))
    decision = "PIT" if example_prob >= threshold else "STAY OUT"
    verb = "needs to pit" if decision == "PIT" else "should stay out"
    return (
        f"Example output: Driver {verb} "
        f"(prob {example_prob:.2f} >= threshold {threshold:.2f})."
    )


def _render_track_demo(
    panel_label: str,
    driver: str,
    lap_text: str,
    circuit_text: str,
    weather_text: str,
    decision: str,
    proba: float,
    threshold: float,
    race_progress: float | None,
    urgency: float,
    pit_window_text: str,
    pit_target_text: str,
    tire_text: str,
    tire_wear_pct: float | None,
    gap_text: str,
    sc_text: str,
    progress_text: str,
    gap_trend_text: str,
    overtake_mode: str | None,
    reason_text: str,
    stint_reset: bool,
    lap_current: int | None,
    lap_min: int | None,
    lap_max: int | None,
    window_start: int | None,
    window_end: int | None,
    rec_lap: int | None,
) -> str:
    progress = 0.5
    if race_progress is not None and not np.isnan(race_progress):
        progress = float(np.clip(race_progress, 0.0, 1.0))

    track_start = 50
    track_len = 700
    car_w = 36
    x_main = track_start + progress * (track_len - car_w)
    rival_progress = min(1.0, progress + 0.08)
    x_rival = track_start + rival_progress * (track_len - car_w)

    lane_main_y = 70
    lane_pit_y = 118
    is_pit_call = decision.startswith("PIT") or decision.startswith("BOX")
    is_standby = decision.startswith("STANDBY")
    main_y = lane_pit_y if is_pit_call else lane_main_y
    if is_pit_call:
        decision_pill = "decision-pit"
    elif is_standby:
        decision_pill = "decision-wait"
    else:
        decision_pill = "decision-stay"
    pit_open = pit_window_text.upper() == "OPEN"
    pit_fill = "#ff9d2b" if pit_open else "#10151d"
    pit_stroke = "#ff9d2b" if pit_open else "#2a3342"
    pit_lane_x = 120.0
    pit_lane_end = 700.0
    pit_lane_w = pit_lane_end - pit_lane_x
    pit_window_x = 600.0
    pit_window_w = 120.0
    pit_in_x = 560.0
    pit_out_x = 700.0
    pit_target_x = 620.0
    pit_window_label_x = pit_window_x + 12.0
    pit_target_label_x = pit_window_x + 12.0
    if (
        lap_min is not None
        and lap_max is not None
        and lap_max > lap_min
        and window_start is not None
        and window_end is not None
    ):
        def _ratio(lap_val: int) -> float:
            return float(np.clip((lap_val - lap_min) / (lap_max - lap_min), 0.0, 1.0))

        start_ratio = _ratio(int(window_start))
        end_ratio = _ratio(int(window_end))
        if end_ratio < start_ratio:
            start_ratio, end_ratio = end_ratio, start_ratio
        pit_window_x = pit_lane_x + start_ratio * pit_lane_w
        pit_window_w = max(18.0, (end_ratio - start_ratio) * pit_lane_w)
        pit_window_x = float(np.clip(pit_window_x, pit_lane_x, pit_lane_end - pit_window_w))
        pit_window_w = float(min(pit_window_w, pit_lane_end - pit_window_x))
        pit_in_x = pit_window_x
        pit_out_x = pit_window_x + pit_window_w
        target_lap = int(window_start)
        if pit_target_text == "NOW" and lap_current is not None:
            target_lap = int(lap_current)
        target_ratio = _ratio(target_lap)
        pit_target_x = pit_lane_x + target_ratio * pit_lane_w
        pit_target_x = float(np.clip(pit_target_x, pit_lane_x + 2.0, pit_lane_end - 2.0))
        pit_window_label_x = pit_window_x + min(12.0, pit_window_w * 0.3)
        pit_target_label_x = float(np.clip(pit_target_x - 18.0, pit_lane_x, pit_lane_end - 70.0))
    main_anim = "car-attack" if overtake_mode == "attack" else ""
    rival_anim = "car-press" if overtake_mode == "defend" else ""
    wear_pct = 0.0
    wear_label = "N/A"
    if tire_wear_pct is not None and not np.isnan(tire_wear_pct):
        wear_pct = float(np.clip(tire_wear_pct, 0.0, 1.0))
        wear_label = f"{wear_pct * 100:.0f}%"

    timeline_html = ""
    if (
        lap_current is not None
        and lap_min is not None
        and lap_max is not None
        and lap_max > lap_min
    ):
        def _pos(lap_val: int) -> float:
            return float(np.clip((lap_val - lap_min) / (lap_max - lap_min), 0.0, 1.0) * 100.0)

        current_pos = _pos(int(lap_current))
        window_start_pos = _pos(window_start) if window_start is not None else None
        window_end_pos = _pos(window_end) if window_end is not None else None
        rec_pos = _pos(rec_lap) if rec_lap is not None else None

        marker_html = f"<div class='timeline-marker timeline-current' style='left:{current_pos:.1f}%;'></div>"
        if window_start_pos is not None:
            marker_html += f"<div class='timeline-marker timeline-window' style='left:{window_start_pos:.1f}%;'></div>"
        if window_end_pos is not None:
            marker_html += f"<div class='timeline-marker timeline-window' style='left:{window_end_pos:.1f}%;'></div>"
        if rec_pos is not None:
            marker_html += f"<div class='timeline-marker timeline-rec' style='left:{rec_pos:.1f}%;'></div>"

        timeline_html = (
            "<div class='timeline-wrap'>"
            "<div class='timeline-title'>Pit Window Timeline</div>"
            "<div class='timeline'>"
            f"{marker_html}"
            "</div>"
            f"<div class='timeline-labels'><span>L{lap_min}</span><span>L{lap_max}</span></div>"
            "</div>"
        )

    stint_chip = "<!-- -->"
    tire_note = ""
    if stint_reset:
        stint_chip = "<div class='signal-chip'>STINT<strong>NEW</strong></div>"
        tire_note = " - New tyres"

    return f"""
<div class='track-card'>
  <div class='track-header'>
    <div>
      <div class='track-title'>{panel_label} Strategy</div>
      <div class='track-sub'>{driver} | {lap_text} | {circuit_text} | {weather_text}</div>
    </div>
    <div class='track-pills'>
      <span class='decision-pill {decision_pill}'>{decision}</span>
      <span class='metric-pill'>P {proba:.2f}</span>
      <span class='metric-pill'>T {threshold:.2f}</span>
    </div>
  </div>
  <svg class='track-svg' viewBox='0 0 800 180' preserveAspectRatio='xMidYMid meet'>
    <rect x='20' y='60' width='760' height='34' rx='16' fill='#1a222f' stroke='#2a3342'/>
    <rect x='20' y='110' width='760' height='22' rx='11' fill='#10151d' stroke='#2a3342'/>
    <rect x='20' y='60' width='6' height='34' fill='#ff2b2b'/>
    <text x='30' y='54' font-size='11' fill='#7a8796'>TRACK</text>
    <text x='30' y='126' font-size='11' fill='#7a8796'>PIT</text>
    <rect x='{pit_window_x:.1f}' y='110' width='{pit_window_w:.1f}' height='22' rx='10' fill='{pit_fill}' stroke='{pit_stroke}'/>
    <rect x='{pit_target_x - 1:.1f}' y='110' width='2' height='22' fill='#ffd15a' opacity='0.85'/>
    <text x='{pit_window_label_x:.1f}' y='126' font-size='10' fill='#ff9d2b'>PIT WINDOW</text>
    <text x='{pit_target_label_x:.1f}' y='102' font-size='10' fill='#ff9d2b'>TARGET {pit_target_text}</text>
    <polygon points='{pit_in_x:.1f},110 {pit_in_x + 10:.1f},110 {pit_in_x + 5:.1f},100' fill='#ff9d2b'/>
    <text x='{pit_in_x - 8:.1f}' y='104' font-size='10' fill='#ff9d2b'>IN</text>
    <polygon points='{pit_out_x:.1f},110 {pit_out_x + 10:.1f},110 {pit_out_x + 5:.1f},100' fill='#ff9d2b'/>
    <text x='{pit_out_x - 8:.1f}' y='104' font-size='10' fill='#ff9d2b'>OUT</text>
    <rect x='740' y='110' width='40' height='22' rx='6' fill='rgba(255,43,43,0.18)' stroke='rgba(255,43,43,0.5)'/>
    <text x='748' y='126' font-size='10' fill='#ff6b6b'>BOX</text>
    <g transform='translate({x_rival:.1f},{lane_main_y:.1f})'>
      <g class='{rival_anim}'>
        <rect x='0' y='4' width='{car_w}' height='12' rx='3' fill='#2c3545' stroke='#111'/>
        <rect x='6' y='0' width='24' height='6' rx='2' fill='#3b4558'/>
        <rect x='4' y='16' width='28' height='4' rx='2' fill='#0b0f16'/>
        <circle cx='6' cy='16' r='2' fill='#0b0f16'/>
        <circle cx='30' cy='16' r='2' fill='#0b0f16'/>
      </g>
    </g>
    <g transform='translate({x_main:.1f},{main_y:.1f})'>
      <g class='{main_anim}'>
        <rect x='0' y='4' width='{car_w}' height='12' rx='3' fill='#ff2b2b' stroke='#111'/>
        <rect x='6' y='0' width='24' height='6' rx='2' fill='#ff6b6b'/>
        <rect x='4' y='16' width='28' height='4' rx='2' fill='#0b0f16'/>
        <circle cx='6' cy='16' r='2' fill='#0b0f16'/>
        <circle cx='30' cy='16' r='2' fill='#0b0f16'/>
      </g>
    </g>
  </svg>
  {timeline_html}
  <div class='track-meter'><div class='track-meter-fill' style='width:{urgency * 100:.0f}%;'></div></div>
  <div class='signal-row'>
    <div class='signal-chip'>PROG<strong>{progress_text}</strong></div>
    <div class='signal-chip'>PIT<strong>{pit_window_text}</strong></div>
    <div class='signal-chip'>TARGET<strong>{pit_target_text}</strong></div>
    <div class='signal-chip'>TIRE<strong>{tire_text}</strong></div>
    {stint_chip}
    <div class='signal-chip'>GAP<strong>{gap_text}</strong></div>
    <div class='signal-chip'>TREND<strong>{gap_trend_text}</strong></div>
    <div class='signal-chip'>TRACK<strong>{sc_text}</strong></div>
    <div class='signal-chip'>WHY<strong>{reason_text}</strong></div>
  </div>
  <div class='tire-gauge'><div class='tire-fill' style='width:{wear_pct * 100:.0f}%;'></div></div>
  <div class='track-sub'>Tire wear {wear_label}{tire_note}</div>
  <div class='track-legend'>
    <span><i class='swatch swatch-my'></i>My car</span>
    <span><i class='swatch swatch-ref'></i>Field</span>
  </div>
</div>
"""


def _resolve_stage4_dataset() -> Path | None:
    primary = ROOT / "personal_datasets" / "fastf1_strategy_dataset.csv"
    secondary = ROOT / "personal_datasets" / "fastf1_strategy_weather_dataset.csv"
    fallback = ROOT / "data" / "strategy_weather_dataset.csv"
    for path in (primary, secondary, fallback):
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def _load_stage4_data() -> pd.DataFrame:
    dataset = _resolve_stage4_dataset()
    if dataset is None:
        return pd.DataFrame()
    df = add_feature_engineering(safe_numeric(load_csv(dataset)))
    return df


def _apply_feature_allowlist(features: list[str]) -> list[str]:
    allow = [f for f in DEMO_SHARED_FEATURES if f in features]
    return allow


def _align_features(features: list[str], *dfs: pd.DataFrame) -> list[str]:
    aligned = []
    for feat in features:
        keep = True
        for df in dfs:
            if feat not in df.columns:
                keep = False
                break
            if df[feat].dropna().empty:
                keep = False
                break
        if keep:
            aligned.append(feat)
    return aligned


def _select_threshold(y_true: np.ndarray, probs: np.ndarray, beta: float = 1.0) -> float:
    if y_true.size == 0 or probs.size == 0:
        return 0.5
    thresholds = np.linspace(0.05, 0.95, 19)
    best_t = 0.5
    best_score = -1.0
    beta2 = beta * beta
    for t in thresholds:
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


def _split_calibration(
    df: pd.DataFrame, y: np.ndarray, group_col: str | None
) -> tuple[np.ndarray, np.ndarray] | None:
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


def _apply_scale_pos_weight(params: dict, y: np.ndarray) -> dict:
    out = dict(params)
    mult = float(out.pop("scale_pos_weight_multiplier", 1.0))
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    spw = float(neg / pos) if pos > 0 else 1.0
    out["scale_pos_weight"] = spw * mult
    return out


def _load_best_params() -> dict:
    if not STAGE4_BEST_PARAMS.exists():
        return {}
    try:
        raw = STAGE4_BEST_PARAMS.read_text(encoding="utf-8")
        data = json.loads(raw)
        params = data.get("params") if isinstance(data, dict) else None
        if isinstance(params, dict):
            return params
    except Exception:
        return {}
    return {}


def _split_groupkfold(
    df: pd.DataFrame, group_col: str, n_splits: int, fold_id: int
) -> tuple[np.ndarray, np.ndarray] | None:
    if group_col not in df.columns:
        return None
    groups = df[group_col]
    if groups.nunique() < 2:
        return None
    try:
        gkf = GroupKFold(n_splits=n_splits)
        splits = list(gkf.split(df, groups=groups))
        if not splits:
            return None
        fold_index = max(0, min(fold_id - 1, len(splits) - 1))
        return splits[fold_index]
    except Exception:
        return None


def _dataset_stats(df: pd.DataFrame, target_col: str, group_col: str | None) -> dict:
    rows = int(len(df))
    pos_rate = None
    if target_col in df.columns and rows > 0:
        pos = float(pd.to_numeric(df[target_col], errors="coerce").fillna(0).mean())
        pos_rate = pos
    groups = int(df[group_col].nunique()) if group_col and group_col in df.columns else None
    return {"rows": rows, "pos_rate": pos_rate, "groups": groups}


def _lap_range(df: pd.DataFrame, lap_col: str | None) -> tuple[int, int] | None:
    if not lap_col or lap_col not in df.columns or df.empty:
        return None
    laps = pd.to_numeric(df[lap_col], errors="coerce").dropna()
    if laps.empty:
        return None
    return int(laps.min()), int(laps.max())


def _pit_window_bounds(df: pd.DataFrame, lap_col: str | None) -> tuple[int, int] | None:
    if not lap_col or lap_col not in df.columns or df.empty:
        return None
    window_col = None
    for cand in ("in_pit_window_prev", "in_pit_window", "pit_window_prev", "pit_window"):
        if cand in df.columns:
            window_col = cand
            break
    if window_col is None:
        return None
    mask = pd.to_numeric(df[window_col], errors="coerce").fillna(0) > 0
    if not mask.any():
        return None
    laps = pd.to_numeric(df.loc[mask, lap_col], errors="coerce").dropna()
    if laps.empty:
        return None
    return int(laps.min()), int(laps.max())


def _demo_decision(
    row: pd.Series,
    proba: float,
    threshold: float,
    lap_value: int | float | None,
    lap_col: str | None,
    tire_max: float,
    lookahead_laps: int,
) -> dict:
    def _get_val(keys: list[str]) -> float | None:
        for key in keys:
            if key in row and pd.notna(row[key]):
                return float(row[key])
        return None

    race_progress = _get_val(["race_progress", "race_progress_prev"])
    if race_progress is None and "nolaps_prev" in row and lap_col and lap_col in row:
        try:
            race_progress = float(row[lap_col]) / float(row["nolaps_prev"])
        except Exception:
            race_progress = None

    pit_window_val = _get_val(["in_pit_window", "in_pit_window_prev", "pit_window", "pit_window_prev"])
    pit_window_text = "OPEN" if pit_window_val is not None and pit_window_val > 0 else "CLOSED"
    if pit_window_val is None:
        pit_window_text = "N/A"

    sc_flag = _get_val(["sc_active", "sc_active_prev"])
    vsc_flag = _get_val(["vsc_active", "vsc_active_prev"])
    if sc_flag is not None and sc_flag > 0:
        sc_text = "SC"
    elif vsc_flag is not None and vsc_flag > 0:
        sc_text = "VSC"
    else:
        sc_text = "CLEAR"

    compound_text = None
    compound_col = None
    for cand in (
        "compound",
        "Compound",
        "tyre_compound",
        "tire_compound",
        "compound_prev",
        "compound_rank",
        "relative_compound",
    ):
        if cand in row and pd.notna(row[cand]):
            compound_col = cand
            val = row[cand]
            if isinstance(val, str):
                name = val.strip().upper()
                if name.startswith("S"):
                    compound_text = "SOFT"
                elif name.startswith("M"):
                    compound_text = "MEDIUM"
                elif name.startswith("H"):
                    compound_text = "HARD"
                elif "INTER" in name:
                    compound_text = "INTER"
                elif "WET" in name:
                    compound_text = "WET"
                else:
                    compound_text = name
            else:
                try:
                    num = float(val)
                    if compound_col and ("rank" in compound_col or "relative" in compound_col):
                        if num <= 1.5:
                            compound_text = "SOFT"
                        elif num <= 2.5:
                            compound_text = "MEDIUM"
                        else:
                            compound_text = "HARD"
                    else:
                        compound_text = f"C{int(round(num))}"
                except Exception:
                    compound_text = None
            break

    tire_age = _get_val(["tireage", "tireage_prev", "stint_laps_prev"])
    if tire_age is not None and compound_text:
        tire_text = f"{tire_age:.0f} laps ({compound_text})"
    elif tire_age is not None:
        tire_text = f"{tire_age:.0f} laps"
    elif compound_text:
        tire_text = compound_text
    else:
        tire_text = "N/A"
    tire_wear_pct = None
    if "tyre_wear_pct_prev" in row and pd.notna(row["tyre_wear_pct_prev"]):
        val = float(row["tyre_wear_pct_prev"])
        tire_wear_pct = val / 100.0 if val > 1.0 else val
    elif "tyre_wear_pct" in row and pd.notna(row["tyre_wear_pct"]):
        val = float(row["tyre_wear_pct"])
        tire_wear_pct = val / 100.0 if val > 1.0 else val
    elif tire_age is not None:
        tire_wear_pct = min(1.0, float(tire_age) / float(tire_max))

    gap_leader = _get_val(["gap", "gap_to_leader_prev"])
    gap_front = _get_val(["interval", "gap_to_front_prev"])
    gap_val = gap_leader if gap_leader is not None else gap_front
    gap_text = _format_seconds(gap_val)
    gap_delta = _get_val(["delta_interval_prev", "delta_best_so_far_prev", "relative_pace_prev"])
    gap_trend_text = "STEADY"
    overtake_mode = None
    if gap_delta is not None:
        if gap_delta < -0.1:
            gap_trend_text = "GAIN"
            overtake_mode = "attack"
        elif gap_delta > 0.1:
            gap_trend_text = "LOSS"
            overtake_mode = "defend"
    elif gap_val is not None:
        if gap_val <= 1.0:
            gap_trend_text = "ATTACK"
            overtake_mode = "attack"
        elif gap_val >= 3.0:
            gap_trend_text = "SAFE"

    decision_reasons = []
    if pit_window_text == "OPEN":
        decision_reasons.append("WINDOW")
    if sc_text in ("SC", "VSC"):
        decision_reasons.append(sc_text)
    if tire_wear_pct is not None and tire_wear_pct >= 0.7:
        decision_reasons.append("WEAR")
    if race_progress is not None and race_progress >= 0.75:
        decision_reasons.append("LATE")
    if not decision_reasons:
        decision_reasons.append("CORE")

    used_threshold = float(threshold)
    if pit_window_text == "OPEN":
        used_threshold = max(0.05, used_threshold - 0.08)
    if sc_text in ("SC", "VSC"):
        used_threshold = max(0.05, used_threshold - 0.12)
    if tire_wear_pct is not None and tire_wear_pct >= 0.7:
        used_threshold = max(0.05, used_threshold - 0.06)
    if race_progress is not None and race_progress >= 0.75:
        used_threshold = max(0.05, used_threshold - 0.03)

    if pit_window_text == "OPEN":
        decision = "BOX BOX" if proba >= used_threshold else "STANDBY"
    else:
        decision = "STAY OUT"
        if proba >= used_threshold:
            decision_reasons.append("NOWINDOW")

    pit_loss_sec = 20.0
    if sc_text in ("SC", "VSC"):
        pit_loss_sec = 12.0
    if gap_val is not None:
        pit_loss_sec = max(8.0, pit_loss_sec - min(6.0, gap_val / 5.0))

    remaining_laps = None
    if lap_value is not None:
        total_laps = _get_val(["nolaps_prev", "nolaps", "n_laps", "laps"])
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

    urgency = float(proba)
    rp = None
    if race_progress is not None:
        rp = float(np.clip(race_progress, 0.0, 1.0))
        urgency = float(np.clip(0.5 * proba + 0.5 * rp, 0.0, 1.0))
    progress_text = f"{rp * 100:.0f}%" if rp is not None else "N/A"

    if pit_window_text == "OPEN":
        pit_target_text = "NOW"
    elif pit_window_text == "CLOSED":
        pit_target_text = "HOLD"
    else:
        pit_target_text = "N/A"

    lap_text = "Latest lap"
    if lap_value is not None and not pd.isna(lap_value):
        lap_text = f"Lap {int(lap_value)}"
    elif lap_col and lap_col in row and pd.notna(row[lap_col]):
        try:
            lap_text = f"Lap {int(row[lap_col])}"
        except Exception:
            lap_text = "Latest lap"

    return {
        "decision": decision,
        "used_threshold": used_threshold,
        "race_progress": race_progress,
        "urgency": urgency,
        "pit_window_text": pit_window_text,
        "pit_target_text": pit_target_text,
        "tire_text": tire_text,
        "tire_wear_pct": tire_wear_pct,
        "gap_text": gap_text,
        "sc_text": sc_text,
        "progress_text": progress_text,
        "gap_trend_text": gap_trend_text,
        "overtake_mode": overtake_mode,
        "reason_text": "+".join(decision_reasons),
        "lap_text": lap_text,
        "compound_text": compound_text,
        "pit_loss_sec": pit_loss_sec,
        "gain_sec": expected_gain_sec,
        "net_gain_sec": net_gain_sec,
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


@st.cache_resource(show_spinner=False)
def _train_demo_model(
    df: pd.DataFrame,
    features: list[str],
    group_col: str | None,
) -> tuple[Pipeline, LogisticRegression | None, float | None]:
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
    y = df["decide_pitstop"].astype(int).values
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
        y_cal = cal_df["decide_pitstop"].astype(int).values
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

    return pipe, calibrator, cal_threshold

def _plot_metric_bar(df: pd.DataFrame, metric: str, std: str) -> plt.Figure:
    colors = ["#ff2b2b" if m == "MyMethod" else "#2c3545" for m in df["method"]]
    x = np.arange(len(df))
    y = df[metric].to_numpy()
    yerr = df[std].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("#0f131b")
    ax.set_facecolor("#0f131b")
    bars = ax.bar(
        x,
        y,
        yerr=yerr,
        capsize=6,
        color=colors,
        edgecolor="#0b0f16",
        linewidth=0.8,
    )
    for bar, val in zip(bars, y):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            min(0.98, val + 0.02),
            f"{val:.3f}",
            ha="center",
            va="bottom",
            color="#e6ebf2",
            fontsize=9,
        )
    ax.set_xticks(x, df["stage_short"].tolist())
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.25, color="#7a8796")
    ax.set_ylabel(metric.replace("mean_", "").upper(), color="#d7dde6")
    ax.tick_params(axis="x", colors="#d7dde6")
    ax.tick_params(axis="y", colors="#d7dde6")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#465267")
    ax.spines["bottom"].set_color("#465267")
    return fig


def _plot_metric_box(df: pd.DataFrame, metric: str) -> plt.Figure:
    stages = df["stage_id"].unique().tolist()
    data = [df.loc[df["stage_id"] == s, metric].to_numpy() for s in stages]
    labels = [f"S{s}" for s in stages]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    fig.patch.set_facecolor("#0f131b")
    ax.set_facecolor("#0f131b")
    bplot = ax.boxplot(data, labels=labels, patch_artist=True, showmeans=True)
    for patch, s in zip(bplot["boxes"], stages):
        patch.set_facecolor("#2c3545" if s % 2 == 0 else "#1a222f")
        patch.set_edgecolor("#7a8796")
    for element in ["whiskers", "caps", "medians", "means"]:
        for item in bplot[element]:
            item.set_color("#d7dde6")
    ax.grid(axis="y", linestyle="--", alpha=0.25, color="#7a8796")
    ax.set_ylabel(metric.upper(), color="#d7dde6")
    ax.tick_params(axis="x", colors="#d7dde6")
    ax.tick_params(axis="y", colors="#d7dde6")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#465267")
    ax.spines["bottom"].set_color("#465267")
    return fig


def main() -> None:
    st.set_page_config(page_title="Pit Stop Dashboard", layout="wide")
    _inject_css()

    st.sidebar.markdown("## Controls")
    mode = st.sidebar.radio("Dataset mode", ["Strict", "Standard"], horizontal=False)
    presenter_mode = st.sidebar.checkbox("Presenter mode", value=True)
    show_explainers = st.sidebar.checkbox("Show metric guide", value=True)
    st.sidebar.markdown("## Overlay")
    team_name = st.sidebar.text_input("Team label", value="MY PITWALL")
    session_name = st.sidebar.text_input("Session", value="RACE")
    event_name = st.sidebar.text_input("Event tag", value="UNSEEN RACES - GROUPKFOLD")
    lap_text = st.sidebar.text_input("Lap counter", value="LAP 12/58")
    weather_text = st.sidebar.text_input("Conditions", value="TRACK: MIXED")

    if mode == "Strict":
        summary_path = SUMMARY_STRICT
        folds_path = FOLDS_STRICT if FOLDS_STRICT.exists() else None
    else:
        summary_path = SUMMARY_STD
        folds_path = FOLDS_STANDARD if FOLDS_STANDARD.exists() else None

    if not summary_path.exists():
        st.error(f"Missing summary file: {summary_path}")
        return

    summary = _load_summary(summary_path, summary_path.stat().st_mtime)
    folds = (
        _load_folds(folds_path, folds_path.stat().st_mtime)
        if folds_path and folds_path.exists()
        else None
    )

    if mode == "Standard":
        st.warning(
            "Standard mode may include same-lap features and is not leakage-safe. "
            "Use Strict mode for thesis claims."
        )

    available_metrics = {
        label: cols
        for label, cols in METRICS.items()
        if cols[0] in summary.columns and cols[1] in summary.columns
    }
    if not available_metrics:
        st.error("No metrics found in the summary file.")
        return

    metric_name = st.sidebar.selectbox(
        "Primary metric",
        list(available_metrics.keys()),
        index=0,
        key="primary_metric",
    )

    metric_col, std_col = available_metrics[metric_name]

    delta_table = _metric_delta_table(summary, available_metrics)
    fold_stats = _fold_delta_stats(folds, available_metrics) if folds is not None else pd.DataFrame()
    thesis_md = _build_thesis_summary(summary, available_metrics, fold_stats)

    st.markdown(
        f"""
<div class="topbar">
  <div class="topbar-left">
    <span class="logo-badge">P1</span>
    <div>
      <div class="top-title">{team_name}</div>
      <div class="top-sub">{event_name}</div>
    </div>
  </div>
  <div class="topbar-center">
    <span class="pill">SESSION <strong>{session_name}</strong></span>
    <span class="pill">{lap_text}</span>
  </div>
  <div class="topbar-right">
    <span class="pill">{weather_text}</span>
    <span class="badge">GroupKFold Summary</span>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='section-title'>Pit Stop Performance Dashboard</div>", unsafe_allow_html=True)

    if show_explainers:
        guide = _metric_guide()
        guide_text = "<br/>".join([f"<strong>{k}:</strong> {v}" for k, v in guide.items()])
        with st.expander("Metric guide", expanded=False):
            st.markdown(f"<div class='callout'>{guide_text}</div>", unsafe_allow_html=True)

    takeaways = _key_takeaways(summary, metric_col)
    example_text = _example_explainer(summary, metric_col, std_col)
    col_a, col_b = st.columns([1.2, 1])
    with col_a:
        st.markdown(
            f"<div class='callout'><strong>Key Takeaways:</strong> {takeaways}</div>",
            unsafe_allow_html=True,
        )
    with col_b:
        st.markdown(
            f"<div class='example-card'><strong>Metric Snapshot:</strong> {example_text}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("### Strategy Demo")
    if not _XGB_OK:
        st.info("Demo prediction requires xgboost + scikit-learn installed.")
    else:
        df_demo = _load_stage4_data()
        if df_demo.empty or "Driver" not in df_demo.columns:
            st.info("Driver column not available in the Stage 4 dataset.")
        else:
            df_demo = df_demo.copy()
            df_demo["weather_label"] = _derive_weather_label(df_demo)
            circuit_col = _pick_column(df_demo, CIRCUIT_COL_CANDIDATES)
            tire_max = None
            if "tireage" in df_demo.columns:
                tire_vals = pd.to_numeric(df_demo["tireage"], errors="coerce").dropna()
                if not tire_vals.empty:
                    tire_max = float(np.nanpercentile(tire_vals, 90))
            elif "stint_laps_prev" in df_demo.columns:
                tire_vals = pd.to_numeric(df_demo["stint_laps_prev"], errors="coerce").dropna()
                if not tire_vals.empty:
                    tire_max = float(np.nanpercentile(tire_vals, 90))
            if tire_max is None or not np.isfinite(tire_max) or tire_max <= 0:
                tire_max = 35.0

            group_col = "race_id" if "race_id" in df_demo.columns else None
            n_splits = 5
            if "n_splits" in summary.columns:
                try:
                    n_splits = int(summary["n_splits"].iloc[0])
                except Exception:
                    n_splits = 5

            fold_id = 1
            split_indices = _split_groupkfold(df_demo, group_col, n_splits, fold_id) if group_col else None

            if split_indices:
                train_idx, test_idx = split_indices
                train_df = df_demo.iloc[train_idx]
                test_df = df_demo.iloc[test_idx]
            else:
                train_df = df_demo
                test_df = df_demo

            def _stat_card(label: str, stats: dict | None) -> None:
                if stats is None:
                    card_html = (
                        "<div class='card'>"
                        f"<div class='card-title'>{label} rows</div>"
                        "<div class='card-value'>N/A</div>"
                        "<div class='card-sub'>Split unavailable</div>"
                        "</div>"
                    )
                else:
                    sub_parts = []
                    if stats["pos_rate"] is not None:
                        sub_parts.append(f"Pos {stats['pos_rate'] * 100:.1f}%")
                    if stats["groups"] is not None:
                        sub_parts.append(f"Races {stats['groups']}")
                    sub_text = " | ".join(sub_parts) if sub_parts else " "
                    card_html = (
                        "<div class='card'>"
                        f"<div class='card-title'>{label} rows</div>"
                        f"<div class='card-value'>{stats['rows']}</div>"
                        f"<div class='card-sub'>{sub_text}</div>"
                        "</div>"
                    )
                st.markdown(card_html, unsafe_allow_html=True)

            stats_train = _dataset_stats(train_df, "decide_pitstop", group_col) if split_indices else None
            stats_test = _dataset_stats(test_df, "decide_pitstop", group_col) if split_indices else None
            stats_all = _dataset_stats(df_demo, "decide_pitstop", group_col)
            stat_cols = st.columns(3)
            with stat_cols[0]:
                _stat_card("Train", stats_train if split_indices else None)
            with stat_cols[1]:
                _stat_card("Test", stats_test if split_indices else None)
            with stat_cols[2]:
                _stat_card("All", stats_all)

            ctrl1, ctrl2, ctrl3 = st.columns(3)
            train_drivers = set(train_df["Driver"].dropna().astype(str))
            test_drivers = set(test_df["Driver"].dropna().astype(str))
            common_drivers = sorted(train_drivers.intersection(test_drivers))
            drivers = common_drivers if common_drivers else sorted(train_drivers.union(test_drivers))
            with ctrl1:
                driver = st.selectbox("Driver", drivers, index=0) if drivers else None
                if not drivers:
                    st.info("No driver names found in the dataset.")

            train_driver = train_df
            test_driver = test_df
            if driver is not None:
                train_driver = train_df[train_df["Driver"] == driver]
                test_driver = test_df[test_df["Driver"] == driver]

            with ctrl2:
                if circuit_col:
                    train_c = train_driver[circuit_col].dropna().astype(str)
                    test_c = test_driver[circuit_col].dropna().astype(str)
                    common_circuits = sorted(set(train_c).intersection(set(test_c)))
                    circuits = common_circuits if common_circuits else sorted(set(train_c).union(set(test_c)))
                    if common_circuits:
                        circuit_choice = st.selectbox("Circuit", ["Auto"] + circuits, index=0)
                        circuit_sel = circuits[0] if circuit_choice == "Auto" else circuit_choice
                    else:
                        circuit_sel = st.selectbox("Circuit", circuits, index=0) if circuits else None
                else:
                    circuit_sel = None

            with ctrl3:
                train_w = train_driver["weather_label"].dropna().astype(str)
                test_w = test_driver["weather_label"].dropna().astype(str)
                common_weather = sorted(set(train_w).intersection(set(test_w)))
                weathers = common_weather if common_weather else sorted(set(train_w).union(set(test_w)))
                if common_weather:
                    weather_choice = st.selectbox("Weather", ["Auto"] + weathers, index=0)
                    weather_sel = weathers[0] if weather_choice == "Auto" else weather_choice
                else:
                    weather_sel = st.selectbox("Weather", weathers, index=0) if weathers else None

            train_filtered = train_driver
            test_filtered = test_driver
            if circuit_col and circuit_sel:
                train_filtered = train_filtered[train_filtered[circuit_col].astype(str) == circuit_sel]
                test_filtered = test_filtered[test_filtered[circuit_col].astype(str) == circuit_sel]
            if weather_sel:
                train_filtered = train_filtered[train_filtered["weather_label"] == weather_sel]
                test_filtered = test_filtered[test_filtered["weather_label"] == weather_sel]

            train_filtered_all = train_filtered
            test_filtered_all = test_filtered

            lap_used = None
            lap_col = "lapno" if "lapno" in df_demo.columns else "lapno_prev" if "lapno_prev" in df_demo.columns else None
            lap_key = "lap_value"
            lap_min = None
            lap_max = None
            lap_pool: list[int] | None = None
            stint_col = _pick_column(df_demo, ["pitstops_so_far", "pitstops_so_far_prev"])
            tire_col = _pick_column(df_demo, ["tireage", "stint_laps_prev"])
            if lap_col and lap_col in df_demo.columns:
                train_laps = pd.to_numeric(train_filtered[lap_col], errors="coerce").dropna().astype(int)
                test_laps = pd.to_numeric(test_filtered[lap_col], errors="coerce").dropna().astype(int)
                common_laps = sorted(set(train_laps).intersection(set(test_laps)))
                if common_laps:
                    lap_pool = common_laps
                else:
                    union_laps = sorted(set(train_laps).union(set(test_laps)))
                    lap_pool = union_laps if union_laps else None

                if lap_pool:
                    lap_min = int(min(lap_pool))
                    lap_max = int(max(lap_pool))
                    if lap_key not in st.session_state:
                        st.session_state[lap_key] = lap_min
                    st.slider(
                        "Lap timeline",
                        min_value=lap_min,
                        max_value=lap_max,
                        step=1,
                        key=lap_key,
                    )
                    lap_value = int(st.session_state[lap_key])
                    lap_used = min(lap_pool, key=lambda x: abs(x - lap_value))
                else:
                    st.text_input("Lap timeline", value="N/A", disabled=True)
            else:
                st.text_input("Lap timeline", value="N/A", disabled=True)

            if driver is not None:
                chip_lap = f"L{lap_used}" if lap_used is not None else "N/A"
                chip_circuit = circuit_sel if circuit_sel else "Any"
                chip_weather = weather_sel if weather_sel else "Any"
                st.markdown(
                    "<div class='demo-chip-row'>"
                    f"<span class='demo-chip'>Driver <strong>{driver}</strong></span>"
                    f"<span class='demo-chip'>Circuit <strong>{chip_circuit}</strong></span>"
                    f"<span class='demo-chip'>Weather <strong>{chip_weather}</strong></span>"
                    f"<span class='demo-chip'>Lap <strong>{chip_lap}</strong></span>"
                    "</div>",
                    unsafe_allow_html=True,
                )

            def _filter_same_stint(
                df_context: pd.DataFrame,
                lap_value: int | None,
            ) -> pd.DataFrame:
                if lap_value is None or lap_col is None or lap_col not in df_context.columns:
                    return df_context
                if stint_col and stint_col in df_context.columns:
                    match = df_context.loc[df_context[lap_col] == lap_value, stint_col].dropna()
                    if not match.empty:
                        stint_val = match.iloc[-1]
                        return df_context[df_context[stint_col] == stint_val]
                    return df_context
                if tire_col and tire_col in df_context.columns:
                    df_laps = df_context.copy()
                    df_laps[lap_col] = pd.to_numeric(df_laps[lap_col], errors="coerce")
                    df_laps[tire_col] = pd.to_numeric(df_laps[tire_col], errors="coerce")
                    df_laps = df_laps.dropna(subset=[lap_col, tire_col]).sort_values(lap_col)
                    if df_laps.empty:
                        return df_context
                    resets = df_laps[tire_col].diff().fillna(0) < -4
                    df_laps["stint_id"] = resets.cumsum()
                    target = df_laps.loc[df_laps[lap_col] == lap_value, "stint_id"]
                    if target.empty:
                        return df_context
                    stint_id = int(target.iloc[-1])
                    return df_context.loc[df_laps.index[df_laps["stint_id"] == stint_id]]
                return df_context

            train_context = _filter_same_stint(train_filtered, lap_used)
            test_context = _filter_same_stint(test_filtered, lap_used)

            win_train = _pit_window_bounds(train_context, lap_col)
            win_test = _pit_window_bounds(test_context, lap_col)

            if lap_used is not None and lap_col and lap_col in train_filtered.columns:
                train_filtered = train_context[train_context[lap_col] == lap_used]
            if lap_used is not None and lap_col and lap_col in test_filtered.columns:
                test_filtered = test_context[test_context[lap_col] == lap_used]

            train_scope = train_filtered if not train_filtered.empty else train_driver
            test_scope = test_filtered if not test_filtered.empty else test_driver

            raw_features = build_feature_list(df_demo, "decide_pitstop", "race_id")
            features = _apply_feature_allowlist(raw_features)
            features = _align_features(features, train_df, test_df)
            if not features:
                st.warning("No shared features available for the demo.")
            elif driver is None:
                st.info("Demo controls are unavailable until driver data is loaded.")
            else:
                with st.spinner("Loading demo model..."):
                    model, calibrator, cal_threshold = _train_demo_model(train_df, features, group_col)

                def _apply_calibration(raw_prob: float) -> float:
                    if calibrator is None:
                        return raw_prob
                    try:
                        return float(calibrator.predict_proba(np.array([[raw_prob]]))[:, 1][0])
                    except Exception:
                        return raw_prob

                base_threshold = 0.5
                if "threshold" in summary.columns:
                    try:
                        base_threshold = float(summary.loc[summary["stage_id"] == 4, "threshold"].iloc[0])
                    except Exception:
                        base_threshold = 0.5
                if not np.isfinite(base_threshold):
                    base_threshold = 0.5
                threshold = _apply_calibration(base_threshold)
                if cal_threshold is not None and np.isfinite(cal_threshold):
                    threshold = float(cal_threshold)

                train_row = train_scope.iloc[-1]
                test_row = test_scope.iloc[-1]

                train_raw = float(model.predict_proba(train_row[features].to_frame().T)[:, 1][0])
                test_raw = float(model.predict_proba(test_row[features].to_frame().T)[:, 1][0])
                train_proba = _apply_calibration(train_raw)
                test_proba = _apply_calibration(test_raw)

                lookahead_laps = 6
                train_payload = _demo_decision(
                    train_row,
                    train_proba,
                    threshold,
                    lap_used,
                    lap_col,
                    tire_max,
                    lookahead_laps,
                )
                test_payload = _demo_decision(
                    test_row,
                    test_proba,
                    threshold,
                    lap_used,
                    lap_col,
                    tire_max,
                    lookahead_laps,
                )

                def _stint_reset_flag(df_context: pd.DataFrame) -> bool:
                    if lap_used is None or lap_col is None or lap_col not in df_context.columns:
                        return False
                    if stint_col and stint_col in df_context.columns:
                        current = df_context.loc[df_context[lap_col] == lap_used, stint_col].dropna()
                        prev = df_context.loc[df_context[lap_col] < lap_used, stint_col].dropna()
                        if current.empty or prev.empty:
                            return False
                        try:
                            return float(current.iloc[-1]) > float(prev.iloc[-1])
                        except Exception:
                            return False
                    if tire_col and tire_col in df_context.columns:
                        df_laps = df_context.copy()
                        df_laps[lap_col] = pd.to_numeric(df_laps[lap_col], errors="coerce")
                        df_laps[tire_col] = pd.to_numeric(df_laps[tire_col], errors="coerce")
                        df_laps = df_laps.dropna(subset=[lap_col, tire_col]).sort_values(lap_col)
                        if df_laps.empty:
                            return False
                        current = df_laps.loc[df_laps[lap_col] == lap_used, tire_col]
                        prev = df_laps.loc[df_laps[lap_col] < lap_used, tire_col]
                        if current.empty or prev.empty:
                            return False
                        curr_val = float(current.iloc[-1])
                        prev_val = float(prev.iloc[-1])
                        return curr_val <= 3.0 and (prev_val - curr_val) >= 5.0
                    return False

                train_reset = _stint_reset_flag(train_context)
                test_reset = _stint_reset_flag(test_context)

                def _recommend_next_lap(df_context: pd.DataFrame) -> dict | None:
                    if lap_col is None or df_context.empty:
                        return None
                    df_laps = df_context.copy()
                    df_laps[lap_col] = pd.to_numeric(df_laps[lap_col], errors="coerce")
                    df_laps = df_laps.dropna(subset=[lap_col])
                    if df_laps.empty:
                        return None
                    if lap_used is not None:
                        df_laps = df_laps[
                            (df_laps[lap_col] >= lap_used)
                            & (df_laps[lap_col] <= lap_used + lookahead_laps)
                        ]
                    if df_laps.empty:
                        return None
                    df_laps = df_laps.sort_values(lap_col).head(lookahead_laps + 1)
                    best = None
                    for _, r in df_laps.iterrows():
                        lap_val = int(r[lap_col])
                        p_raw = float(model.predict_proba(r[features].to_frame().T)[:, 1][0])
                        p = _apply_calibration(p_raw)
                        payload = _demo_decision(r, p, threshold, lap_val, lap_col, tire_max, lookahead_laps)
                        score = float(payload["net_gain_sec"])
                        if best is None or score > best["net_gain_sec"]:
                            best = {
                                "lap": lap_val,
                                "net_gain_sec": score,
                                "proba": p,
                            }
                    return best

                train_rec = _recommend_next_lap(train_context)
                test_rec = _recommend_next_lap(test_context)

                def _driver_label(row: pd.Series, fallback: str) -> str:
                    if "Driver" in row and pd.notna(row["Driver"]):
                        return str(row["Driver"])
                    return fallback

                def _display_text(row: pd.Series, fallback: str, col_name: str | None) -> str:
                    if col_name and col_name in row and pd.notna(row[col_name]):
                        return str(row[col_name])
                    return fallback

                train_circuit_text = _display_text(train_row, circuit_sel, circuit_col)
                if not train_circuit_text or train_circuit_text == "Any":
                    train_circuit_text = "Circuit"
                test_circuit_text = _display_text(test_row, circuit_sel, circuit_col)
                if not test_circuit_text or test_circuit_text == "Any":
                    test_circuit_text = "Circuit"

                train_weather_text = str(train_row["weather_label"]) if "weather_label" in train_row else weather_sel
                if not train_weather_text or train_weather_text == "Any":
                    train_weather_text = "Weather"
                test_weather_text = str(test_row["weather_label"]) if "weather_label" in test_row else weather_sel
                if not test_weather_text or test_weather_text == "Any":
                    test_weather_text = "Weather"

                train_range = _lap_range(train_context, lap_col)
                test_range = _lap_range(test_context, lap_col)
                group_cols = []
                if group_col and group_col in df_demo.columns:
                    group_cols.append(group_col)
                if "Driver" in df_demo.columns:
                    group_cols.append("Driver")

                demo_cols = st.columns(2)
                with demo_cols[0]:
                    st.markdown(
                        _render_track_demo(
                            panel_label="Train (Learned)",
                            driver=_driver_label(train_row, driver or "Driver"),
                            lap_text=train_payload["lap_text"],
                            circuit_text=train_circuit_text,
                            weather_text=train_weather_text,
                            decision=train_payload["decision"],
                            proba=train_proba,
                            threshold=train_payload["used_threshold"],
                            race_progress=train_payload["race_progress"],
                            urgency=train_payload["urgency"],
                            pit_window_text=train_payload["pit_window_text"],
                            pit_target_text=train_payload["pit_target_text"],
                            tire_text=train_payload["tire_text"],
                            tire_wear_pct=train_payload["tire_wear_pct"],
                            gap_text=train_payload["gap_text"],
                            sc_text=train_payload["sc_text"],
                            progress_text=train_payload["progress_text"],
                            gap_trend_text=train_payload["gap_trend_text"],
                            overtake_mode=train_payload["overtake_mode"],
                            reason_text=train_payload["reason_text"],
                            stint_reset=train_reset,
                            lap_current=int(lap_used) if lap_used is not None else None,
                            lap_min=train_range[0] if train_range else None,
                            lap_max=train_range[1] if train_range else None,
                            window_start=win_train[0] if win_train else None,
                            window_end=win_train[1] if win_train else None,
                            rec_lap=train_rec["lap"] if train_rec else None,
                        ),
                        unsafe_allow_html=True,
                    )
                with demo_cols[1]:
                    st.markdown(
                        _render_track_demo(
                            panel_label="Test (Unseen)",
                            driver=_driver_label(test_row, driver or "Driver"),
                            lap_text=test_payload["lap_text"],
                            circuit_text=test_circuit_text,
                            weather_text=test_weather_text,
                            decision=test_payload["decision"],
                            proba=test_proba,
                            threshold=test_payload["used_threshold"],
                            race_progress=test_payload["race_progress"],
                            urgency=test_payload["urgency"],
                            pit_window_text=test_payload["pit_window_text"],
                            pit_target_text=test_payload["pit_target_text"],
                            tire_text=test_payload["tire_text"],
                            tire_wear_pct=test_payload["tire_wear_pct"],
                            gap_text=test_payload["gap_text"],
                            sc_text=test_payload["sc_text"],
                            progress_text=test_payload["progress_text"],
                            gap_trend_text=test_payload["gap_trend_text"],
                            overtake_mode=test_payload["overtake_mode"],
                            reason_text=test_payload["reason_text"],
                            stint_reset=test_reset,
                            lap_current=int(lap_used) if lap_used is not None else None,
                            lap_min=test_range[0] if test_range else None,
                            lap_max=test_range[1] if test_range else None,
                            window_start=win_test[0] if win_test else None,
                            window_end=win_test[1] if win_test else None,
                            rec_lap=test_rec["lap"] if test_rec else None,
                        ),
                        unsafe_allow_html=True,
                    )
                train_rows = int(len(train_filtered_all))
                test_rows = int(len(test_filtered_all))
                train_strength, train_gap = _decision_strength(train_proba, threshold)
                test_strength, test_gap = _decision_strength(test_proba, threshold)
                helper_html = (
                    "<div class='helper-card'>"
                    "<div class='helper-title'>Presenter Helper</div>"
                    "<div class='helper-grid'>"
                    "<div>"
                    "<div class='helper-pill'>Train data <strong>learned behavior</strong></div>"
                    f"<div class='helper-note'>Decision strength: {train_strength} (|P-T| {train_gap:.2f})</div>"
                    f"<div class='helper-note'>Data reliability: {_reliability_label(train_rows)} ({train_rows} rows)</div>"
                    f"<div class='helper-note'>{_decision_sentence(train_payload, train_proba, threshold)}</div>"
                    "</div>"
                    "<div>"
                    "<div class='helper-pill'>Test data <strong>unseen races</strong></div>"
                    f"<div class='helper-note'>Decision strength: {test_strength} (|P-T| {test_gap:.2f})</div>"
                    f"<div class='helper-note'>Data reliability: {_reliability_label(test_rows)} ({test_rows} rows)</div>"
                    f"<div class='helper-note'>{_decision_sentence(test_payload, test_proba, threshold)}</div>"
                    "</div>"
                    "</div>"
                    "</div>"
                )
                st.markdown(helper_html, unsafe_allow_html=True)
                summary_cols = st.columns(2)

                def _diff_summary(left: dict, right: dict) -> str:
                    diffs = []

                    def _add(label: str, a: str, b: str) -> None:
                        if a != b:
                            diffs.append(f"{label} {a} vs {b}")

                    _add("Service window", left["pit_window_text"], right["pit_window_text"])
                    _add("Tyre age", left["tire_text"], right["tire_text"])
                    _add("Time gap", left["gap_text"], right["gap_text"])
                    _add("Track status", left["sc_text"], right["sc_text"])
                    if not diffs:
                        return "No major differences"
                    return " | ".join(diffs[:3])

                diff_text = _diff_summary(train_payload, test_payload)
                train_rec_text = "N/A"
                if train_rec is not None:
                    train_rec_text = f"L{train_rec['lap']} (net {train_rec['net_gain_sec']:+.1f}s)"
                test_rec_text = "N/A"
                if test_rec is not None:
                    test_rec_text = f"L{test_rec['lap']} (net {test_rec['net_gain_sec']:+.1f}s)"

                def _input_summary(
                    label: str,
                    driver_name: str,
                    circuit_text: str,
                    weather_text: str,
                    payload: dict,
                    diff_note: str,
                    rec_note: str,
                    row: pd.Series,
                ) -> str:
                    telemetry_html = _telemetry_sections(row, payload)
                    return (
                        "<div class='card summary-card'>"
                        f"<div class='card-title'>{label} Summary</div>"
                        f"<div class='card-sub'>Driver: {driver_name}</div>"
                        f"<div class='card-sub'>Current lap: {payload['lap_text']}</div>"
                        f"<div class='card-sub'>Service window: {payload['pit_window_text']}</div>"
                        f"<div class='card-sub'>Suggested stop: {rec_note}</div>"
                        f"<div class='card-sub'>Current call: {payload['decision']}</div>"
                        f"<div class='card-sub summary-diff'>Key differences: {diff_note}</div>"
                        f"{telemetry_html}"
                        "</div>"
                    )

                with summary_cols[0]:
                    st.markdown(
                        _input_summary(
                            "Train",
                            _driver_label(train_row, driver or "Driver"),
                            train_circuit_text,
                            train_weather_text,
                            train_payload,
                            diff_text,
                            train_rec_text,
                            train_row,
                        ),
                        unsafe_allow_html=True,
                    )
                with summary_cols[1]:
                    st.markdown(
                        _input_summary(
                            "Test",
                            _driver_label(test_row, driver or "Driver"),
                            test_circuit_text,
                            test_weather_text,
                            test_payload,
                            diff_text,
                            test_rec_text,
                            test_row,
                        ),
                        unsafe_allow_html=True,
                    )
                st.markdown("### Strategy Impact (Estimated)")
                impact_train = _strategy_impact(
                    train_filtered_all,
                    features,
                    model,
                    calibrator,
                    threshold,
                    lap_col,
                    tire_max,
                    lookahead_laps,
                    group_cols,
                )
                impact_test = _strategy_impact(
                    test_filtered_all,
                    features,
                    model,
                    calibrator,
                    threshold,
                    lap_col,
                    tire_max,
                    lookahead_laps,
                    group_cols,
                )
                impact_cols = st.columns(2)
                for col, title, impact in (
                    (impact_cols[0], "Train impact", impact_train),
                    (impact_cols[1], "Test impact", impact_test),
                ):
                    with col:
                        if impact is None:
                            st.info("Impact backtest is unavailable for this selection.")
                            continue
                        summary_impact, table = impact
                        card_html = (
                            "<div class='card'>"
                            f"<div class='card-title'>{title}</div>"
                            f"<div class='card-value'>{summary_impact['avg_delta']:+.1f}s</div>"
                            f"<div class='card-sub'>Median {summary_impact['median_delta']:+.1f}s</div>"
                            f"<div class='card-sub'>Improved {summary_impact['improve_rate'] * 100:.0f}%</div>"
                            f"<div class='card-sub'>Groups {summary_impact['groups']}</div>"
                            f"<div class='card-sub'>Rows {summary_impact['rows']}</div>"
                            "</div>"
                        )
                        st.markdown(card_html, unsafe_allow_html=True)
                        top = table.sort_values("delta", ascending=False).head(5)
                        show_cols = [c for c in group_cols if c in top.columns] + ["delta"]
                        st.dataframe(top[show_cols], use_container_width=True, hide_index=True)

                st.markdown("### Baseline Model Check (Test Split)")
                compare = _baseline_compare(
                    train_filtered_all,
                    test_filtered_all,
                    features,
                    model,
                    calibrator,
                    threshold,
                    group_col,
                )
                if compare is None:
                    st.info("Baseline comparison is unavailable for this selection.")
                else:
                    metrics_df, sign_stats = compare
                    st.dataframe(metrics_df, use_container_width=True, hide_index=True)
                    if sign_stats:
                        lr_p = sign_stats.get("xgb_vs_lr_p")
                        rf_p = sign_stats.get("xgb_vs_rf_p")
                        lr_p_text = f"{lr_p:.3f}" if lr_p is not None else "N/A"
                        rf_p_text = f"{rf_p:.3f}" if rf_p is not None else "N/A"
                        st.caption(
                            "XGB vs LogReg win rate "
                            f"{sign_stats['xgb_vs_lr_win_rate'] * 100:.0f}% "
                            f"(p={lr_p_text})"
                        )
                        st.caption(
                            "XGB vs RandomForest win rate "
                            f"{sign_stats['xgb_vs_rf_win_rate'] * 100:.0f}% "
                            f"(p={rf_p_text})"
                        )

                st.caption("Estimated impact from simplified cost model; not a full race-time simulation.")

    st.markdown("### Stage Snapshot")
    cols = st.columns(4)
    for idx, row in summary.iterrows():
        delta_text = ""
        delta_badge = ""
        if row["stage_id"] in (2, 4):
            prev = summary.loc[idx - 1, metric_col]
            delta = row[metric_col] - prev
            delta_text = f"Delta vs prev: {delta:+.3f}"
            delta_badge = _delta_badge(delta)
        else:
            delta_badge = "<span class='delta-flat'>Baseline</span>"

        card_html = (
            "<div class='card'>"
            f"<div class='card-title'>{row['stage_short']} - {row['method']}</div>"
            f"<div class='card-value'>{_fmt(row[metric_col])}</div>"
            f"<div class='card-sub'>Std: {_fmt(row[std_col])}</div>"
            f"<div class='card-sub'>{row['dataset']}</div>"
            f"<div class='card-sub'>{delta_text}</div>"
            f"<div class='card-sub'>{delta_badge}</div>"
            "</div>"
        )
        cols[idx].markdown(card_html, unsafe_allow_html=True)

    st.markdown(
        """
<div class="legend">
  <span><i class="swatch swatch-ref"></i>RefTech</span>
  <span><i class="swatch swatch-my"></i>MyMethod</span>
</div>
""",
        unsafe_allow_html=True,
    )

    st.markdown("### Metric Comparison")
    st.pyplot(_plot_metric_bar(summary, metric_col, std_col), use_container_width=True)

    if presenter_mode:
        tabs = ["Summary Table", "Stage Deltas", "CRISP-DM", "Project Details", "Thesis Summary"]
    else:
        tabs = [
            "Summary Table",
            "Stage Deltas",
            "Fold Details",
            "PowerBI",
            "CRISP-DM",
            "Project Details",
            "Thesis Summary",
        ]
    tab_map = dict(zip(tabs, st.tabs(tabs)))

    with tab_map["Summary Table"]:
        show_cols = ["stage", "method", "dataset"]
        for mean_col, stdc in available_metrics.values():
            show_cols.extend([mean_col, stdc])
        show_cols = [c for c in show_cols if c in summary.columns]
        st.dataframe(summary[show_cols], use_container_width=True, hide_index=True)

        with st.expander("All metrics view", expanded=not presenter_mode):
            fig, axes = plt.subplots(2, 3, figsize=(12, 7))
            fig.patch.set_facecolor("#0f131b")
            axes = axes.flatten()
            for ax, (label, (mean_col, stdc)) in zip(axes, available_metrics.items()):
                x = np.arange(len(summary))
                ax.bar(
                    x,
                    summary[mean_col].to_numpy(),
                    yerr=summary[stdc].to_numpy(),
                    capsize=4,
                    color=["#ff2b2b" if m == "MyMethod" else "#2c3545" for m in summary["method"]],
                    edgecolor="#111",
                    linewidth=0.5,
                )
                ax.set_title(label)
                ax.set_xticks(x, summary["stage_short"].tolist())
                ax.set_ylim(0.0, 1.0)
                ax.grid(axis="y", linestyle="--", alpha=0.25, color="#7a8796")
                ax.set_facecolor("#0f131b")
                ax.tick_params(axis="x", colors="#d7dde6")
                ax.tick_params(axis="y", colors="#d7dde6")
                ax.title.set_color("#d7dde6")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.spines["left"].set_color("#465267")
                ax.spines["bottom"].set_color("#465267")
            for ax in axes[len(available_metrics) :]:
                ax.axis("off")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)

    with tab_map["Stage Deltas"]:
        if delta_table.empty:
            st.info("Delta table is not available for this dataset.")
        else:
            st.markdown("**Stage Improvements (mean deltas)**")
            st.dataframe(delta_table, use_container_width=True, hide_index=True)
        if fold_stats.empty:
            st.info("Fold-level paired tests are only available for Strict mode with fold data.")
        else:
            st.markdown("**Fold-level paired deltas + significance**")
            st.dataframe(fold_stats, use_container_width=True, hide_index=True)
            if (fold_stats["test"] == "sign").any():
                st.caption("Note: Sign test is used when SciPy Wilcoxon is unavailable.")

    with tab_map["CRISP-DM"]:
        crisp_lines = [
            "**Business Understanding**: Define a decision-support goal for pit-stop timing and compare to a reference method.",
            "**Data Understanding**: Inspect race, lap, and weather signals; verify class balance and leakage risks.",
            "**Data Preparation**: Clean CSVs and engineer pace, pit-window, and weather features.",
            "**Modeling**: Train XGBoost models with calibrated probabilities across the four stages.",
            "**Evaluation**: Use GroupKFold by race and report F1/F2/PR-AUC/Recall with fold-level summaries.",
            "**Deployment**: Deliver the Streamlit pitwall dashboard and strategy-impact backtest panel.",
        ]
        st.markdown("\n".join(crisp_lines))

    if not presenter_mode and "Fold Details" in tab_map:
        with tab_map["Fold Details"]:
            if folds is None:
                st.info("Fold details are only available for the Strict dataset.")
            else:
                fold_metric = st.selectbox("Fold metric", list(available_metrics.keys()), index=0, key="fold_metric")
                fold_col = available_metrics[fold_metric][0].replace("mean_", "")
                st.pyplot(_plot_metric_box(folds, fold_col), use_container_width=True)
                st.dataframe(folds, use_container_width=True, hide_index=True)

    if not presenter_mode and "PowerBI" in tab_map:
        with tab_map["PowerBI"]:
            st.markdown(
                "Use the CSVs below in PowerBI (Get Data -> Text/CSV). "
                "Stage names can be used as categories, and mean metrics as values."
            )
            st.download_button(
                "Download summary CSV",
                summary.to_csv(index=False).encode("utf-8"),
                file_name=summary_path.name,
                mime="text/csv",
            )
            if folds is not None:
                st.download_button(
                    "Download fold CSV",
                    folds.to_csv(index=False).encode("utf-8"),
                    file_name=folds_path.name if folds_path else "stage_folds.csv",
                    mime="text/csv",
                )

            st.markdown("Paste a PowerBI embed URL to view it here (optional):")
            embed_url = st.text_input("PowerBI embed URL", value="")
            if embed_url:
                st.components.v1.iframe(embed_url, height=700, scrolling=True)

    with tab_map["Project Details"]:
        if PROJECT_DETAILS.exists():
            project_md = PROJECT_DETAILS.read_text(encoding="utf-8")
            st.markdown(project_md)
            st.download_button(
                "Download Project Details",
                project_md.encode("utf-8"),
                file_name=PROJECT_DETAILS.name,
                mime="text/markdown",
            )
        else:
            st.info("Project Details file not found.")

    with tab_map["Thesis Summary"]:
        st.markdown("**Thesis-ready summary**")
        st.text_area("Summary", thesis_md, height=320)
        st.download_button(
            "Download summary (Markdown)",
            thesis_md.encode("utf-8"),
            file_name="thesis_results_summary.md",
            mime="text/markdown",
        )


if __name__ == "__main__":
    main()
