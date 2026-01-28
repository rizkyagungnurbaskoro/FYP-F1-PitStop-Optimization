from __future__ import annotations

import html
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
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, brier_score_loss
    from sklearn.calibration import calibration_curve
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder
    from xgboost import XGBClassifier

    _XGB_OK = True
except Exception:
    _XGB_OK = False

SUMMARY_STRICT = ROOT / "results" / "summary_plots" / "stage_summary_strict.csv"
SUMMARY_STD = ROOT / "results" / "summary_plots" / "stage_summary.csv"
SUMMARY_HOLDOUT = ROOT / "results" / "summary_plots" / "stage34_holdout_summary.csv"
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
  --bg1: #0b0d12;
  --bg2: #111521;
  --panel: #111620;
  --panel-2: #0c111a;
  --ink: #f5f7fb;
  --muted: #b2bccb;
  --accent: #e10600;
  --accent2: #ff9d2b;
  --accent3: #17c3ff;
  --good: #2bd97f;
  --warn: #ffb703;
  --border: rgba(255,255,255,0.08);
}
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;500;600;700&family=Oxanium:wght@500;600;700&display=swap');
html, body, [class*="css"]  {
  font-family: "Rajdhani", "Segoe UI", sans-serif;
  color: var(--ink);
}
.stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6,
.stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
  font-family: "Oxanium", "Rajdhani", sans-serif;
  letter-spacing: 0.04em;
}
.stApp {
  background:
    radial-gradient(900px 580px at 12% -10%, rgba(225,6,0,0.16) 0%, rgba(0,0,0,0) 65%),
    radial-gradient(860px 560px at 92% -20%, rgba(23,195,255,0.12) 0%, rgba(0,0,0,0) 70%),
    radial-gradient(700px 420px at 70% 10%, rgba(255,157,43,0.12) 0%, rgba(0,0,0,0) 68%),
    linear-gradient(135deg, var(--bg1), var(--bg2));
}
section[data-testid="stAppViewContainer"]::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background-image: repeating-linear-gradient(
    90deg,
    rgba(255,255,255,0.02) 0 1px,
    rgba(255,255,255,0.0) 1px 80px
  );
  opacity: 0.35;
}
  section[data-testid="stSidebar"] {
    display: block;
    background: linear-gradient(180deg, rgba(10, 12, 18, 0.98), rgba(8, 10, 16, 0.98));
    border-right: 1px solid rgba(255, 255, 255, 0.06);
  }
  div[data-testid="stSidebarNav"] {
    padding-top: 12px;
  }
  div[data-testid="stSidebarNav"] span {
    font-family: "Oxanium", "Rajdhani", sans-serif;
    letter-spacing: 0.06em;
  }
.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 16px;
  padding: 14px 18px;
  border: 1px solid var(--border);
  border-radius: 18px;
  background: linear-gradient(160deg, rgba(16,20,28,0.96), rgba(10,12,18,0.98));
  box-shadow: 0 16px 32px rgba(0, 0, 0, 0.45);
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
.topbar:after {
  content: "";
  position: absolute;
  right: 0;
  top: 0;
  width: 140px;
  height: 100%;
  opacity: 0.12;
  background: repeating-linear-gradient(
    90deg,
    rgba(255, 255, 255, 0.15) 0 12px,
    rgba(0, 0, 0, 0.0) 12px 24px
  );
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
  font-family: "Oxanium", "Rajdhani", sans-serif;
  font-size: 1.1rem;
  font-weight: 700;
  letter-spacing: 0.04em;
}
.top-title {
  font-family: "Oxanium", "Rajdhani", sans-serif;
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #f8fbff;
  text-shadow: 0 0 16px rgba(225,6,0,0.25);
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
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
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
  background: linear-gradient(160deg, rgba(16,20,28,0.96), rgba(10,12,18,0.98));
  border: 1px solid var(--border);
  border-radius: 18px;
  padding: 16px 18px;
  box-shadow: 0 14px 30px rgba(0, 0, 0, 0.45);
  backdrop-filter: blur(6px);
  animation: fadein 0.6s ease-out;
  position: relative;
  overflow: hidden;
}
.card:before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  width: 4px;
  height: 100%;
  background: linear-gradient(180deg, rgba(255, 43, 43, 0.8), rgba(255, 43, 43, 0.0));
  opacity: 0.35;
}
.card-title {
  font-size: 0.95rem;
  color: var(--muted);
  margin-bottom: 6px;
  letter-spacing: 0.04em;
  text-transform: uppercase;
}
.card-value {
  font-family: "Oxanium", "Rajdhani", sans-serif;
  font-size: 2.2rem;
  font-weight: 700;
  margin-bottom: 2px;
  letter-spacing: 0.02em;
}
  .card-sub {
    font-size: 0.85rem;
    color: var(--muted);
    line-height: 1.35;
  }
  .hero {
    background: linear-gradient(120deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01));
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 18px 22px;
    margin-bottom: 18px;
    box-shadow: 0 16px 34px rgba(0,0,0,0.45);
    position: relative;
    overflow: hidden;
  }
  .hero:after {
    content: "";
    position: absolute;
    inset: 0;
    background: radial-gradient(220px 120px at 86% -20%, rgba(255, 157, 43, 0.15), transparent 60%);
    pointer-events: none;
  }
  .hero-title {
    font-family: "Oxanium", "Rajdhani", sans-serif;
    font-size: 2rem;
    font-weight: 800;
    letter-spacing: 0.08em;
  }
  .hero-tagline {
    margin-top: 6px;
    color: #ff6b6b;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.12em;
  }
  .hero-sub {
    margin-top: 8px;
    color: var(--muted);
    font-size: 0.9rem;
  }
  .summary-row {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 12px;
    margin-top: 8px;
  }
  .summary-item {
    background: rgba(15, 20, 32, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 10px 12px;
    text-align: center;
  }
  .summary-item span {
    display: block;
    font-size: 0.7rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }
  .summary-item strong {
    font-family: "Oxanium", "Rajdhani", sans-serif;
    font-size: 1.1rem;
    color: #ffffff;
  }
.badge {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  background: rgba(255, 43, 43, 0.12);
  color: #ff6b6b;
  font-weight: 600;
  font-size: 0.8rem;
  border: 1px solid rgba(255, 43, 43, 0.28);
  text-transform: uppercase;
  letter-spacing: 0.05em;
}
.section-title {
  font-family: "Oxanium", "Rajdhani", sans-serif;
  font-size: 1.5rem;
  font-weight: 700;
  margin: 6px 0 12px 0;
  letter-spacing: 0.04em;
  text-transform: uppercase;
  padding-bottom: 6px;
  border-bottom: 2px solid rgba(255, 43, 43, 0.35);
}
.section-title::after {
  content: "";
  display: block;
  height: 3px;
  width: 90px;
  margin-top: 6px;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--accent), var(--accent2));
}
.callout {
  background: linear-gradient(160deg, rgba(255,43,43,0.08), rgba(18,22,31,0.9));
  border: 1px solid rgba(255,43,43,0.3);
  border-radius: 12px;
  padding: 12px 14px;
  color: #e9edf5;
  font-size: 0.95rem;
  backdrop-filter: blur(6px);
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
  backdrop-filter: blur(6px);
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
  backdrop-filter: blur(6px);
}
.policy-card {
  background: linear-gradient(160deg, rgba(23,195,255,0.08), rgba(18,22,31,0.92));
  border: 1px solid rgba(23,195,255,0.28);
  border-radius: 14px;
  padding: 12px 14px;
  color: #e9edf5;
  font-size: 0.9rem;
  box-shadow: 0 12px 24px rgba(0,0,0,0.35);
  backdrop-filter: blur(6px);
}
.policy-title {
  font-family: "Oxanium", "Rajdhani", sans-serif;
  font-size: 0.95rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 6px;
}
.policy-item {
  color: var(--muted);
  font-size: 0.85rem;
  margin-bottom: 4px;
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
.source-pill {
  display: inline-block;
  padding: 4px 10px;
  border-radius: 999px;
  font-weight: 700;
  font-size: 0.75rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  border: 1px solid rgba(255,255,255,0.16);
}
.source-model {
  background: rgba(23,195,255,0.15);
  color: #8de2ff;
}
.source-policy {
  background: rgba(255,157,43,0.16);
  color: #ffce85;
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
  background: linear-gradient(160deg, rgba(255,43,43,0.06), rgba(14,18,26,0.95));
  border: 1px solid rgba(255,43,43,0.28);
  border-radius: 18px;
  padding: 14px 16px;
  color: #e9edf5;
  box-shadow: 0 14px 30px rgba(0, 0, 0, 0.45);
}
.strategy-title {
  font-family: "Oxanium", "Rajdhani", sans-serif;
  font-size: 1.05rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
}
.strategy-sub {
  color: var(--muted);
  font-size: 0.85rem;
  margin-top: 4px;
}
.track-card {
  background: linear-gradient(160deg, rgba(23,195,255,0.08), rgba(14,18,26,0.95));
  border: 1px solid rgba(23,195,255,0.32);
  border-radius: 18px;
  padding: 14px 16px;
  color: #e9edf5;
  box-shadow: 0 14px 30px rgba(0, 0, 0, 0.45);
}
.track-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  flex-wrap: wrap;
}
.track-title {
  font-family: "Oxanium", "Rajdhani", sans-serif;
  font-size: 1.05rem;
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
.legend-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin: 4px 0 10px 0;
}
.legend-item {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 10px;
  border-radius: 999px;
  border: 1px solid #2b3342;
  background: #0b0f16;
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  color: var(--muted);
}
.legend-note {
  color: var(--muted);
  font-size: 0.72rem;
  letter-spacing: 0.02em;
  text-transform: none;
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
.telemetry-panel {
  background: linear-gradient(160deg, rgba(16,20,28,0.98), rgba(10,12,18,0.98));
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 18px;
  padding: 14px 16px;
  box-shadow: 0 14px 30px rgba(0, 0, 0, 0.45);
  color: #e9edf5;
  backdrop-filter: blur(6px);
}
.telemetry-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 10px;
}
.telemetry-head-right {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 6px;
}
.telemetry-head-left {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.telemetry-main {
  font-family: "Oxanium", "Rajdhani", sans-serif;
  font-size: 0.88rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.telemetry-sub {
  font-size: 0.75rem;
  color: var(--muted);
}
.telemetry-call {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 4px 10px;
  border-radius: 999px;
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.04);
}
.telemetry-call.pit {
  border-color: rgba(255,43,43,0.6);
  color: #ff6b6b;
}
.telemetry-call.wait {
  border-color: rgba(255,157,43,0.6);
  color: #ffb25c;
}
.telemetry-call.stay {
  border-color: rgba(23,195,255,0.6);
  color: #7be7f3;
}
.telemetry-signal {
  font-size: 0.72rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.telemetry-confidence {
  display: flex;
  flex-direction: column;
  gap: 4px;
  min-width: 120px;
}
.telemetry-conf-label {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #cdd6e2;
}
.telemetry-conf-rail {
  height: 6px;
  border-radius: 999px;
  background: #151c28;
  border: 1px solid rgba(255,255,255,0.08);
  overflow: hidden;
}
.telemetry-conf-fill {
  height: 100%;
  background: linear-gradient(90deg, rgba(23,195,255,0.7), rgba(255,157,43,0.85), rgba(225,6,0,0.95));
}
.telemetry-chip {
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.telemetry-chart {
  background: #0c111a;
  border: 1px solid rgba(255,255,255,0.08);
  border-radius: 14px;
  padding: 8px 10px;
}
.telemetry-svg {
  width: 100%;
  height: 140px;
  display: block;
}
.telemetry-delta-row {
  display: grid;
  grid-template-columns: repeat(13, minmax(0, 1fr));
  gap: 4px;
  margin-top: 6px;
}
.telemetry-delta-seg {
  height: 6px;
  border-radius: 999px;
  background: rgba(255,255,255,0.08);
}
.telemetry-delta-seg.up {
  background: linear-gradient(90deg, rgba(23,195,255,0.3), rgba(43,217,127,0.8));
}
.telemetry-delta-seg.down {
  background: linear-gradient(90deg, rgba(255,157,43,0.4), rgba(225,6,0,0.85));
}
.telemetry-delta-seg.flat {
  background: rgba(255,255,255,0.12);
}
.telemetry-delta-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.65rem;
  color: var(--muted);
  margin-top: 4px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.telemetry-bars {
  display: grid;
  gap: 8px;
  margin-top: 10px;
}
.telemetry-bar {
  display: grid;
  grid-template-columns: 80px 1fr 52px;
  align-items: center;
  gap: 10px;
  font-size: 0.74rem;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.telemetry-rail {
  height: 8px;
  border-radius: 999px;
  background: #151c28;
  border: 1px solid rgba(255,255,255,0.08);
  overflow: hidden;
}
.telemetry-fill {
  height: 100%;
}
.telemetry-fill.throttle {
  background: linear-gradient(90deg, rgba(23,195,255,0.7), rgba(43,217,127,0.9));
}
.telemetry-fill.brake {
  background: linear-gradient(90deg, rgba(255,183,3,0.7), rgba(225,6,0,0.9));
}
.telemetry-fill.wear {
  background: linear-gradient(90deg, rgba(255,183,3,0.5), rgba(225,6,0,0.8));
}
.telemetry-foot {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 8px;
}
.telemetry-tag {
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(255,255,255,0.04);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #cdd6e2;
}
.driver-card {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 14px;
  border-radius: 16px;
  border: 1px solid rgba(255,255,255,0.12);
  background: linear-gradient(160deg, rgba(16,20,28,0.96), rgba(10,12,18,0.98));
  position: relative;
  overflow: hidden;
  margin-bottom: 8px;
}
.driver-card:before {
  content: "";
  position: absolute;
  left: 0;
  top: 0;
  width: 4px;
  height: 100%;
  background: var(--team-color, #17c3ff);
}
.driver-badge {
  width: 46px;
  height: 46px;
  border-radius: 12px;
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(255,255,255,0.12);
  display: flex;
  align-items: center;
  justify-content: center;
  font-family: "Oxanium", "Rajdhani", sans-serif;
  font-size: 1.05rem;
  letter-spacing: 0.08em;
  color: #f7fafc;
}
.driver-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}
.driver-label {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
}
.driver-name {
  font-family: "Oxanium", "Rajdhani", sans-serif;
  font-size: 0.9rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.driver-meta {
  font-size: 0.74rem;
  color: var(--muted);
}
.driver-tag {
  margin-left: auto;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.18);
  background: rgba(255,255,255,0.04);
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #f1f5f9;
}
.driver-tag.pit {
  border-color: rgba(255,43,43,0.6);
  color: #ff6b6b;
}
.driver-tag.wait {
  border-color: rgba(255,157,43,0.6);
  color: #ffb25c;
}
.driver-tag.stay {
  border-color: rgba(23,195,255,0.6);
  color: #7be7f3;
}
.ladder-card {
  background: linear-gradient(160deg, rgba(16,20,28,0.96), rgba(10,12,18,0.98));
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
  padding: 12px 14px;
  box-shadow: 0 12px 26px rgba(0, 0, 0, 0.4);
  margin-top: 8px;
}
.ladder-title {
  font-size: 0.7rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin-bottom: 8px;
}
.ladder-steps {
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 8px;
}
.ladder-step {
  padding: 8px 6px;
  border-radius: 12px;
  border: 1px solid rgba(255,255,255,0.1);
  background: rgba(255,255,255,0.03);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  text-align: center;
  color: #cdd6e2;
}
.ladder-step.active.pit {
  border-color: rgba(225,6,0,0.7);
  color: #ff6b6b;
  box-shadow: 0 0 12px rgba(225,6,0,0.25);
}
.ladder-step.active.wait {
  border-color: rgba(255,157,43,0.7);
  color: #ffb25c;
  box-shadow: 0 0 12px rgba(255,157,43,0.25);
}
.ladder-step.active.stay {
  border-color: rgba(23,195,255,0.7);
  color: #7be7f3;
  box-shadow: 0 0 12px rgba(23,195,255,0.25);
}
.reliability-ribbon {
  display: flex;
  gap: 10px;
  align-items: center;
  padding: 10px 12px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  background: rgba(16,20,28,0.9);
  margin: 10px 0 6px;
  flex-wrap: wrap;
}
.reliability-pill {
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.14);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #cdd6e2;
}
.reliability-pill.high {
  border-color: rgba(43,217,127,0.6);
  color: #2bd97f;
}
.reliability-pill.med {
  border-color: rgba(255,157,43,0.6);
  color: #ffb25c;
}
.reliability-pill.low {
  border-color: rgba(225,6,0,0.6);
  color: #ff6b6b;
}
.audit-badge {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  padding: 6px 12px;
  border-radius: 999px;
  border: 1px solid rgba(255,255,255,0.12);
  font-size: 0.72rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  margin-bottom: 8px;
}
.audit-pass {
  border-color: rgba(43,217,127,0.6);
  color: #2bd97f;
}
.audit-fail {
  border-color: rgba(225,6,0,0.6);
  color: #ff6b6b;
}
.telemetry-split {
  margin-top: 8px;
}
.telemetry-split-label {
  font-size: 0.68rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--muted);
  margin-bottom: 4px;
}
.telemetry-split-rail {
  position: relative;
  height: 6px;
  border-radius: 999px;
  background: #151c28;
  border: 1px solid rgba(255,255,255,0.08);
}
.telemetry-split-marker {
  position: absolute;
  top: -4px;
  width: 3px;
  height: 14px;
  background: #17c3ff;
  box-shadow: 0 0 8px rgba(23,195,255,0.6);
}
.telemetry-split-meta {
  font-size: 0.68rem;
  color: #cdd6e2;
  margin-top: 4px;
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
.mini-tower {
  background: linear-gradient(160deg, rgba(16,20,28,0.96), rgba(10,12,18,0.98));
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 16px;
  padding: 12px 14px;
  box-shadow: 0 12px 26px rgba(0, 0, 0, 0.45);
  margin-bottom: 12px;
}
.mini-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  font-size: 0.8rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #cdd6e2;
  margin-bottom: 8px;
}
.mini-head span {
  color: var(--muted);
  font-size: 0.7rem;
}
.mini-rows {
  display: grid;
  gap: 6px;
}
.mini-row {
  display: grid;
  grid-template-columns: 34px 1fr 62px 62px;
  align-items: center;
  gap: 8px;
  padding: 6px 8px;
  border-radius: 12px;
  background: rgba(14,18,26,0.9);
  border: 1px solid rgba(255,255,255,0.08);
}
.mini-row.best {
  border-color: rgba(255,157,43,0.6);
  box-shadow: 0 0 12px rgba(255,157,43,0.18);
}
.mini-stage {
  font-family: "Oxanium", "Rajdhani", sans-serif;
  font-weight: 700;
  letter-spacing: 0.08em;
}
.mini-bar {
  height: 8px;
  border-radius: 999px;
  background: #151c28;
  border: 1px solid rgba(255,255,255,0.08);
  overflow: hidden;
}
.mini-fill {
  height: 100%;
  background: linear-gradient(90deg, #2c3545, #6e7888);
}
.mini-row.my .mini-fill {
  background: linear-gradient(90deg, #ff9d2b, #ff2b2b);
}
.mini-val {
  text-align: right;
  font-weight: 700;
  font-size: 0.78rem;
}
.mini-delta {
  text-align: right;
  font-weight: 700;
  font-size: 0.74rem;
}
div[data-baseweb="tab-list"] {
  gap: 8px;
}
button[data-baseweb="tab"] {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(255,255,255,0.12);
  border-radius: 999px !important;
  padding: 6px 14px;
  font-family: "Oxanium", "Rajdhani", sans-serif;
  font-size: 0.78rem;
  text-transform: uppercase;
  letter-spacing: 0.06em;
  color: #cdd6e2;
}
button[data-baseweb="tab"][aria-selected="true"] {
  background: linear-gradient(90deg, rgba(225,6,0,0.25), rgba(15,19,27,0.95));
  border-color: rgba(225,6,0,0.6);
  color: #fff5f5;
  box-shadow: 0 0 16px rgba(225,6,0,0.25);
}
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div,
div[data-baseweb="textarea"] > div {
  background: rgba(255,255,255,0.04) !important;
  border: 1px solid rgba(255,255,255,0.12) !important;
  border-radius: 12px !important;
  color: var(--ink);
}
@keyframes fadein {
  from { opacity: 0; transform: translateY(6px); }
  to { opacity: 1; transform: translateY(0); }
}
</style>
""",
        unsafe_allow_html=True,
    )


def _broadcast_ticker_html(items: list[str]) -> str:
    safe_items = [html.escape(item) for item in items if item]
    payload = json.dumps(safe_items[:4])
    return f"""
<div class="rc-strip">
  <div class="rc-title">Race Control</div>
  <div class="rc-items" id="rc-items"></div>
</div>
<script>
const rcItems = {payload};
const rcRoot = document.getElementById("rc-items");
rcItems.forEach((text, idx) => {{
  const item = document.createElement("div");
  item.className = "rc-item" + (idx === 0 ? " active" : "");
  item.textContent = text;
  rcRoot.appendChild(item);
}});
let idx = 0;
setInterval(() => {{
  const nodes = rcRoot.querySelectorAll(".rc-item");
  if (!nodes.length) return;
  nodes.forEach(n => n.classList.remove("active"));
  nodes[idx % nodes.length].classList.add("active");
  idx += 1;
}}, 2400);
</script>
<style>
  .rc-strip {{
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 999px;
    background: linear-gradient(90deg, rgba(225,6,0,0.18), rgba(12,16,24,0.95));
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 6px 12px;
    margin: 6px 0 12px 0;
    overflow: hidden;
    font-family: "Rajdhani", sans-serif;
  }}
  .rc-title {{
    font-family: "Oxanium", "Rajdhani", sans-serif;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #ffd166;
    font-weight: 700;
    padding-right: 10px;
    border-right: 1px solid rgba(255,255,255,0.15);
  }}
  .rc-items {{
    display: flex;
    gap: 8px;
    flex-wrap: wrap;
  }}
  .rc-item {{
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #cdd6e2;
    padding: 4px 10px;
    border-radius: 999px;
    border: 1px solid rgba(255,255,255,0.12);
    background: rgba(255,255,255,0.04);
    transition: all 0.2s ease;
  }}
  .rc-item.active {{
    border-color: rgba(255,157,43,0.8);
    color: #fff5e6;
    box-shadow: 0 0 12px rgba(255,157,43,0.35);
  }}
</style>
"""


def _timing_tower_html(
    rows: list[dict],
    metric_label: str,
    dom_id: str = "tower-rows-main",
    title: str = "Timing Tower",
    show_legend: bool = True,
) -> str:
    safe_label = html.escape(metric_label)
    safe_title = html.escape(title)
    safe_id = html.escape(dom_id)
    js_id = dom_id.replace("-", "_")
    payload = json.dumps(rows)
    legend_html = (
        '<div class="tower-legend">'
        '<span class="tower-dot ref"></span>RefTech'
        '<span class="tower-dot my"></span>MyMethod'
        "</div>"
        if show_legend
        else ""
    )
    return f"""
<div class="tower-wrap">
  <div class="tower-head">
    <div class="tower-title">{safe_title}</div>
    <div class="tower-sub">{safe_label}</div>
  </div>
  <div class="tower-rows" id="{safe_id}"></div>
  {legend_html}
</div>
<script>
const towerRows_{js_id} = {payload};
const maxVal_{js_id} = Math.max(...towerRows_{js_id}.map(r => r.value || 0), 0.0001);
const bestVal_{js_id} = Math.max(...towerRows_{js_id}.map(r => r.value || 0), 0);
const root_{js_id} = document.getElementById("{safe_id}");
if (root_{js_id}) {{
  towerRows_{js_id}.forEach((row) => {{
    const wrap = document.createElement("div");
    const isBest = Math.abs((row.value || 0) - bestVal_{js_id}) < 1e-6;
    wrap.className = "tower-row" + (row.method === "MyMethod" ? " is-my" : " is-ref") + (isBest ? " is-best" : "");
    const stage = document.createElement("div");
    stage.className = "tower-stage";
    stage.textContent = row.stage;
    const bar = document.createElement("div");
    bar.className = "tower-bar";
    const fill = document.createElement("div");
    fill.className = "tower-fill";
    fill.style.width = ((row.value || 0) / maxVal_{js_id} * 100).toFixed(1) + "%";
    bar.appendChild(fill);
    const val = document.createElement("div");
    val.className = "tower-value";
    val.textContent = (row.value ?? 0).toFixed(3);
    const delta = document.createElement("div");
    delta.className = "tower-delta";
    if (row.delta === null || row.delta === undefined) {{
      delta.textContent = "N/A";
      delta.classList.add("delta-flat");
    }} else {{
      const d = row.delta;
      delta.textContent = (d >= 0 ? "+" : "") + d.toFixed(3);
      delta.classList.add(d > 0 ? "delta-up" : d < 0 ? "delta-down" : "delta-flat");
    }}
    wrap.appendChild(stage);
    wrap.appendChild(bar);
    wrap.appendChild(val);
    wrap.appendChild(delta);
    root_{js_id}.appendChild(wrap);
  }});
}}
</script>
<style>
  .tower-wrap {{
    font-family: "Rajdhani", sans-serif;
    background: linear-gradient(180deg, rgba(16,20,28,0.95), rgba(10,12,18,0.98));
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 18px;
    padding: 12px 16px 14px 16px;
    color: #e9edf5;
  }}
  .tower-head {{
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: 8px;
  }}
  .tower-title {{
    font-family: "Oxanium", "Rajdhani", sans-serif;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #e6ebf2;
  }}
  .tower-sub {{
    font-size: 0.7rem;
    color: #a6adbb;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}
  .tower-rows {{
    display: grid;
    gap: 8px;
  }}
  .tower-row {{
    display: grid;
    grid-template-columns: 48px 1fr 70px 70px;
    align-items: center;
    gap: 10px;
    padding: 6px 8px;
    border-radius: 12px;
    background: rgba(14,18,26,0.92);
    border: 1px solid rgba(255,255,255,0.08);
  }}
  .tower-row.is-best {{
    border-color: rgba(255,157,43,0.7);
    box-shadow: 0 0 12px rgba(255,157,43,0.2);
  }}
  .tower-stage {{
    font-family: "Oxanium", "Rajdhani", sans-serif;
    font-weight: 700;
    color: #d9e0ea;
    letter-spacing: 0.08em;
  }}
  .tower-bar {{
    height: 8px;
    border-radius: 999px;
    background: #151c28;
    border: 1px solid rgba(255,255,255,0.08);
    overflow: hidden;
  }}
  .tower-fill {{
    height: 100%;
    background: linear-gradient(90deg, #2c3545, #6e7888);
  }}
  .tower-row.is-my .tower-fill {{
    background: linear-gradient(90deg, #ff9d2b, #ff2b2b);
  }}
  .tower-value {{
    text-align: right;
    font-weight: 700;
    font-size: 0.78rem;
  }}
  .tower-delta {{
    text-align: right;
    font-size: 0.75rem;
    font-weight: 700;
  }}
  .tower-legend {{
    margin-top: 8px;
    display: flex;
    gap: 12px;
    align-items: center;
    font-size: 0.72rem;
    color: #9aa4b3;
  }}
  .tower-dot {{
    width: 8px;
    height: 8px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
  }}
  .tower-dot.ref {{ background: #6e7888; }}
  .tower-dot.my {{ background: #ff2b2b; }}
</style>
"""


def _mini_tower_html(rows: list[dict], metric_label: str) -> str:
    if not rows:
        return ""
    best_val = max((r.get("value", 0.0) or 0.0) for r in rows)
    max_val = max(best_val, 1e-6)
    html_rows = []
    for row in rows:
        stage = html.escape(str(row.get("stage", "S?")))
        val = float(row.get("value", 0.0) or 0.0)
        width = min(max(val / max_val * 100.0, 0.0), 100.0)
        method = row.get("method", "RefTech")
        is_my = method == "MyMethod"
        is_best = abs(val - best_val) < 1e-6
        delta = row.get("delta")
        if delta is None or not np.isfinite(delta):
            delta_text = "-"
            delta_class = "delta-flat"
        else:
            delta_text = f"{delta:+.3f}"
            delta_class = "delta-up" if delta > 0 else "delta-down" if delta < 0 else "delta-flat"
        row_class = "mini-row"
        if is_my:
            row_class += " my"
        if is_best:
            row_class += " best"
        html_rows.append(
            "<div class='{row_class}'>"
            f"<div class='mini-stage'>{stage}</div>"
            "<div class='mini-bar'>"
            f"<div class='mini-fill' style='width:{width:.1f}%'></div>"
            "</div>"
            f"<div class='mini-val'>{val:.3f}</div>"
            f"<div class='mini-delta {delta_class}'>{delta_text}</div>"
            "</div>".format(row_class=row_class)
        )
    safe_label = html.escape(metric_label)
    return (
        "<div class='mini-tower'>"
        "<div class='mini-head'>Mini timing tower <span>"
        f"{safe_label}"
        "</span></div>"
        "<div class='mini-rows'>"
        + "".join(html_rows)
        + "</div></div>"
    )


def _pit_window_gauge_html(
    title: str,
    lap_min: int | None,
    lap_max: int | None,
    lap_now: int | None,
    window_start: int | None,
    window_end: int | None,
    target_lap: int | None,
) -> str:
    def _pos(val: int | None) -> float:
        if val is None or lap_min is None or lap_max is None or lap_max <= lap_min:
            return 0.0
        return float(max(0.0, min(1.0, (val - lap_min) / (lap_max - lap_min))))

    now_pos = _pos(lap_now) * 100.0
    win_start = _pos(window_start) * 100.0
    win_end = _pos(window_end) * 100.0
    target_pos = _pos(target_lap) * 100.0
    if window_start is None or window_end is None:
        win_start = 0.0
        win_end = 0.0
    safe_title = html.escape(title)
    return f"""
<div class="gauge-wrap">
  <div class="gauge-title">{safe_title}</div>
  <div class="gauge-track">
    <div class="gauge-window" style="left:{win_start:.1f}%; width:{max(2.0, win_end - win_start):.1f}%;"></div>
    <div class="gauge-tick in" style="left:{win_start:.1f}%;"><span>IN</span></div>
    <div class="gauge-tick target" style="left:{target_pos:.1f}%;"><span>TARGET</span></div>
    <div class="gauge-tick out" style="left:{win_end:.1f}%;"><span>OUT</span></div>
    <div class="gauge-now" style="left:{now_pos:.1f}%;"><span>NOW</span></div>
  </div>
  <div class="gauge-labels"><span>L{lap_min if lap_min is not None else 'N/A'}</span><span>L{lap_max if lap_max is not None else 'N/A'}</span></div>
</div>
<style>
  .gauge-wrap {{
    font-family: "Rajdhani", sans-serif;
    background: rgba(14,18,26,0.92);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 16px;
    padding: 10px 12px;
    color: #e9edf5;
  }}
  .gauge-title {{
    font-family: "Oxanium", "Rajdhani", sans-serif;
    font-size: 0.72rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #a6adbb;
    margin-bottom: 6px;
  }}
  .gauge-track {{
    position: relative;
    height: 10px;
    border-radius: 999px;
    background: #151c28;
    border: 1px solid rgba(255,255,255,0.08);
    overflow: hidden;
  }}
  .gauge-window {{
    position: absolute;
    top: 0;
    bottom: 0;
    background: linear-gradient(90deg, rgba(255,157,43,0.25), rgba(255,157,43,0.95));
  }}
  .gauge-now {{
    position: absolute;
    top: -8px;
    width: 3px;
    height: 24px;
    background: #28c1d6;
    box-shadow: 0 0 10px rgba(40,193,214,0.7);
  }}
  .gauge-now span {{
    position: absolute;
    top: -14px;
    left: -10px;
    font-size: 0.6rem;
    color: #7be7f3;
    letter-spacing: 0.08em;
  }}
  .gauge-tick {{
    position: absolute;
    top: -6px;
    width: 2px;
    height: 20px;
    background: rgba(255,255,255,0.4);
  }}
  .gauge-tick span {{
    position: absolute;
    top: -14px;
    left: -10px;
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #cdd6e2;
  }}
  .gauge-tick.in {{
    background: rgba(255,157,43,0.8);
  }}
  .gauge-tick.in span {{
    color: #ffb25c;
  }}
  .gauge-tick.out {{
    background: rgba(225,6,0,0.8);
  }}
  .gauge-tick.out span {{
    color: #ff6b6b;
  }}
  .gauge-tick.target {{
    background: rgba(23,195,255,0.8);
  }}
  .gauge-tick.target span {{
    color: #7be7f3;
  }}
  .gauge-labels {{
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    color: #a6adbb;
    margin-top: 4px;
  }}
</style>
"""


@st.cache_data
def _load_summary(path: Path, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path)
    stage_match = df["stage"].astype(str).str.extract(r"Stage\s+(\d+)")
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
    stage_match = df["stage"].astype(str).str.extract(r"Stage\s+(\d+)")
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


def _gap_percentile(df_context: pd.DataFrame, row: pd.Series) -> float | None:
    gap_col = _pick_column(
        df_context,
        [
            "gap_to_leader_prev",
            "gap_to_leader",
            "gap",
            "gap_to_front_prev",
            "interval",
        ],
    )
    if not gap_col or gap_col not in df_context.columns or gap_col not in row:
        return None
    vals = pd.to_numeric(df_context[gap_col], errors="coerce").dropna()
    if vals.empty:
        return None
    try:
        target = float(row[gap_col])
    except Exception:
        return None
    if not np.isfinite(target):
        return None
    rank = float((vals <= target).mean())
    return float((1.0 - rank) * 100.0)


def _threshold_for_precision(
    y_true: np.ndarray, probs: np.ndarray, target_precision: float = 0.6
) -> float | None:
    if y_true.size == 0 or probs.size == 0:
        return None
    order = np.argsort(probs)[::-1]
    y_sorted = y_true[order]
    p_sorted = probs[order]
    tp = 0
    fp = 0
    best = None
    for i in range(len(p_sorted)):
        if y_sorted[i] == 1:
            tp += 1
        else:
            fp += 1
        prec = tp / max(1, tp + fp)
        if prec >= target_precision:
            best = float(p_sorted[i])
            break
    return best


def _apply_confirm(
    df_context: pd.DataFrame,
    proba_col: str,
    lap_col: str | None,
    threshold: float,
    confirm_laps: int = 2,
) -> pd.Series:
    if lap_col is None or lap_col not in df_context.columns:
        return pd.Series([False] * len(df_context), index=df_context.index)
    df_sorted = df_context.copy()
    df_sorted[lap_col] = pd.to_numeric(df_sorted[lap_col], errors="coerce")
    df_sorted = df_sorted.dropna(subset=[lap_col]).sort_values(lap_col)
    hits = (df_sorted[proba_col] >= threshold).astype(int).rolling(confirm_laps).sum() >= confirm_laps
    out = pd.Series(False, index=df_context.index)
    out.loc[df_sorted.index] = hits.fillna(False)
    return out


def _smooth_prob_by_lap(
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


def _leakage_audit(df: pd.DataFrame) -> list[str]:
    risk_fields = {
        "lapno",
        "pitstops_so_far",
        "position",
        "race_progress",
        "tireage",
        "stint_laps",
        "gap",
        "interval",
    }
    flagged = []
    for col in df.columns:
        lower = col.lower()
        if lower in risk_fields and not lower.endswith("_prev"):
            flagged.append(col)
    return sorted(flagged)


def _telemetry_panel_html(
    panel_label: str,
    driver: str,
    lap_text: str,
    circuit: str,
    weather: str,
    row: pd.Series,
    payload: dict,
    lap_value: int | None,
    proba: float | None,
    proba_raw: float | None,
    threshold: float | None,
    fold_std: float | None,
    gap_percentile: float | None,
) -> str:
    def _num(val: float | str | None, default: float) -> float:
        if val is None:
            return default
        try:
            out = float(val)
            if not np.isfinite(out):
                return default
            return out
        except Exception:
            return default

    pace = _first_valid(row, ["relative_pace_prev", "relative_pace"])
    pace = _num(pace, 1.0)
    pace = float(np.clip(pace, 0.75, 1.25))
    base_speed = 320.0 / pace
    base_speed = float(np.clip(base_speed, 260.0, 340.0))

    wear_pct = payload.get("tire_wear_pct")
    wear_ratio = _num(wear_pct, 40.0)
    wear_ratio = wear_ratio / 100.0 if wear_ratio > 1.5 else wear_ratio
    wear_ratio = float(np.clip(wear_ratio, 0.05, 0.95))

    volatility = 18.0 + wear_ratio * 12.0
    phase = (lap_value or 0) * 0.35
    xs = np.linspace(0, 2 * np.pi, 14)
    series = (
        base_speed
        + volatility * np.sin(xs + phase)
        + volatility * 0.35 * np.sin(xs * 1.8 + phase * 0.5)
    )
    series = np.clip(series, 180.0, 360.0)
    min_s = float(np.min(series))
    max_s = float(np.max(series))
    if max_s - min_s < 1.0:
        max_s = min_s + 1.0

    width = 320.0
    height = 100.0
    points = []
    for idx, val in enumerate(series):
        x = idx / (len(series) - 1) * width
        y = height - (float(val) - min_s) / (max_s - min_s) * height
        points.append((x, y))
    path = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in points)
    area = f"{path} L {width:.1f} {height:.1f} L 0 {height:.1f} Z"

    deltas = np.diff(series)
    max_delta = float(np.max(np.abs(deltas))) if deltas.size else 0.0
    if max_delta <= 0.0:
        max_delta = 1.0
    delta_segments = []
    for d in deltas:
        cls = "flat"
        if d > 0.5:
            cls = "up"
        elif d < -0.5:
            cls = "down"
        opacity = 0.35 + 0.65 * min(abs(d) / max_delta, 1.0)
        delta_segments.append(
            f"<div class='telemetry-delta-seg {cls}' title='{d:+.1f} km/h' "
            f"style='opacity:{opacity:.2f};'></div>"
        )
    delta_html = "".join(delta_segments)

    sc_active = _num(_first_valid(row, ["sc_active_prev", "sc_active"]), 0.0) > 0
    throttle = 92.0 - wear_ratio * 30.0 - (12.0 if sc_active else 0.0)
    brake = 8.0 + wear_ratio * 28.0 + (12.0 if sc_active else 0.0)
    throttle = float(np.clip(throttle, 5.0, 100.0))
    brake = float(np.clip(brake, 5.0, 100.0))
    wear_display = float(np.clip(wear_ratio * 100.0, 0.0, 100.0))

    delta_val = _first_valid(row, ["delta_interval_prev", "delta_interval"])
    delta_text = _format_seconds(_num(delta_val, np.nan))
    if delta_text != "N/A":
        delta_text = f"{delta_text}"

    decision = str(payload.get("decision", "STAY OUT")).upper()
    if "BOX" in decision:
        call_class = "pit"
    elif "STANDBY" in decision:
        call_class = "wait"
    else:
        call_class = "stay"

    gap = None
    if proba is not None and threshold is not None and np.isfinite(proba) and np.isfinite(threshold):
        gap = abs(float(proba) - float(threshold))
    conf_pct = 0.0 if gap is None else min(gap / 0.25, 1.0) * 100.0
    std_text = f"STD {fold_std:.2f}" if fold_std is not None and np.isfinite(fold_std) else "STD N/A"
    conf_label = "CONF N/A" if gap is None else f"|P-T| {gap:.2f}"
    conf_label = f"{conf_label} | {std_text}"

    signal_text = "Signal N/A"
    if proba is not None and np.isfinite(proba):
        if proba_raw is not None and np.isfinite(proba_raw):
            signal_text = f"Signal RAW {proba_raw:.2f} | SMOOTH {proba:.2f}"
        else:
            signal_text = f"Signal {proba:.2f}"

    subtitle = f"{driver} | {lap_text} | {circuit} | {weather}"
    safe_title = html.escape(panel_label)
    safe_sub = html.escape(subtitle)
    safe_decision = html.escape(decision)
    safe_conf = html.escape(conf_label)
    safe_signal = html.escape(signal_text)
    speed_text = f"{base_speed:.0f} km/h"
    speed_tag = html.escape(speed_text)
    delta_tag = html.escape(delta_text)
    wear_tag = f"{wear_display:.0f}% wear"
    gap_pct = None
    if gap_percentile is not None and np.isfinite(gap_percentile):
        gap_pct = float(np.clip(gap_percentile, 0.0, 100.0))

    return f"""
<div class="telemetry-panel">
  <div class="telemetry-head">
    <div class="telemetry-head-left">
      <div class="telemetry-main">{safe_title}</div>
      <div class="telemetry-sub">{safe_sub}</div>
    </div>
    <div class="telemetry-head-right">
      <div class="telemetry-call {call_class}">{safe_decision}</div>
      <div class="telemetry-confidence">
        <div class="telemetry-conf-label">{safe_conf}</div>
        <div class="telemetry-conf-rail">
          <div class="telemetry-conf-fill" style="width:{conf_pct:.1f}%;"></div>
        </div>
      </div>
      <div class="telemetry-signal">{safe_signal}</div>
    </div>
  </div>
  <div class="telemetry-chart">
    <svg viewBox="0 0 320 110" class="telemetry-svg" preserveAspectRatio="none">
      <defs>
        <linearGradient id="telemetryLine" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stop-color="#17c3ff"/>
          <stop offset="55%" stop-color="#ff9d2b"/>
          <stop offset="100%" stop-color="#e10600"/>
        </linearGradient>
        <linearGradient id="telemetryFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stop-color="rgba(23,195,255,0.25)"/>
          <stop offset="100%" stop-color="rgba(11,13,18,0.0)"/>
        </linearGradient>
      </defs>
      <path d="{area}" fill="url(#telemetryFill)"></path>
      <path d="{path}" stroke="url(#telemetryLine)" stroke-width="2.2" fill="none"></path>
    </svg>
  </div>
  <div class="telemetry-delta-row">
    {delta_html}
  </div>
  <div class="telemetry-delta-labels">
    <span>Delta band</span>
    <span>+/-{max_delta:.1f} km/h</span>
  </div>
  <div class="telemetry-bars">
    <div class="telemetry-bar">
      <span>Throttle</span>
      <div class="telemetry-rail"><div class="telemetry-fill throttle" style="width:{throttle:.1f}%;"></div></div>
      <strong>{throttle:.0f}%</strong>
    </div>
    <div class="telemetry-bar">
      <span>Brake</span>
      <div class="telemetry-rail"><div class="telemetry-fill brake" style="width:{brake:.1f}%;"></div></div>
      <strong>{brake:.0f}%</strong>
    </div>
    <div class="telemetry-bar">
      <span>Tyre</span>
      <div class="telemetry-rail"><div class="telemetry-fill wear" style="width:{wear_display:.1f}%;"></div></div>
      <strong>{wear_display:.0f}%</strong>
    </div>
  </div>
  <div class="telemetry-split">
    <div class="telemetry-split-label">Gap percentile</div>
  <div class="telemetry-split-rail">
      <div class="telemetry-split-marker" style="left:{(0 if gap_pct is None else gap_pct):.1f}%;"></div>
  </div>
    <div class="telemetry-split-meta">{'N/A' if gap_pct is None else f'{gap_pct:.0f}% closer than field'}</div>
  </div>
  <div class="telemetry-foot">
    <span class="telemetry-chip">Speed trap {speed_tag}</span>
    <span class="telemetry-tag">Delta {delta_tag}</span>
    <span class="telemetry-tag">Wear {wear_tag}</span>
  </div>
</div>
"""


def _driver_team_info(driver_code: str) -> tuple[str, str]:
    mapping = {
        "VER": ("Red Bull", "#1e5bc6"),
        "PER": ("Red Bull", "#1e5bc6"),
        "LEC": ("Ferrari", "#dc0000"),
        "SAI": ("Ferrari", "#dc0000"),
        "HAM": ("Mercedes", "#00d2be"),
        "RUS": ("Mercedes", "#00d2be"),
        "NOR": ("McLaren", "#ff8700"),
        "PIA": ("McLaren", "#ff8700"),
        "ALO": ("Aston Martin", "#006f62"),
        "STR": ("Aston Martin", "#006f62"),
        "GAS": ("Alpine", "#0090ff"),
        "OCO": ("Alpine", "#0090ff"),
        "ALB": ("Williams", "#005aff"),
        "SAR": ("Williams", "#005aff"),
        "TSU": ("RB", "#2b4562"),
        "RIC": ("RB", "#2b4562"),
        "BOT": ("Kick Sauber", "#00e701"),
        "ZHO": ("Kick Sauber", "#00e701"),
        "HUL": ("Haas", "#b6babd"),
        "MAG": ("Haas", "#b6babd"),
    }
    code = driver_code.upper()
    return mapping.get(code, ("Independent", "#6e7888"))


def _driver_card_html(
    label: str,
    driver: str,
    lap_text: str,
    circuit: str,
    weather: str,
    decision: str,
) -> str:
    code = (driver or "DRV").upper()[:3]
    team_name, team_color = _driver_team_info(code)
    decision_text = (decision or "STAY OUT").upper()
    if "BOX" in decision_text:
        tag_class = "pit"
    elif "STANDBY" in decision_text:
        tag_class = "wait"
    else:
        tag_class = "stay"
    meta = f"{lap_text} | {circuit} | {weather}"
    return (
        f"<div class='driver-card' style='--team-color:{team_color};'>"
        f"<div class='driver-badge'>{html.escape(code)}</div>"
        "<div class='driver-info'>"
        f"<div class='driver-label'>{html.escape(label)}</div>"
        f"<div class='driver-name'>{html.escape(driver)}</div>"
        f"<div class='driver-meta'>{html.escape(team_name)} | {html.escape(meta)}</div>"
        "</div>"
        f"<div class='driver-tag {tag_class}'>{html.escape(decision_text)}</div>"
        "</div>"
    )


def _decision_ladder_html(label: str, decision: str) -> str:
    steps = [("STAY OUT", "stay"), ("STANDBY", "wait"), ("BOX BOX", "pit")]
    decision = (decision or "STAY OUT").upper()
    items = []
    for name, cls in steps:
        active = "active" if name in decision else ""
        items.append(
            f"<div class='ladder-step {active} {cls}'>{html.escape(name)}</div>"
        )
    return (
        "<div class='ladder-card'>"
        f"<div class='ladder-title'>{html.escape(label)}</div>"
        "<div class='ladder-steps'>"
        + "".join(items)
        + "</div></div>"
    )


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
            decision_margin=0.05,
            window_start=None,
            window_end=None,
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


def _reliability_label(rows: int, groups: int | None, pos_rate: float | None) -> str:
    score = 0
    if rows >= 800:
        score += 2
    elif rows >= 300:
        score += 1
    if groups is not None:
        if groups >= 10:
            score += 2
        elif groups >= 4:
            score += 1
    if pos_rate is not None and pos_rate >= 0.02:
        score += 1
    if score >= 4:
        return "High"
    if score >= 2:
        return "Medium"
    return "Low"


def _scenario_key_cols(circuit_col: str | None) -> list[str]:
    cols = ["Driver", "weather_label"]
    if circuit_col:
        cols.insert(1, circuit_col)
    return cols


def _pick_default_scenario(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    circuit_col: str | None,
) -> dict[str, str] | None:
    key_cols = _scenario_key_cols(circuit_col)
    for col in key_cols:
        if col not in train_df.columns or col not in test_df.columns:
            return None
    train_keys = train_df[key_cols].dropna()
    test_keys = test_df[key_cols].dropna()
    if train_keys.empty or test_keys.empty:
        return None
    common = train_keys.drop_duplicates().merge(
        test_keys.drop_duplicates(), on=key_cols, how="inner"
    )
    if common.empty:
        return None
    counts = train_keys.value_counts().reset_index(name="count")
    common = common.merge(counts, on=key_cols, how="left").sort_values(
        "count", ascending=False
    )
    row = common.iloc[0]
    return {col: str(row[col]) for col in key_cols}


def _find_monaco_double_stack(df: pd.DataFrame, lap_col: str | None) -> list[pd.Series]:
    if df.empty or "race_id" not in df.columns or "Driver" not in df.columns:
        return []
    monaco = df[df["race_id"].astype(str) == "2022_Monaco"].copy()
    if monaco.empty:
        return []
    drivers = ["LEC", "SAI"]
    rows: list[pd.Series] = []
    target_lap = 21
    for drv in drivers:
        subset = monaco[monaco["Driver"].astype(str) == drv]
        if subset.empty:
            continue
        prefer = subset[subset["decide_pitstop"] == 1]
        pick = prefer if not prefer.empty else subset
        if lap_col and lap_col in pick.columns:
            laps = pd.to_numeric(pick[lap_col], errors="coerce").fillna(target_lap)
            idx = (laps - target_lap).abs().idxmin()
            rows.append(pick.loc[idx])
        else:
            rows.append(pick.iloc[0])
    return rows


def _apply_scenario_filters(
    df: pd.DataFrame,
    driver: str | None,
    circuit_col: str | None,
    circuit_sel: str | None,
    weather_sel: str | None,
) -> pd.DataFrame:
    out = df
    if driver is not None:
        out = out[out["Driver"] == driver]
    if circuit_col and circuit_sel:
        out = out[out[circuit_col].astype(str) == circuit_sel]
    if weather_sel:
        out = out[out["weather_label"] == weather_sel]
    return out


def _reason_phrase(reason_text: str) -> str:
    mapping = {
        "WINDOW": "window open",
        "WEAR": "high tyre wear",
        "WEAR-URGENT": "critical tyre wear",
        "WEAR-CRIT": "extreme tyre wear",
        "WINDOW-SOON": "pit window soon",
        "SC": "safety car",
        "VSC": "virtual safety car",
        "LATE": "late race",
        "NOWINDOW": "window closed",
        "COST-": "costly stop",
        "COST+": "time gain",
        "COST-HOLD": "cost warning",
        "COOLDOWN": "new tyres cooldown",
        "CONFIRM": "confirm 2 laps",
        "CAP": "alert cap",
        "LOCK": "policy lock",
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
    window_text = window.lower() if window != "N/A" else "unknown"
    if decision.startswith("BOX"):
        return f"Window {window_text} with strong signal: BOX (signals: {reasons})."
    if decision.startswith("STANDBY"):
        return f"Window {window_text} with mixed signal: STANDBY (signals: {reasons})."
    return f"Window {window_text} and signal low: STAY OUT (signals: {reasons})."


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


def _policy_summary_html(
    decision_threshold: float,
    target_precision: float,
    precision_guard: float,
    alert_cap: float,
    smooth_window: int,
    confirm_laps: int,
) -> str:
    alert_pct = int(round(alert_cap * 100))
    return (
        "<div class='policy-card'>"
        "<div class='policy-title'>Decision Policy (Demo)</div>"
        f"<div class='policy-item'>Signal: calibrated probability, smoothed over {smooth_window} laps.</div>"
        f"<div class='policy-item'>Threshold: {decision_threshold:.2f} "
        f"(precision target {target_precision:.2f}, guard +{precision_guard:.2f}).</div>"
        f"<div class='policy-item'>Stability: confirm {confirm_laps} laps, alert cap top {alert_pct}%.</div>"
        "<div class='policy-item'>Overrides: pit window, tyre wear, SC/VSC, cooldown.</div>"
        "</div>"
    )


def _holdout_tower_data(
    holdout: pd.DataFrame | None,
) -> tuple[list[dict] | None, str, str]:
    if holdout is None or holdout.empty:
        return None, "", "Holdout 70/30 (Stage 3/4): summary not available."
    if "stage_id" not in holdout.columns and "stage" in holdout.columns:
        stage_match = holdout["stage"].astype(str).str.extract(r"Stage\s+(\d+)")
        holdout = holdout.copy()
        holdout["stage_id"] = pd.to_numeric(stage_match[0], errors="coerce").fillna(0).astype(int)
    s3 = holdout.loc[holdout["stage_id"] == 3]
    s4 = holdout.loc[holdout["stage_id"] == 4]
    if s3.empty or s4.empty:
        return None, "", "Holdout 70/30 (Stage 3/4): missing Stage 3 or Stage 4 rows."
    f1_s3 = float(s3["mean_f1"].iloc[0])
    f1_s4 = float(s4["mean_f1"].iloc[0])
    delta = f1_s4 - f1_s3
    rows = [
        {"stage": "S3", "value": f1_s3, "method": "RefTech", "delta": None},
        {"stage": "S4", "value": f1_s4, "method": "MyMethod", "delta": delta},
    ]
    note = "Group holdout by race (train/test = 70/30)."
    return rows, note, ""


def _render_track_demo(
    panel_label: str,
    driver: str,
    lap_text: str,
    circuit_text: str,
    weather_text: str,
    decision: str,
    decision_source: str | None,
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
    source_text = (decision_source or "MODEL").upper()
    if source_text not in ("MODEL", "POLICY"):
        source_text = "MODEL"
    source_class = "source-model" if source_text == "MODEL" else "source-policy"
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
      <span class='source-pill {source_class}'>{source_text}</span>
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
    return allow if allow else features


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
    decision_margin: float = 0.05,
    window_start: int | None = None,
    window_end: int | None = None,
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
        if (
            window_start is not None
            and window_end is not None
            and lap_value is not None
        ):
            if window_start <= lap_value <= window_end:
                pit_window_text = "OPEN"
            else:
                pit_window_text = "CLOSED"
        else:
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
    if tire_wear_pct is not None and tire_age is not None and tire_age <= 2:
        tire_wear_pct = min(float(tire_wear_pct), 0.1)

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
        elif urgent_wear:
            decision = "BOX BOX"
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
        if urgent_wear:
            decision = "STANDBY"
            decision_reasons.append("NOWINDOW")
        elif window_soon and high_wear:
            decision = "STANDBY"
        else:
            decision = "STAY OUT"
            if proba >= used_threshold + margin:
                decision_reasons.append("NOWINDOW")

    cooldown_laps = 2
    if tire_age is not None and tire_age <= cooldown_laps:
        if decision != "STAY OUT":
            decision = "STAY OUT"
            decision_reasons.append("COOLDOWN")

    baseline_reasons = []
    if pit_window_text == "OPEN":
        baseline_reasons.append("WINDOW")
        if tire_wear_pct is not None and tire_wear_pct >= 0.7:
            baseline_reasons.append("WEAR")
        if race_progress is not None and race_progress >= 0.8:
            baseline_reasons.append("LATE")
        if "WEAR" in baseline_reasons or "LATE" in baseline_reasons:
            baseline_decision = "BOX BOX"
        else:
            baseline_decision = "STANDBY"
    else:
        baseline_reasons.append("NOWINDOW")
        baseline_decision = "STANDBY" if urgent_wear else "STAY OUT"
    baseline_reason_text = "+".join(baseline_reasons) if baseline_reasons else "CORE"

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
        pit_target_text = "SOON" if urgent_wear else "HOLD"
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

    decision_source = "MODEL"
    if lock_decision:
        decision_source = "POLICY"
    elif decision in ("BOX BOX", "PIT NOW") and proba < used_threshold:
        decision_source = "POLICY"
    elif decision in ("STAY OUT", "STANDBY") and proba >= used_threshold + margin:
        decision_source = "POLICY"
    if any(tag in decision_reasons for tag in ("COOLDOWN", "COST-", "NOWINDOW")):
        decision_source = "POLICY"

    return {
        "decision": decision,
        "baseline_decision": baseline_decision,
        "baseline_reason_text": baseline_reason_text,
        "decision_source": decision_source,
        "lock_decision": lock_decision,
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




def _render_strategy_demo(summary: pd.DataFrame, mode: str, presenter_mode: bool) -> None:
        st.markdown("### Strategy Demo")
        st.markdown(
            "<div class='legend-row'>"
            "<div class='legend-item'><span class='decision-pill decision-pit'>BOX BOX</span>"
            "<span class='legend-note'>pit now</span></div>"
            "<div class='legend-item'><span class='decision-pill decision-wait'>STANDBY</span>"
            "<span class='legend-note'>prepare / window soon</span></div>"
            "<div class='legend-item'><span class='decision-pill decision-stay'>STAY OUT</span>"
            "<span class='legend-note'>continue</span></div>"
            "</div>",
            unsafe_allow_html=True,
        )


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
                leakage_cols = _leakage_audit(df_demo)
                if leakage_cols:
                    leaked = ", ".join(leakage_cols[:6])
                    extra = "..." if len(leakage_cols) > 6 else ""
                    st.markdown(
                        f"<div class='audit-badge audit-fail'>Leakage risk: {html.escape(leaked + extra)}</div>",
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        "<div class='audit-badge audit-pass'>Leakage guard: PASS</div>",
                        unsafe_allow_html=True,
                    )
                def _calc_tire_max(df: pd.DataFrame) -> float | None:
                    if "tireage" in df.columns:
                        tire_vals = pd.to_numeric(df["tireage"], errors="coerce").dropna()
                    elif "stint_laps_prev" in df.columns:
                        tire_vals = pd.to_numeric(df["stint_laps_prev"], errors="coerce").dropna()
                    else:
                        return None
                    if tire_vals.empty:
                        return None
                    return float(np.nanpercentile(tire_vals, 90))

                tire_max_global = _calc_tire_max(df_demo)
                if tire_max_global is None or not np.isfinite(tire_max_global) or tire_max_global <= 0:
                    tire_max_global = 35.0

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
                if "decide_pitstop" not in df_demo.columns:
                    st.warning("Demo requires 'decide_pitstop' column in the dataset.")
                    return

                lap_col = "lapno_prev" if "lapno_prev" in df_demo.columns else "lapno" if "lapno" in df_demo.columns else None
                lap_bounds = _lap_range(df_demo, lap_col)
                lap_min, lap_max = (lap_bounds if lap_bounds else (1, 70))

                drivers = sorted(df_demo["Driver"].dropna().astype(str).unique().tolist())
                if not drivers:
                    st.info("No driver values found in the Stage 4 dataset.")
                    return

                circuits = []
                if circuit_col and circuit_col in df_demo.columns:
                    circuits = sorted(df_demo[circuit_col].dropna().astype(str).unique().tolist())
                weather_vals = sorted(df_demo["weather_label"].dropna().astype(str).unique().tolist())

                default_scenario = _pick_default_scenario(train_df, test_df, circuit_col) or {}
                default_driver = default_scenario.get("Driver", drivers[0])
                default_weather = default_scenario.get("weather_label")
                default_circuit = default_scenario.get(circuit_col) if circuit_col else None

                def _safe_index(options: list[str], value: str | None, fallback: int = 0) -> int:
                    if value is None:
                        return fallback
                    try:
                        return options.index(value)
                    except ValueError:
                        return fallback

                lap_value = int(lap_min)

                def _resolve_context(df: pd.DataFrame) -> pd.DataFrame:
                    if df.empty:
                        return df
                    attempts = [
                        (driver_sel, circuit_filter, weather_filter),
                        (driver_sel, circuit_filter, None),
                        (driver_sel, None, None),
                        (None, None, None),
                    ]
                    for d, c, w in attempts:
                        out = _apply_scenario_filters(df, d, circuit_col, c, w)
                        if not out.empty:
                            return out
                    return df

                def _pick_row(df_context: pd.DataFrame, lap_target: int) -> pd.Series | None:
                    if df_context.empty:
                        return None
                    if lap_col and lap_col in df_context.columns:
                        laps = pd.to_numeric(df_context[lap_col], errors="coerce")
                        laps = laps.fillna(lap_target)
                        idx = (laps - lap_target).abs().idxmin()
                        return df_context.loc[idx]
                    return df_context.iloc[-1]

                def _int_val(val: object, fallback: int | None = None) -> int | None:
                    try:
                        return int(float(val))
                    except Exception:
                        return fallback


                group_col = "race_id" if "race_id" in df_demo.columns else None
                try:
                    features = build_feature_list(df_demo, "decide_pitstop", group_col or "race_id")
                except Exception:
                    features = [c for c in df_demo.columns if c not in ("decide_pitstop", group_col)]
                features = _apply_feature_allowlist(features)
                features = _align_features(features, train_df, test_df)

                if not features:
                    st.warning("No shared features available for the demo model.")
                    return

                model, calibrator, cal_threshold = _train_demo_model(train_df, features, group_col)
                train_df = train_df.copy()
                test_df = test_df.copy()
                train_probs_raw = model.predict_proba(train_df[features])[:, 1]
                test_probs_raw = model.predict_proba(test_df[features])[:, 1]
                if calibrator is not None:
                    try:
                        train_probs = calibrator.predict_proba(train_probs_raw.reshape(-1, 1))[:, 1]
                        test_probs = calibrator.predict_proba(test_probs_raw.reshape(-1, 1))[:, 1]
                    except Exception:
                        train_probs = train_probs_raw
                        test_probs = test_probs_raw
                else:
                    train_probs = train_probs_raw
                    test_probs = test_probs_raw

                train_df["proba_raw"] = train_probs_raw
                train_df["proba"] = train_probs
                test_df["proba_raw"] = test_probs_raw
                test_df["proba"] = test_probs

                target_precision = 0.6
                precision_guard = 0.03
                decision_threshold = cal_threshold if cal_threshold is not None else _select_threshold(
                    train_df["decide_pitstop"].astype(int).values,
                    train_probs,
                    beta=1.0,
                )
                prec_thresh = _threshold_for_precision(
                    train_df["decide_pitstop"].astype(int).values, train_probs, target_precision
                )
                if prec_thresh is not None:
                    decision_threshold = max(decision_threshold, prec_thresh + precision_guard)
                decision_threshold = float(np.clip(decision_threshold, 0.05, 0.95))

                smooth_window = 3
                confirm_laps = 2
                alert_cap = 0.1
                lookahead_laps = 4

                summary_use = summary.copy()
                if "stage_id" not in summary_use.columns and "stage" in summary_use.columns:
                    stage_match = summary_use["stage"].astype(str).str.extract(r"Stage\s+(\d+)")
                    summary_use["stage_id"] = pd.to_numeric(stage_match[0], errors="coerce").fillna(0).astype(int)

                def _get_stage_val(stage_id: int) -> float | None:
                    row = summary_use.loc[summary_use["stage_id"] == stage_id]
                    if row.empty:
                        return None
                    return float(row.iloc[0]["mean_f1"])

                s1 = _get_stage_val(1)
                s2 = _get_stage_val(2)
                s3 = _get_stage_val(3)
                s4 = _get_stage_val(4)
                delta21 = None if s1 is None or s2 is None else s2 - s1
                delta43 = None if s3 is None or s4 is None else s4 - s3

                quick_lines = []
                if s1 is not None and s2 is not None:
                    quick_lines.append(f"S2 vs S1: {delta21:+.3f} F1")
                if s3 is not None and s4 is not None:
                    quick_lines.append(f"S4 vs S3: {delta43:+.3f} F1")
                quick_text = " | ".join(quick_lines) if quick_lines else "F1 summary unavailable."
                quick_html = (
                    "<div class='card'>"
                    "<div class='card-title'>Quick Result (F1)</div>"
                    f"<div class='card-value'>{_fmt(s2) if s2 is not None else 'N/A'}</div>"
                    f"<div class='card-sub'>{html.escape(quick_text)}</div>"
                    "</div>"
                )

                tower_rows = []
                for sid in sorted(summary_use["stage_id"].dropna().unique().tolist()):
                    row = summary_use.loc[summary_use["stage_id"] == sid].iloc[0]
                    stage_label = f"S{int(sid)}"
                    method = "MyMethod" if int(sid) in (2, 4) else "RefTech"
                    delta = None
                    if int(sid) == 2 and delta21 is not None:
                        delta = delta21
                    if int(sid) == 4 and delta43 is not None:
                        delta = delta43
                    tower_rows.append(
                        {"stage": stage_label, "value": float(row["mean_f1"]), "method": method, "delta": delta}
                    )

                summary_cols = st.columns([1.2, 1.8])
                with summary_cols[0]:
                    st.markdown(quick_html, unsafe_allow_html=True)
                with summary_cols[1]:
                    st.markdown(
                        _timing_tower_html(tower_rows, "F1", dom_id="tower-demo", title="Timing Tower"),
                        unsafe_allow_html=True,
                    )

                st.markdown("#### Iconic Scenario: Monaco 2022 Ferrari Double-Stack")
                scenario_rows = _find_monaco_double_stack(df_demo, lap_col)
                hero_payload = None
                hero_prob = None
                hero_driver = None
                hero_lap = None
                hero_delta = None
                hero_hist_call = None
                hero_model_call = None
                if scenario_rows:
                    st.caption(
                        "Historical pit call vs model what‑if (demo only). "
                        "Estimated impact uses the same pit-loss heuristic as the demo."
                    )
                    scen_cols = st.columns(len(scenario_rows))
                    for col, row in zip(scen_cols, scenario_rows):
                        lap_val = _int_val(row.get(lap_col, lap_value) if lap_col else lap_value, lap_value)
                        row_df = pd.DataFrame([row])
                        prob_raw = float(model.predict_proba(row_df[features])[:, 1][0])
                        if calibrator is not None:
                            try:
                                prob = float(calibrator.predict_proba(np.array([[prob_raw]]))[:, 1][0])
                            except Exception:
                                prob = prob_raw
                        else:
                            prob = prob_raw
                        payload = _demo_decision(
                            row,
                            prob,
                            decision_threshold,
                            lap_val,
                            lap_col,
                            tire_max_global,
                            lookahead_laps,
                            decision_margin=0.05,
                            window_start=None,
                            window_end=None,
                        )
                        hist_call = "PIT" if int(row.get("decide_pitstop", 0)) == 1 else "STAY OUT"
                        model_call = payload["decision"]
                        net_gain = float(payload.get("net_gain_sec", 0.0))
                        impact_hist = net_gain if hist_call == "PIT" else 0.0
                        impact_model = net_gain if model_call.startswith("BOX") else 0.0
                        delta = impact_model - impact_hist
                        driver = str(row.get("Driver", "DRV"))
                        if hero_payload is None:
                            hero_payload = payload
                            hero_prob = prob
                            hero_driver = driver
                            hero_lap = lap_val
                            hero_delta = delta
                            hero_hist_call = hist_call
                            hero_model_call = model_call
                        lap_txt = f"Lap {lap_val}" if lap_val is not None else "Lap N/A"
                        card = (
                            "<div class='card'>"
                            f"<div class='card-title'>{driver} | Monaco 2022 | {lap_txt}</div>"
                            f"<div class='card-value'>{delta:+.1f}s</div>"
                            f"<div class='card-sub'>Historical: {hist_call} · Model: {model_call}</div>"
                            f"<div class='card-sub'>Net pit gain estimate: {net_gain:+.1f}s</div>"
                            "</div>"
                        )
                        with col:
                            st.markdown(card, unsafe_allow_html=True)
                else:
                    st.info("Monaco 2022 scenario not found in the dataset. Using general demo below.")

                if hero_payload is not None and hero_prob is not None:
                    call_html = (
                        "<div class='card'>"
                        "<div class='card-title'>Model Call (Monaco 2022)</div>"
                        f"<div class='card-value'>{html.escape(str(hero_model_call))}</div>"
                        f"<div class='card-sub'>Historical: {hero_hist_call} · Δ {hero_delta:+.1f}s · "
                        f"P(pit) {hero_prob:.2f}</div>"
                        "</div>"
                    )
                    st.markdown(call_html, unsafe_allow_html=True)

                with st.expander("Details (advanced)", expanded=False):
                    stat_cols = st.columns(3)
                    with stat_cols[0]:
                        _stat_card("Train", stats_train if split_indices else None)
                    with stat_cols[1]:
                        _stat_card("Test", stats_test if split_indices else None)
                    with stat_cols[2]:
                        _stat_card("All", stats_all)

                    select_cols = st.columns([1.1, 1.1, 1.0])
                    with select_cols[0]:
                        driver_sel = st.selectbox("Driver", drivers, index=_safe_index(drivers, default_driver))
                    with select_cols[1]:
                        if circuits:
                            circuit_options = ["Auto"] + circuits
                            circuit_sel = st.selectbox(
                                "Circuit",
                                circuit_options,
                                index=_safe_index(circuit_options, default_circuit, fallback=0),
                            )
                        else:
                            circuit_sel = "Auto"
                            st.selectbox("Circuit", ["Auto"], index=0, disabled=True)
                    with select_cols[2]:
                        if weather_vals:
                            weather_options = ["Auto"] + weather_vals
                            weather_sel = st.selectbox(
                                "Weather",
                                weather_options,
                                index=_safe_index(weather_options, default_weather, fallback=0),
                            )
                        else:
                            weather_sel = "Auto"
                            st.selectbox("Weather", ["Auto"], index=0, disabled=True)

                    lap_value = st.slider(
                        "Lap timeline",
                        min_value=int(lap_min),
                        max_value=int(lap_max),
                        value=int(lap_min),
                        step=1,
                    )

                    circuit_filter = None if circuit_sel == "Auto" else circuit_sel
                    weather_filter = None if weather_sel == "Auto" else weather_sel

                    train_context = _apply_scenario_filters(train_df, driver_sel, circuit_col, circuit_filter, weather_filter)
                    test_context = _apply_scenario_filters(test_df, driver_sel, circuit_col, circuit_filter, weather_filter)
                    if train_context.empty:
                        train_context = _resolve_context(train_df)
                    if test_context.empty:
                        test_context = _resolve_context(test_df)

                    train_row = _pick_row(train_context, lap_value)
                    test_row = _pick_row(test_context, lap_value)
                    if train_row is None or test_row is None:
                        st.warning("Unable to resolve demo rows after filtering.")
                        return

                    train_lap_val = _int_val(train_row.get(lap_col, lap_value) if lap_col else lap_value, lap_value)
                    test_lap_val = _int_val(test_row.get(lap_col, lap_value) if lap_col else lap_value, lap_value)
    
                    train_window = _pit_window_bounds(train_context, lap_col)
                    test_window = _pit_window_bounds(test_context, lap_col)
                    train_win_start, train_win_end = train_window if train_window else (None, None)
                    test_win_start, test_win_end = test_window if test_window else (None, None)
    
                    train_range = _lap_range(train_context, lap_col) or (lap_min, lap_max)
                    test_range = _lap_range(test_context, lap_col) or (lap_min, lap_max)
    
                    train_prob = float(train_row.get("proba", train_row.get("proba_raw", 0.0)))
                    test_prob = float(test_row.get("proba", test_row.get("proba_raw", 0.0)))
                    train_prob_smooth = _smooth_prob_by_lap(train_context, lap_col, "proba", train_lap_val, smooth_window)
                    test_prob_smooth = _smooth_prob_by_lap(test_context, lap_col, "proba", test_lap_val, smooth_window)
                    if train_prob_smooth is not None:
                        train_prob = float(train_prob_smooth)
                    if test_prob_smooth is not None:
                        test_prob = float(test_prob_smooth)
    
                    train_payload = _demo_decision(
                        train_row,
                        train_prob,
                        decision_threshold,
                        train_lap_val,
                        lap_col,
                        tire_max_global,
                        lookahead_laps,
                        decision_margin=0.05,
                        window_start=train_win_start,
                        window_end=train_win_end,
                    )
                    test_payload = _demo_decision(
                        test_row,
                        test_prob,
                        decision_threshold,
                        test_lap_val,
                        lap_col,
                        tire_max_global,
                        lookahead_laps,
                        decision_margin=0.05,
                        window_start=test_win_start,
                        window_end=test_win_end,
                    )
    
                    circuit_text = str(train_row.get(circuit_col, "N/A")) if circuit_col else "N/A"
                    weather_text = str(train_row.get("weather_label", "N/A"))
    
                    st.markdown(
                        "<div class='demo-chip-row'>"
                        f"<div class='demo-chip'>Driver <strong>{html.escape(driver_sel)}</strong></div>"
                        f"<div class='demo-chip'>Circuit <strong>{html.escape(circuit_text)}</strong></div>"
                        f"<div class='demo-chip'>Weather <strong>{html.escape(weather_text)}</strong></div>"
                        f"<div class='demo-chip'>Lap <strong>L{train_lap_val}</strong></div>"
                        "</div>",
                        unsafe_allow_html=True,
                    )
    
                    pit_cols = st.columns(2)
                    with pit_cols[0]:
                        st.markdown(
                            _pit_window_gauge_html(
                                "Train Pit Window",
                                train_range[0],
                                train_range[1],
                                train_lap_val,
                                train_win_start,
                                train_win_end,
                                train_win_start or train_lap_val,
                            ),
                            unsafe_allow_html=True,
                        )
                    with pit_cols[1]:
                        st.markdown(
                            _pit_window_gauge_html(
                                "Test Pit Window",
                                test_range[0],
                                test_range[1],
                                test_lap_val,
                                test_win_start,
                                test_win_end,
                                test_win_start or test_lap_val,
                            ),
                            unsafe_allow_html=True,
                        )
    
                    policy_cols = st.columns(2)
                    with policy_cols[0]:
                        st.markdown(
                            _policy_summary_html(
                                decision_threshold,
                                target_precision,
                                precision_guard,
                                alert_cap,
                                smooth_window,
                                confirm_laps,
                            ),
                            unsafe_allow_html=True,
                        )
                    with policy_cols[1]:
                        train_sentence = _decision_sentence(train_payload, train_prob, decision_threshold)
                        test_sentence = _decision_sentence(test_payload, test_prob, decision_threshold)
                        summary_html = (
                            "<div class='card'>"
                            "<div class='card-title'>Decision Summary</div>"
                            f"<div class='card-sub'><strong>Train:</strong> {html.escape(train_sentence)}</div>"
                            f"<div class='card-sub'><strong>Test:</strong> {html.escape(test_sentence)}</div>"
                            "</div>"
                        )
                        st.markdown(summary_html, unsafe_allow_html=True)
    
                    track_cols = st.columns(2)
                    with track_cols[0]:
                        st.markdown(
                            _render_track_demo(
                                "Train (Learned)",
                                driver_sel,
                                train_payload["lap_text"],
                                circuit_text,
                                weather_text,
                                train_payload["decision"],
                                train_payload["decision_source"],
                                train_prob,
                                decision_threshold,
                                train_payload["race_progress"],
                                train_payload["urgency"],
                                train_payload["pit_window_text"],
                                train_payload["pit_target_text"],
                                train_payload["tire_text"],
                                train_payload["tire_wear_pct"],
                                train_payload["gap_text"],
                                train_payload["sc_text"],
                                train_payload["progress_text"],
                                train_payload["gap_trend_text"],
                                train_payload["overtake_mode"],
                                train_payload["reason_text"],
                                bool(train_payload.get("tire_wear_pct") and train_payload.get("tire_wear_pct") <= 0.1),
                                train_lap_val,
                                train_range[0],
                                train_range[1],
                                train_win_start,
                                train_win_end,
                                train_win_start,
                            ),
                            unsafe_allow_html=True,
                        )
                    with track_cols[1]:
                        test_circuit = str(test_row.get(circuit_col, circuit_text)) if circuit_col else circuit_text
                        test_weather = str(test_row.get("weather_label", weather_text))
                        st.markdown(
                            _render_track_demo(
                                "Test (Unseen)",
                                driver_sel,
                                test_payload["lap_text"],
                                test_circuit,
                                test_weather,
                                test_payload["decision"],
                                test_payload["decision_source"],
                                test_prob,
                                decision_threshold,
                                test_payload["race_progress"],
                                test_payload["urgency"],
                                test_payload["pit_window_text"],
                                test_payload["pit_target_text"],
                                test_payload["tire_text"],
                                test_payload["tire_wear_pct"],
                                test_payload["gap_text"],
                                test_payload["sc_text"],
                                test_payload["progress_text"],
                                test_payload["gap_trend_text"],
                                test_payload["overtake_mode"],
                                test_payload["reason_text"],
                                bool(test_payload.get("tire_wear_pct") and test_payload.get("tire_wear_pct") <= 0.1),
                                test_lap_val,
                                test_range[0],
                                test_range[1],
                                test_win_start,
                                test_win_end,
                                test_win_start,
                            ),
                            unsafe_allow_html=True,
                        )
    
                    telemetry_cols = st.columns(2)
                    with telemetry_cols[0]:
                        gap_pct = _gap_percentile(train_context, train_row)
                        tele_html = _telemetry_panel_html(
                            "Train telemetry",
                            driver_sel,
                            train_payload["lap_text"],
                            circuit_text,
                            weather_text,
                            train_row,
                            train_payload,
                            train_lap_val,
                            train_prob,
                            float(train_row.get("proba_raw", train_prob)),
                            decision_threshold,
                            None,
                            gap_pct,
                        )
                        st.markdown(tele_html, unsafe_allow_html=True)
                    with telemetry_cols[1]:
                        gap_pct = _gap_percentile(test_context, test_row)
                        tele_html = _telemetry_panel_html(
                            "Test telemetry",
                            driver_sel,
                            test_payload["lap_text"],
                            test_circuit,
                            test_weather,
                            test_row,
                            test_payload,
                            test_lap_val,
                            test_prob,
                            float(test_row.get("proba_raw", test_prob)),
                            decision_threshold,
                            None,
                            gap_pct,
                        )
                        st.markdown(tele_html, unsafe_allow_html=True)
    
                    ladder_cols = st.columns(2)
                    with ladder_cols[0]:
                        st.markdown(
                            _decision_ladder_html("Train Decision Ladder", train_payload["decision"]),
                            unsafe_allow_html=True,
                        )
                    with ladder_cols[1]:
                        st.markdown(
                            _decision_ladder_html("Test Decision Ladder", test_payload["decision"]),
                            unsafe_allow_html=True,
                        )
    
                    def _rel_class(label: str) -> str:
                        low = label.lower()
                        if low.startswith("h"):
                            return "high"
                        if low.startswith("m"):
                            return "med"
                        return "low"
    
                    train_label = _reliability_label(
                        stats_train["rows"] if stats_train else len(train_df),
                        stats_train["groups"] if stats_train else None,
                        stats_train["pos_rate"] if stats_train else None,
                    )
                    test_label = _reliability_label(
                        stats_test["rows"] if stats_test else len(test_df),
                        stats_test["groups"] if stats_test else None,
                        stats_test["pos_rate"] if stats_test else None,
                    )
                    rel_html = (
                        "<div class='reliability-ribbon'>"
                        "<span class='reliability-pill'>Reliability</span>"
                        f"<span class='reliability-pill {_rel_class(train_label)}'>Train {train_label}</span>"
                        f"<span class='reliability-pill {_rel_class(test_label)}'>Test {test_label}</span>"
                        "</div>"
                    )
                    st.markdown(rel_html, unsafe_allow_html=True)
    
                    if presenter_mode:
                        train_strength, train_gap = _decision_strength(train_prob, decision_threshold)
                        test_strength, test_gap = _decision_strength(test_prob, decision_threshold)
                        helper_html = (
                            "<div class='helper-card'>"
                            "<div class='helper-title'>Presenter helper</div>"
                            "<div class='helper-grid'>"
                            "<div>"
                            "<span class='helper-pill'>Train data <strong>learned behavior</strong></span>"
                            f"<div class='helper-note'>Decision strength: {train_strength} (|P-T| {train_gap:.2f})</div>"
                            f"<div class='helper-note'>Data reliability: {train_label}</div>"
                            f"<div class='helper-note'>{html.escape(train_sentence)}</div>"
                            "</div>"
                            "<div>"
                            "<span class='helper-pill'>Test data <strong>unseen races</strong></span>"
                            f"<div class='helper-note'>Decision strength: {test_strength} (|P-T| {test_gap:.2f})</div>"
                            f"<div class='helper-note'>Data reliability: {test_label}</div>"
                            f"<div class='helper-note'>{html.escape(test_sentence)}</div>"
                            "</div>"
                            "</div>"
                            "</div>"
                        )
                        st.markdown(helper_html, unsafe_allow_html=True)
    
    
    
def main() -> None:
    st.set_page_config(page_title="Pit Stop Dashboard", layout="wide")
    _inject_css()

    st.markdown(
        """
        <div class="hero">
          <div class="hero-title">PIT STOP PERFORMANCE DASHBOARD</div>
          <div class="hero-tagline">Every second matters</div>
          <div class="hero-sub">Use the sidebar to switch between Overview and Demo.</div>
        </div>
        """
        ,
        unsafe_allow_html=True,
    )

    cols = st.columns(2)
    with cols[0]:
        st.markdown(
            "<div class='card'><div class='card-title'>Overview</div>"
            "<div class='card-sub'>Key metrics, timing tower, holdout snapshot.</div></div>",
            unsafe_allow_html=True,
        )
    with cols[1]:
        st.markdown(
            "<div class='card'><div class='card-title'>Demo</div>"
            "<div class='card-sub'>Strategy decision demo (train vs test).</div></div>",
            unsafe_allow_html=True,
        )

    st.info("Tip: Use the page list in the sidebar for Overview, Demo, and Project Details.")


if __name__ == "__main__":
    main()
