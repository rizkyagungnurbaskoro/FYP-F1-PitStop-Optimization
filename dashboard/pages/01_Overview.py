from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dashboard import streamlit_app as core


def _metric_delta(summary: pd.DataFrame, stage_a: int, stage_b: int, col: str) -> float:
    a = summary.loc[summary["stage_id"] == stage_a]
    b = summary.loc[summary["stage_id"] == stage_b]
    if a.empty or b.empty:
        return 0.0
    return float(b[col].iloc[0] - a[col].iloc[0])


def _card(title: str, value: str, sub: str) -> None:
    st.markdown(
        (
            "<div class='card'>"
            f"<div class='card-title'>{title}</div>"
            f"<div class='card-value'>{value}</div>"
            f"<div class='card-sub'>{sub}</div>"
            "</div>"
        ),
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(page_title="Pit Stop Dashboard - Overview", layout="wide")
    core._inject_css()

    summary_path = core.SUMMARY_STRICT if core.SUMMARY_STRICT.exists() else core.SUMMARY_STD
    if not summary_path.exists():
        st.error(f"Missing summary file: {summary_path}")
        return

    summary = core._load_summary(summary_path, summary_path.stat().st_mtime)
    metric_col = core.METRICS["F1"][0]

    delta_ref = _metric_delta(summary, 1, 2, metric_col)
    delta_my = _metric_delta(summary, 3, 4, metric_col)

    s2 = summary.loc[summary["stage_id"] == 2]
    s4 = summary.loc[summary["stage_id"] == 4]
    f1_s2 = float(s2[metric_col].iloc[0]) if not s2.empty else 0.0
    f1_s4 = float(s4[metric_col].iloc[0]) if not s4.empty else 0.0

    holdout = None
    if core.SUMMARY_HOLDOUT.exists():
        holdout = core._load_summary(core.SUMMARY_HOLDOUT, core.SUMMARY_HOLDOUT.stat().st_mtime)
    holdout_rows, holdout_note, _holdout_err = core._holdout_tower_data(holdout)

    st.markdown(
        """
        <div style="margin-bottom:10px;">
          <div style="font-family: 'Oxanium', 'Rajdhani', sans-serif; font-size:2rem; font-weight:800; letter-spacing:0.06em;">
            PIT STOP PERFORMANCE DASHBOARD
          </div>
          <div style="color:#ff6b6b; font-weight:600; letter-spacing:0.08em; text-transform:uppercase;">
            Every second matters
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    kpi_cols = st.columns(4)
    with kpi_cols[0]:
        _card("RefData gain (S2 vs S1)", f"{delta_ref:+.3f}", "Higher is better")
    with kpi_cols[1]:
        _card("MyData+W gain (S4 vs S3)", f"{delta_my:+.3f}", "Higher is better")
    with kpi_cols[2]:
        _card("F1 (MyMethod on RefData)", f"{f1_s2:.3f}", "Stage 2")
    with kpi_cols[3]:
        _card("F1 (MyMethod on MyData+W)", f"{f1_s4:.3f}", "Stage 4")

    st.markdown("<div class='section-title'>Quick Comparison</div>", unsafe_allow_html=True)

    tower_rows = []
    prev_val = None
    for _, row in summary.iterrows():
        val = float(row.get(metric_col, 0.0))
        delta = None if prev_val is None else val - prev_val
        tower_rows.append(
            {
                "stage": row.get("stage_short", "S?"),
                "value": val,
                "method": row.get("method", "RefTech"),
                "delta": delta,
            }
        )
        prev_val = val

    st.components.v1.html(
        core._timing_tower_html(
            tower_rows,
            "F1 (Primary metric)",
            dom_id="tower-rows-overview",
            title="Timing Tower",
        ),
        height=240,
    )

    if holdout_rows:
        st.components.v1.html(
            core._timing_tower_html(
                holdout_rows,
                "Holdout 70/30 (Stage 3/4) | F1",
                dom_id="tower-rows-overview-holdout",
                title="Holdout Tower",
                show_legend=False,
            ),
            height=170,
        )
        st.markdown(f"<div class='card-sub'>{holdout_note}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
