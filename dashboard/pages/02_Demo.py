from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dashboard import streamlit_app as core


def main() -> None:
    st.set_page_config(page_title="Pit Stop Dashboard - Demo", layout="wide")
    core._inject_css()

    st.markdown(
        """
        <div style="margin-bottom:10px;">
          <div style="font-family: 'Oxanium', 'Rajdhani', sans-serif; font-size:2rem; font-weight:800; letter-spacing:0.06em;">
            STRATEGY DEMO
          </div>
          <div style="color:#ff6b6b; font-weight:600; letter-spacing:0.08em; text-transform:uppercase;">
            Every second matters
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    mode = "Strict"
    summary_path = core.SUMMARY_STRICT if core.SUMMARY_STRICT.exists() else core.SUMMARY_STD
    if not summary_path.exists():
        st.error(f"Missing summary file: {summary_path}")
        return

    summary = core._load_summary(summary_path, summary_path.stat().st_mtime)
    core._render_strategy_demo(summary, mode, presenter_mode=True)


if __name__ == "__main__":
    main()
