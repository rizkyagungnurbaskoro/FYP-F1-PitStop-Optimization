from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dashboard import streamlit_app as core


def main() -> None:
    st.set_page_config(page_title="Project Details", layout="wide")
    core._inject_css()

    st.markdown("<div class='section-title'>Project Details</div>", unsafe_allow_html=True)
    if core.PROJECT_DETAILS.exists():
        project_md = core.PROJECT_DETAILS.read_text(encoding="utf-8")
        st.markdown(project_md)
        st.download_button(
            "Download Project Details",
            project_md.encode("utf-8"),
            file_name=core.PROJECT_DETAILS.name,
            mime="text/markdown",
        )
    else:
        st.info("Project Details file not found.")


if __name__ == "__main__":
    main()
