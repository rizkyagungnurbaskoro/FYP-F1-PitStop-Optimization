import streamlit as st

try:
    st.switch_page("pages/01_Overview.py")
except Exception:
    st.set_page_config(page_title="Pit Stop Dashboard - Home", layout="wide")
    st.markdown(
        """
        <div style="font-family: 'Oxanium','Rajdhani',sans-serif; font-size:1.6rem; font-weight:700;">
          Home
        </div>
        <div style="color:#b2bccb; margin-top:6px;">
          Use the Overview or Demo pages from the sidebar.
        </div>
        """,
        unsafe_allow_html=True,
    )
