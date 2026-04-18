"""HSR Perception — Streamlit entrypoint.

Training + evaluation pipeline over a pybullet_hsr dataset. Use the
sidebar or the cards below to navigate. Data collection, annotation,
and object registry have been removed from this repo.
"""

import sys
from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="HSR Perception",
    page_icon="app/img/favicon.png",
    layout="wide",
    initial_sidebar_state="expanded",
)

app_dir = Path(__file__).parent
if str(app_dir) not in sys.path:
    sys.path.insert(0, str(app_dir))

from components.common_sidebar import render_common_sidebar


def main() -> None:
    render_common_sidebar()

    st.title("🤖 HSR Perception")
    st.markdown(
        "YOLOv8 fine-tuning + evaluation over a dataset produced by "
        "`pybullet_hsr`."
    )

    st.subheader("Pipeline")

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        with st.container(border=True):
            st.page_link("pages/1_Dashboard.py", label="📊 Dashboard", use_container_width=True)
            st.caption("Status snapshot")
    with c2:
        with st.container(border=True):
            st.page_link("pages/5_Training.py", label="🎯 Training", use_container_width=True)
            st.caption("Prepare dataset + fine-tune")
    with c3:
        with st.container(border=True):
            st.page_link("pages/6_Evaluation.py", label="📈 Evaluation", use_container_width=True)
            st.caption("mAP / inference-time checks")
    with c4:
        with st.container(border=True):
            st.page_link("pages/7_Settings.py", label="⚙️ Settings", use_container_width=True)
            st.caption("Env overrides + paths")

    st.markdown("---")
    st.markdown("**Pipeline:** prepare-dataset → Training → Evaluation")
    st.caption("Target: mAP ≥ 85%, Inference ≤ 100ms")


if __name__ == "__main__":
    main()
