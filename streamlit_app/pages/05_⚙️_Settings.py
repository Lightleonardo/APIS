import streamlit as st
from streamlit_app.utils.session_state import SessionState
from backend.config import settings


st.header("⚙️ Settings")

st.subheader("🎯 Advisor Settings")
tone = st.selectbox("Default Tone", ["encouraging", "direct", "analytical"], index=0)
st.session_state.advisor_tone = tone

st.divider()
st.subheader("📦 Model Information")
st.info(f"**Model Directory:** `{settings.MODEL_DIR}`")
st.info(f"**Next GPA Model:** `{settings.NEXT_GPA_MODEL}`")
st.info(f"**Final CGPA Model:** `{settings.FINAL_CGPA_MODEL}`")
st.info(f"**Graduation Class Model:** `{settings.GRADUATION_CLASS_MODEL}`")
st.info(f"**Academic Risk Model:** `{settings.ACADEMIC_RISK_MODEL}`")

st.divider()
st.subheader("🔧 Advanced")
if st.button("🗑️ Clear Session", type="secondary"):
    SessionState.clear()
    st.rerun()

st.caption("APIS v1.0 — Academic Performance Intelligence System")