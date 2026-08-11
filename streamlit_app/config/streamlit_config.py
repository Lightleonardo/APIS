import streamlit as st

def configure_page():
    st.set_page_config(
        page_title="APIS — Academic Performance Intelligence System",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    # Custom CSS for tone badges
    st.markdown("""
    <style>
    .tone-badge { padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; }
    .tone-encouraging { background: #d4edda; color: #155724; }
    .tone-direct { background: #cce5ff; color: #004085; }
    .tone-analytical { background: #fff3cd; color: #856404; }
    .stMetric { background: #f8f9fa; padding: 1rem; border-radius: 0.5rem; }
    </style>
    """, unsafe_allow_html=True)