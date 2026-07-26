import time 
import streamlit as st
from backend.schemas import PipelineResult


class SessionState:
    @staticmethod
    def init():
        defaults = {
            "pipeline_result": None,
            "advisor_response": None,
            "what_if_gpas": None,
            "form_data": {},
            "last_run": 0.0,
        }
        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val

    @staticmethod
    def set_results(pipeline: PipelineResult, advice: str | None = None):
        st.session_state.pipeline_result = pipeline
        st.session_state.advisor_response = advice
        st.session_state.last_run = time.time()

    @staticmethod
    def clear():
        for key in ["pipeline_result", "advisor_response", "what_if_gpas", "last_run"]:
            if key in st.session_state:
                del st.session_state[key]

    @staticmethod
    def get_pipeline() -> PipelineResult | None:
        return st.session_state.get("pipeline_result")

    @staticmethod
    def get_advice() -> str | None:
        return st.session_state.get("advisor_response")