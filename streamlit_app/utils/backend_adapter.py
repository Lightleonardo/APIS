from backend.orchestrator import run_pipeline, run_full_pipeline_with_advice
from backend.schemas import StudentInput, PipelineResult
import streamlit as st


@st.cache_data(ttl=300, show_spinner="Analyzing academic data...")
def _run_analysis_cached(student_input_json: str) -> PipelineResult:
    student_input = StudentInput.model_validate_json(student_input_json)
    return run_pipeline(student_input)


def run_analysis(student_input: StudentInput) -> PipelineResult:
    return _run_analysis_cached(student_input.model_dump_json())


@st.cache_data(ttl=300, show_spinner="Getting AI advice...")
def _run_analysis_with_advice_cached(student_input_json: str) -> tuple[PipelineResult, str]:
    student_input = StudentInput.model_validate_json(student_input_json)
    return run_full_pipeline_with_advice(student_input)


def run_analysis_with_advice(student_input: StudentInput) -> tuple[PipelineResult, str]:
    return _run_analysis_with_advice_cached(student_input.model_dump_json())