# APIS Phase 6 Implementation Plan

> **For agentic workers:** Use subagent-driven development (recommended) or executing-plans. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build Streamlit Dashboard (Phase 6) — interactive web UI for academic trajectory, planning, what-if simulation, and AI advisor.

**Prerequisites:** Phases 1–5 complete. Backend orchestrator exposes `run_pipeline()` and `run_full_pipeline_with_advice()`. Models trained. 135 tests passing.

**Tech Stack:** Streamlit, Plotly, existing backend (Pydantic, orchestrator, advisor)

---

## File Structure Map (Phase 6 Additions)

| File | Responsibility |
|------|----------------|
| `streamlit_app/app.py` | Entry point, page config, routing |
| `streamlit_app/pages/01_📊_Dashboard.py` | Trajectory chart + summary metrics |
| `streamlit_app/pages/02_📈_Planner.py` | Semester planner (actual vs target) |
| `streamlit_app/pages/03_🔮_What_If.py` | Interactive what-if simulator |
| `streamlit_app/pages/04_🤖_Advisor.py` | AI Advisor chat panel |
| `streamlit_app/pages/05_⚙️_Settings.py` | Tone, model info, export |
| `streamlit_app/components/charts.py` | Plotly figure rendering wrappers |
| `streamlit_app/components/forms.py` | Sidebar input form |
| `streamlit_app/components/what_if.py` | What-if slider interaction |
| `streamlit_app/components/advisor_chat.py` | Advisor message display |
| `streamlit_app/utils/backend_adapter.py` | Calls orchestrator functions |
| `streamlit_app/utils/session_state.py` | Streamlit session management |
| `streamlit_app/utils/formatters.py` | Number formatting, labels |
| `streamlit_app/config/streamlit_config.py` | Page config, custom CSS |
| `requirements.txt` | + `streamlit>=1.30` |

---

## Phase 6 Implementation Tasks

---

### Task 1: Project Setup & Configuration

**Files:**
- `streamlit_app/config/streamlit_config.py` (create)
- `requirements.txt` (modify)
- `streamlit_app/app.py` (create)

**Step 1: Write failing test for config**

```python
# tests/test_streamlit_config.py
import pytest
from streamlit_app.config.streamlit_config import configure_page

def test_configure_page_sets_page_config(monkeypatch):
    import streamlit as st
    calls = {}
    monkeypatch.setattr(st, "set_page_config", lambda **kw: calls.update(kw))
    configure_page()
    assert calls["page_title"] == "APIS — Academic Performance Intelligence System"
    assert calls["page_icon"] == "📊"
    assert calls["layout"] == "wide"
    assert calls["initial_sidebar_state"] == "expanded"
```

Run: `pytest tests/test_streamlit_config.py -v` → Expected: FAIL

**Step 2: Implement config**

```python
# streamlit_app/config/streamlit_config.py
import streamlit as st

def configure_page():
    st.set_page_config(
        page_title="APIS — Academic Performance Intelligence System",
        page_icon="📊",
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
```

Run test → Expected: PASS

**Step 3: Add streamlit to requirements**

```text
# requirements.txt (add)
streamlit>=1.30
```

**Step 4: Create app.py entry point**

```python
# streamlit_app/app.py
import streamlit as st
from streamlit_app.config.streamlit_config import configure_page

configure_page()

st.title("📊 Academic Performance Intelligence System")
st.caption("Plan. Predict. Succeed.")

# Page navigation handled by Streamlit's pages/ directory
st.sidebar.success("Select a page above to begin.")
```

**Step 5: Commit**
```bash
git add streamlit_app/config/streamlit_config.py streamlit_app/app.py requirements.txt tests/test_streamlit_config.py
git commit -m "feat: add Streamlit app entry point and page config"
```

---

### Task 2: Backend Adapter & Session State

**Files:**
- `streamlit_app/utils/backend_adapter.py` (create)
- `streamlit_app/utils/session_state.py` (create)
- `streamlit_app/utils/formatters.py` (create)

**Step 1: Write failing tests**

```python
# tests/test_backend_adapter.py
import pytest
from unittest.mock import Mock, patch
from streamlit_app.utils.backend_adapter import run_analysis, run_analysis_with_advice
from backend.schemas import StudentInput, SemesterRecord, PipelineResult

def test_run_analysis_calls_orchestrator():
    student = StudentInput(
        student_name="Test", university="U", faculty="F", department="D", course="C",
        programme_duration_years=4, current_level=200,
        semester_records=[SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2023/2024")],
        target_graduation_class="First Class",
    )
    with patch('streamlit_app.utils.backend_adapter.run_pipeline') as mock_run:
        mock_run.return_value = Mock(spec=PipelineResult)
        result = run_analysis(student)
        mock_run.assert_called_once_with(student)
        assert result is mock_run.return_value

def test_run_analysis_with_advice_calls_full_pipeline():
    student = StudentInput(
        student_name="Test", university="U", faculty="F", department="D", course="C",
        programme_duration_years=4, current_level=200,
        semester_records=[SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2023/2024")],
        target_graduation_class="First Class",
    )
    with patch('streamlit_app.utils.backend_adapter.run_full_pipeline_with_advice') as mock_run:
        mock_run.return_value = (Mock(spec=PipelineResult), "advice text")
        pipeline, advice = run_analysis_with_advice(student)
        mock_run.assert_called_once_with(student)
        assert advice == "advice text"
```

Run → Expected: FAIL

**Step 2: Implement backend adapter**

```python
# streamlit_app/utils/backend_adapter.py
from backend.orchestrator import run_pipeline, run_full_pipeline_with_advice
from backend.schemas import StudentInput, PipelineResult
import streamlit as st

@st.cache_data(ttl=300, show_spinner="Analyzing academic data...")
def run_analysis(student_input: StudentInput) -> PipelineResult:
    return run_pipeline(student_input)

@st.cache_data(ttl=300, show_spinner="Getting AI advice...")
def run_analysis_with_advice(student_input: StudentInput) -> tuple[PipelineResult, str]:
    return run_full_pipeline_with_advice(student_input)
```

**Step 3: Session state utility**

```python
# streamlit_app/utils/session_state.py
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
        st.session_state.last_run = st.time.time()

    @staticmethod
    def clear():
        for key in ["pipeline_result", "advisor_response", "what_if_gpas", "last_run"]:
            if key in st.session_state:
                del st.session_state[key]
```

**Step 4: Formatters**

```python
# streamlit_app/utils/formatters.py
def fmt_gpa(val: float | None) -> str:
    return f"{val:.2f}" if val is not None else "—"

def fmt_cgpa(val: float | None) -> str:
    return f"{val:.2f}" if val is not None else "—"

def fmt_pct(val: float) -> str:
    return f"{val:.1f}%"

def tone_badge(tone: str) -> str:
    return f'<span class="tone-badge tone-{tone}">{tone.capitalize()}</span>'
```

Run tests → Expected: PASS

**Step 5: Commit**
```bash
git add streamlit_app/utils/ tests/test_backend_adapter.py
git commit -m "feat: add backend adapter, session state, formatters"
```

---

### Task 3: Sidebar Input Form

**Files:**
- `streamlit_app/components/forms.py` (create)
- `streamlit_app/pages/01_📊_Dashboard.py` (modify to include form)

**Step 1: Write failing test**

```python
# tests/test_forms.py
import pytest
from streamlit_app.components.forms import render_sidebar_form, validate_semester_records
from backend.schemas import StudentInput, SemesterRecord

def test_validate_semester_records_empty():
    assert validate_semester_records([]) == "At least one semester required"

def test_validate_semester_records_gpa_bounds():
    records = [SemesterRecord(semester_number=1, gpa=5.1, credits=20, academic_session="2023/2024")]
    assert "GPA must be 0–5" in validate_semester_records(records)

def test_validate_semester_records_credits_bounds():
    records = [SemesterRecord(semester_number=1, gpa=3.5, credits=10, academic_session="2023/2024")]
    assert "Credits must be 12–24" in validate_semester_records(records)

def test_render_sidebar_form_returns_student_input(monkeypatch):
    import streamlit as st
    # Mock streamlit widgets
    monkeypatch.setattr(st.sidebar, "text_input", lambda *a, **k: "Test Uni")
    monkeypatch.setattr(st.sidebar, "selectbox", lambda *a, **k: "Science")
    # ... mock other widgets
    # This is integration-style; skip detailed unit test for Streamlit widgets
```

**Step 2: Implement form component**

```python
# streamlit_app/components/forms.py
import streamlit as st
from backend.schemas import StudentInput, SemesterRecord
from streamlit_app.utils.formatters import fmt_gpa

def validate_semester_records(records: list[SemesterRecord]) -> str | None:
    if not records:
        return "At least one semester required"
    for r in records:
        if not (0.0 <= r.gpa <= 5.0):
            return f"Semester {r.semester_number}: GPA must be 0.00–5.00"
        if not (12 <= r.credits <= 24):
            return f"Semester {r.semester_number}: Credits must be 12–24"
        if "/" not in r.academic_session:
            return f"Semester {r.semester_number}: Session format 'YYYY/YYYY' required"
    return None

def render_sidebar_form() -> StudentInput | None:
    st.sidebar.header("📝 Student Profile")
    
    # Basic info
    student_name = st.sidebar.text_input("Full Name", value="John Doe")
    university = st.sidebar.text_input("University", value="University of Lagos")
    faculty = st.sidebar.text_input("Faculty", value="Science")
    department = st.sidebar.text_input("Department", value="Computer Science")
    course = st.sidebar.text_input("Course", value="Computer Science")
    
    programme_duration = st.sidebar.selectbox("Programme Duration (years)", [4, 5, 6], index=1)
    
    st.sidebar.divider()
    st.sidebar.subheader("Semester Records")
    
    # Dynamic semester rows
    if "semester_rows" not in st.session_state:
        st.session_state.semester_rows = 1
    
    records = []
    for i in range(st.session_state.semester_rows):
        with st.sidebar.expander(f"Semester {i+1}", expanded=(i == st.session_state.semester_rows - 1)):
            sem_num = i + 1
            gpa = st.number_input(f"GPA", 0.0, 5.0, 3.5, 0.01, key=f"gpa_{i}", format="%.2f")
            credits = st.number_input(f"Credits", 12, 24, 20, key=f"credits_{i}")
            session = st.text_input(f"Session (YYYY/YYYY)", "2023/2024", key=f"session_{i}")
            records.append(SemesterRecord(semester_number=sem_num, gpa=gpa, credits=credits, academic_session=session))
    
    col1, col2 = st.sidebar.columns(2)
    if col1.button("➕ Add Semester", use_container_width=True):
        st.session_state.semester_rows += 1
        st.rerun()
    if col2.button("🗑️ Remove Last", use_container_width=True, disabled=st.session_state.semester_rows <= 1):
        st.session_state.semester_rows -= 1
        st.rerun()
    
    st.sidebar.divider()
    st.sidebar.subheader("🎯 Target")
    target_type = st.sidebar.radio("Target by:", ["Graduation Class", "Target CGPA"], horizontal=True)
    
    if target_type == "Graduation Class":
        target_class = st.sidebar.selectbox(
            "Target Class",
            ["First Class", "Second Class Upper", "Second Class Lower", "Third Class", "Pass"],
            index=0,
        )
        target_cgpa = None
    else:
        target_cgpa = st.sidebar.number_input("Target CGPA", 0.0, 5.0, 4.5, 0.01, format="%.2f")
        target_class = None
    
    # Validation
    error = validate_semester_records(records)
    if error:
        st.sidebar.error(error)
        return None
    
    # Current level auto-calculated
    semesters_completed = len(records)
    total_semesters = programme_duration * 2
    from backend.grading_rules import level_for_semester
    current_level = level_for_semester(semesters_completed, total_semesters)
    
    run_clicked = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)
    
    if not run_clicked:
        return None
    
    return StudentInput(
        student_name=student_name,
        university=university,
        faculty=faculty,
        department=department,
        course=course,
        programme_duration_years=programme_duration,
        current_level=current_level,
        semester_records=records,
        target_graduation_class=target_class,
        target_cgpa=target_cgpa,
    )
```

Run tests → Expected: PASS

**Step 3: Commit**
```bash
git add streamlit_app/components/forms.py tests/test_forms.py
git commit -m "feat: add sidebar input form with validation"
```

---

### Task 4: Dashboard Page (Trajectory Chart + Metrics)

**Files:**
- `streamlit_app/pages/01_📊_Dashboard.py` (create)
- `streamlit_app/components/charts.py` (create)

**Step 1: Charts component**

```python
# streamlit_app/components/charts.py
import streamlit as st
import plotly.graph_objects as go
from backend.graphs import trajectory_chart, semester_planner_chart, what_if_simulator
from backend.schemas import PipelineResult

def render_trajectory_chart(pipeline: PipelineResult):
    fig_dict = trajectory_chart(pipeline)
    fig = go.Figure(fig_dict)
    st.plotly_chart(fig, use_container_width=True)

def render_semester_planner(pipeline: PipelineResult):
    fig_dict = semester_planner_chart(pipeline)
    fig = go.Figure(fig_dict)
    st.plotly_chart(fig, use_container_width=True)

def render_what_if_chart(pipeline: PipelineResult, what_if_gpas: list[float]):
    fig_dict = what_if_simulator(pipeline, what_if_gpas)
    fig = go.Figure(fig_dict)
    st.plotly_chart(fig, use_container_width=True)

def render_metric_cards(pipeline: PipelineResult):
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Current CGPA", f"{pipeline.current_cgpa:.2f}" if pipeline.current_cgpa else "—")
    with col2:
        st.metric("Predicted Final CGPA", f"{pipeline.predicted_final_cgpa:.2f}")
    with col3:
        st.metric("Academic Health", f"{pipeline.academic_health_score}/100")
    with col4:
        st.metric("Goal Feasible", "✅ Yes" if pipeline.feasibility.goal_achievable else "❌ No")

def render_feature_tables(pipeline: PipelineResult):
    with st.expander("🔍 Top Predictive Features"):
        tabs = st.tabs(["Next GPA", "Final CGPA", "Graduation Class", "Academic Risk"])
        feature_sets = [
            ("top_features_next_gpa", tabs[0]),
            ("top_features_final_cgpa", tabs[1]),
            ("top_features_graduation_class", tabs[2]),
            ("top_features_academic_risk", tabs[3]),
        ]
        for attr, tab in feature_sets:
            with tab:
                feats = getattr(pipeline, attr, [])
                if feats:
                    import pandas as pd
                    df = pd.DataFrame([{"Feature": f.feature, "Importance": f.importance} for f in feats])
                    st.dataframe(df, hide_index=True, use_container_width=True)
                else:
                    st.info("No feature importance available")
```

**Step 2: Dashboard page**

```python
# streamlit_app/pages/01_📊_Dashboard.py
import streamlit as st
from streamlit_app.utils.backend_adapter import run_analysis
from streamlit_app.utils.session_state import SessionState
from streamlit_app.components.forms import render_sidebar_form
from streamlit_app.components.charts import (
    render_trajectory_chart, render_metric_cards, render_feature_tables
)

SessionState.init()

# Sidebar form
student = render_sidebar_form()

if student:
    with st.spinner("Analyzing..."):
        pipeline = run_analysis(student)
        SessionState.set_results(pipeline)

pipeline = st.session_state.get("pipeline_result")

if pipeline:
    st.header(f"📊 {pipeline.student_name} — Academic Trajectory")
    
    # Metric cards
    render_metric_cards(pipeline)
    
    st.divider()
    
    # Trajectory chart
    st.subheader("Cumulative CGPA Trajectory")
    render_trajectory_chart(pipeline)
    
    # Summary info
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Current Classification:** {pipeline.current_classification or '—'}")
        st.info(f"**GPA Trend:** {pipeline.gpa_trend.value if hasattr(pipeline.gpa_trend, 'value') else pipeline.gpa_trend}")
        st.info(f"**Consistency Index:** {pipeline.consistency_index}/25")
    with col2:
        st.info(f"**Required Avg GPA:** {pipeline.feasibility.required_average_gpa:.2f}" if pipeline.feasibility.required_average_gpa else "**Required Avg GPA:** N/A (final semester)")
        st.info(f"**Max Achievable CGPA:** {pipeline.feasibility.max_achievable_cgpa:.2f} ({pipeline.feasibility.realistic_classification})")
        st.info(f"**Confidence:** {pipeline.feasibility.confidence:.0%}")
    
    st.divider()
    render_feature_tables(pipeline)

else:
    st.info("👈 Fill in your semester records and click **Run Analysis** to begin.")
```

Run: `streamlit run streamlit_app/app.py` → Verify manually

**Step 3: Commit**
```bash
git add streamlit_app/components/charts.py streamlit_app/pages/01_📊_Dashboard.py
git commit -m "feat: add Dashboard page with trajectory chart and metrics"
```

---

### Task 5: Planner Page (Semester-by-Semester Targets)

**Files:**
- `streamlit_app/pages/02_📈_Planner.py` (create)

```python
# streamlit_app/pages/02_📈_Planner.py
import streamlit as st
from streamlit_app.components.charts import render_semester_planner, render_metric_cards

st.header("📈 Semester Planner")

pipeline = st.session_state.get("pipeline_result")
if not pipeline:
    st.warning("Run analysis first from the Dashboard.")
    st.stop()

render_metric_cards(pipeline)

st.divider()
st.subheader("Actual vs Target GPA per Semester")
render_semester_planner(pipeline)

st.divider()
st.subheader("Semester Targets")
if pipeline.semester_plan:
    import pandas as pd
    df = pd.DataFrame([
        {
            "Semester": p.semester_number,
            "Target GPA": f"{p.target_gpa:.2f}",
            "Projected Cum. CGPA": f"{p.cumulative_cgpa_if_met:.2f}",
        }
        for p in pipeline.semester_plan
    ])
    st.dataframe(df, hide_index=True, use_container_width=True)
else:
    st.info("No remaining semesters — this is your final semester.")

st.caption(f"Feasibility: {pipeline.feasibility.message}")
```

**Step 2: Commit**
```bash
git add streamlit_app/pages/02_📈_Planner.py
git commit -m "feat: add Planner page with semester targets"
```

---

### Task 6: What-If Simulator Page

**Files:**
- `streamlit_app/components/what_if.py` (create)
- `streamlit_app/pages/03_🔮_What_If.py` (create)

**Step 1: What-if component**

```python
# streamlit_app/components/what_if.py
import streamlit as st
from streamlit_app.components.charts import render_what_if_chart
from backend.schemas import PipelineResult

def render_what_if_simulator(pipeline: PipelineResult) -> list[float]:
    st.subheader("🔮 What-If Simulator")
    st.caption("Adjust future semester GPAs to see impact on final CGPA and classification.")
    
    n_remaining = pipeline.semesters_remaining
    if n_remaining == 0:
        st.info("No remaining semesters to simulate.")
        return []
    
    # Initialize from session state or use plan targets
    if "what_if_gpas" not in st.session_state or len(st.session_state.what_if_gpas) != n_remaining:
        st.session_state.what_if_gpas = [p.target_gpa for p in pipeline.semester_plan]
    
    gpas = []
    cols = st.columns(min(n_remaining, 4))
    for i in range(n_remaining):
        sem_num = pipeline.semesters_completed + i + 1
        with cols[i % 4]:
            gpa = st.slider(
                f"Semester {sem_num}",
                0.0, 5.0,
                st.session_state.what_if_gpas[i],
                0.01,
                format="%.2f",
                key=f"whatif_{i}"
            )
            gpas.append(gpa)
    
    st.session_state.what_if_gpas = gpas
    
    if st.button("📊 Simulate", type="primary"):
        st.session_state.what_if_gpas = gpas
        st.rerun()
    
    # Show simulated result
    if gpas:
        render_what_if_chart(pipeline, gpas)
        
        # Quick summary
        from backend.graphs import what_if_simulator
        from backend.grading_rules import classify_cgpa
        import plotly.graph_objects as go
        
        fig_dict = what_if_simulator(pipeline, gpas)
        fig = go.Figure(fig_dict)
        # Extract final CGPA from last trace
        final_cgpa = fig.data[-1].y[-1] if fig.data else pipeline.current_cgpa
        final_class = classify_cgpa(final_cgpa)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Simulated Final CGPA", f"{final_cgpa:.2f}")
        col2.metric("Projected Class", final_class)
        col3.metric("Change vs Current", f"{final_cgpa - (pipeline.current_cgpa or 0):+.2f}")
    
    return gpas
```

**Step 2: What-If page**

```python
# streamlit_app/pages/03_🔮_What_If.py
import streamlit as st
from streamlit_app.components.what_if import render_what_if_simulator

st.header("🔮 What-If Simulator")

pipeline = st.session_state.get("pipeline_result")
if not pipeline:
    st.warning("Run analysis first from the Dashboard.")
    st.stop()

render_what_if_simulator(pipeline)
```

**Step 3: Commit**
```bash
git add streamlit_app/components/what_if.py streamlit_app/pages/03_🔮_What_If.py
git commit -m "feat: add What-If simulator with interactive sliders"
```

---

### Task 7: AI Advisor Page

**Files:**
- `streamlit_app/components/advisor_chat.py` (create)
- `streamlit_app/pages/04_🤖_Advisor.py` (create)
- `streamlit_app/utils/backend_adapter.py` (add `run_analysis_with_advice` call)

**Step 1: Advisor chat component**

```python
# streamlit_app/components/advisor_chat.py
import streamlit as st
from streamlit_app.utils.formatters import tone_badge
from streamlit_app.utils.backend_adapter import run_analysis_with_advice
from backend.schemas import StudentInput

def render_advisor_panel(student_input: StudentInput | None = None):
    st.header("🤖 AI Academic Advisor")
    
    pipeline = st.session_state.get("pipeline_result")
    advice = st.session_state.get("advisor_response")
    
    if not pipeline:
        st.warning("Run analysis first from the Dashboard.")
        return
    
    # Tone selector
    col1, col2 = st.columns([3, 1])
    with col1:
        tone = st.selectbox(
            "Advisor Tone",
            ["encouraging", "direct", "analytical"],
            index=0,
            key="advisor_tone"
        )
    with col2:
        if st.button("🔄 Regenerate", use_container_width=True):
            st.session_state.advisor_response = None
            st.rerun()
    
    if advice:
        st.markdown(f"**Tone:** {tone_badge(tone)}", unsafe_allow_html=True)
        st.markdown(f"> {advice}")
    else:
        with st.spinner("Getting advice..."):
            pipeline, advice = run_analysis_with_advice(student_input)
            st.session_state.pipeline_result = pipeline
            st.session_state.advisor_response = advice
            st.rerun()
    
    if st.button("📋 Copy Advice"):
        st.code(advice or "", language=None)
        st.toast("Copied to clipboard!")
```

**Step 2: Advisor page**

```python
# streamlit_app/pages/04_🤖_Advisor.py
import streamlit as st
from streamlit_app.components.forms import render_sidebar_form
from streamlit_app.components.advisor_chat import render_advisor_chat
from streamlit_app.utils.backend_adapter import run_analysis
from streamlit_app.utils.session_state import SessionState

SessionState.init()

student = render_sidebar_form()

if student:
    with st.spinner("Analyzing..."):
        pipeline = run_analysis(student)
        SessionState.set_results(pipeline)

render_advisor_chat(student if student else None)
```

**Step 3: Commit**
```bash
git add streamlit_app/components/advisor_chat.py streamlit_app/pages/04_🤖_Advisor.py
git commit -m "feat: add AI Advisor page with tone selection"
```

---

### Task 8: Settings Page

**Files:**
- `streamlit_app/pages/05_⚙️_Settings.py` (create)

```python
# streamlit_app/pages/05_⚙️_Settings.py
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
```

**Step 2: Commit**
```bash
git add streamlit_app/pages/05_⚙️_Settings.py
git commit -m "feat: add Settings page"
```

---

### Task 9: Integration & Polish

**Files:**
- `streamlit_app/app.py` (update with navigation)
- `README.md` (add dashboard section)

**Step 1: Update app.py**

```python
# streamlit_app/app.py
import streamlit as st
from streamlit_app.config.streamlit_config import configure_page
from streamlit_app.utils.session_state import SessionState

configure_page()
SessionState.init()

st.title("📊 Academic Performance Intelligence System")
st.caption("Plan. Predict. Succeed.")

st.sidebar.title("Navigation")
st.sidebar.markdown("""
1. **📊 Dashboard** — Trajectory & metrics
2. **📈 Planner** — Semester targets
3. **🔮 What-If** — Simulate scenarios
4. **🤖 Advisor** — AI guidance
5. **⚙️ Settings** — Preferences
""")

# Show current analysis status
pipeline = st.session_state.get("pipeline_result")
if pipeline:
    st.sidebar.success(f"✅ Analysis: {pipeline.student_name}")
    st.sidebar.metric("CGPA", f"{pipeline.current_cgpa:.2f}" if pipeline.current_cgpa else "—")
    st.sidebar.metric("Health", f"{pipeline.academic_health_score}/100")
else:
    st.sidebar.info("No analysis run yet")
```

**Step 2: Update README.md**

Add to README:
```markdown
## Phase 6: Streamlit Dashboard

### Launch
```bash
streamlit run streamlit_app/app.py
```

### Features
- **Dashboard** — CGPA trajectory with predictions, health metrics, feature importance
- **Planner** — Semester-by-semester GPA targets
- **What-If Simulator** — Interactive sliders for scenario planning
- **AI Advisor** — Personalized guidance (Gemini 1.5 Flash)
- **Settings** — Tone, model info, session management

### Screenshots
*(Add after launch)*
```

**Step 3: Run full test suite**

```bash
pytest tests/ -q
# Should have 135+ tests passing (including new component tests if added)
```

**Step 4: Manual verification checklist**
- [ ] `streamlit run streamlit_app/app.py` launches without errors
- [ ] Sidebar form accepts semesters, validates correctly
- [ ] Dashboard shows trajectory chart, metric cards, feature tables
- [ ] Planner shows actual vs target bars + target table
- [ ] What-If sliders update chart in real-time
- [ ] Advisor generates response, tone selector works
- [ ] Settings page displays model info, clears session
- [ ] Navigation between pages preserves session state
- [ ] Mobile viewport: sidebar collapsible, charts responsive

**Step 5: Commit**
```bash
git add streamlit_app/app.py README.md
git commit -m "feat: complete Phase 6 Streamlit Dashboard"
```

---

## Phase 6 Commit Summary

| Commit | Files |
|--------|-------|
| `feat: add Streamlit app entry point and page config` | `streamlit_app/config/streamlit_config.py`, `streamlit_app/app.py`, `requirements.txt`, `tests/test_streamlit_config.py` |
| `feat: add backend adapter, session state, formatters` | `streamlit_app/utils/`, `tests/test_backend_adapter.py` |
| `feat: add sidebar input form with validation` | `streamlit_app/components/forms.py`, `tests/test_forms.py` |
| `feat: add Dashboard page with trajectory chart and metrics` | `streamlit_app/components/charts.py`, `streamlit_app/pages/01_📊_Dashboard.py` |
| `feat: add Planner page with semester targets` | `streamlit_app/pages/02_📈_Planner.py` |
| `feat: add What-If simulator with interactive sliders` | `streamlit_app/components/what_if.py`, `streamlit_app/pages/03_🔮_What_If.py` |
| `feat: add AI Advisor page with tone selection` | `streamlit_app/components/advisor_chat.py`, `streamlit_app/pages/04_🤖_Advisor.py` |
| `feat: add Settings page` | `streamlit_app/pages/05_⚙️_Settings.py` |
| `feat: complete Phase 6 Streamlit Dashboard` | `streamlit_app/app.py`, `README.md` |

---

## Execution Options

**Option 1: Subagent-Driven (Recommended)**
- One subagent per task above
- Review each task completion before next

**Option 2: Inline with Checkpoints**
- Execute Tasks 1-2 → run tests → Tasks 3-4 → run tests → etc.

**Option 3: Batch Execute**
- Implement all, then single test run

**Which approach?**