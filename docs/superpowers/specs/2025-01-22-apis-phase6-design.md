# Academic Performance Intelligence System (APIS) — Phase 6 Design Specification

**Version:** 1.0  
**Date:** 2026-07-22  
**Status:** Approved for Implementation  
**Prerequisite:** Phases 1–5 Complete (Backend logic, ML models, AI Advisor)

---

## 1. Phase 6 Scope

**Streamlit Dashboard** — Interactive web UI for students to:
- View academic trajectory with predictions
- See semester-by-semester GPA plan
- Run what-if scenarios
- Get AI advisor guidance
- Export reports

| In Scope | Out of Scope (Phase 7+) |
|----------|------------------------|
| Multi-page Streamlit app | User authentication/accounts |
| Trajectory chart (historical + predicted + goal) | Persistence/database |
| Semester planner (actual vs target bars) | Multi-user support |
| What-if simulator (interactive GPA adjustment) | Real-time collaboration |
| AI Advisor chat panel | Export to PDF/Word |
| Responsive layout (mobile-friendly) | Custom theming beyond Streamlit defaults |

---

## 2. Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Streamlit App (frontend)                 │
│  ┌─────────┐  ┌──────────────┐  ┌────────────┐  ┌────────┐  │
│  │ Sidebar │  │ Trajectory   │  │ Semester   │  │ What-If │  │
│  │ Input   │  │ Chart        │  │ Planner    │  │ Sim    │  │
│  └────┬────┘  └──────┬───────┘  └─────┬──────┘  └────┬───┘  │
│       │              │                │              │      │
│       ▼              ▼                ▼              ▼      │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Backend Adapter (thin wrapper)          │  │
│  │  run_pipeline() │ run_full_pipeline_with_advice()   │  │
│  └────────────────────────────┬─────────────────────────┘  │
│                               │                             │
└───────────────────────────────┼─────────────────────────────┘
                                ▼
                    ┌───────────────────────┐
                    │   Phase 4/5 Backend   │
                    │  calculator + planner │
                    │  + predictor + advisor│
                    └───────────────────────┘
```

**Key Principle:** Streamlit handles **UI only**. All computation (CGPA, feasibility, predictions, what-if math) stays in the backend. The dashboard calls `orchestrator.run_pipeline()` and `orchestrator.run_full_pipeline_with_advice()`.

---

## 3. Page Structure

```
streamlit_app/
├── app.py                    # Entry point, page routing
├── pages/
│   ├── 01_📊_Dashboard.py    # Main trajectory view
│   ├── 02_📈_Planner.py      # Semester-by-semester plan
│   ├── 03_🔮_What_If.py      # Interactive what-if simulator
│   ├── 04_🤖_Advisor.py      # AI Academic Advisor chat
│   └── 05_⚙️_Settings.py     # Tone, model info, export
├── components/
│   ├── charts.py             # Plotly figure rendering
│   ├── forms.py              # Input forms (sidebar)
│   ├── what_if.py            # What-if slider/interaction
│   └── advisor_chat.py       # Advisor message display
├── utils/
│   ├── backend_adapter.py    # Calls orchestrator functions
│   ├── session_state.py      # Streamlit session management
│   └── formatters.py         # Number formatting, labels
└── config/
    └── streamlit_config.py   # Page config, theme
```

---

## 4. Data Flow

### 4.1 Input (Sidebar Form)
```
StudentInput → orchestrator.run_pipeline() → PipelineResult
```

Sidebar collects:
- Student name, university, faculty, department, course
- Programme duration (4/5/6 years)
- Current level (auto-calculated from semesters)
- Semester records: GPA, credits, session per semester
- Target: graduation class OR target CGPA

### 4.2 Processing (Backend)
```
PipelineResult → PipelineResult + AdvisorResponse
```

Backend returns:
- Current CGPA, classification, trend, health score
- Feasibility (required avg GPA, max achievable, confidence)
- Semester plan (target GPA per remaining semester)
- Predictions (next GPA, final CGPA, class, risk)
- Feature importance tables
- Advisor response (string)

### 4.3 Output (Dashboard Components)

| Component | Data Source | Visualization |
|-----------|-------------|---------------|
| Trajectory Chart | `semester_history` + `predicted_next_gpa` + `target_cgpa_resolved` | Plotly line: actual CGPA, predicted next, goal line, First Class threshold |
| Semester Planner | `semester_history.gpa` + `semester_plan.target_gpa` | Plotly grouped bars: actual vs target per semester |
| What-If Simulator | User-adjusted future GPAs → recomputed trajectory | Interactive sliders → real-time chart update |
| Advisor Panel | `run_full_pipeline_with_advice()` response | Chat-style message with tone badge |
| Feature Tables | `top_features_*` | Sortable dataframes |

---

## 5. UI Components Detail

### 5.1 Sidebar Input Form (`components/forms.py`)

**Semester Entry:**
- Dynamic form: "Add Semester" button adds new row
- Each row: Semester # (auto), GPA (0.00–5.00), Credits (12–24), Session (e.g., "2023/2024")
- Validation: GPA bounds, credit bounds, session format
- "Run Analysis" button triggers backend call

**Target Selection:**
- Radio: "Target Graduation Class" (dropdown: 6 classes) OR "Target CGPA" (number input)
- Default: First Class (4.50)

### 5.2 Trajectory Chart (`components/charts.py`)

Uses `backend.graphs.trajectory_chart()` (Plotly figure dict):
- **Blue line+markers**: Historical cumulative CGPA per semester
- **Orange diamond**: Predicted next semester CGPA (if not final)
- **Green dashed line**: Target CGPA (goal)
- **Gray dotted line**: First Class threshold (4.50)
- Y-axis: 0–5.0, X-axis: Semester number
- Hover: semester, GPA, cumulative CGPA, credits

### 5.3 Semester Planner (`components/charts.py`)

Uses `backend.graphs.semester_planner_chart()`:
- **Blue bars**: Actual semester GPA (historical)
- **Green bars (70% opacity)**: Target GPA (future semesters)
- Grouped by semester number
- Y-axis: 0–5.0

### 5.4 What-If Simulator (`components/what_if.py`)

**Interaction:**
- Sliders for each remaining semester (default = target GPA from plan)
- Range: 0.00–5.00, step 0.01
- "Simulate" button calls `backend.graphs.what_if_simulator()`
- Updates trajectory chart in real-time
- Shows: "If you get X, Y, Z → Final CGPA = W, Class = V"

**Backend:** `graphs.what_if_simulator(pipeline, what_if_gpas)` → modified PipelineResult → `trajectory_chart()`

### 5.5 Advisor Chat (`components/advisor_chat.py`)

- Single message display (no chat history for MVP)
- Tone badge: 🟢 Encouraging / 🔵 Direct / 🟠 Analytical
- "Regenerate with different tone" button (re-runs advisor)
- Copy to clipboard button

### 5.6 Feature Importance Tables (`components/charts.py`)

- Four expandable sections: Next GPA, Final CGPA, Graduation Class, Academic Risk
- Columns: Feature, Importance (bar + value)
- Sorted by importance descending

---

## 6. Session State Management

```python
# utils/session_state.py
class SessionState:
    pipeline_result: PipelineResult | None = None
    advisor_response: str | None = None
    what_if_gpas: list[float] | None = None
    input_form_data: dict = {}
    last_run_timestamp: float = 0
```

- Persists across page navigation
- Cleared on "New Analysis" button
- Cache invalidation: re-run only when form data changes

---

## 7. Configuration

**`config/streamlit_config.py`:**
```python
st.set_page_config(
    page_title="APIS — Academic Performance Intelligence System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for tone badges, card styling
st.markdown("""
<style>
.tone-encouraging { background: #d4edda; color: #155724; }
.tone-direct { background: #cce5ff; color: #004085; }
.tone-analytical { background: #fff3cd; color: #856404; }
</style>
""", unsafe_allow_html=True)
```

**Environment Variables (from Phase 5):**
- `GEMINI_API_KEY` — for AI Advisor
- Backend model paths (from `backend.config.Settings`)

---

## 8. Dependencies

Add to `requirements.txt`:
```
streamlit>=1.30
streamlit-plotly-events>=0.1  # optional, for click interactions
```

---

## 9. Acceptance Criteria

- [ ] App launches with `streamlit run streamlit_app/app.py`
- [ ] Sidebar form accepts semester data, validates inputs
- [ ] "Run Analysis" calls backend, displays all 4 tabs correctly
- [ ] Trajectory chart shows historical, predicted, goal, First Class line
- [ ] Semester planner shows actual vs target bars
- [ ] What-if sliders update trajectory chart in real-time
- [ ] Advisor panel shows response with tone badge, regenerates on tone change
- [ ] Feature tables display correctly sorted
- [ ] Mobile-responsive (collapsible sidebar, stacked charts)
- [ ] No console errors, clean Streamlit reruns

---

## 10. File Structure (Phase 6 Additions)

```
APIS/
├── streamlit_app/
│   ├── app.py
│   ├── pages/
│   │   ├── 01_📊_Dashboard.py
│   │   ├── 02_📈_Planner.py
│   │   ├── 03_🔮_What_If.py
│   │   ├── 04_🤖_Advisor.py
│   │   └── 05_⚙️_Settings.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── charts.py
│   │   ├── forms.py
│   │   ├── what_if.py
│   │   └── advisor_chat.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── backend_adapter.py
│   │   ├── session_state.py
│   │   └── formatters.py
│   └── config/
│       └── streamlit_config.py
├── requirements.txt          # + streamlit
└── README.md                 # + Dashboard section
```

---

## 11. Deferred to Phase 7+

- User accounts / persistence (PostgreSQL)
- Multi-user, role-based access
- Export to PDF/Word
- Email reports
- Real-time collaboration
- Mobile app
- Custom theming/branding
- Admin panel for model management

---

## 12. Approval

This design has been reviewed and approved for Phase 6 implementation.

**Next Step:** Create implementation plan.
