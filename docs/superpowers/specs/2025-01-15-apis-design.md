# Academic Performance Intelligence System (APIS) — Design Specification

**Version:** 1.0 (Concept Phase)  
**Date:** 2026-07-17  
**Status:** Approved for Implementation  

---

## 1. Project Overview

APIS is an AI-powered academic planning and decision support platform for university students. It combines deterministic academic calculations, machine learning predictions, and generative AI explanations to answer questions like:

- Where do I currently stand academically?
- Can I still achieve a First Class?
- What GPA must I maintain each semester?
- What is my likely graduating CGPA?
- Am I at academic risk?

### Layered Intelligence Architecture

```
Student Data
      │
      ▼
Academic Analytics Engine (Deterministic Calculations)
      │
      ▼
Machine Learning Engine (Predictive Analytics)
      │
      ▼
AI Academic Advisor (Explanation & Guidance)
      │
      ▼
Interactive Dashboard
```

---

## 2. Phasing Strategy

| Phase | Scope | Deliverables |
|-------|-------|--------------|
| **Phase 1** | Research & Requirements | This design document |
| **Phase 2** | Dataset Engineering | Synthetic data generator, validated CSVs |
| **Phase 3** | Model Development | 4 trained models (`.pkl`), experiment log |
| **Phase 4** | Backend Logic | Deterministic engine, ML integration, orchestrator, frozen Advisor contract |
| **Phase 5** (Deferred) | AI Advisor | LLM integration, prompt engineering |
| **Phase 6** (Deferred) | Streamlit Dashboard | Web UI |
| **Phase 7** (Deferred) | Production | Auth, DB, API, cloud deployment |

**This spec covers Phases 1–4 only.**

---

## 3. Directory Structure

```
APIS/
├── data/
│   ├── synthetic_dataset.csv
│   └── processed_dataset.csv
├── data_generation/
│   ├── __init__.py
│   ├── structural_layer.py
│   ├── trajectory_noise.py
│   └── generator.py
├── notebooks/
│   ├── EDA.ipynb
│   └── Model_Training.ipynb
├── models/
│   ├── next_gpa.pkl
│   ├── final_cgpa.pkl
│   ├── graduation_class.pkl
│   ├── academic_risk.pkl
│   └── training_log.jsonl
├── backend/
│   ├── __init__.py
│   ├── config.py
│   ├── schemas.py
│   ├── grading_rules.py
│   ├── trajectory_features.py
│   ├── calculator.py
│   ├── planner.py
│   ├── predictor.py
│   ├── orchestrator.py
│   ├── graphs.py
│   └── mock_advisor.py
├── prompts/
│   └── advisor_prompt.txt
├── tests/
│   ├── test_calculator.py
│   ├── test_planner.py
│   ├── test_predictor.py
│   ├── test_schemas.py
│   └── test_grading_rules.py
├── requirements.txt
└── README.md
```

---

## 4. Core Design Decisions

### 4.1 GPA Scale (MVP)
- **5.0 scale only** (Nigerian university standard)
- `gpa_scale: Literal[5.0]` in schema — structural placeholder for future 4.0 support

### 4.2 Architecture Pattern: Functional Core + Thin Orchestrator
- Pure functions for deterministic math (`calculator.py`, `planner.py`)
- Classes only for stateful components (ML models, LLM client)
- `orchestrator.py` composes functions: `calculator → planner → predictor`
- Pydantic models at every function boundary (single source of truth)

### 4.3 Feature Engineering: Single Implementation
- `backend/trajectory_features.py::compute_trajectory_features()` — **one implementation** used by:
  - `data_generation/generator.py` (training data)
  - `backend/calculator.py` (inference)
- Eliminates train/inference feature drift

### 4.4 ML Label Encoding
- `LabelEncoder` fitted on canonical class lists, saved with each model artifact
- `predictor.py::_decode_label()` uses artifact's encoder — never assumes static order

---

## 5. Mathematical Design (Deterministic Engine)

### 5.1 Weighted CGPA
```
CGPA = Σ(GPAᵢ × Creditsᵢ) / Σ(Creditsᵢ)
```
Returns `None` if no semesters completed (distinct from 0.0 = failing).

### 5.2 Graduation Classification (5.0 Scale — 6 Classes)
| Classification | CGPA Range |
|----------------|------------|
| First Class | 4.50 – 5.00 |
| Second Class Upper | 3.50 – 4.49 |
| Second Class Lower | 2.40 – 3.49 |
| Third Class | 1.50 – 2.39 |
| Pass | 1.00 – 1.49 |
| Fail | < 1.00 |

**Implementation:** `grading_rules.py::classify_cgpa()` iterates `GRADUATION_CLASSES` (ordered highest→lowest), returns first match.

### 5.3 Remaining Semesters
```
semesters_completed = len(semester_records)
semesters_remaining = programme_duration_years × 2 - semesters_completed
```

### 5.4 Target GPA (Required Average)
```
G_required = (G_target × (C_current + C_remaining) - G_current × C_current) / C_remaining
```
Returns `None` if `semesters_remaining == 0` (final semester — no projection).

### 5.5 Goal Feasibility
```
max_achievable = (G_current × C_current + 5.0 × C_remaining) / (C_current + C_remaining)
goal_achievable = required_gpa ≤ 5.0
confidence = 1 - |required_gpa - avg_historical_gpa| / 5.0  (clipped to [0,1])
```

### 5.6 Academic Health Score (0–100)
| Component | Weight | Calculation |
|-----------|--------|-------------|
| Current CGPA (normalized) | 30% | `CGPA / 5.0 × 30` |
| Trend (HEURISTIC) | 25% | Improving=25, Stable=15, Declining=5, InsufficientData=15 |
| Consistency Index | 25% | std_dev ≤0.3→25, ≤0.6→15, >0.6→5 (single semester→25) |
| Goal Progress | 20% | `min(1, current/target) × 20` (no goal→10) |

### 5.7 Improvement Index
Linear regression slope on semester GPAs:
- slope > +0.1 → Improving
- slope < -0.1 → Declining
- Else → Stable
- <2 semesters → InsufficientData

### 5.8 Credit Estimation
```python
CREDITS_PER_LEVEL = {100: 20, 200: 20, 300: 17, 400: 17, 500: 15}  # Midpoints
```
6-year programmes: Level 500 for semesters 9–12 (no Level 600).

---

## 6. Dataset Design (Phase 2)

### 6.1 Schema: `DatasetRow` (One Row = One Student-Semester)
```python
class DatasetRow(BaseModel):
    student_id: str
    university: str
    faculty: str
    department: str
    course: str
    gpa_scale: Literal[5.0]
    programme_duration_years: int
    current_level: int
    semester_number: int
    academic_session: str
    semester_gpa: float
    semester_credits: int
    cumulative_cgpa: float
    cumulative_credits: int
    semesters_completed: int
    semesters_remaining: int
    is_final_semester: bool
    # Engineered features (computed by shared function)
    gpa_trend_slope: float
    gpa_volatility: float
    recent_gpa_avg_3: float
    credits_velocity: float
    # Target labels
    next_semester_gpa: Optional[float]
    final_cgpa: Optional[float]
    graduation_class: Optional[str]
    academic_risk: Optional[str]
```

### 6.2 Two-Layer Generation

**Layer 1: Structural** (`structural_layer.py`) — Deterministic validity
- Credit units per level, GPA bounds, level progression, session format
- Computes derived fields (cumulative CGPA, credits, etc.)

**Layer 2: Trajectory Noise** (`trajectory_noise.py`) — Statistical realism
- Latent `TrajectoryProfile`: base_ability, volatility, trend, shock_probability, shock_magnitude
- Mixture distributions for diverse student archetypes
- Clipped truncated-normal sampling for `base_ability ∈ [2.0, 4.8]`

### 6.3 Target Labels
| Label | Computation |
|-------|-------------|
| `next_semester_gpa` | Next row's `semester_gpa` (None for final) |
| `final_cgpa` | Final semester's cumulative CGPA (same for all rows of student) |
| `graduation_class` | `classify_cgpa(final_cgpa)` — 6 classes |
| `academic_risk` | HEURISTIC: High if CGPA<2.0 or latest GPA<1.5; Medium if CGPA<3.0 or latest GPA<2.5; else Low |

### 6.4 Feature Columns (Single Source of Truth)
```python
FEATURE_COLUMNS = [
    "programme_duration_years",
    "current_level",
    "semester_number",
    "semester_credits",
    "semesters_completed",
    "semesters_remaining",
    "cumulative_cgpa",
    "cumulative_credits",
    "gpa_trend_slope",
    "gpa_volatility",
    "recent_gpa_avg_3",
    "credits_velocity",
]
```
Computed as `ALL_SCHEMA_COLUMNS - EXCLUDED_FROM_FEATURES`.

### 6.5 Validation Checklist (Automated)
- Schema conformance (Pydantic)
- No null targets except final semester
- CGPA recomputation matches (±0.001)
- GPA bounds [0, 5], credit bounds [12, 24]
- Level progression matches `level_for_semester()`
- Target consistency across student's rows
- Class distribution ±10pp of expected (First Class ~15%, 2:1 ~30%, 2:2 ~35%, Third ~15%, Pass ~5%, Fail ~0%)

---

## 7. ML Design (Phase 3)

### 7.1 Tasks & Targets
| Model | Type | Target | Horizion |
|-------|------|--------|----------|
| Model 1 | Regression | `next_semester_gpa` | Next semester |
| Model 2 | Regression | `final_cgpa` | Graduation |
| Model 3 | Classification | `graduation_class` | 6 classes |
| Model 4 | Classification | `academic_risk` | 3 classes |

All use identical `FEATURE_COLUMNS`.

### 7.2 Candidate Models
**Regression:** Linear, Ridge, Random Forest, Gradient Boosting, XGBoost, CatBoost  
**Classification:** Logistic Regression, Random Forest, XGBoost, CatBoost

### 7.3 Evaluation Metrics & Acceptance Thresholds

**Regression:**
| Model | MAE ≤ | RMSE ≤ | R² ≥ |
|-------|-------|--------|------|
| Next GPA | 0.35 | 0.45 | 0.70 |
| Final CGPA | 0.25 | 0.35 | 0.80 |

**Classification:**
| Metric | Target |
|--------|--------|
| Accuracy | ≥ 0.80 |
| Macro F1 | ≥ 0.75 |
| Per-class F1 | All ≥ 0.65 |
| Confusion | No cell > 20% (row-normalized) |

### 7.4 Train/Val/Test Protocol
- Split by `student_id` (70/15/15) — no semester leakage
- `GroupKFold(n=3, groups=student_id)` for CV
- `RandomizedSearchCV` (50 iter) → `GridSearchCV` (fine) → Val selection → Test report

### 7.5 Model Persistence
```python
# Each .pkl contains:
{
    'model': fitted_model,
    'feature_columns': FEATURE_COLUMNS,
    'metrics': {...},
    'feature_importance': [...],
    'label_encoder': LabelEncoder  # For classification models
}
```

### 7.6 ⚠️ Synthetic Data Caveat
> Phase 3 metrics validate the **pipeline**, not real-world accuracy. Models are evaluated on how well they recover the trajectory-noise generator's function. Real-world thresholds must be re-validated with anonymized real data.

---

## 8. Backend Logic (Phase 4)

### 8.1 Input Schema: `StudentInput`
```python
class SemesterRecord(BaseModel):
    semester_number: int
    gpa: float
    credits: int
    academic_session: str

class StudentInput(BaseModel):
    student_name: str
    university: str
    faculty: str
    department: str
    course: str
    gpa_scale: Literal[5.0]
    programme_duration_years: int
    current_level: int
    semester_records: List[SemesterRecord]
    target_graduation_class: Optional[str]
    target_cgpa: Optional[float]
```

### 8.2 Calculator Output: `CalculatorOutput`
```python
class CalculatorOutput(BaseModel):
    current_cgpa: Optional[float]
    total_credits: int
    semesters_completed: int
    semesters_remaining: int
    current_classification: Optional[str]
    gpa_trend: ImprovementTrend
    consistency_index: int
    academic_health_score: int
    # Engineered features (shared with generator)
    gpa_trend_slope: float
    gpa_volatility: float
    recent_gpa_avg_3: float
    credits_velocity: float
```

### 8.3 Planner Output: `PlannerOutput`
```python
class FeasibilityResult(BaseModel):
    goal_achievable: bool
    max_achievable_cgpa: float
    required_average_gpa: Optional[float]  # None if final semester
    realistic_classification: str
    confidence: float
    message: str

class SemesterTarget(BaseModel):
    semester_number: int
    target_gpa: float
    cumulative_cgpa_if_met: float

class PlannerOutput(BaseModel):
    target_cgpa: float
    feasibility: FeasibilityResult
    best_possible_classification: str
    semester_plan: List[SemesterTarget]
```

### 8.4 Predictor Output: `PredictorOutput`
```python
class PredictorOutput(BaseModel):
    predicted_next_gpa: Optional[float]
    predicted_final_cgpa: float
    predicted_graduation_class: str
    predicted_academic_risk: str
    top_features_next_gpa: List[FeatureImportance]
    top_features_final_cgpa: List[FeatureImportance]
    top_features_graduation_class: List[FeatureImportance]
    top_features_academic_risk: List[FeatureImportance]
```

### 8.5 Pipeline Result: `PipelineResult` (Phase 4 Complete Output)
Combines all above + `semester_history` for charts.

### 8.6 Orchestrator: `run_pipeline(student_input) → PipelineResult`
Pure function composition: `calculator → planner → predictor → assemble`

### 8.7 Frozen AI Advisor Contract: `AdvisorInput`
```python
class AdvisorInput(BaseModel):
    student_name: str
    course: str
    current_cgpa: Optional[float]
    target_graduation_class: Optional[str]
    target_cgpa: Optional[float]
    remaining_semesters: int
    required_average_gpa: Optional[float]
    predicted_final_cgpa: float
    predicted_graduation_class: str
    academic_risk: str
    goal_feasible: bool
    best_possible_classification: str
    academic_health_score: int
    gpa_trend: ImprovementTrend
    consistency_index: int
    semester_plan: List[SemesterTarget]
    top_features_final_cgpa: List[FeatureImportance]
    top_features_graduation_class: List[FeatureImportance]
    top_features_academic_risk: List[FeatureImportance]
    tone: Literal["encouraging", "direct", "analytical"] = "encouraging"
    language: str = "en"
```
**FROZEN** — Phase 5 uses exactly this structure. `pipeline_to_advisor_input()` converts `PipelineResult → AdvisorInput`.

### 8.8 Visualization Data Prep: `graphs.py`
Returns Plotly figure dicts (JSON-serializable):
- `trajectory_chart()` — Historical + predicted + goal line
- `semester_planner_chart()` — Actual vs target GPA bars
- `what_if_simulator()` — User-modified future GPAs → updated trajectory

---

## 9. Key Constants (Single Source of Truth)

### 9.1 Graduation Classes (6)
```python
GRADUATION_CLASSES = [
    "First Class", "Second Class Upper", "Second Class Lower",
    "Third Class", "Pass", "Fail"
]
CLASS_MIN_CGPA = {
    "First Class": 4.50, "Second Class Upper": 3.50,
    "Second Class Lower": 2.40, "Third Class": 1.50,
    "Pass": 1.00, "Fail": 0.00
}
```

### 9.2 Academic Risk Classes
```python
ACADEMIC_RISK_CLASSES = ["Low", "Medium", "High"]
RISK_HIGH_CGPA_THRESHOLD = 2.0
RISK_HIGH_GPA_THRESHOLD = 1.5
RISK_MEDIUM_CGPA_THRESHOLD = 3.0
RISK_MEDIUM_GPA_THRESHOLD = 2.5
```

### 9.3 Credit Rules
```python
CREDITS_PER_LEVEL = {100: 20, 200: 20, 300: 17, 400: 17, 500: 15}
```

### 9.4 Heuristic Constants (Explicitly Labeled)
| Constant | Value | Label |
|----------|-------|-------|
| Trend scores | Improving=25, Stable=15, Declining=5 | HEURISTIC |
| Slope thresholds | >0.1 / <-0.1 | HEURISTIC |
| Consistency bins | ≤0.3, ≤0.6, >0.6 | HEURISTIC |
| Risk thresholds | CGPA 2.0/3.0, GPA 1.5/2.5 | HEURISTIC |

---

## 10. Testing Strategy

| Module | Tests |
|--------|-------|
| `calculator.py` | CGPA, trend, consistency, health score, edge cases (0 semesters, 1 semester) |
| `planner.py` | Feasibility (achievable/unachievable), final semester (None required), semester plan |
| `predictor.py` | Load validation, prediction shape, label decoding, mock model injection |
| `schemas.py` | Round-trip validation, boundary values |
| `grading_rules.py` | Classification boundaries (4.50, 4.49, 3.50, 3.49, 2.40, 2.39, 1.50, 1.49, 1.00, 0.99, 0.00) |

---

## 11. Configuration

```python
# backend/config.py
class Settings(BaseSettings):
    MODEL_DIR: str = "models"
    NEXT_GPA_MODEL: str = "next_gpa.pkl"
    FINAL_CGPA_MODEL: str = "final_cgpa.pkl"
    GRADUATION_CLASS_MODEL: str = "graduation_class.pkl"
    ACADEMIC_RISK_MODEL: str = "academic_risk.pkl"
    # Future: OPENAI_API_KEY, GEMINI_API_KEY
    class Config:
        env_file = ".env"
```

---

## 12. Phase 4 Deliverables Checklist

- [ ] `backend/grading_rules.py` — Classification, credits, level mapping
- [ ] `backend/trajectory_features.py` — Shared feature computation
- [ ] `backend/calculator.py` — Pure functions, typed returns
- [ ] `backend/planner.py` — Goal resolution, feasibility, semester plan
- [ ] `backend/predictor.py` — Lazy model loading, feature building, label decoding
- [ ] `backend/orchestrator.py` — `run_pipeline()` composition
- [ ] `backend/graphs.py` — Plotly figure dicts
- [ ] `backend/mock_advisor.py` — Contract validation
- [ ] `backend/schemas.py` — All Pydantic models
- [ ] `backend/config.py` — Settings
- [ ] `tests/` — Unit tests for all modules
- [ ] `data_generation/` — Generator with shared features
- [ ] `notebooks/` — EDA + Model_Training
- [ ] `models/` — 4 trained `.pkl` + `training_log.jsonl`

---

## 13. Deferred to Phase 5+

- AI Academic Advisor (LLM integration)
- Streamlit Dashboard
- User accounts / persistence
- Multi-university / 4.0 scale support
- Production deployment (FastAPI, PostgreSQL, Docker)

---

## 14. Approval

This design has been reviewed and approved for implementation of Phases 1–4.

**Next Step:** Invoke `writing-plans` skill to create detailed implementation plan.