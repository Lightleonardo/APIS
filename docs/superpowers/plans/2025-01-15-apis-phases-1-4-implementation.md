# APIS Phases 1–4 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the Academic Performance Intelligence System (APIS) Phases 1–4: deterministic academic analytics engine, synthetic dataset generator, ML model training pipeline, and backend orchestrator with frozen AI Advisor contract.

**Architecture:** Functional core (pure functions for calculations) + thin orchestrator. Pydantic models at every boundary. Shared feature engineering between data generation and inference. ML models trained on synthetic data, persisted as `.pkl` with label encoders.

**Tech Stack:** Python 3.11+, Pydantic v2, Pandas, NumPy, Scikit-learn, XGBoost, CatBoost, Joblib, Plotly, Pytest

---

## File Structure Map

| File | Responsibility |
|------|----------------|
| `backend/schemas.py` | All Pydantic models (single source of truth) |
| `backend/config.py` | Settings (model paths, future API keys) |
| `backend/grading_rules.py` | Classification, credit rules, level mapping (6 classes) |
| `backend/trajectory_features.py` | Shared feature computation (generator + calculator) |
| `backend/calculator.py` | Pure functions: CGPA, trend, consistency, health score |
| `backend/planner.py` | Goal resolution, feasibility, semester plan |
| `backend/predictor.py` | Lazy model loading, feature building, label decoding |
| `backend/orchestrator.py` | `run_pipeline()` composition |
| `backend/graphs.py` | Plotly figure dicts for frontend |
| `backend/mock_advisor.py` | AdvisorInput contract validation |
| `data_generation/structural_layer.py` | Deterministic validity (credits, levels, sessions) |
| `data_generation/trajectory_noise.py` | Statistical realism (trajectory profiles) |
| `data_generation/generator.py` | Orchestrates both layers, outputs CSV |
| `notebooks/EDA.ipynb` | Data quality, distributions, correlations |
| `notebooks/Model_Training.ipynb` | Full training pipeline with CV, selection, persistence |
| `tests/test_*.py` | Unit tests for every module |

---

## Phase 1: Foundation (Schemas, Config, Grading Rules)

### Task 1: `backend/config.py` — Settings

**Files:**
- Create: `backend/config.py`
- Test: `tests/test_config.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_config.py
import pytest
from backend.config import Settings

def test_settings_defaults():
    s = Settings()
    assert s.MODEL_DIR == "models"
    assert s.NEXT_GPA_MODEL == "next_gpa.pkl"
    assert s.FINAL_CGPA_MODEL == "final_cgpa.pkl"
    assert s.GRADUATION_CLASS_MODEL == "graduation_class.pkl"
    assert s.ACADEMIC_RISK_MODEL == "academic_risk.pkl"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_config.py::test_settings_defaults -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write minimal implementation**

```python
# backend/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    MODEL_DIR: str = "models"
    NEXT_GPA_MODEL: str = "next_gpa.pkl"
    FINAL_CGPA_MODEL: str = "final_cgpa.pkl"
    GRADUATION_CLASS_MODEL: str = "graduation_class.pkl"
    ACADEMIC_RISK_MODEL: str = "academic_risk.pkl"

    class Config:
        env_file = ".env"

settings = Settings()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_config.py::test_settings_defaults -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/config.py tests/test_config.py
git commit -m "feat: add config settings with model paths"
```

---

### Task 2: `backend/grading_rules.py` — Classification & Credit Rules

**Files:**
- Create: `backend/grading_rules.py`
- Test: `tests/test_grading_rules.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_grading_rules.py
import pytest
from backend.grading_rules import (
    classify_cgpa,
    GRADUATION_CLASSES,
    CLASS_MIN_CGPA,
    CREDITS_PER_LEVEL,
    estimate_credits_for_semester,
    estimate_remaining_credits,
    level_for_semester,
)

class TestClassifyCGPA:
    @pytest.mark.parametrize("cgpa,expected", [
        (5.00, "First Class"),
        (4.50, "First Class"),
        (4.49, "Second Class Upper"),
        (3.50, "Second Class Upper"),
        (3.49, "Second Class Lower"),
        (2.40, "Second Class Lower"),
        (2.39, "Third Class"),
        (1.50, "Third Class"),
        (1.49, "Pass"),
        (1.00, "Pass"),
        (0.99, "Fail"),
        (0.00, "Fail"),
    ])
    def test_boundaries(self, cgpa, expected):
        assert classify_cgpa(cgpa) == expected

class TestCredits:
    def test_credits_per_level_midpoints(self):
        assert CREDITS_PER_LEVEL[100] == 20
        assert CREDITS_PER_LEVEL[200] == 20
        assert CREDITS_PER_LEVEL[300] == 17
        assert CREDITS_PER_LEVEL[400] == 17
        assert CREDITS_PER_LEVEL[500] == 15

    def test_estimate_credits_for_semester(self):
        assert estimate_credits_for_semester(100) == 20
        assert estimate_credits_for_semester(300) == 17
        assert estimate_credits_for_semester(999) == 17  # fallback

    def test_estimate_remaining_credits(self):
        assert estimate_remaining_credits(300, 4) == 68  # 17 * 4

class TestLevelMapping:
    @pytest.mark.parametrize("sem,total,expected", [
        (1, 8, 100), (2, 8, 100), (3, 8, 200), (4, 8, 200),
        (5, 8, 300), (6, 8, 300), (7, 8, 400), (8, 8, 400),
        (1, 10, 100), (5, 10, 300), (9, 10, 500), (10, 10, 500),
        (1, 12, 100), (9, 12, 500), (12, 12, 500),
    ])
    def test_level_for_semester(self, sem, total, expected):
        assert level_for_semester(sem, total) == expected
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grading_rules.py -v`
Expected: FAIL (module not found)

- [ ] **Step 3: Write implementation**

```python
# backend/grading_rules.py
from typing import Dict, List

GRADUATION_CLASSES: List[str] = [
    "First Class",
    "Second Class Upper",
    "Second Class Lower",
    "Third Class",
    "Pass",
    "Fail",
]

CLASS_MIN_CGPA: Dict[str, float] = {
    "First Class": 4.50,
    "Second Class Upper": 3.50,
    "Second Class Lower": 2.40,
    "Third Class": 1.50,
    "Pass": 1.00,
    "Fail": 0.00,
}

CREDITS_PER_LEVEL: Dict[int, int] = {
    100: 20,
    200: 20,
    300: 17,
    400: 17,
    500: 15,
}

def classify_cgpa(cgpa: float) -> str:
    """Returns one of GRADUATION_CLASSES. Ordered highest→lowest."""
    for cls in GRADUATION_CLASSES:
        if cgpa >= CLASS_MIN_CGPA[cls]:
            return cls
    return "Fail"

def estimate_credits_for_semester(current_level: int) -> int:
    return CREDITS_PER_LEVEL.get(current_level, 17)

def estimate_remaining_credits(current_level: int, semesters_remaining: int) -> int:
    return estimate_credits_for_semester(current_level) * semesters_remaining

def level_for_semester(semester_number: int, total_semesters: int) -> int:
    if total_semesters == 8:
        boundaries = [(1, 2, 100), (3, 4, 200), (5, 6, 300), (7, 8, 400)]
    elif total_semesters == 10:
        boundaries = [(1, 2, 100), (3, 4, 200), (5, 6, 300), (7, 8, 400), (9, 10, 500)]
    elif total_semesters == 12:
        boundaries = [(1, 2, 100), (3, 4, 200), (5, 6, 300), (7, 8, 400), (9, 12, 500)]
    else:
        raise ValueError(f"Unsupported programme length: {total_semesters} semesters")
    
    for start, end, level in boundaries:
        if start <= semester_number <= end:
            return level
    raise ValueError(f"Semester {semester_number} out of range for {total_semesters}-semester programme")
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_grading_rules.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/grading_rules.py tests/test_grading_rules.py
git commit -m "feat: add grading rules with 6-class classification and credit estimation"
```

---

### Task 3: `backend/schemas.py` — All Pydantic Models

**Files:**
- Create: `backend/schemas.py`
- Test: `tests/test_schemas.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_schemas.py
import pytest
from pydantic import ValidationError
from backend.schemas import (
    StudentInput, SemesterRecord, CalculatorOutput, PlannerOutput,
    FeasibilityResult, SemesterTarget, PredictorOutput, PipelineResult,
    AdvisorInput, ImprovementTrend, DatasetRow, FeatureImportance,
    FEATURE_COLUMNS, EXCLUDED_FROM_FEATURES, GRADUATION_CLASSES,
)

class TestSemesterRecord:
    def test_valid_record(self):
        r = SemesterRecord(semester_number=1, gpa=4.5, credits=18, academic_session="2023/2024")
        assert r.gpa == 4.5

    def test_gpa_bounds(self):
        with pytest.raises(ValidationError):
            SemesterRecord(semester_number=1, gpa=5.1, credits=18, academic_session="2023/2024")
        with pytest.raises(ValidationError):
            SemesterRecord(semester_number=1, gpa=-0.1, credits=18, academic_session="2023/2024")

class TestStudentInput:
    def test_valid_minimal(self):
        s = StudentInput(
            student_name="Test Student",
            university="Test Uni",
            faculty="Science",
            department="Physics",
            course="Physics",
            gpa_scale=5.0,
            programme_duration_years=5,
            current_level=300,
            semester_records=[
                SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="2021/2022"),
                SemesterRecord(semester_number=2, gpa=4.2, credits=20, academic_session="2021/2022"),
            ],
        )
        assert s.current_level == 300

    def test_level_consistency_validation(self):
        with pytest.raises(ValidationError):
            StudentInput(
                student_name="Test",
                university="U", faculty="F", department="D", course="C",
                programme_duration_years=5, current_level=100,
                semester_records=[SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="2021/2022")],
            )

class TestCalculatorOutput:
    def test_all_fields_present(self):
        out = CalculatorOutput(
            current_cgpa=4.2, total_credits=40, semesters_completed=2,
            semesters_remaining=8, current_classification="First Class",
            gpa_trend=ImprovementTrend.STABLE, consistency_index=25,
            academic_health_score=85,
            gpa_trend_slope=0.1, gpa_volatility=0.2,
            recent_gpa_avg_3=4.1, credits_velocity=20.0,
        )
        assert out.current_cgpa == 4.2

class TestDatasetRow:
    def test_feature_columns_excludes_targets(self):
        targets = {"next_semester_gpa", "final_cgpa", "graduation_class", "academic_risk"}
        for t in targets:
            assert t not in FEATURE_COLUMNS
        assert "student_id" not in FEATURE_COLUMNS

    def test_6_classes_in_graduation(self):
        assert len(GRADUATION_CLASSES) == 6
        assert "Fail" in GRADUATION_CLASSES
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_schemas.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# backend/schemas.py
from typing import List, Optional, Literal, Dict, Any, Set
from pydantic import BaseModel, Field, model_validator
from enum import Enum

from backend.grading_rules import GRADUATION_CLASSES, CLASS_MIN_CGPA, level_for_semester

class ImprovementTrend(str, Enum):
    IMPROVING = "Improving"
    STABLE = "Stable"
    DECLINING = "Declining"
    INSUFFICIENT_DATA = "InsufficientData"

class SemesterRecord(BaseModel):
    semester_number: int = Field(ge=1, le=12)
    gpa: float = Field(ge=0.0, le=5.0)
    credits: int = Field(ge=12, le=24)
    academic_session: str

class StudentInput(BaseModel):
    student_name: str
    university: str
    faculty: str
    department: str
    course: str
    gpa_scale: Literal[5.0] = 5.0
    programme_duration_years: int = Field(ge=4, le=6)
    current_level: int = Field(ge=100, le=500)
    semester_records: List[SemesterRecord] = Field(min_length=1)
    target_graduation_class: Optional[str] = None
    target_cgpa: Optional[float] = Field(default=None, ge=0.0, le=5.0)

    @model_validator(mode='after')
    def validate_level_consistency(self):
        expected_level = level_for_semester(
            len(self.semester_records),
            self.programme_duration_years * 2
        )
        if self.current_level != expected_level:
            raise ValueError(
                f"current_level ({self.current_level}) inconsistent with "
                f"{len(self.semester_records)} semesters completed "
                f"(expected {expected_level})"
            )
        return self

class CalculatorOutput(BaseModel):
    current_cgpa: Optional[float]
    total_credits: int
    semesters_completed: int
    semesters_remaining: int
    current_classification: Optional[str]
    gpa_trend: ImprovementTrend
    consistency_index: int
    academic_health_score: int
    gpa_trend_slope: float
    gpa_volatility: float
    recent_gpa_avg_3: float
    credits_velocity: float

class FeasibilityResult(BaseModel):
    goal_achievable: bool
    max_achievable_cgpa: float
    required_average_gpa: Optional[float]
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

class FeatureImportance(BaseModel):
    feature: str
    importance: float

class PredictorOutput(BaseModel):
    predicted_next_gpa: Optional[float]
    predicted_final_cgpa: float
    predicted_graduation_class: str
    predicted_academic_risk: str
    top_features_next_gpa: List[FeatureImportance] = []
    top_features_final_cgpa: List[FeatureImportance] = []
    top_features_graduation_class: List[FeatureImportance] = []
    top_features_academic_risk: List[FeatureImportance] = []

class SemesterHistoryPoint(BaseModel):
    semester_number: int
    gpa: float
    cumulative_cgpa: float
    credits: int
    academic_session: str

class PipelineResult(BaseModel):
    student_name: str
    university: str
    faculty: str
    department: str
    course: str
    programme_duration_years: int
    current_level: int
    target_graduation_class: Optional[str]
    target_cgpa: Optional[float]
    current_cgpa: Optional[float]
    total_credits: int
    semesters_completed: int
    semesters_remaining: int
    current_classification: Optional[str]
    gpa_trend: ImprovementTrend
    consistency_index: int
    academic_health_score: int
    target_cgpa_resolved: float
    feasibility: FeasibilityResult
    best_possible_classification: str
    semester_plan: List[SemesterTarget]
    predicted_next_gpa: Optional[float]
    predicted_final_cgpa: float
    predicted_graduation_class: str
    predicted_academic_risk: str
    top_features_next_gpa: List[FeatureImportance]
    top_features_final_cgpa: List[FeatureImportance]
    top_features_graduation_class: List[FeatureImportance]
    top_features_academic_risk: List[FeatureImportance]
    semester_history: List[SemesterHistoryPoint]

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
    gpa_trend_slope: float
    gpa_volatility: float
    recent_gpa_avg_3: float
    credits_velocity: float
    next_semester_gpa: Optional[float]
    final_cgpa: Optional[float]
    graduation_class: Optional[str]
    academic_risk: Optional[str]

# Feature/Target column definitions (single source of truth)
EXCLUDED_FROM_FEATURES: Set[str] = {
    "student_id",
    "next_semester_gpa", "final_cgpa", "graduation_class", "academic_risk",
    "university", "faculty", "department", "course", "gpa_scale",
    "academic_session", "is_final_semester",
}

ALL_SCHEMA_COLUMNS: List[str] = list(DatasetRow.model_fields.keys())

FEATURE_COLUMNS: List[str] = [
    col for col in ALL_SCHEMA_COLUMNS
    if col not in EXCLUDED_FROM_FEATURES
]

TARGET_NEXT_GPA = "next_semester_gpa"
TARGET_FINAL_CGPA = "final_cgpa"
TARGET_GRADUATION_CLASS = "graduation_class"
TARGET_ACADEMIC_RISK = "academic_risk"

ACADEMIC_RISK_CLASSES: List[str] = ["Low", "Medium", "High"]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_schemas.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/schemas.py tests/test_schemas.py
git commit -m "feat: add all Pydantic schemas with feature/target column definitions"
```

---

## Phase 2: Trajectory Features & Calculator

### Task 4: `backend/trajectory_features.py` — Shared Feature Computation

**Files:**
- Create: `backend/trajectory_features.py`
- Test: `tests/test_trajectory_features.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_trajectory_features.py
import pytest
from backend.trajectory_features import compute_trajectory_features

class TestComputeTrajectoryFeatures:
    def test_empty_gpas(self):
        feat = compute_trajectory_features([], 0)
        assert feat == {
            "gpa_trend_slope": 0.0,
            "gpa_volatility": 0.0,
            "recent_gpa_avg_3": 0.0,
            "credits_velocity": 0.0,
        }

    def test_single_gpa(self):
        feat = compute_trajectory_features([4.0], 20)
        assert feat["gpa_trend_slope"] == 0.0
        assert feat["gpa_volatility"] == 0.0
        assert feat["recent_gpa_avg_3"] == 4.0
        assert feat["credits_velocity"] == 20.0

    def test_two_gpas_perfect_line(self):
        # GPAs: 3.0, 4.0 → slope = 1.0
        feat = compute_trajectory_features([3.0, 4.0], 40)
        assert feat["gpa_trend_slope"] == 1.0
        assert feat["gpa_volatility"] == pytest.approx(0.7071, rel=1e-3)
        assert feat["recent_gpa_avg_3"] == 3.5
        assert feat["credits_velocity"] == 20.0

    def test_three_gpas(self):
        feat = compute_trajectory_features([3.0, 3.5, 4.0], 60)
        assert feat["recent_gpa_avg_3"] == pytest.approx(3.5, rel=1e-3)
        assert feat["credits_velocity"] == 20.0

    def test_four_gpas_recent_avg_3(self):
        feat = compute_trajectory_features([3.0, 3.5, 4.0, 4.5], 80)
        assert feat["recent_gpa_avg_3"] == pytest.approx(4.0, rel=1e-3)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trajectory_features.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# backend/trajectory_features.py
import statistics
from typing import List, Dict

def compute_trajectory_features(gpas: List[float], total_credits: int) -> Dict[str, float]:
    """
    Single source of truth for the 4 engineered features.
    gpas must be ONLY semesters completed so far (chronological, no future data).
    """
    n = len(gpas)

    gpa_trend_slope = 0.0
    if n >= 2:
        x_mean = (n + 1) / 2  # mean of 1..n
        y_mean = sum(gpas) / n
        num = sum((i + 1 - x_mean) * (gpas[i] - y_mean) for i in range(n))
        den = sum((i + 1 - x_mean) ** 2 for i in range(n))
        gpa_trend_slope = num / den if den != 0 else 0.0

    gpa_volatility = statistics.stdev(gpas) if n >= 2 else 0.0
    recent_gpa_avg_3 = sum(gpas[-3:]) / min(3, n) if n > 0 else 0.0
    credits_velocity = total_credits / n if n > 0 else 0.0

    return {
        "gpa_trend_slope": gpa_trend_slope,
        "gpa_volatility": gpa_volatility,
        "recent_gpa_avg_3": recent_gpa_avg_3,
        "credits_velocity": credits_velocity,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trajectory_features.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/trajectory_features.py tests/test_trajectory_features.py
git commit -m "feat: add shared trajectory feature computation"
```

---

### Task 5: `backend/calculator.py` — Deterministic Calculations

**Files:**
- Create: `backend/calculator.py`
- Test: `tests/test_calculator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_calculator.py
import pytest
from backend.calculator import (
    calculate_cgpa, calculate_gpa_trend, calculate_consistency_index,
    calculate_academic_health_score, run_calculator
)
from backend.schemas import StudentInput, SemesterRecord, CalculatorOutput, ImprovementTrend

class TestCalculateCGPA:
    def test_empty_returns_none(self):
        assert calculate_cgpa([]) is None

    def test_single_semester(self):
        records = [SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="2023/2024")]
        assert calculate_cgpa(records) == 4.0

    def test_weighted_multiple(self):
        records = [
            SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="2023/2024"),
            SemesterRecord(semester_number=2, gpa=3.0, credits=18, academic_session="2023/2024"),
        ]
        # (4*20 + 3*18) / 38 = 124/38 = 3.263...
        assert calculate_cgpa(records) == pytest.approx(3.26, abs=0.01)

class TestCalculateGPATrend:
    def test_insufficient_data(self):
        assert calculate_gpa_trend([SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="x")]) \
            == ImprovementTrend.INSUFFICIENT_DATA

    def test_improving(self):
        records = [
            SemesterRecord(semester_number=1, gpa=3.0, credits=20, academic_session="x"),
            SemesterRecord(semester_number=2, gpa=4.0, credits=20, academic_session="x"),
        ]
        # slope = 1.0 > 0.1
        assert calculate_gpa_trend(records) == ImprovementTrend.IMPROVING

    def test_declining(self):
        records = [
            SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="x"),
            SemesterRecord(semester_number=2, gpa=3.0, credits=20, academic_session="x"),
        ]
        assert calculate_gpa_trend(records) == ImprovementTrend.DECLINING

    def test_stable(self):
        records = [
            SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="x"),
            SemesterRecord(semester_number=2, gpa=3.6, credits=20, academic_session="x"),
        ]
        assert calculate_gpa_trend(records) == ImprovementTrend.STABLE

class TestCalculateConsistencyIndex:
    def test_insufficient_data(self):
        assert calculate_consistency_index([SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="x")]) == 25

    def test_high_consistency(self):
        records = [SemesterRecord(semester_number=i, gpa=4.0, credits=20, academic_session="x") for i in range(1, 5)]
        assert calculate_consistency_index(records) == 25

    def test_medium_consistency(self):
        records = [
            SemesterRecord(semester_number=1, gpa=3.8, credits=20, academic_session="x"),
            SemesterRecord(semester_number=2, gpa=4.0, credits=20, academic_session="x"),
            SemesterRecord(semester_number=3, gpa=3.9, credits=20, academic_session="x"),
        ]
        assert calculate_consistency_index(records) == 15

    def test_low_consistency(self):
        records = [
            SemesterRecord(semester_number=1, gpa=5.0, credits=20, academic_session="x"),
            SemesterRecord(semester_number=2, gpa=2.0, credits=20, academic_session="x"),
        ]
        assert calculate_consistency_index(records) == 5

class TestCalculateAcademicHealthScore:
    def test_full_score_components(self):
        # CGPA 5.0 (30), Improving (25), Consistency 25, Goal 4.5 reached (20) = 100
        score = calculate_academic_health_score(
            current_cgpa=5.0,
            trend=ImprovementTrend.IMPROVING,
            consistency_index=25,
            target_cgpa=4.5,
        )
        assert score == 100

    def test_no_goal(self):
        score = calculate_academic_health_score(
            current_cgpa=4.0, trend=ImprovementTrend.STABLE,
            consistency_index=15, target_cgpa=None
        )
        # CGPA: 4/5*30=24, Trend: 15, Consistency: 15, No goal: 10 = 64
        assert score == 64

    def test_insufficient_data_trend_neutral(self):
        score = calculate_academic_health_score(
            current_cgpa=3.0, trend=ImprovementTrend.INSUFFICIENT_DATA,
            consistency_index=25, target_cgpa=None
        )
        assert score == int(3/5*30) + 15 + 25 + 10  # 18+15+25+10=68

class TestRunCalculator:
    def test_full_pipeline(self):
        student = StudentInput(
            student_name="Test",
            university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=300,
            semester_records=[
                SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2021/2022"),
                SemesterRecord(semester_number=2, gpa=4.0, credits=20, academic_session="2021/2022"),
                SemesterRecord(semester_number=3, gpa=4.2, credits=18, academic_session="2022/2023"),
                SemesterRecord(semester_number=4, gpa=4.1, credits=18, academic_session="2022/2023"),
            ],
            target_graduation_class="First Class",
        )
        out = run_calculator(student)
        assert isinstance(out, CalculatorOutput)
        assert out.current_cgpa == pytest.approx(3.94, abs=0.02)
        assert out.semesters_completed == 4
        assert out.semesters_remaining == 6
        assert out.gpa_trend in [ImprovementTrend.IMPROVING, ImprovementTrend.STABLE]
        assert 0 <= out.academic_health_score <= 100
        assert out.gpa_trend_slope != 0.0  # has trend
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_calculator.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# backend/calculator.py
import statistics
from typing import List, Optional
from backend.schemas import (
    StudentInput, SemesterRecord, CalculatorOutput, ImprovementTrend
)
from backend.grading_rules import classify_cgpa
from backend.trajectory_features import compute_trajectory_features

def calculate_cgpa(records: List[SemesterRecord]) -> Optional[float]:
    if not records:
        return None
    total_points = sum(r.gpa * r.credits for r in records)
    total_credits = sum(r.credits for r in records)
    return round(total_points / total_credits, 2)

def calculate_gpa_trend(records: List[SemesterRecord]) -> ImprovementTrend:
    if len(records) < 2:
        return ImprovementTrend.INSUFFICIENT_DATA
    gpas = [r.gpa for r in records]
    n = len(gpas)
    x = list(range(1, n + 1))
    x_mean = sum(x) / n
    y_mean = sum(gpas) / n
    numerator = sum((x[i] - x_mean) * (gpas[i] - y_mean) for i in range(n))
    denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
    slope = numerator / denominator if denominator != 0 else 0.0
    if slope > 0.1:
        return ImprovementTrend.IMPROVING
    elif slope < -0.1:
        return ImprovementTrend.DECLINING
    else:
        return ImprovementTrend.STABLE

def calculate_consistency_index(records: List[SemesterRecord]) -> int:
    if len(records) < 2:
        return 25
    gpas = [r.gpa for r in records]
    std_dev = statistics.stdev(gpas)
    if std_dev <= 0.3:
        return 25
    elif std_dev <= 0.6:
        return 15
    else:
        return 5

def calculate_academic_health_score(
    current_cgpa: Optional[float],
    trend: ImprovementTrend,
    consistency_index: int,
    target_cgpa: Optional[float],
) -> int:
    score = 0
    if current_cgpa is not None:
        score += int((current_cgpa / 5.0) * 30)
    trend_scores = {
        ImprovementTrend.IMPROVING: 25,
        ImprovementTrend.STABLE: 15,
        ImprovementTrend.DECLINING: 5,
        ImprovementTrend.INSUFFICIENT_DATA: 15,
    }
    score += trend_scores[trend]
    score += consistency_index
    if target_cgpa is not None and current_cgpa is not None:
        progress = min(1.0, current_cgpa / target_cgpa)
        score += int(progress * 20)
    else:
        score += 10
    return min(100, max(0, score))

def run_calculator(student_input: StudentInput) -> CalculatorOutput:
    records = student_input.semester_records
    current_cgpa = calculate_cgpa(records)
    total_credits = sum(r.credits for r in records)
    semesters_completed = len(records)
    total_semesters = student_input.programme_duration_years * 2
    semesters_remaining = total_semesters - semesters_completed

    trend = calculate_gpa_trend(records)
    consistency = calculate_consistency_index(records)
    health_score = calculate_academic_health_score(
        current_cgpa=current_cgpa,
        trend=trend,
        consistency_index=consistency,
        target_cgpa=student_input.target_cgpa,
    )
    classification = classify_cgpa(current_cgpa) if current_cgpa is not None else None

    features = compute_trajectory_features(
        gpas=[r.gpa for r in records],
        total_credits=total_credits,
    )

    return CalculatorOutput(
        current_cgpa=current_cgpa,
        total_credits=total_credits,
        semesters_completed=semesters_completed,
        semesters_remaining=semesters_remaining,
        current_classification=classification,
        gpa_trend=trend,
        consistency_index=consistency,
        academic_health_score=health_score,
        gpa_trend_slope=features["gpa_trend_slope"],
        gpa_volatility=features["gpa_volatility"],
        recent_gpa_avg_3=features["recent_gpa_avg_3"],
        credits_velocity=features["credits_velocity"],
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_calculator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/calculator.py tests/test_calculator.py
git commit -m "feat: add calculator with CGPA, trend, consistency, health score"
```

---

## Phase 3: Planner (Goal Feasibility & Semester Plan)

### Task 6: `backend/planner.py` — Goal Planning & Feasibility

**Files:**
- Create: `backend/planner.py`
- Test: `tests/test_planner.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_planner.py
import pytest
from backend.planner import (
    resolve_target_cgpa, compute_feasibility, compute_semester_plan, run_planner
)
from backend.schemas import StudentInput, SemesterRecord, CalculatorOutput, FeasibilityResult
from backend.grading_rules import CLASS_MIN_CGPA

class TestResolveTargetCGPA:
    def test_explicit_cgpa_target(self):
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=300,
            semester_records=[SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="x")],
            target_cgpa=4.75,
        )
        assert resolve_target_cgpa(student) == 4.75

    def test_class_target_first_class(self):
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=300,
            semester_records=[SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="x")],
            target_graduation_class="First Class",
        )
        assert resolve_target_cgpa(student) == CLASS_MIN_CGPA["First Class"]

    def test_default_first_class(self):
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=300,
            semester_records=[SemesterRecord(semester_number=1, gpa=4.0, credits=20, academic_session="x")],
        )
        assert resolve_target_cgpa(student) == CLASS_MIN_CGPA["First Class"]

class TestComputeFeasibility:
    def test_achievable_goal(self):
        calc = CalculatorOutput(
            current_cgpa=4.0, total_credits=40, semesters_completed=2,
            semesters_remaining=8, current_classification="Second Class Upper",
            gpa_trend=None, consistency_index=0, academic_health_score=0,
            gpa_trend_slope=0, gpa_volatility=0, recent_gpa_avg_3=0, credits_velocity=0,
        )
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=200,
            semester_records=[SemesterRecord(semester_number=i, gpa=4.0, credits=20, academic_session="x") for i in range(1,3)],
            target_graduation_class="First Class",
        )
        feasibility = compute_feasibility(calc, student, 4.50)
        assert feasibility.goal_achievable is True
        assert feasibility.required_average_gpa is not None
        assert feasibility.required_average_gpa <= 5.0
        assert feasibility.confidence > 0

    def test_unachievable_goal(self):
        calc = CalculatorOutput(
            current_cgpa=2.0, total_credits=40, semesters_completed=2,
            semesters_remaining=2, current_classification="Pass",
            gpa_trend=None, consistency_index=0, academic_health_score=0,
            gpa_trend_slope=0, gpa_volatility=0, recent_gpa_avg_3=0, credits_velocity=0,
        )
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=200,
            semester_records=[SemesterRecord(semester_number=i, gpa=2.0, credits=20, academic_session="x") for i in range(1,3)],
            target_graduation_class="First Class",
        )
        feasibility = compute_feasibility(calc, student, 4.50)
        assert feasibility.goal_achievable is False
        assert feasibility.required_average_gpa is not None
        assert feasibility.required_average_gpa > 5.0 or feasibility.max_achievable_cgpa < 4.50
        assert "NOT achievable" in feasibility.message

    def test_final_semester_none_required(self):
        calc = CalculatorOutput(
            current_cgpa=4.2, total_credits=120, semesters_completed=10,
            semesters_remaining=0, current_classification="First Class",
            gpa_trend=None, consistency_index=0, academic_health_score=0,
            gpa_trend_slope=0, gpa_volatility=0, recent_gpa_avg_3=0, credits_velocity=0,
        )
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=500,
            semester_records=[SemesterRecord(semester_number=i, gpa=4.0, credits=15, academic_session="x") for i in range(1,11)],
            target_graduation_class="First Class",
        )
        feasibility = compute_feasibility(calc, student, 4.50)
        assert feasibility.required_average_gpa is None
        assert feasibility.goal_achievable == (calc.current_cgpa >= 4.50)

class TestComputeSemesterPlan:
    def test_plan_returns_correct_length(self):
        calc = CalculatorOutput(
            current_cgpa=3.5, total_credits=40, semesters_completed=2,
            semesters_remaining=4, current_classification="2:1",
            gpa_trend=None, consistency_index=0, academic_health_score=0,
            gpa_trend_slope=0, gpa_volatility=0, recent_gpa_avg_3=0, credits_velocity=0,
        )
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=200,
            semester_records=[SemesterRecord(semester_number=i, gpa=3.5, credits=20, academic_session="x") for i in range(1,3)],
            target_cgpa=4.0,
        )
        feasibility = FeasibilityResult(
            goal_achievable=True, max_achievable_cgpa=4.2,
            required_average_gpa=4.25, realistic_classification="First Class",
            confidence=0.8, message="ok",
        )
        plan = compute_semester_plan(calc, student, 4.0, feasibility)
        assert len(plan) == 4
        for p in plan:
            assert p.target_gpa == 4.25
            assert p.cumulative_cgpa_if_met >= 3.5

    def test_plan_uses_target_cgpa_when_required_is_none(self):
        calc = CalculatorOutput(
            current_cgpa=4.5, total_credits=120, semesters_completed=10,
            semesters_remaining=0, current_classification="First Class",
            gpa_trend=None, consistency_index=0, academic_health_score=0,
            gpa_trend_slope=0, gpa_volatility=0, recent_gpa_avg_3=0, credits_velocity=0,
        )
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=500,
            semester_records=[SemesterRecord(semester_number=i, gpa=4.0, credits=15, academic_session="x") for i in range(1,11)],
            target_cgpa=4.5,
        )
        feasibility = FeasibilityResult(
            goal_achievable=True, max_achievable_cgpa=4.5,
            required_average_gpa=None, realistic_classification="First Class",
            confidence=1.0, message="final semester",
        )
        plan = compute_semester_plan(calc, student, 4.5, feasibility)
        assert plan == []

class TestRunPlanner:
    def test_integration(self):
        student = StudentInput(
            student_name="Test",
            university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=300,
            semester_records=[
                SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2021/2022"),
                SemesterRecord(semester_number=2, gpa=3.8, credits=20, academic_session="2021/2022"),
                SemesterRecord(semester_number=3, gpa=4.0, credits=18, academic_session="2022/2023"),
                SemesterRecord(semester_number=4, gpa=4.1, credits=18, academic_session="2022/2023"),
            ],
            target_graduation_class="First Class",
        )
        from backend.calculator import run_calculator
        calc = run_calculator(student)
        planner_out = run_planner(student, calc)
        assert planner_out.target_cgpa == 4.50
        assert isinstance(planner_out.feasibility, FeasibilityResult)
        assert len(planner_out.semester_plan) == 6
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_planner.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# backend/planner.py
from typing import List, Optional
from backend.schemas import (
    StudentInput, CalculatorOutput, PlannerOutput, FeasibilityResult, SemesterTarget
)
from backend.grading_rules import (
    classify_cgpa, CLASS_MIN_CGPA, estimate_remaining_credits, estimate_credits_for_semester
)
from backend.calculator import compute_trajectory_features

def resolve_target_cgpa(student_input: StudentInput) -> float:
    if student_input.target_cgpa is not None:
        return student_input.target_cgpa
    class_target = student_input.target_graduation_class or "First Class"
    return CLASS_MIN_CGPA.get(class_target, 4.50)

def compute_feasibility_confidence(required_avg: float, historical_gpas: List[float]) -> float:
    if not historical_gpas:
        return 0.5
    avg_historical = sum(historical_gpas) / len(historical_gpas)
    distance = abs(required_avg - avg_historical)
    normalized = distance / 5.0
    confidence = 1.0 - normalized
    return max(0.0, min(1.0, confidence))

def compute_feasibility(
    calculator_out: CalculatorOutput,
    student_input: StudentInput,
    target_cgpa: float,
) -> FeasibilityResult:
    current_cgpa = calculator_out.current_cgpa
    current_credits = calculator_out.total_credits
    semesters_remaining = calculator_out.semesters_remaining

    if semesters_remaining == 0:
        final_cgpa = current_cgpa or 0.0
        return FeasibilityResult(
            goal_achievable=(final_cgpa >= target_cgpa),
            max_achievable_cgpa=final_cgpa,
            required_average_gpa=None,
            realistic_classification=classify_cgpa(final_cgpa),
            confidence=1.0 if final_cgpa >= target_cgpa else 0.0,
            message="Final semester — no remaining semesters to improve."
        )

    c_remaining = estimate_remaining_credits(
        student_input.current_level,
        semesters_remaining,
    )

    max_achievable = (
        (current_cgpa * current_credits + 5.0 * c_remaining) / 
        (current_credits + c_remaining)
    ) if current_cgpa is not None else 5.0

    if current_cgpa is not None:
        required = (
            (target_cgpa * (current_credits + c_remaining) - current_cgpa * current_credits) / 
            c_remaining
        )
        required = max(0.0, min(required, 5.0))
    else:
        required = target_cgpa

    goal_achievable = required <= 5.0
    realistic_classification = classify_cgpa(max_achievable)

    historical_gpas = [r.gpa for r in student_input.semester_records]
    confidence = compute_feasibility_confidence(required, historical_gpas)

    if goal_achievable:
        message = f"Goal achievable. Required average: {required:.2f} per semester."
    else:
        message = (
            f"Goal NOT achievable. Even with perfect 5.0 GPAs, "
            f"max CGPA = {max_achievable:.2f} ({realistic_classification})."
        )

    return FeasibilityResult(
        goal_achievable=goal_achievable,
        max_achievable_cgpa=round(max_achievable, 2),
        required_average_gpa=round(required, 2) if required is not None else None,
        realistic_classification=realistic_classification,
        confidence=round(confidence, 2),
        message=message,
    )

def compute_semester_plan(
    calculator_out: CalculatorOutput,
    student_input: StudentInput,
    target_cgpa: float,
    feasibility: FeasibilityResult,
) -> List[SemesterTarget]:
    if calculator_out.semesters_remaining == 0:
        return []

    required_avg = (
        feasibility.required_average_gpa 
        if feasibility.required_average_gpa is not None 
        else target_cgpa
    )
    current_cgpa = calculator_out.current_cgpa or 0.0
    current_credits = calculator_out.total_credits
    c_per_sem = estimate_credits_for_semester(student_input.current_level)

    plan = []
    running_cgpa = current_cgpa
    running_credits = current_credits

    for i in range(calculator_out.semesters_remaining):
        sem_num = calculator_out.semesters_completed + i + 1
        target_gpa = required_avg

        running_credits += c_per_sem
        running_cgpa = (
            (running_cgpa * (running_credits - c_per_sem) + target_gpa * c_per_sem) / 
            running_credits
        )

        plan.append(SemesterTarget(
            semester_number=sem_num,
            target_gpa=round(target_gpa, 2),
            cumulative_cgpa_if_met=round(running_cgpa, 2),
        ))

    return plan

def run_planner(
    student_input: StudentInput,
    calculator_out: CalculatorOutput,
) -> PlannerOutput:
    target_cgpa = resolve_target_cgpa(student_input)
    feasibility = compute_feasibility(calculator_out, student_input, target_cgpa)
    best_possible = classify_cgpa(feasibility.max_achievable_cgpa)
    semester_plan = compute_semester_plan(calculator_out, student_input, target_cgpa, feasibility)

    return PlannerOutput(
        target_cgpa=target_cgpa,
        feasibility=feasibility,
        best_possible_classification=best_possible,
        semester_plan=semester_plan,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_planner.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/planner.py tests/test_planner.py
git commit -m "feat: add planner with goal feasibility and semester targets"
```

---

## Phase 4: Predictor (ML Integration)

### Task 7: `backend/predictor.py` — Model Loading & Prediction

**Files:**
- Create: `backend/predictor.py`
- Test: `tests/test_predictor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_predictor.py
import pytest
import pandas as pd
from unittest.mock import Mock, MagicMock
from backend.predictor import (
    load_all_models, get_models, build_feature_vector, run_predictor, _decode_label
)
from backend.schemas import StudentInput, SemesterRecord, CalculatorOutput, PredictorOutput, FEATURE_COLUMNS

class MockModel:
    def __init__(self, return_value):
        self.return_value = return_value
    def predict(self, X):
        return [self.return_value]

class TestBuildFeatureVector:
    def test_vector_shape_and_columns(self):
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=300,
            semester_records=[
                SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="x"),
                SemesterRecord(semester_number=2, gpa=4.0, credits=20, academic_session="x"),
                SemesterRecord(semester_number=3, gpa=4.2, credits=18, academic_session="x"),
                SemesterRecord(semester_number=4, gpa=4.1, credits=18, academic_session="x"),
            ],
        )
        calc = CalculatorOutput(
            current_cgpa=3.94, total_credits=76, semesters_completed=4,
            semesters_remaining=6, current_classification="2:1",
            gpa_trend=None, consistency_index=0, academic_health_score=0,
            gpa_trend_slope=0.1, gpa_volatility=0.25,
            recent_gpa_avg_3=4.1, credits_velocity=19.0,
        )
        X = build_feature_vector(student, calc)
        assert isinstance(X, pd.DataFrame)
        assert list(X.columns) == FEATURE_COLUMNS
        assert X.shape == (1, len(FEATURE_COLUMNS))

class TestDecodeLabel:
    def test_string_label_passthrough(self):
        artifact = {'model': MockModel("First Class"), 'label_encoder': None}
        result = _decode_label(artifact, None)
        assert result == "First Class"

    def test_encoded_label_with_encoder(self):
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        le.fit(["Fail", "Pass", "Third Class", "Second Class Lower", "Second Class Upper", "First Class"])
        artifact = {'model': MockModel(0), 'label_encoder': le}  # 0 = "Fail" after fit
        result = _decode_label(artifact, None)
        assert result == "Fail"

class TestRunPredictorWithMocks:
    def test_predictor_output_structure(self):
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=5, current_level=300,
            semester_records=[
                SemesterRecord(semester_number=i, gpa=4.0, credits=20, academic_session="x") for i in range(1,5)
            ],
        )
        calc = CalculatorOutput(
            current_cgpa=4.0, total_credits=80, semesters_completed=4,
            semesters_remaining=6, current_classification="First Class",
            gpa_trend=None, consistency_index=25, academic_health_score=90,
            gpa_trend_slope=0.0, gpa_volatility=0.0,
            recent_gpa_avg_3=4.0, credits_velocity=20.0,
        )
        mock_models = {
            'next_gpa': {'model': MockModel(4.2), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}},
            'final_cgpa': {'model': MockModel(4.1), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}},
            'graduation_class': {'model': MockModel("First Class"), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'label_encoder': None},
            'academic_risk': {'model': MockModel("Low"), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'label_encoder': None},
        }
        out = run_predictor(student, calc, models=mock_models)
        assert isinstance(out, PredictorOutput)
        assert out.predicted_next_gpa == 4.2
        assert out.predicted_final_cgpa == 4.1
        assert out.predicted_graduation_class == "First Class"
        assert out.predicted_academic_risk == "Low"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_predictor.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# backend/predictor.py
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Optional
from backend.config import settings
from backend.schemas import (
    StudentInput, CalculatorOutput, PredictorOutput, FeatureImportance, FEATURE_COLUMNS
)

MODEL_PATHS = {
    'next_gpa': f"{settings.MODEL_DIR}/{settings.NEXT_GPA_MODEL}",
    'final_cgpa': f"{settings.MODEL_DIR}/{settings.FINAL_CGPA_MODEL}",
    'graduation_class': f"{settings.MODEL_DIR}/{settings.GRADUATION_CLASS_MODEL}",
    'academic_risk': f"{settings.MODEL_DIR}/{settings.ACADEMIC_RISK_MODEL}",
}

_MODELS: Optional[Dict[str, Dict]] = None

def load_all_models() -> Dict[str, Dict]:
    models = {}
    for name, path in MODEL_PATHS.items():
        artifact = joblib.load(path)
        if artifact['feature_columns'] != FEATURE_COLUMNS:
            raise ValueError(
                f"Model {name}: feature column mismatch.\n"
                f"  Model expects: {artifact['feature_columns']}\n"
                f"  Current schema: {FEATURE_COLUMNS}\n"
                f"Retrain model or update schema consistently."
            )
        models[name] = artifact
    return models

def get_models() -> Dict[str, Dict]:
    global _MODELS
    if _MODELS is None:
        _MODELS = load_all_models()
    return _MODELS

def build_feature_vector(
    student_input: StudentInput,
    calculator_out: CalculatorOutput,
) -> pd.DataFrame:
    records = student_input.semester_records
    feature_dict = {
        "programme_duration_years": student_input.programme_duration_years,
        "current_level": student_input.current_level,
        "semester_number": calculator_out.semesters_completed,
        "semester_credits": records[-1].credits if records else 18,
        "semesters_completed": calculator_out.semesters_completed,
        "semesters_remaining": calculator_out.semesters_remaining,
        "cumulative_cgpa": calculator_out.current_cgpa or 0.0,
        "cumulative_credits": calculator_out.total_credits,
        "gpa_trend_slope": calculator_out.gpa_trend_slope,
        "gpa_volatility": calculator_out.gpa_volatility,
        "recent_gpa_avg_3": calculator_out.recent_gpa_avg_3,
        "credits_velocity": calculator_out.credits_velocity,
    }
    row = {col: feature_dict.get(col, 0.0) for col in FEATURE_COLUMNS}
    return pd.DataFrame([row], columns=FEATURE_COLUMNS)

def extract_top_features(model_artifact: Dict, top_k: int = 5) -> list[FeatureImportance]:
    importances = model_artifact.get('feature_importance', [])
    if not importances:
        return []
    sorted_imp = sorted(importances, key=lambda x: x['importance'], reverse=True)
    return [
        FeatureImportance(feature=item['feature'], importance=item['importance'])
        for item in sorted_imp[:top_k]
    ]

def _decode_label(model_artifact: Dict, X: pd.DataFrame) -> str:
    raw = model_artifact['model'].predict(X)[0]
    encoder = model_artifact.get('label_encoder')
    if encoder is not None and isinstance(raw, (int, np.integer)):
        return str(encoder.inverse_transform([raw])[0])
    return str(raw)

def run_predictor(
    student_input: StudentInput,
    calculator_out: CalculatorOutput,
    models: Optional[Dict[str, Dict]] = None,
) -> PredictorOutput:
    MODELS = models or get_models()
    X = build_feature_vector(student_input, calculator_out)

    pred_next_gpa = None
    if calculator_out.semesters_remaining > 0:
        pred_next_gpa = float(MODELS['next_gpa']['model'].predict(X)[0])
        pred_next_gpa = max(0.0, min(5.0, pred_next_gpa))

    pred_final_cgpa = float(MODELS['final_cgpa']['model'].predict(X)[0])
    pred_final_cgpa = max(0.0, min(5.0, pred_final_cgpa))

    pred_class = _decode_label(MODELS['graduation_class'], X)
    pred_risk = _decode_label(MODELS['academic_risk'], X)

    return PredictorOutput(
        predicted_next_gpa=pred_next_gpa,
        predicted_final_cgpa=round(pred_final_cgpa, 2),
        predicted_graduation_class=pred_class,
        predicted_academic_risk=pred_risk,
        top_features_next_gpa=extract_top_features(MODELS['next_gpa']),
        top_features_final_cgpa=extract_top_features(MODELS['final_cgpa']),
        top_features_graduation_class=extract_top_features(MODELS['graduation_class']),
        top_features_academic_risk=extract_top_features(MODELS['academic_risk']),
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_predictor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/predictor.py tests/test_predictor.py
git commit -m "feat: add predictor with lazy model loading and label decoding"
```

---

## Phase 5: Orchestrator, Graphs, Mock Advisor

### Task 8: `backend/orchestrator.py` — Pipeline Composition

**Files:**
- Create: `backend/orchestrator.py`
- Test: `tests/test_orchestrator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_orchestrator.py
import pytest
from backend.orchestrator import run_pipeline, build_semester_history
from backend.schemas import StudentInput, SemesterRecord, PipelineResult

class TestRunPipeline:
    def test_full_pipeline_returns_pipeline_result(self):
        student = StudentInput(
            student_name="Test Student",
            university="Test Uni", faculty="Science", department="Physics", course="Physics",
            programme_duration_years=5, current_level=300,
            semester_records=[
                SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2021/2022"),
                SemesterRecord(semester_number=2, gpa=3.8, credits=20, academic_session="2021/2022"),
                SemesterRecord(semester_number=3, gpa=4.0, credits=18, academic_session="2022/2023"),
                SemesterRecord(semester_number=4, gpa=4.1, credits=18, academic_session="2022/2023"),
            ],
            target_graduation_class="First Class",
        )
        result = run_pipeline(student)
        assert isinstance(result, PipelineResult)
        assert result.student_name == "Test Student"
        assert result.current_cgpa is not None
        assert result.feasibility is not None
        assert result.predicted_final_cgpa is not None
        assert len(result.semester_history) == 4

class TestBuildSemesterHistory:
    def test_reconstructs_cumulative_cgpa(self):
        student = StudentInput(
            student_name="T", university="U", faculty="F", department="D", course="C",
            programme_duration_years=4, current_level=200,
            semester_records=[
                SemesterRecord(semester_number=1, gpa=3.0, credits=20, academic_session="2022/2023"),
                SemesterRecord(semester_number=2, gpa=4.0, credits=20, academic_session="2022/2023"),
            ],
        )
        from backend.calculator import run_calculator
        calc = run_calculator(student)
        history = build_semester_history(student, calc)
        assert len(history) == 2
        # Sem 1: 3.0 * 20 / 20 = 3.0
        assert history[0].cumulative_cgpa == 3.0
        # Sem 2: (3*20 + 4*20) / 40 = 3.5
        assert history[1].cumulative_cgpa == 3.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_orchestrator.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# backend/orchestrator.py
from typing import List
from backend.schemas import StudentInput, PipelineResult, SemesterHistoryPoint
from backend.calculator import run_calculator
from backend.planner import run_planner
from backend.predictor import run_predictor
from backend.grading_rules import estimate_credits_for_semester

def build_semester_history(
    student_input: StudentInput,
    calculator_out,
) -> List[SemesterHistoryPoint]:
    history = []
    running_points = 0.0
    running_credits = 0
    for r in student_input.semester_records:
        running_points += r.gpa * r.credits
        running_credits += r.credits
        cum_cgpa = running_points / running_credits if running_credits > 0 else 0.0
        history.append(SemesterHistoryPoint(
            semester_number=r.semester_number,
            gpa=r.gpa,
            cumulative_cgpa=round(cum_cgpa, 2),
            credits=r.credits,
            academic_session=r.academic_session,
        ))
    return history

def run_pipeline(student_input: StudentInput) -> PipelineResult:
    calculator_out = run_calculator(student_input)
    planner_out = run_planner(student_input, calculator_out)
    predictor_out = run_predictor(student_input, calculator_out)
    history = build_semester_history(student_input, calculator_out)

    return PipelineResult(
        student_name=student_input.student_name,
        university=student_input.university,
        faculty=student_input.faculty,
        department=student_input.department,
        course=student_input.course,
        programme_duration_years=student_input.programme_duration_years,
        current_level=student_input.current_level,
        target_graduation_class=student_input.target_graduation_class,
        target_cgpa=student_input.target_cgpa,
        current_cgpa=calculator_out.current_cgpa,
        total_credits=calculator_out.total_credits,
        semesters_completed=calculator_out.semesters_completed,
        semesters_remaining=calculator_out.semesters_remaining,
        current_classification=calculator_out.current_classification,
        gpa_trend=calculator_out.gpa_trend,
        consistency_index=calculator_out.consistency_index,
        academic_health_score=calculator_out.academic_health_score,
        target_cgpa_resolved=planner_out.target_cgpa,
        feasibility=planner_out.feasibility,
        best_possible_classification=planner_out.best_possible_classification,
        semester_plan=planner_out.semester_plan,
        predicted_next_gpa=predictor_out.predicted_next_gpa,
        predicted_final_cgpa=predictor_out.predicted_final_cgpa,
        predicted_graduation_class=predictor_out.predicted_graduation_class,
        predicted_academic_risk=predictor_out.predicted_academic_risk,
        top_features_next_gpa=predictor_out.top_features_next_gpa,
        top_features_final_cgpa=predictor_out.top_features_final_cgpa,
        top_features_graduation_class=predictor_out.top_features_graduation_class,
        top_features_academic_risk=predictor_out.top_features_academic_risk,
        semester_history=history,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_orchestrator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/orchestrator.py tests/test_orchestrator.py
git commit -m "feat: add orchestrator pipeline composition"
```

---

### Task 9: `backend/graphs.py` — Plotly Figure Dicts

**Files:**
- Create: `backend/graphs.py`
- Test: `tests/test_graphs.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_graphs.py
import pytest
from backend.graphs import trajectory_chart, semester_planner_chart
from backend.schemas import PipelineResult, SemesterHistoryPoint, SemesterTarget, FeasibilityResult

def make_mock_pipeline():
    return PipelineResult(
        student_name="Test", university="U", faculty="F", department="D", course="C",
        programme_duration_years=5, current_level=300,
        target_graduation_class="First Class", target_cgpa=4.5,
        current_cgpa=3.9, total_credits=80, semesters_completed=4, semesters_remaining=6,
        current_classification="2:1", gpa_trend=None, consistency_index=20, academic_health_score=75,
        target_cgpa_resolved=4.5, feasibility=FeasibilityResult(
            goal_achievable=True, max_achievable_cgpa=4.6, required_average_gpa=4.7,
            realistic_classification="First Class", confidence=0.8, message="ok"
        ),
        best_possible_classification="First Class", semester_plan=[
            SemesterTarget(semester_number=5, target_gpa=4.7, cumulative_cgpa_if_met=4.1),
            SemesterTarget(semester_number=6, target_gpa=4.7, cumulative_cgpa_if_met=4.2),
        ],
        predicted_next_gpa=4.2, predicted_final_cgpa=4.4,
        predicted_graduation_class="First Class", predicted_academic_risk="Low",
        top_features_next_gpa=[], top_features_final_cgpa=[],
        top_features_graduation_class=[], top_features_academic_risk=[],
        semester_history=[
            SemesterHistoryPoint(semester_number=1, gpa=3.5, cumulative_cgpa=3.5, credits=20, academic_session="2021/2022"),
            SemesterHistoryPoint(semester_number=2, gpa=3.8, cumulative_cgpa=3.65, credits=20, academic_session="2021/2022"),
            SemesterHistoryPoint(semester_number=3, gpa=4.0, cumulative_cgpa=3.77, credits=18, academic_session="2022/2023"),
            SemesterHistoryPoint(semester_number=4, gpa=4.2, cumulative_cgpa=3.9, credits=18, academic_session="2022/2023"),
        ],
    )

class TestTrajectoryChart:
    def test_returns_dict_with_expected_keys(self):
        pipeline = make_mock_pipeline()
        fig_dict = trajectory_chart(pipeline)
        assert isinstance(fig_dict, dict)
        assert "data" in fig_dict
        assert "layout" in fig_dict

    def test_has_historical_trace(self):
        pipeline = make_mock_pipeline()
        fig_dict = trajectory_chart(pipeline)
        trace_names = [t.get("name", "") for t in fig_dict["data"]]
        assert "Actual CGPA" in trace_names

    def test_has_goal_line(self):
        pipeline = make_mock_pipeline()
        fig_dict = trajectory_chart(pipeline)
        shapes = fig_dict.get("layout", {}).get("shapes", [])
        assert any(s.get("type") == "line" and s.get("line", {}).get("dash") == "dash" for s in shapes)

class TestSemesterPlannerChart:
    def test_returns_dict(self):
        pipeline = make_mock_pipeline()
        fig_dict = semester_planner_chart(pipeline)
        assert isinstance(fig_dict, dict)
        assert "data" in fig_dict

    def test_has_both_traces(self):
        pipeline = make_mock_pipeline()
        fig_dict = semester_planner_chart(pipeline)
        trace_names = [t.get("name", "") for t in fig_dict["data"]]
        assert "Actual GPA" in trace_names
        assert "Target GPA" in trace_names
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graphs.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# backend/graphs.py
from typing import List
from backend.schemas import PipelineResult
import plotly.graph_objects as go

def trajectory_chart(pipeline: PipelineResult) -> dict:
    fig = go.Figure()

    # Historical trajectory
    semesters = [p.semester_number for p in pipeline.semester_history]
    cgpas = [p.cumulative_cgpa for p in pipeline.semester_history]
    fig.add_trace(go.Scatter(
        x=semesters, y=cgpas, mode='lines+markers',
        name='Actual CGPA', line=dict(color='blue')
    ))

    # Predicted next semester
    if pipeline.predicted_next_gpa is not None and pipeline.semesters_remaining > 0:
        next_sem = pipeline.semesters_completed + 1
        next_credits = estimate_credits_for_semester(pipeline.current_level)
        pred_cum_cgpa = (
            (pipeline.current_cgpa * pipeline.total_credits + pipeline.predicted_next_gpa * next_credits) / 
            (pipeline.total_credits + next_credits)
        ) if pipeline.current_cgpa is not None else pipeline.predicted_next_gpa

        fig.add_trace(go.Scatter(
            x=[next_sem], y=[pred_cum_cgpa], mode='markers',
            name='Predicted Next', marker=dict(color='orange', size=10, symbol='diamond')
        ))

    # Goal line
    fig.add_hline(
        y=pipeline.target_cgpa_resolved, line_dash='dash', line_color='green',
        annotation_text=f"Target: {pipeline.target_cgpa_resolved:.2f}"
    )

    # First Class threshold
    fig.add_hline(
        y=4.50, line_dash='dot', line_color='gray',
        annotation_text="First Class threshold"
    )

    fig.update_layout(
        title="Academic Trajectory",
        xaxis_title="Semester",
        yaxis_title="Cumulative CGPA",
        yaxis_range=[0, 5.0],
        template="plotly_white",
        showlegend=True,
    )

    return fig.to_dict()

def semester_planner_chart(pipeline: PipelineResult) -> dict:
    fig = go.Figure()

    # Historical GPAs
    sem_hist = [p.semester_number for p in pipeline.semester_history]
    gpa_hist = [p.gpa for p in pipeline.semester_history]
    fig.add_trace(go.Bar(x=sem_hist, y=gpa_hist, name='Actual GPA', marker_color='blue'))

    # Target GPAs
    sem_targets = [p.semester_number for p in pipeline.semester_plan]
    gpa_targets = [p.target_gpa for p in pipeline.semester_plan]
    fig.add_trace(go.Bar(x=sem_targets, y=gpa_targets, name='Target GPA', marker_color='green', opacity=0.7))

    fig.update_layout(
        title="Semester GPA Plan",
        xaxis_title="Semester",
        yaxis_title="GPA",
        yaxis_range=[0, 5.0],
        barmode='group',
        template="plotly_white",
    )
    return fig.to_dict()

def what_if_simulator(pipeline: PipelineResult, what_if_gpas: List[float]) -> dict:
    """What-if: replace future semester GPAs with user values, recompute trajectory."""
    # Build modified semester history
    history = pipeline.semester_history.copy()
    current_cgpa = pipeline.current_cgpa or 0.0
    current_credits = pipeline.total_credits

    for i, what_if_gpa in enumerate(what_if_gpas):
        sem_num = pipeline.semesters_completed + i + 1
        next_credits = estimate_credits_for_semester(pipeline.current_level)
        current_credits += next_credits
        current_cgpa = (
            (current_cgpa * (current_credits - next_credits) + what_if_gpa * next_credits) / 
            current_credits
        )
        history.append(type(pipeline.semester_history[0])(
            semester_number=sem_num,
            gpa=what_if_gpa,
            cumulative_cgpa=round(current_cgpa, 2),
            credits=next_credits,
            academic_session=f"Projected {sem_num}"
        ))

    # Create modified pipeline for charting
    from copy import deepcopy
    mod_pipeline = deepcopy(pipeline)
    mod_pipeline.semester_history = history
    mod_pipeline.semesters_remaining = 0
    mod_pipeline.predicted_next_gpa = None

    return trajectory_chart(mod_pipeline)

# Import at bottom to avoid circular
from backend.grading_rules import estimate_credits_for_semester
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graphs.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/graphs.py tests/test_graphs.py
git commit -m "feat: add visualization data prep with Plotly figure dicts"
```

---

### Task 10: `backend/mock_advisor.py` — Contract Validation

**Files:**
- Create: `backend/mock_advisor.py`
- Test: `tests/test_mock_advisor.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_mock_advisor.py
import pytest
from backend.mock_advisor import mock_advisor, test_advisor_contract
from backend.schemas import AdvisorInput, SemesterTarget, FeatureImportance

def test_mock_advisor_returns_string():
    advisor_in = AdvisorInput(
        student_name="Test", course="Physics", current_cgpa=4.0,
        target_graduation_class="First Class", target_cgpa=4.5,
        remaining_semesters=4, required_average_gpa=4.7,
        predicted_final_cgpa=4.3, predicted_graduation_class="First Class",
        academic_risk="Low", goal_feasible=True,
        best_possible_classification="First Class",
        academic_health_score=85, gpa_trend=None, consistency_index=20,
        semester_plan=[], top_features_final_cgpa=[], top_features_graduation_class=[],
        top_features_academic_risk=[],
    )
    result = mock_advisor(advisor_in)
    assert isinstance(result, str)
    assert "Test" in result
    assert "4.0" in result
    assert "4.3" in result

def test_mock_advisor_validates_required_fields():
    # Missing student_name should raise
    with pytest.raises(AssertionError):
        mock_advisor(AdvisorInput(
            student_name="", course="C", current_cgpa=None,
            target_graduation_class=None, target_cgpa=None,
            remaining_semesters=0, required_average_gpa=None,
            predicted_final_cgpa=0, predicted_graduation_class="Pass",
            academic_risk="Low", goal_feasible=False,
            best_possible_classification="Pass",
            academic_health_score=0, gpa_trend=None, consistency_index=0,
            semester_plan=[], top_features_final_cgpa=[], top_features_graduation_class=[],
            top_features_academic_risk=[],
        ))

def test_test_advisor_contract_e2e():
    from backend.orchestrator import run_pipeline
    from backend.schemas import StudentInput, SemesterRecord
    student = StudentInput(
        student_name="Test Student", university="U", faculty="F", department="D", course="C",
        programme_duration_years=4, current_level=200,
        semester_records=[
            SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2022/2023"),
            SemesterRecord(semester_number=2, gpa=4.0, credits=20, academic_session="2022/2023"),
        ],
        target_graduation_class="First Class",
    )
    result = test_advisor_contract(student)
    assert isinstance(result, str)
    assert "Test Student" in result
    assert "[MOCK ADVISOR" in result
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_mock_advisor.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# backend/mock_advisor.py
from backend.schemas import AdvisorInput, PipelineResult
from backend.orchestrator import pipeline_to_advisor_input, run_pipeline

def mock_advisor(advisor_input: AdvisorInput) -> str:
    """Validates AdvisorInput contract with a deterministic mock response."""
    assert advisor_input.student_name
    assert advisor_input.predicted_final_cgpa >= 0.0
    assert advisor_input.academic_risk in ["Low", "Medium", "High"]
    assert advisor_input.goal_feasible in [True, False]
    assert isinstance(advisor_input.semester_plan, list)

    return (
        f"Hello {advisor_input.student_name}! "
        f"Your current CGPA is {advisor_input.current_cgpa or 'N/A'}. "
        f"Predicted final CGPA: {advisor_input.predicted_final_cgpa:.2f}. "
        f"Goal feasible: {advisor_input.goal_feasible}. "
        f"[MOCK ADVISOR — replace with LLM in Phase 5]"
    )

def test_advisor_contract(student_input) -> str:
    """End-to-end test: pipeline → advisor_input → mock_advisor."""
    pipeline = run_pipeline(student_input)
    advisor_input = pipeline_to_advisor_input(pipeline)
    return mock_advisor(advisor_input)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_mock_advisor.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/mock_advisor.py tests/test_mock_advisor.py
git commit -m "feat: add mock advisor for contract validation"
```

---

## Phase 6: Data Generation (Phase 2 of Project)

### Task 11: `data_generation/structural_layer.py` — Deterministic Validity

**Files:**
- Create: `data_generation/structural_layer.py`
- Test: `tests/test_structural_layer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_structural_layer.py
import pytest
from data_generation.structural_layer import (
    build_static_attributes, build_semester_skeleton, level_for_semester
)
from backend.grading_rules import CREDITS_PER_LEVEL

class TestBuildStaticAttributes:
    def test_returns_dict_with_all_fields(self):
        attrs = build_static_attributes(0, 5)
        assert "student_id" in attrs
        assert "university" in attrs
        assert "faculty" in attrs
        assert "department" in attrs
        assert "course" in attrs
        assert "gpa_scale" in attrs
        assert attrs["gpa_scale"] == 5.0
        assert attrs["programme_duration_years"] == 5

class TestBuildSemesterSkeleton:
    def test_correct_number_of_semesters(self):
        attrs = build_static_attributes(0, 5)
        rows = build_semester_skeleton(attrs, 0)
        assert len(rows) == 10  # 5 years * 2

    def test_level_progression_5yr(self):
        attrs = build_static_attributes(0, 5)
        rows = build_semester_skeleton(attrs, 0)
        expected = [100, 100, 200, 200, 300, 300, 400, 400, 500, 500]
        actual = [r["current_level"] for r in rows]
        assert actual == expected

    def test_level_progression_6yr(self):
        attrs = build_static_attributes(0, 6)
        rows = build_semester_skeleton(attrs, 0)
        expected = [100, 100, 200, 200, 300, 300, 400, 400, 500, 500, 500, 500]
        actual = [r["current_level"] for r in rows]
        assert actual == expected

    def test_credits_within_bounds(self):
        attrs = build_static_attributes(0, 5)
        rows = build_semester_skeleton(attrs, 0)
        for r in rows:
            assert 12 <= r["semester_credits"] <= 24

    def test_session_format(self):
        attrs = build_static_attributes(0, 5)
        rows = build_semester_skeleton(attrs, 0)
        for r in rows:
            assert "/" in r["academic_session"]
            year = int(r["academic_session"].split("/")[0])
            assert 2020 <= year <= 2030
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_structural_layer.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# data_generation/structural_layer.py
import random
from typing import Dict, List, Any
from backend.grading_rules import level_for_semester, CREDITS_PER_LEVEL

UNIVERSITIES = ["University of Lagos", "University of Ibadan", "Ahmadu Bello University", "Obafemi Awolowo University"]
FACULTIES = ["Science", "Engineering", "Arts", "Social Sciences", "Medical Sciences"]
DEPARTMENTS = {
    "Science": ["Physics", "Chemistry", "Mathematics", "Biology", "Computer Science"],
    "Engineering": ["Electrical", "Mechanical", "Civil", "Chemical", "Computer"],
    "Arts": ["English", "History", "Philosophy", "Linguistics", "Theatre Arts"],
    "Social Sciences": ["Economics", "Political Science", "Sociology", "Psychology", "Geography"],
    "Medical Sciences": ["Medicine", "Nursing", "Pharmacy", "Physiotherapy", "Public Health"],
}
COURSES = {
    "Physics": "Physics with Electronics", "Chemistry": "Industrial Chemistry",
    "Mathematics": "Mathematics", "Biology": "Cell Biology", "Computer Science": "Computer Science",
    "Electrical": "Electrical Engineering", "Mechanical": "Mechanical Engineering",
    "Civil": "Civil Engineering", "Chemical": "Chemical Engineering", "Computer": "Computer Engineering",
    "English": "English Language", "History": "History", "Philosophy": "Philosophy",
    "Linguistics": "Linguistics", "Theatre Arts": "Theatre Arts",
    "Economics": "Economics", "Political Science": "Political Science",
    "Sociology": "Sociology", "Psychology": "Psychology", "Geography": "Geography",
    "Medicine": "Medicine and Surgery", "Nursing": "Nursing Science",
    "Pharmacy": "Pharmacy", "Physiotherapy": "Physiotherapy", "Public Health": "Public Health",
}

def build_static_attributes(student_idx: int, programme_duration_years: int) -> Dict[str, Any]:
    uni = random.choice(UNIVERSITIES)
    faculty = random.choice(FACULTIES)
    dept = random.choice(DEPARTMENTS[faculty])
    course = COURSES.get(dept, dept)
    
    return {
        "student_id": f"STU_{student_idx:04d}",
        "university": uni,
        "faculty": faculty,
        "department": dept,
        "course": course,
        "gpa_scale": 5.0,
        "programme_duration_years": programme_duration_years,
    }

def build_semester_skeleton(static_attrs: Dict[str, Any], student_idx: int) -> List[Dict[str, Any]]:
    total_semesters = static_attrs["programme_duration_years"] * 2
    rows = []
    base_year = 2020 + (student_idx % 5)
    
    for sem_num in range(1, total_semesters + 1):
        level = level_for_semester(sem_num, total_semesters)
        credits = random.randint(CREDITS_PER_LEVEL[level] - 2, CREDITS_PER_LEVEL[level] + 2)
        credits = max(12, min(24, credits))
        
        year = base_year + (sem_num - 1) // 2
        session = f"{year}/{year+1}"
        
        rows.append({
            "student_id": static_attrs["student_id"],
            "university": static_attrs["university"],
            "faculty": static_attrs["faculty"],
            "department": static_attrs["department"],
            "course": static_attrs["course"],
            "gpa_scale": static_attrs["gpa_scale"],
            "programme_duration_years": static_attrs["programme_duration_years"],
            "current_level": level,
            "semester_number": sem_num,
            "academic_session": session,
            "semester_credits": credits,
            # semester_gpa filled by trajectory_noise layer
            # cumulative fields computed after full trajectory
        })
    return rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_structural_layer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data_generation/structural_layer.py tests/test_structural_layer.py
git commit -m "feat: add structural layer for synthetic data generation"
```

---

### Task 12: `data_generation/trajectory_noise.py` — Statistical Realism

**Files:**
- Create: `data_generation/trajectory_noise.py`
- Test: `tests/test_trajectory_noise.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_trajectory_noise.py
import pytest
from data_generation.trajectory_noise import (
    TrajectoryProfile, sample_trajectory_profile, apply_trajectory_noise
)
from backend.grading_rules import CREDITS_PER_LEVEL

class TestSampleTrajectoryProfile:
    def test_base_ability_in_range(self):
        for _ in range(100):
            profile = sample_trajectory_profile()
            assert 2.0 <= profile.base_ability <= 4.8

    def test_volatility_positive(self):
        for _ in range(100):
            profile = sample_trajectory_profile()
            assert 0.1 <= profile.volatility <= 0.6

    def test_trend_in_range(self):
        for _ in range(100):
            profile = sample_trajectory_profile()
            assert -0.15 <= profile.trend <= 0.15

    def test_shock_params(self):
        for _ in range(100):
            profile = sample_trajectory_profile()
            assert 0.0 <= profile.shock_probability <= 0.2
            assert 0.0 <= profile.shock_magnitude <= 2.0

class TestApplyTrajectoryNoise:
    def test_preserves_row_count(self):
        skeleton = [{"semester_number": i, "semester_credits": 20} for i in range(1, 5)]
        profile = TrajectoryProfile(base_ability=3.5, volatility=0.2, trend=0.05, shock_probability=0.0, shock_magnitude=0.0)
        result = apply_trajectory_noise(skeleton, profile)
        assert len(result) == 4

    def test_gpa_bounds(self):
        skeleton = [{"semester_number": i, "semester_credits": 20} for i in range(1, 11)]
        profile = TrajectoryProfile(base_ability=3.5, volatility=0.2, trend=0.05, shock_probability=0.0, shock_magnitude=0.0)
        result = apply_trajectory_noise(skeleton, profile)
        for r in result:
            assert 0.0 <= r["semester_gpa"] <= 5.0

    def test_cumulative_fields_computed(self):
        skeleton = [{"semester_number": i, "semester_credits": 20} for i in range(1, 5)]
        profile = TrajectoryProfile(base_ability=3.5, volatility=0.2, trend=0.05, shock_probability=0.0, shock_magnitude=0.0)
        result = apply_trajectory_noise(skeleton, profile)
        for r in result:
            assert "cumulative_cgpa" in r
            assert "cumulative_credits" in r
            assert "semesters_completed" in r
            assert "semesters_remaining" in r
            assert "is_final_semester" in r
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_trajectory_noise.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# data_generation/trajectory_noise.py
import random
import numpy as np
from typing import List, Dict, Any
from dataclasses import dataclass
from scipy.stats import truncnorm
from backend.trajectory_features import compute_trajectory_features

@dataclass
class TrajectoryProfile:
    base_ability: float      # Mean GPA this student gravitates toward (2.0–4.8)
    volatility: float        # Std dev of semester-to-semester noise (0.15–0.6)
    trend: float             # Slope per semester (-0.15 to +0.15)
    shock_probability: float # Chance of one bad semester (0.05–0.15)
    shock_magnitude: float   # GPA drop if shock occurs (0.5–1.5)

def sample_base_ability() -> float:
    component = random.choices(
        population=['avg', 'high', 'low'],
        weights=[0.60, 0.30, 0.10],
        k=1
    )[0]
    if component == 'avg':
        a, b = (2.0 - 3.5) / 0.5, (4.8 - 3.5) / 0.5
        return float(truncnorm.rvs(a, b, loc=3.5, scale=0.5))
    elif component == 'high':
        a, b = (2.0 - 4.2) / 0.3, (4.8 - 4.2) / 0.3
        return float(truncnorm.rvs(a, b, loc=4.2, scale=0.3))
    else:
        a, b = (2.0 - 2.0) / 0.4, (4.8 - 2.0) / 0.4
        return float(truncnorm.rvs(a, b, loc=2.0, scale=0.4))

def sample_trajectory_profile() -> TrajectoryProfile:
    return TrajectoryProfile(
        base_ability=sample_base_ability(),
        volatility=float(np.random.lognormal(mean=-1.5, sigma=0.3)),
        trend=float(np.random.normal(0.0, 0.05)),
        shock_probability=float(np.random.beta(2, 20)),
        shock_magnitude=float(np.random.uniform(0.5, 1.5)),
    )

def apply_trajectory_noise(skeleton: List[Dict[str, Any]], profile: TrajectoryProfile) -> List[Dict[str, Any]]:
    result = []
    gpas_so_far = []
    credits_so_far = 0
    
    for i, row in enumerate(skeleton):
        sem_num = row["semester_number"]
        
        # Generate GPA for this semester
        raw_gpa = profile.base_ability + profile.trend * sem_num + np.random.normal(0, profile.volatility)
        if random.random() < profile.shock_probability:
            raw_gpa -= profile.shock_magnitude
        semester_gpa = max(0.0, min(5.0, raw_gpa))
        
        gpas_so_far.append(semester_gpa)
        credits_so_far += row["semester_credits"]
        
        # Compute engineered features
        features = compute_trajectory_features(gpas_so_far, credits_so_far)
        
        # Build complete row
        complete_row = row.copy()
        complete_row.update({
            "semester_gpa": round(semester_gpa, 2),
            "cumulative_cgpa": round(
                sum(g * row["semester_credits"] for g, row in zip(gpas_so_far, skeleton[:i+1])) / credits_so_far, 2
            ),
            "cumulative_credits": credits_so_far,
            "semesters_completed": sem_num,
            "semesters_remaining": len(skeleton) - sem_num,
            "is_final_semester": (sem_num == len(skeleton)),
            "gpa_trend_slope": features["gpa_trend_slope"],
            "gpa_volatility": features["gpa_volatility"],
            "recent_gpa_avg_3": features["recent_gpa_avg_3"],
            "credits_velocity": features["credits_velocity"],
        })
        result.append(complete_row)
    
    # Compute target labels from full trajectory
    final_cgpa = result[-1]["cumulative_cgpa"]
    from backend.grading_rules import classify_cgpa
    graduation_class = classify_cgpa(final_cgpa)
    
    for i, row in enumerate(result):
        if i < len(result) - 1:
            row["next_semester_gpa"] = result[i + 1]["semester_gpa"]
        else:
            row["next_semester_gpa"] = None
        row["final_cgpa"] = final_cgpa
        row["graduation_class"] = graduation_class
        
        # Academic risk label (heuristic)
        cum_cgpa = row["cumulative_cgpa"]
        latest_gpa = row["semester_gpa"]
        if cum_cgpa < 2.0 or latest_gpa < 1.5:
            row["academic_risk"] = "High"
        elif cum_cgpa < 3.0 or latest_gpa < 2.5:
            row["academic_risk"] = "Medium"
        else:
            row["academic_risk"] = "Low"
    
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_trajectory_noise.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data_generation/trajectory_noise.py tests/test_trajectory_noise.py
git commit -m "feat: add trajectory noise layer for synthetic data"
```

---

### Task 13: `data_generation/generator.py` — Orchestrates Generation

**Files:**
- Create: `data_generation/generator.py`
- Test: `tests/test_generator.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_generator.py
import pytest
from data_generation.generator import generate_dataset
from backend.schemas import DatasetRow

class TestGenerateDataset:
    def test_returns_list_of_dataset_rows(self):
        rows = generate_dataset(n_students=10, programme_durations=[4], seed=42)
        assert len(rows) > 0
        assert all(isinstance(r, DatasetRow) for r in rows)

    def test_student_count_approximately_correct(self):
        rows = generate_dataset(n_students=100, programme_durations=[4, 5], seed=42)
        unique_students = len(set(r.student_id for r in rows))
        assert 90 <= unique_students <= 110  # Some variation

    def test_rows_have_all_required_fields(self):
        rows = generate_dataset(n_students=5, programme_durations=[4], seed=42)
        for r in rows:
            assert r.student_id
            assert r.gpa_scale == 5.0
            assert 0.0 <= r.semester_gpa <= 5.0
            assert 12 <= r.semester_credits <= 24
            assert r.cumulative_cgpa >= 0.0
            assert r.graduation_class in ["First Class", "Second Class Upper", "Second Class Lower", "Third Class", "Pass", "Fail"]
            assert r.academic_risk in ["Low", "Medium", "High"]

    def test_final_semester_has_null_next_gpa(self):
        rows = generate_dataset(n_students=10, programme_durations=[4], seed=42)
        final_rows = [r for r in rows if r.is_final_semester]
        for r in final_rows:
            assert r.next_semester_gpa is None
            assert r.final_cgpa is not None
            assert r.graduation_class is not None
            assert r.academic_risk is not None

    def test_final_cgpa_consistent_per_student(self):
        rows = generate_dataset(n_students=10, programme_durations=[4], seed=42)
        by_student = {}
        for r in rows:
            by_student.setdefault(r.student_id, []).append(r.final_cgpa)
        for student_id, cgpas in by_student.items():
            assert len(set(cgpas)) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_generator.py -v`
Expected: FAIL

- [ ] **Step 3: Write implementation**

```python
# data_generation/generator.py
from typing import List
from data_generation.structural_layer import build_static_attributes, build_semester_skeleton
from data_generation.trajectory_noise import sample_trajectory_profile, apply_trajectory_noise
from backend.schemas import DatasetRow

def generate_dataset(
    n_students: int = 10000,
    programme_durations: List[int] = [4, 5, 6],
    seed: int = 42,
) -> List[DatasetRow]:
    import random
    random.seed(seed)
    
    all_rows = []
    
    for student_idx in range(n_students):
        duration = random.choice(programme_durations)
        static_attrs = build_static_attributes(student_idx, duration)
        skeleton = build_semester_skeleton(static_attrs, student_idx)
        profile = sample_trajectory_profile()
        complete_rows = apply_trajectory_noise(skeleton, profile)
        
        for row_dict in complete_rows:
            all_rows.append(DatasetRow(**row_dict))
    
    return all_rows
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_generator.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add data_generation/generator.py tests/test_generator.py
git commit -m "feat: add generator orchestrating structural and noise layers"
```

---

### Task 14: `notebooks/EDA.ipynb` — Exploratory Data Analysis

**Files:**
- Create: `notebooks/EDA.ipynb`

- [ ] **Step 1: Create notebook with EDA cells**

```json
{
  "cells": [
    {"cell_type": "markdown", "metadata": {}, "source": ["# APIS Synthetic Dataset EDA\n\nExploratory analysis of generated synthetic student data."]},
    {"cell_type": "code", "metadata": {}, "source": ["import pandas as pd\nimport numpy as np\nimport matplotlib.pyplot as plt\nimport seaborn as sns\n\nfrom data_generation.generator import generate_dataset\nfrom backend.schemas import FEATURE_COLUMNS, TARGET_NEXT_GPA, TARGET_FINAL_CGPA, TARGET_GRADUATION_CLASS, TARGET_ACADEMIC_RISK"]},
    {"cell_type": "code", "metadata": {}, "source": ["rows = generate_dataset(n_students=5000, programme_durations=[4,5,6], seed=42)\ndf = pd.DataFrame([r.model_dump() for r in rows])\nprint(f\"Shape: {df.shape}\")\nprint(f\"Students: {df['student_id'].nunique()}\")\ndf.head()"]},
    {"cell_type": "code", "metadata": {}, "source": ["df.describe(include='all')"]},
    {"cell_type": "code", "metadata": {}, "source": ["# Target distributions\nfig, axes = plt.subplots(2, 2, figsize=(12, 10))\ndf_final = df[df['is_final_semester']]\n\naxes[0,0].hist(df_final[TARGET_FINAL_CGPA].dropna(), bins=30, edgecolor='black')\naxes[0,0].set_title('Final CGPA Distribution')\n\ndf_final[TARGET_GRADUATION_CLASS].value_counts().plot(kind='bar', ax=axes[0,1])\naxes[0,1].set_title('Graduation Class Distribution')\n\ndf_final[TARGET_ACADEMIC_RISK].value_counts().plot(kind='bar', ax=axes[1,0])\naxes[1,0].set_title('Academic Risk Distribution')\n\ndf[df['semester_number'] < df.groupby('student_id')['semester_number'].transform('max')][TARGET_NEXT_GPA].hist(bins=30, ax=axes[1,1], edgecolor='black')\naxes[1,1].set_title('Next Semester GPA Distribution')\nplt.tight_layout()\nplt.show()"]},
    {"cell_type": "code", "metadata": {}, "source": ["# Feature correlations with targets\nfeature_cols = FEATURE_COLUMNS\ntargets = [TARGET_NEXT_GPA, TARGET_FINAL_CGPA, TARGET_GRADUATION_CLASS, TARGET_ACADEMIC_RISK]\n\n# Encode categorical targets for correlation\ndf_enc = df.copy()\ndf_enc[TARGET_GRADUATION_CLASS] = pd.Categorical(df_enc[TARGET_GRADUATION_CLASS], \n    categories=['Fail', 'Pass', 'Third Class', 'Second Class Lower', 'Second Class Upper', 'First Class']).codes\ndf_enc[TARGET_ACADEMIC_RISK] = pd.Categorical(df_enc[TARGET_ACADEMIC_RISK], \n    categories=['Low', 'Medium', 'High']).codes\n\ncorr_matrix = df_enc[feature_cols + targets].corr()\nplt.figure(figsize=(14, 10))\nsns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r', center=0)\nplt.title('Feature-Target Correlations')\nplt.show()"]},
    {"cell_type": "code", "metadata": {}, "source": ["# Class balance check\nprint('Graduation Class Balance:')\nprint(df_final[TARGET_GRADUATION_CLASS].value_counts(normalize=True))\nprint()\nprint('Academic Risk Balance:')\nprint(df_final[TARGET_ACADEMIC_RISK].value_counts(normalize=True))"]},
    {"cell_type": "code", "metadata": {}, "source": ["# Trajectory patterns\n# Average GPA trajectory by graduation class\ntraj = df.groupby(['student_id', 'semester_number'])['semester_gpa'].mean().reset_index()\ntraj = traj.merge(df_final[['student_id', TARGET_GRADUATION_CLASS']], on='student_id')\n\nfor cls in sorted(df_final[TARGET_GRADUATION_CLASS].unique()):\n    subset = traj[traj[TARGET_GRADUATION_CLASS] == cls]\n    avg_traj = subset.groupby('semester_number')['semester_gpa'].mean()\n    plt.plot(avg_traj.index, avg_traj.values, label=cls, marker='o')\nplt.xlabel('Semester')\nplt.ylabel('Average Semester GPA')\nplt.title('Average GPA Trajectory by Graduation Class')\nplt.legend()\nplt.grid(True, alpha=0.3)\nplt.show()"]},
  ],
  "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
  "nbformat": 4, "nbformat_minor": 4
}
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/EDA.ipynb
git commit -m "feat: add EDA notebook for synthetic data exploration"
```

---

### Task 15: `notebooks/Model_Training.ipynb` — Full Training Pipeline

**Files:**
- Create: `notebooks/Model_Training.ipynb`

- [ ] **Step 1: Create notebook with training pipeline**

```json
{
  "cells": [
    {"cell_type": "markdown", "metadata": {}, "source": ["# APIS Model Training Pipeline\n\nTrains 4 models: Next Semester GPA (reg), Final CGPA (reg), Graduation Class (clf), Academic Risk (clf)."]},
    {"cell_type": "code", "metadata": {}, "source": ["import pandas as pd\nimport numpy as np\nimport joblib\nimport json\nfrom datetime import datetime\n\nfrom data_generation.generator import generate_dataset\nfrom backend.schemas import (\n    FEATURE_COLUMNS, TARGET_NEXT_GPA, TARGET_FINAL_CGPA,\n    TARGET_GRADUATION_CLASS, TARGET_ACADEMIC_RISK,\n    GRADUATION_CLASSES, ACADEMIC_RISK_CLASSES\n)\n\nfrom sklearn.model_selection import train_test_split, GroupKFold, RandomizedSearchCV, GridSearchCV\nfrom sklearn.linear_model import LinearRegression, Ridge, LogisticRegression\nfrom sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor\nfrom sklearn.metrics import (\n    mean_absolute_error, mean_squared_error, r2_score,\n    accuracy_score, f1_score, confusion_matrix, classification_report\n)\nfrom sklearn.preprocessing import LabelEncoder\nimport xgboost as xgb\nimport catboost as cb"]},
    {"cell_type": "code", "metadata": {}, "source": ["# Generate / load dataset\nrows = generate_dataset(n_students=15000, programme_durations=[4,5,6], seed=42)\ndf = pd.DataFrame([r.model_dump() for r in rows])\nprint(f\"Dataset shape: {df.shape}\")"]},
    {"cell_type": "code", "metadata": {}, "source": ["# Split BY STUDENT (no leakage)\nstudent_ids = df['student_id'].unique()\ntrain_ids, temp_ids = train_test_split(student_ids, test_size=0.30, random_state=42)\nval_ids, test_ids = train_test_split(temp_ids, test_size=0.50, random_state=42)\n\ntrain_df = df[df['student_id'].isin(train_ids)]\nval_df = df[df['student_id'].isin(val_ids)]\ntest_df = df[df['student_id'].isin(test_ids)]\n\nprint(f\"Train: {len(train_ids)} students, {len(train_df)} rows\")\nprint(f\"Val: {len(val_ids)} students, {len(val_df)} rows\")\nprint(f\"Test: {len(test_ids)} students, {len(test_df)} rows\")"]},
    {"cell_type": "code", "metadata": {}, "source": ["# Prepare X/y for each target\ndef prepare_xy(dataframe, target_col):\n    y = dataframe[target_col].dropna()\n    X = dataframe.loc[y.index, FEATURE_COLUMNS]\n    return X, y\n\nX1_train, y1_train = prepare_xy(train_df, TARGET_NEXT_GPA)\nX1_val, y1_val = prepare_xy(val_df, TARGET_NEXT_GPA)\nX1_test, y1_test = prepare_xy(test_df, TARGET_NEXT_GPA)\n\nX2_train, y2_train = prepare_xy(train_df, TARGET_FINAL_CGPA)\nX2_val, y2_val = prepare_xy(val_df, TARGET_FINAL_CGPA)\nX2_test, y2_test = prepare_xy(test_df, TARGET_FINAL_CGPA)\n\nX3_train, y3_train = prepare_xy(train_df, TARGET_GRADUATION_CLASS)\nX3_val, y3_val = prepare_xy(val_df, TARGET_GRADUATION_CLASS)\nX3_test, y3_test = prepare_xy(test_df, TARGET_GRADUATION_CLASS)\n\nX4_train, y4_train = prepare_xy(train_df, TARGET_ACADEMIC_RISK)\nX4_val, y4_val = prepare_xy(val_df, TARGET_ACADEMIC_RISK)\nX4_test, y4_test = prepare_xy(test_df, TARGET_ACADEMIC_RISK)\n\n# Encode classification targets\nle_class = LabelEncoder()\nle_class.fit(GRADUATION_CLASSES)\ny3_train_enc = le_class.transform(y3_train)\ny3_val_enc = le_class.transform(y3_val)\ny3_test_enc = le_class.transform(y3_test)\n\nle_risk = LabelEncoder()\nle_risk.fit(ACADEMIC_RISK_CLASSES)\ny4_train_enc = le_risk.transform(y4_train)\ny4_val_enc = le_risk.transform(y4_val)\ny4_test_enc = le_risk.transform(y4_test)"]},
    {"cell_type": "code", "metadata": {}, "source": ["# Training function with CV\ndef train_with_cv(model_class, param_dist, X, y, cv_groups, task_type='regression', n_iter=50):\n    cv = GroupKFold(n_splits=3)\n    scoring = 'neg_mean_absolute_error' if task_type == 'regression' else 'f1_macro'\n    \n    search = RandomizedSearchCV(\n        model_class(), param_dist, n_iter=n_iter,\n        cv=cv, scoring=scoring, n_jobs=-1, random_state=42, verbose=1\n    )\n    search.fit(X, y, groups=cv_groups)\n    return search.best_estimator_, search.best_params_, search.best_score_\n\n# Model definitions\nreg_models = {\n    'LinearRegression': (LinearRegression, {}),\n    'Ridge': (Ridge, {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}),\n    'RandomForest': (RandomForestRegressor, {'n_estimators': [200, 500], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5]}),\n    'GradientBoosting': (GradientBoostingRegressor, {'n_estimators': [200, 500], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5]}),\n    'XGBoost': (xgb.XGBRegressor, {'n_estimators': [200, 500], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7], 'subsample': [0.8, 1.0]}),\n    'CatBoost': (cb.CatBoostRegressor, {'iterations': [200, 500], 'learning_rate': [0.01, 0.05, 0.1], 'depth': [4, 6, 8], 'verbose': [False]}),\n}\n\nclf_models = {\n    'LogisticRegression': (LogisticRegression, {'C': [0.01, 0.1, 1.0, 10.0], 'max_iter': [1000], 'class_weight': ['balanced']}),\n    'RandomForest': (RandomForestClassifier, {'n_estimators': [200, 500], 'max_depth': [5, 10, None], 'min_samples_split': [2, 5], 'class_weight': ['balanced']}),\n    'XGBoost': (xgb.XGBClassifier, {'n_estimators': [200, 500], 'learning_rate': [0.01, 0.05, 0.1], 'max_depth': [3, 5, 7]}),\n    'CatBoost': (cb.CatBoostClassifier, {'iterations': [200, 500], 'learning_rate': [0.01, 0.05, 0.1], 'depth': [4, 6, 8], 'verbose': [False]}),\n}"]},
    {"cell_type": "code", "metadata": {}, "source": ["# Train all models\ndef train_task(name, models_dict, X_train, y_train, X_val, y_val, cv_groups, task_type, label_encoder=None):\n    print(f\"\\n=== {name} ===\")\n    best_model = None\n    best_val_score = -np.inf if task_type == 'classification' else np.inf\n    best_artifact = None\n    \n    for model_name, (model_class, param_dist) in models_dict.items():\n        print(f\"  Training {model_name}...\")\n        try:\n            model, best_params, cv_score = train_with_cv(\n                model_class, param_dist, X_train, y_train, cv_groups, task_type\n            )\n            \n            # Evaluate on validation\n            if task_type == 'regression':\n                val_pred = model.predict(X_val)\n                val_mae = mean_absolute_error(y_val, val_pred)\n                val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))\n                val_r2 = r2_score(y_val, val_pred)\n                print(f\"    CV MAE: {-cv_score:.4f}, Val MAE: {val_mae:.4f}, RMSE: {val_rmse:.4f}, R²: {val_r2:.4f}\")\n                score = val_mae\n                better = score < best_val_score\n            else:\n                val_pred = model.predict(X_val)\n                val_acc = accuracy_score(y_val, val_pred)\n                val_f1 = f1_score(y_val, val_pred, average='macro')\n                print(f\"    CV F1: {cv_score:.4f}, Val Acc: {val_acc:.4f}, Macro F1: {val_f1:.4f}\")\n                score = val_f1\n                better = score > best_val_score\n            \n            if better:\n                best_val_score = score\n                best_model = model\n                \n                # Feature importance\n                if hasattr(model, 'feature_importances_'):\n                    fi = list(zip(FEATURE_COLUMNS, model.feature_importances_))\n                elif hasattr(model, 'coef_'):\n                    fi = list(zip(FEATURE_COLUMNS, np.abs(model.coef_).flatten() if model.coef_.ndim > 1 else np.abs(model.coef_)))\n                else:\n                    fi = []\n                fi = [{'feature': f, 'importance': float(i)} for f, i in fi]\n                fi.sort(key=lambda x: x['importance'], reverse=True)\n                \n                best_artifact = {\n                    'model': model,\n                    'feature_columns': FEATURE_COLUMNS,\n                    'metrics': {'cv_score': float(cv_score), 'val_score': float(score)},\n                    'feature_importance': fi,\n                }\n                if label_encoder is not None:\n                    best_artifact['label_encoder'] = label_encoder\n        except Exception as e:\n            print(f\"    Failed: {e}\")\n    \n    return best_model, best_artifact\n\n# Train 4 tasks\nprint(\"Training Next Semester GPA (Regression)\")\nbest_next_gpa, artifact_next_gpa = train_task(\n    'Next GPA', reg_models, X1_train, y1_train, X1_val, y1_val, train_df.loc[X1_train.index, 'student_id'], 'regression'\n)\n\nprint(\"\\nTraining Final CGPA (Regression)\")\nbest_final_cgpa, artifact_final_cgpa = train_task(\n    'Final CGPA', reg_models, X2_train, y2_train, X2_val, y2_val, train_df.loc[X2_train.index, 'student_id'], 'regression'\n)\n\nprint(\"\\nTraining Graduation Class (Classification)\")\nbest_grad_class, artifact_grad_class = train_task(\n    'Grad Class', clf_models, X3_train, y3_train_enc, X3_val, y3_val_enc, train_df.loc[X3_train.index, 'student_id'], 'classification', le_class\n)\n\nprint(\"\\nTraining Academic Risk (Classification)\")\nbest_acad_risk, artifact_acad_risk = train_task(\n    'Acad Risk', clf_models, X4_train, y4_train_enc, X4_val, y4_val_enc, train_df.loc[X4_train.index, 'student_id'], 'classification', le_risk\n)"]},
    {"cell_type": "code", "metadata": {}, "source": ["# Final evaluation on test set\ndef evaluate_final(model, X_test, y_test, task_type, label_encoder=None, target_name=''):\n    pred = model.predict(X_test)\n    if task_type == 'regression':\n        mae = mean_absolute_error(y_test, pred)\n        rmse = np.sqrt(mean_squared_error(y_test, pred))\n        r2 = r2_score(y_test, pred)\n        print(f\"{target_name} Test: MAE={mae:.4f}, RMSE={rmse:.4f}, R²={r2:.4f}\")\n        return {'mae': mae, 'rmse': rmse, 'r2': r2}\n    else:\n        if label_encoder:\n            y_test_labels = label_encoder.inverse_transform(y_test)\n            pred_labels = label_encoder.inverse_transform(pred)\n        else:\n            y_test_labels = y_test\n            pred_labels = pred\n        acc = accuracy_score(y_test_labels, pred_labels)\n        f1 = f1_score(y_test_labels, pred_labels, average='macro')\n        print(f\"{target_name} Test: Acc={acc:.4f}, Macro F1={f1:.4f}\")\n        print(classification_report(y_test_labels, pred_labels))\n        # Confusion matrix check\n        cm = confusion_matrix(y_test_labels, pred_labels, normalize='true')\n        classes = label_encoder.classes_ if label_encoder else np.unique(y_test_labels)\n        for i, cls in enumerate(classes):\n            for j, cls2 in enumerate(classes):\n                if i != j and cm[i, j] > 0.20:\n                    print(f\"  WARNING: {cls} -> {cls2}: {cm[i, j]:.1%} > 20%\")\n        return {'accuracy': acc, 'macro_f1': f1}\n\nprint(\"\\n=== FINAL TEST EVALUATION ===\")\nmetrics_next = evaluate_final(best_next_gpa, X1_test, y1_test, 'regression', target_name='Next GPA')\nmetrics_final = evaluate_final(best_final_cgpa, X2_test, y2_test, 'regression', target_name='Final CGPA')\nmetrics_class = evaluate_final(best_grad_class, X3_test, y3_test_enc, 'classification', le_class, 'Graduation Class')\nmetrics_risk = evaluate_final(best_acad_risk, X4_test, y4_test_enc, 'classification', le_risk, 'Academic Risk')"]},
    {"cell_type": "code", "metadata": {}, "source": ["# Save models\nimport os\nos.makedirs('models', exist_ok=True)\n\njoblib.dump(artifact_next_gpa, 'models/next_gpa.pkl')\njoblib.dump(artifact_final_cgpa, 'models/final_cgpa.pkl')\njoblib.dump(artifact_grad_class, 'models/graduation_class.pkl')\njoblib.dump(artifact_acad_risk, 'models/academic_risk.pkl')\n\nprint(\"Models saved to models/\")"]},
    {"cell_type": "code", "metadata": {}, "source": ["# Log training run\nlog_entry = {\n    'timestamp': datetime.now().isoformat(),\n    'task': 'next_semester_gpa',\n    'model_type': type(best_next_gpa).__name__,\n    'hyperparameters': best_next_gpa.get_params() if hasattr(best_next_gpa, 'get_params') else {},\n    'cv_mae': -artifact_next_gpa['metrics']['cv_score'],\n    'val_mae': artifact_next_gpa['metrics']['val_score'],\n    'test_mae': metrics_next['mae'],\n    'test_rmse': metrics_next['rmse'],\n    'test_r2': metrics_next['r2'],\n    'selected': True\n}\n\nwith open('models/training_log.jsonl', 'a') as f:\n    f.write(json.dumps(log_entry) + '\\n')\n\n# Repeat for other 3 models...\nprint(\"Training log appended to models/training_log.jsonl\")"]},
  ],
  "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
  "nbformat": 4, "nbformat_minor": 4
}
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/Model_Training.ipynb
git commit -m "feat: add model training notebook with full pipeline"
```

---

## Phase 7: Validation & Integration

### Task 16: `requirements.txt` — Dependencies

**Files:**
- Create: `requirements.txt`

- [ ] **Step 1: Write file**

```text
# requirements.txt
pydantic>=2.0
pydantic-settings>=2.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
catboost>=1.2
scipy>=1.11
joblib>=1.3
plotly>=5.15
pytest>=7.0
matplotlib>=3.7
seaborn>=0.12
notebook>=7.0
ipykernel>=6.0
```

- [ ] **Step 2: Commit**

```bash
git add requirements.txt
git commit -m "feat: add requirements.txt with all dependencies"
```

---

### Task 17: End-to-End Integration Test

**Files:**
- Test: `tests/test_integration.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_integration.py
import pytest
from backend.orchestrator import run_pipeline
from backend.schemas import StudentInput, SemesterRecord

def test_e2e_pipeline_with_mock_models(monkeypatch):
    """Test full pipeline with mocked ML models."""
    from backend.predictor import get_models
    from backend.schemas import FEATURE_COLUMNS
    from unittest.mock import Mock
    
    class MockModel:
        def __init__(self, val): self.val = val
        def predict(self, X): return [self.val]
    
    mock_models = {
        'next_gpa': {'model': MockModel(4.2), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'feature_importance': []},
        'final_cgpa': {'model': MockModel(4.1), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'feature_importance': []},
        'graduation_class': {'model': MockModel("First Class"), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'feature_importance': [], 'label_encoder': None},
        'academic_risk': {'model': MockModel("Low"), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'feature_importance': [], 'label_encoder': None},
    }
    
    def mock_get_models():
        return mock_models
    
    monkeypatch.setattr('backend.predictor.get_models', mock_get_models)
    
    student = StudentInput(
        student_name="Integration Test",
        university="Test Uni", faculty="Science", department="Physics", course="Physics",
        programme_duration_years=5, current_level=300,
        semester_records=[
            SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2021/2022"),
            SemesterRecord(semester_number=2, gpa=3.8, credits=20, academic_session="2021/2022"),
            SemesterRecord(semester_number=3, gpa=4.0, credits=18, academic_session="2022/2023"),
            SemesterRecord(semester_number=4, gpa=4.1, credits=18, academic_session="2022/2023"),
        ],
        target_graduation_class="First Class",
    )
    
    result = run_pipeline(student)
    
    assert result.student_name == "Integration Test"
    assert result.current_cgpa is not None
    assert result.predicted_next_gpa == 4.2
    assert result.predicted_final_cgpa == 4.1
    assert result.predicted_graduation_class == "First Class"
    assert result.predicted_academic_risk == "Low"
    assert result.feasibility is not None
    assert len(result.semester_plan) == 6
    assert len(result.semester_history) == 4

def test_advisor_contract_e2e(monkeypatch):
    from backend.mock_advisor import test_advisor_contract
    from backend.predictor import get_models
    from backend.schemas import FEATURE_COLUMNS
    from unittest.mock import Mock
    
    class MockModel:
        def __init__(self, val): self.val = val
        def predict(self, X): return [self.val]
    
    mock_models = {
        'next_gpa': {'model': MockModel(4.2), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'feature_importance': []},
        'final_cgpa': {'model': MockModel(4.1), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'feature_importance': []},
        'graduation_class': {'model': MockModel("First Class"), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'feature_importance': [], 'label_encoder': None},
        'academic_risk': {'model': MockModel("Low"), 'feature_columns': FEATURE_COLUMNS, 'metrics': {}, 'feature_importance': [], 'label_encoder': None},
    }
    
    def mock_get_models():
        return mock_models
    
    monkeypatch.setattr('backend.predictor.get_models', mock_get_models)
    
    student = StudentInput(
        student_name="Advisor Test",
        university="U", faculty="F", department="D", course="C",
        programme_duration_years=4, current_level=200,
        semester_records=[
            SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2022/2023"),
            SemesterRecord(semester_number=2, gpa=4.0, credits=20, academic_session="2022/2023"),
        ],
        target_graduation_class="First Class",
    )
    
    result = test_advisor_contract(student)
    assert "Advisor Test" in result
    assert "[MOCK ADVISOR" in result
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_integration.py -v`
Expected: FAIL (until all previous tasks done)

- [ ] **Step 3: After all prior tasks pass, run test**

Run: `pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "feat: add end-to-end integration test"
```

---

## Self-Review Checklist

- [ ] **Spec Coverage:** Every section of the design spec maps to at least one task
- [ ] **No Placeholders:** All code blocks are complete, no "TBD" or "implement later"
- [ ] **Type Consistency:** Pydantic models match across all tasks (schemas.py is single source)
- [ ] **TDD Order:** Tests written first, then implementation, then verify
- [ ] **Commit Per Task:** Each task ends with a git commit

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2025-01-15-apis-phases-1-4-implementation.md`. Two execution options:**

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**