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

RISK_HIGH_CGPA_THRESHOLD = 2.0
RISK_HIGH_GPA_THRESHOLD = 1.5
RISK_MEDIUM_CGPA_THRESHOLD = 3.0
RISK_MEDIUM_GPA_THRESHOLD = 2.5