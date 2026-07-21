import joblib
import pandas as pd
import numpy as np
from typing import Dict, Optional, List
from backend.config import settings
from backend.schemas import (
    StudentInput, CalculatorOutput, PredictorOutput, FeatureImportance,
    FEATURE_COLUMNS, GRADUATION_CLASSES, ACADEMIC_RISK_CLASSES
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


def extract_top_features(model_artifact: Dict, top_k: int = 5) -> List[FeatureImportance]:
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