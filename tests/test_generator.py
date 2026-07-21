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