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
            assert 0.05 <= profile.volatility <= 0.7

    def test_trend_in_range(self):
        for _ in range(100):
            profile = sample_trajectory_profile()
            assert -0.2 <= profile.trend <= 0.2

    def test_shock_params(self):
        for _ in range(100):
            profile = sample_trajectory_profile()
            assert 0.0 <= profile.shock_probability <= 0.5
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

    def test_final_semester_has_null_next_gpa(self):
        skeleton = [{"semester_number": i, "semester_credits": 20} for i in range(1, 5)]
        profile = TrajectoryProfile(base_ability=3.5, volatility=0.2, trend=0.05, shock_probability=0.0, shock_magnitude=0.0)
        result = apply_trajectory_noise(skeleton, profile)
        assert result[-1]["next_semester_gpa"] is None
        for r in result[:-1]:
            assert r["next_semester_gpa"] is not None

    def test_final_cgpa_consistent_per_student(self):
        skeleton = [{"semester_number": i, "semester_credits": 20} for i in range(1, 5)]
        profile = TrajectoryProfile(base_ability=3.5, volatility=0.2, trend=0.05, shock_probability=0.0, shock_magnitude=0.0)
        result = apply_trajectory_noise(skeleton, profile)
        final_cgpas = [r["final_cgpa"] for r in result]
        assert len(set(final_cgpas)) == 1

    def test_graduation_class_from_final_cgpa(self):
        skeleton = [{"semester_number": i, "semester_credits": 20} for i in range(1, 5)]
        profile = TrajectoryProfile(base_ability=3.5, volatility=0.2, trend=0.05, shock_probability=0.0, shock_magnitude=0.0)
        result = apply_trajectory_noise(skeleton, profile)
        for r in result:
            assert r["graduation_class"] in ["First Class", "Second Class Upper", "Second Class Lower", "Third Class", "Pass", "Fail"]

    def test_academic_risk_labels(self):
        skeleton = [{"semester_number": i, "semester_credits": 20} for i in range(1, 5)]
        profile = TrajectoryProfile(base_ability=3.5, volatility=0.2, trend=0.05, shock_probability=0.0, shock_magnitude=0.0)
        result = apply_trajectory_noise(skeleton, profile)
        for r in result:
            assert r["academic_risk"] in ["Low", "Medium", "High"]