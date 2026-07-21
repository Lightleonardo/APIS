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