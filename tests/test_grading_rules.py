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