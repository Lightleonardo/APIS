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


class TestLevelForSemester:
    @pytest.mark.parametrize("sem,total,expected", [
        (1, 8, 100), (2, 8, 100), (3, 8, 200), (4, 8, 200),
        (5, 8, 300), (6, 8, 300), (7, 8, 400), (8, 8, 400),
        (1, 10, 100), (5, 10, 300), (9, 10, 500), (10, 10, 500),
        (1, 12, 100), (9, 12, 500), (12, 12, 500),
    ])
    def test_level_for_semester(self, sem, total, expected):
        assert level_for_semester(sem, total) == expected