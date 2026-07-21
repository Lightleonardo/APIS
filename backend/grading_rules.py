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

RISK_HIGH_CGPA_THRESHOLD = 2.0
RISK_HIGH_GPA_THRESHOLD = 1.5
RISK_MEDIUM_CGPA_THRESHOLD = 3.0
RISK_MEDIUM_GPA_THRESHOLD = 2.5


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