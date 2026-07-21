import random
from typing import Dict, List, Any
from backend.grading_rules import level_for_semester, CREDITS_PER_LEVEL


UNIVERSITIES = [
    "University of Lagos", "University of Ibadan", "Ahmadu Bello University",
    "Obafemi Awolowo University", "University of Nigeria, Nsukka",
]

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
        })
    return rows