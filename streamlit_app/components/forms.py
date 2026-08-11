import streamlit as st
from backend.schemas import StudentInput, SemesterRecord
from backend.grading_rules import level_for_semester


def validate_semester_records(records: list[SemesterRecord]) -> str | None:
    if not records:
        return "At least one semester required"
    for r in records:
        if not (0.0 <= r.gpa <= 5.0):
            return f"Semester {r.semester_number}: GPA must be 0.00–5.00"
        if not (12 <= r.credits <= 24):
            return f"Semester {r.semester_number}: Credits must be 12–24"
        if "/" not in r.academic_session:
            return f"Semester {r.semester_number}: Session format 'YYYY/YYYY' required"
    return None


def render_sidebar_form() -> StudentInput | None:
    st.sidebar.header("📝 Student Profile")

    # Basic info
    student_name = st.sidebar.text_input("Full Name", value="")
    university = st.sidebar.text_input("University", value="")
    faculty = st.sidebar.text_input("Faculty", value="")
    department = st.sidebar.text_input("Department", value="")
    course = st.sidebar.text_input("Course", value="")

    programme_duration = st.sidebar.selectbox("Programme Duration (years)", [4, 5, 6], index=1)

    st.sidebar.divider()
    st.sidebar.subheader("Semester Records")

    # Dynamic semester rows
    if "semester_rows" not in st.session_state:
        st.session_state.semester_rows = 1

    records = []
    for i in range(st.session_state.semester_rows):
        with st.sidebar.expander(f"Semester {i+1}", expanded=(i == st.session_state.semester_rows - 1)):
            sem_num = i + 1
            gpa = st.number_input(f"GPA", 0.0, 5.0, 3.5, 0.01, key=f"gpa_{i}", format="%.2f")
            credits = st.number_input(f"Credits", 12, 24, 20, key=f"credits_{i}")
            session = st.text_input(f"Session (YYYY/YYYY)", "2025/2026", key=f"session_{i}")
            records.append(SemesterRecord(semester_number=sem_num, gpa=gpa, credits=credits, academic_session=session))

    col1, col2 = st.sidebar.columns(2)
    if col1.button("➕ Add Semester", use_container_width=True):
        st.session_state.semester_rows += 1
        st.rerun()
    if col2.button("🗑️ Remove Last", use_container_width=True, disabled=st.session_state.semester_rows <= 1):
        st.session_state.semester_rows -= 1
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("🎯 Target")
    target_type = st.sidebar.radio("Target by:", ["Graduation Class", "Target CGPA"], horizontal=True)

    if target_type == "Graduation Class":
        target_class = st.sidebar.selectbox(
            "Target Class",
            ["First Class", "Second Class Upper", "Second Class Lower", "Third Class", "Pass"],
            index=0,
        )
        target_cgpa = None
    else:
        target_cgpa = st.sidebar.number_input("Target CGPA", 0.0, 5.0, 4.5, 0.01, format="%.2f")
        target_class = None

    # Validation
    error = validate_semester_records(records)
    if error:
        st.sidebar.error(error)
        return None

    # Current level auto-calculated
    semesters_completed = len(records)
    total_semesters = programme_duration * 2
    current_level = level_for_semester(semesters_completed, total_semesters)

    run_clicked = st.sidebar.button("🚀 Run Analysis", type="primary", use_container_width=True)

    if not run_clicked:
        return None

    return StudentInput(
        student_name=student_name,
        university=university,
        faculty=faculty,
        department=department,
        course=course,
        programme_duration_years=programme_duration,
        current_level=current_level,
        semester_records=records,
        target_graduation_class=target_class,
        target_cgpa=target_cgpa,
    )