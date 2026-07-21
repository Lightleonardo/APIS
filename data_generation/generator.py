from typing import Any, List
import pandas as pd
from pathlib import Path
from data_generation.structural_layer import build_static_attributes, build_semester_skeleton
from data_generation.trajectory_noise import sample_trajectory_profile, apply_trajectory_noise
from backend.schemas import DatasetRow


def generate_dataset(
    n_students: int = 10000,
    programme_durations: List[int] = [4, 5, 6],
    seed: int = 42,
) -> List[DatasetRow]:
    import random
    random.seed(seed)

    all_rows = []

    for student_idx in range(n_students):
        duration = random.choice(programme_durations)
        static_attrs = build_static_attributes(student_idx, duration)
        skeleton = build_semester_skeleton(static_attrs, student_idx)
        profile = sample_trajectory_profile()
        complete_rows = apply_trajectory_noise(skeleton, profile)

        for row_dict in complete_rows:
            all_rows.append(DatasetRow(**row_dict))

    return all_rows


def save_dataset(rows: List[DatasetRow], output_path: str) -> None:
    """Write rows to CSV. Creates parent directory if needed."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([row.model_dump() for row in rows])
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    rows = generate_dataset(n_students=100, seed=42)
    print(f"Generated {len(rows)} rows for {len(set(r.student_id for r in rows))} students")

    # Quick validation
    print(f"Columns: {list(DatasetRow.model_fields.keys())}")
    print(f"Sample row: {rows[0].student_id}, CGPA: {rows[0].cumulative_cgpa}")

    save_dataset(rows, "data/synthetic_dataset.csv")
