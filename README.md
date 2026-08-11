# Academic Performance Intelligence System (APIS)

Tells you exactly where you stand academically, what's realistically still possible, and what you need to do about it. Backed by real math and machine learning, explained in plain language.

## Architecture

```
Student Data
      │
      ▼
Academic Analytics Engine (Deterministic Calculations)
      │
      ▼
Machine Learning Engine (Predictive Analytics) 
      │
      ▼
AI Academic Advisor (Explanation & Guidance) 
      │
      ▼
Interactive Dashboard 
```

## Phases

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** | Research & Requirements | ✅ Complete |
| **Phase 2** | Dataset Engineering | ✅ Complete |
| **Phase 3** | Model Development | ✅ Complete |
| **Phase 4** | Backend Logic | ✅ Complete |
| **Phase 5** | AI Advisor (LLM Integration) | ✅ Complete |
| **Phase 6** | Streamlit Dashboard | ✅ Complete |
| **Phase 7** | Production (Deployment, API) | ✅ Complete |

---

## Quick Start

### Prerequisites
- Python 3.11+
- Google Gemini (AI Advisor)

### Installation

```bash
# Clone and enter project
cd APIS

# Install dependencies
pip install -r requirements.txt

# Set up environment (for AI Advisor)
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY
```

### Generate Synthetic Data

```bash
# Generate dataset and train models
python -c "from data_generation.generator import generate_dataset; rows = generate_dataset(n_students=5000, seed=42); print(f'Generated {len(rows)} rows')"
```

### Run Tests

```bash
# All tests
pytest tests/ -v

# Specific module
pytest tests/test_calculator.py -v
pytest tests/test_advisor.py -v
```

---

## Phase 5: AI Academic Advisor

### Setup

1. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/)
2. Add to `.env`:
```
GEMINI_API_KEY=your_api_key_here
```

### Usage

```python
from backend.orchestrator import run_full_pipeline_with_advice
from backend.schemas import StudentInput, SemesterRecord

student = StudentInput(
    student_name="Jane Doe",
    university="University of Lagos",
    faculty="Science",
    department="Computer Science",
    course="Computer Science",
    programme_duration_years=4,
    current_level=200,
    semester_records=[
        SemesterRecord(semester_number=1, gpa=3.5, credits=20, academic_session="2023/2024"),
        SemesterRecord(semester_number=2, gpa=3.8, credits=20, academic_session="2023/2024"),
    ],
    target_graduation_class="First Class",
)

# Run full pipeline + get AI advice
pipeline_result, advice = run_full_pipeline_with_advice(student)

print(f"Current CGPA: {pipeline_result.current_cgpa}")
print(f"Predicted Final CGPA: {pipeline_result.predicted_final_cgpa}")
print(f"Goal Feasible: {pipeline_result.feasibility.goal_achievable}")
print(f"\nAI Advisor: {advice}")
```

### Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `GEMINI_API_KEY` | `""` | Required for AI Advisor |
| `LLM_MODEL` | `gemini 3.1 flash lite` | Model to use |
| `LLM_TEMPERATURE` | `0.3` | Generation temperature |
| `LLM_MAX_TOKENS` | `200` | Max output tokens |
| `LLM_TOP_P` | `0.9` | Nucleus sampling |

### Safety Guardrails

The AI Advisor includes multiple safety layers:

1. **Content Filtering** - Gemini built-in safety settings (`BLOCK_MEDIUM_AND_ABOVE`)
2. **Numeric Echo-Check** - Extracts all numbers from LLM response, validates against `AdvisorInput` (±0.02 tolerance)
3. **Prompt Constraints** - "Never invent numbers. Only reference values explicitly provided."
4. **Fallback** - On ANY failure (API error, timeout, echo-check fail, empty response) → falls back to deterministic `mock_advisor()`

### Tones

Three tones supported via `AdvisorInput.tone`:
- `encouraging` (default) - Supportive, motivational
- `direct` - Concise, factual
- `analytical` - Data-driven, detailed

---

## Project Structure

```
APIS/
├── backend/
│   ├── __init__.py
│   ├── config.py              # Settings (model paths, LLM config)
│   ├── schemas.py             # All Pydantic models (single source of truth)
│   ├── grading_rules.py       # Classification, credits, level mapping (6 classes)
│   ├── trajectory_features.py # Shared feature computation (generator + calculator)
│   ├── calculator.py          # Pure functions: CGPA, trend, consistency, health score
│   ├── planner.py             # Goal feasibility, semester targets
│   ├── predictor.py           # Lazy model loading, feature building, label decoding
│   ├── orchestrator.py        # run_pipeline(), run_full_pipeline_with_advice()
│   ├── graphs.py              # Plotly figure dicts (JSON-serializable)
│   ├── mock_advisor.py        # Deterministic fallback
│   ├── llm_client.py          # LLMClient abstraction + GeminiClient
│   └── advisor.py             # run_advisor(), prompt building, echo-check, fallback
├── data_generation/
│   ├── structural_layer.py    # Deterministic validity (credits, levels, sessions)
│   ├── trajectory_noise.py    # Statistical realism (trajectory profiles)
│   └── generator.py           # Orchestrates both layers
├── notebooks/
│   ├── EDA.ipynb              # Exploratory data analysis
│   └── Model_Training.ipynb   # Full training pipeline (CV, selection, persistence)
├── models/                    # Trained .pkl artifacts (gitignored)
├── prompts/
│   └── advisor_prompt.txt     # Single prompt template with tone variable
├── tests/
│   ├── test_*.py              # Unit tests for every module
│   └── test_integration.py    # End-to-end tests with mocked models
├── .env.example               # Template for API keys
├── requirements.txt
└── README.md
```

---

## Key Design Decisions

### Functional Core + Thin Orchestrator
- Pure functions for deterministic math (`calculator.py`, `planner.py`)
- Classes only for stateful components (ML models, LLM client)
- `orchestrator.py` composes: `calculator → planner → predictor → advisor`

### Single Source of Truth for Features
- `backend/trajectory_features.py::compute_trajectory_features()` used by:
  - `data_generation/generator.py` (training data)
  - `backend/calculator.py` (inference)
- Eliminates train/inference feature drift

### ML Label Encoding
- `LabelEncoder` fitted on canonical class lists, saved with each model
- `predictor.py::_decode_label()` uses artifact's encoder — never assumes static order

### 6 Graduation Classes (5.0 Scale)
| Classification | CGPA Range |
|----------------|------------|
| First Class | 4.50 – 5.00 |
| Second Class Upper | 3.50 – 4.49 |
| Second Class Lower | 2.40 – 3.49 |
| Third Class | 1.50 – 2.39 |
| Pass | 1.00 – 1.49 |
| Fail | < 1.00 |

### Heuristic Constants (Explicitly Labeled)
| Constant | Value | Label |
|----------|-------|-------|
| Trend scores | Improving=25, Stable=15, Declining=5, Insufficient=15 | HEURISTIC |
| Slope thresholds | >0.1 / <-0.1 | HEURISTIC |
| Consistency bins | std ≤0.3, ≤0.6, >0.6 | HEURISTIC |
| Risk thresholds | CGPA 2.0/3.0, GPA 1.5/2.5 | HEURISTIC |

---

## Testing

```bash
# Run all 135 tests
pytest tests/ -v

# Phase 5 specific
pytest tests/test_advisor.py tests/test_llm_client.py tests/test_integration.py -v
```

---

## Model Training (Phase 3)

Run the training notebook:

```bash
jupyter notebook notebooks/Model_Training.ipynb
```

Or programmatically:

```python
from data_generation.generator import generate_dataset
rows = generate_dataset(n_students=15000, programme_durations=[4,5,6], seed=42)
# Train 4 models: Next GPA (reg), Final CGPA (reg), Graduation Class (clf), Academic Risk (clf)
# Uses GroupKFold by student_id, RandomizedSearchCV → GridSearchCV
# Persists .pkl with model, feature_columns, metrics, feature_importance, label_encoder
```
✌️