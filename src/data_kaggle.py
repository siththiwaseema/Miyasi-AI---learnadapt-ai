"""Module A data: Kaggle "AI-Powered Personalized Learning Dataset" (CC0).

One row per learner. Loads, checks and splits the CSV.
"""
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
KAGGLE_CSV = ROOT / "data" / "AI_Personalized_Learning.csv"

ID = "student_id"
TARGET_CLASS = "learning_outcome"
TARGET_REG = "final_assessment_score"
# Outcomes measured after learning: never used as model inputs (would leak the answer).
LEAKY = ["path_efficiency_score", TARGET_REG, TARGET_CLASS, ID]

NUMERIC = [
    "age", "previous_gpa", "completed_modules", "avg_time_per_module",
    "engagement_score", "distraction_events", "quiz_accuracy", "feedback_score",
]
CATEGORICAL = [
    "gender", "education_level", "learning_style",
    "contextual_difficulty_level", "recommended_path", "actual_path_followed",
]
FEATURES = NUMERIC + CATEGORICAL
OUTCOME_ORDER = ["Fail", "Fair", "Good", "Excellent"]

# Strengths / gaps: (column, higher_is_better)
PROFILE_METRICS = [
    ("previous_gpa", True),
    ("completed_modules", True),
    ("engagement_score", True),
    ("distraction_events", False),
    ("quiz_accuracy", True),
    ("feedback_score", True),
]
STRENGTH_PCT = 75  # our choice: top 25% = strength
GAP_PCT = 25       # our choice: bottom 25% = gap

TEST_SIZE = 0.2    # our choice (from brief): 80/20, stratified, random_state=42
RANDOM_STATE = 42


def load_kaggle(path=KAGGLE_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    missing_cols = set(FEATURES + LEAKY) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Kaggle CSV is missing columns: {sorted(missing_cols)}")
    for c in CATEGORICAL + [TARGET_CLASS, ID]:
        df[c] = df[c].astype(str).str.strip()
    return df


def split_kaggle(df: pd.DataFrame):
    """80/20 split stratified by learning_outcome. Returns (train_df, test_df)."""
    return train_test_split(
        df, test_size=TEST_SIZE, stratify=df[TARGET_CLASS], random_state=RANDOM_STATE
    )


def percentile_profile(df_all: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    """Percentile of one learner vs all learners for each profile metric.

    For distraction_events lower is better, so the percentile is flipped
    (a high percentile always means "better than most learners").
    """
    out = []
    for col, higher_better in PROFILE_METRICS:
        v = row[col]
        col_vals = df_all[col]
        # mean of "strictly below" and "at or below" handles ties fairly
        pct = 100 * ((col_vals < v).mean() + (col_vals <= v).mean()) / 2
        if not higher_better:
            pct = 100 - pct
        status = "Strength" if pct >= STRENGTH_PCT else "Gap" if pct <= GAP_PCT else "Typical"
        out.append({"metric": col, "value": v, "percentile (higher = better)": round(pct, 1),
                    "status": status})
    return pd.DataFrame(out)


# Rule-based support actions (our choice), adapted to learning style.
_STYLE_TIPS = {
    "Visual": "use diagrams, mind maps and colour-coded notes",
    "Auditory": "use recorded explanations, discussion and reading notes aloud",
    "Kinesthetic": "use hands-on exercises, simulations and short practice tasks",
}
_GAP_ACTIONS = {
    "previous_gpa": "Revisit foundation topics before new modules; {tip}.",
    "completed_modules": "Set a weekly module target with small milestones; {tip}.",
    "engagement_score": "Shorter, more interactive sessions to lift engagement; {tip}.",
    "distraction_events": "Focus blocks (e.g. 25 min) with notifications off; {tip}.",
    "quiz_accuracy": "Extra low-stakes practice quizzes on missed items; {tip}.",
    "feedback_score": "Check in with a tutor about what is not working; {tip}.",
}


def support_actions(profile: pd.DataFrame, learning_style: str) -> list[str]:
    tip = _STYLE_TIPS.get(learning_style, "use the format that works best for you")
    gaps = profile.loc[profile["status"] == "Gap", "metric"]
    return [f"[{m}] " + _GAP_ACTIONS[m].format(tip=tip) for m in gaps]
