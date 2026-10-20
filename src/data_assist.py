"""Module B data: ASSISTments 2009-2010 Skill Builder (corrected file).

Quiz-by-quiz attempts. These learners are NOT the Kaggle learners; the two
datasets are never joined.
"""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

ROOT = Path(__file__).resolve().parents[1]
ASSIST_CSV = ROOT / "data" / "raw" / "skill_builder_data_corrected.csv"
# Git-ignored: the ASSISTments Terms of Use say "I agree to not give this data to
# anyone else", so neither the raw file nor per-learner derived files are committed.
DERIVED_DIR = ROOT / "data" / "derived"
DERIVED_FILES = ["assist_test_attempts.csv.gz", "assist_test_labels.csv.gz", "assist_popularity.csv"]

# Direct download of the "corrected version" linked from the official data page
# (Google Drive file id 1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE).
ASSIST_DOWNLOAD_URL = ("https://drive.usercontent.google.com/download"
                       "?id=1NNXHFRxcArrU0ZJSb9BIL56vmUt5FhlE&export=download&confirm=t")
RANDOM_STATE = 42
TEST_SIZE = 0.25

# All thresholds below are our choice.
MIN_SKILL_ATTEMPTS = 300   # keep skills with >= 300 attempts
MIN_USER_ATTEMPTS = 20     # keep learners with >= 20 attempts
HISTORY_FRAC = 0.8         # first 80% of a learner's attempts = history
WEAK_ACC = 0.6             # weak spot: next-window accuracy < 60% ...
WEAK_MIN_ATTEMPTS = 2      # ... with >= 2 next-window attempts
RECENT_N = 5               # "recent accuracy" = last 5 attempts on the skill
K = 3

FEATURE_COLS = ["attempts", "smoothed_acc", "recent_acc", "since_last", "overall_acc"]

USECOLS = ["order_id", "user_id", "problem_id", "original", "correct", "skill_name"]


def load_assist(path=ASSIST_CSV):
    """Load and clean. Returns (clean_df, cleaning_stats)."""
    raw = pd.read_csv(path, encoding="latin-1", usecols=USECOLS, low_memory=False)
    stats = {"raw_rows": len(raw), "raw_users": int(raw.user_id.nunique())}

    df = raw.dropna(subset=["skill_name"])
    stats["dropped_no_skill"] = len(raw) - len(df)

    # original: 1 = main problem, 0 = scaffolding sub-question. Our choice:
    # keep main problems only so a scaffolded problem is not counted twice.
    n = len(df)
    df = df[df.original == 1]
    stats["dropped_scaffolding"] = n - len(df)

    n = len(df)
    skill_counts = df.skill_name.value_counts()
    keep_skills = skill_counts[skill_counts >= MIN_SKILL_ATTEMPTS].index
    df = df[df.skill_name.isin(keep_skills)]
    stats["dropped_rare_skills"] = n - len(df)

    n = len(df)
    user_counts = df.user_id.value_counts()
    df = df[df.user_id.isin(user_counts[user_counts >= MIN_USER_ATTEMPTS].index)]
    stats["dropped_short_learners"] = n - len(df)

    # Attempt order (our choice): order_id. We checked that within a learner
    # and skill, the dataset's own `opportunity` counter increases with
    # order_id in 99.8% of groups, so we use it as the time order.
    df = df.sort_values(["user_id", "order_id"]).reset_index(drop=True)
    df["correct"] = df["correct"].astype(int)
    df["skill_name"] = df["skill_name"].astype(str)
    stats.update(rows=len(df), users=int(df.user_id.nunique()),
                 skills=int(df.skill_name.nunique()))
    return df, stats


def split_history_next(df: pd.DataFrame) -> pd.DataFrame:
    """Mark each attempt as history (first 80%) or next window (last 20%)."""
    df = df.copy()
    pos = df.groupby("user_id").cumcount()
    n = df.groupby("user_id")["user_id"].transform("size")
    df["is_history"] = pos < np.floor(n * HISTORY_FRAC)
    return df


def skill_features(hist: pd.DataFrame) -> pd.DataFrame:
    """Features per (user_id, skill_name) from HISTORY attempts only.

    hist must be sorted by user then attempt order.
    """
    h = hist[["user_id", "skill_name", "correct"]].copy()
    h["pos"] = h.groupby("user_id").cumcount()
    n_hist = h.groupby("user_id").size().rename("n_hist")
    overall = h.groupby("user_id")["correct"].mean().rename("overall_acc")

    g = h.groupby(["user_id", "skill_name"])
    feats = pd.DataFrame({
        "attempts": g.size(),
        "n_correct": g["correct"].sum(),
        "recent_acc": g["correct"].apply(lambda s: s.tail(RECENT_N).mean()),
        "last_pos": g["pos"].max(),
    }).reset_index()
    # Laplace smoothing (our choice): (correct + 1) / (attempts + 2)
    feats["smoothed_acc"] = (feats.n_correct + 1) / (feats.attempts + 2)
    feats = feats.join(n_hist, on="user_id").join(overall, on="user_id")
    feats["since_last"] = feats.n_hist - feats.last_pos - 1
    return feats.drop(columns=["last_pos", "n_hist"])


def next_window_labels(nxt: pd.DataFrame) -> pd.DataFrame:
    g = nxt.groupby(["user_id", "skill_name"])["correct"]
    lab = pd.DataFrame({"next_attempts": g.size(), "next_acc": g.mean()}).reset_index()
    lab["weak"] = ((lab.next_attempts >= WEAK_MIN_ATTEMPTS) & (lab.next_acc < WEAK_ACC)).astype(int)
    lab["labelled"] = lab.next_attempts >= WEAK_MIN_ATTEMPTS
    return lab


def download_assist(path=ASSIST_CSV, url=ASSIST_DOWNLOAD_URL) -> Path:
    """Download the corrected file from the official link (not stored in git)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    with open(tmp, "r", encoding="latin-1") as f:
        header = f.readline()
    if "order_id" not in header or "skill_name" not in header:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"Download from {url} did not return the ASSISTments CSV "
                           f"(first line: {header[:120]!r}).")
    tmp.replace(path)
    return path


def split_users(df: pd.DataFrame):
    """GroupShuffleSplit on user_id: no learner is in both train and test."""
    users = df.user_id.unique()
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    tr_idx, te_idx = next(gss.split(users, groups=users))
    tr_users, te_users = set(users[tr_idx]), set(users[te_idx])
    assert not tr_users & te_users
    return tr_users, te_users


def build_tables(path=ASSIST_CSV) -> dict:
    """Everything Module B needs, built the same way for train.py and the app."""
    df, stats = load_assist(path)
    df = split_history_next(df)
    feats = skill_features(df[df.is_history])
    labels = next_window_labels(df[~df.is_history])
    tr_users, te_users = split_users(df)
    weak = labels[labels.weak == 1]
    popularity = weak[weak.user_id.isin(tr_users)].skill_name.value_counts()
    return {"df": df, "stats": stats, "feats": feats, "labels": labels,
            "tr_users": tr_users, "te_users": te_users, "popularity": popularity}


def write_derived(t: dict, out_dir=DERIVED_DIR) -> None:
    """Save only what the app needs for TEST learners (git-ignored)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    te = t["te_users"]
    t["df"][t["df"].user_id.isin(te)].to_csv(out_dir / DERIVED_FILES[0], index=False)
    t["labels"][t["labels"].user_id.isin(te)].to_csv(out_dir / DERIVED_FILES[1], index=False)
    t["popularity"].rename("weak_count").to_csv(out_dir / DERIVED_FILES[2])


def read_derived(out_dir=DERIVED_DIR):
    out_dir = Path(out_dir)
    att = pd.read_csv(out_dir / DERIVED_FILES[0])
    lab = pd.read_csv(out_dir / DERIVED_FILES[1])
    pop = pd.read_csv(out_dir / DERIVED_FILES[2], index_col=0)["weak_count"]
    return att, lab, pop


def ensure_derived(download=True, out_dir=DERIVED_DIR):
    """Return (attempts, labels, popularity) for test learners.

    Uses the derived files if present; otherwise builds them from the raw CSV,
    downloading the raw CSV from the official link first if needed.
    """
    out_dir = Path(out_dir)
    if not all((out_dir / f).exists() for f in DERIVED_FILES):
        if not ASSIST_CSV.exists():
            if not download:
                raise FileNotFoundError(ASSIST_CSV)
            download_assist()
        write_derived(build_tables(), out_dir)
    return read_derived(out_dir)


def reason(row) -> str:
    return (f"{int(row.n_correct)}/{int(row.attempts)} correct so far; "
            f"recent accuracy {row.recent_acc:.0%}; "
            f"last practised {int(row.since_last)} attempts ago")


def top_k(feats_user: pd.DataFrame, scores: np.ndarray, k=K) -> pd.DataFrame:
    f = feats_user.assign(p_weak=scores).sort_values("p_weak", ascending=False).head(k)
    f["reason"] = f.apply(reason, axis=1)
    return f
