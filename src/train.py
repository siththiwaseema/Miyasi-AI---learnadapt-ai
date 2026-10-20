"""Train all models, save them to models/ and write reports/results.md + results.json.

Run from the repo root:  python src/train.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support, r2_score,
                             roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import data_assist as da  # noqa: E402
import data_kaggle as dk  # noqa: E402

ROOT = dk.ROOT
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
RS = 42


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = [f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def make_pipe(model):
    pre = ColumnTransformer([
        ("num", StandardScaler(), dk.NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), dk.CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("model", model)])


# --------------------------------------------------------------------------- Module A
def module_a():
    df = dk.load_kaggle()
    train, test = dk.split_kaggle(df)
    Xtr, Xte = train[dk.FEATURES], test[dk.FEATURES]
    out = {"rows": len(df), "train_rows": len(train), "test_rows": len(test),
           "test_ids": test[dk.ID].tolist()}

    # A1: classify learning_outcome
    ytr, yte = train[dk.TARGET_CLASS], test[dk.TARGET_CLASS]
    clfs = {
        "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
        "Dummy (stratified)": DummyClassifier(strategy="stratified", random_state=RS),
        "Logistic Regression": LogisticRegression(max_iter=2000),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RS),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RS)
    rows, cms = [], {}
    for name, m in clfs.items():
        pipe = make_pipe(m)
        cv = cross_val_score(pipe, Xtr, ytr, cv=skf, scoring="f1_macro")
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        p, r, f, _ = precision_recall_fscore_support(yte, pred, average="macro", zero_division=0)
        rows.append({"model": name, "accuracy": accuracy_score(yte, pred),
                     "macro_precision": p, "macro_recall": r, "macro_f1": f,
                     "cv_macro_f1_mean": cv.mean(), "cv_macro_f1_std": cv.std()})
        cms[name] = confusion_matrix(yte, pred, labels=dk.OUTCOME_ORDER).tolist()
        if name == "Logistic Regression":
            joblib.dump(pipe, MODELS / "kaggle_clf_logreg.joblib")
        if name == "Random Forest":
            joblib.dump(pipe, MODELS / "kaggle_clf_rf.joblib")
    out["a1"] = rows
    out["a1_confusion"] = cms
    out["a1_labels"] = dk.OUTCOME_ORDER

    # A2: regress final_assessment_score
    ytr, yte = train[dk.TARGET_REG], test[dk.TARGET_REG]
    regs = {
        "Dummy (mean)": DummyRegressor(strategy="mean"),
        "Ridge": Ridge(alpha=1.0),
        "Random Forest": RandomForestRegressor(n_estimators=300, random_state=RS),
    }
    kf = KFold(n_splits=5, shuffle=True, random_state=RS)
    rows, fitted = [], {}
    for name, m in regs.items():
        pipe = make_pipe(m)
        cv = cross_val_score(pipe, Xtr, ytr, cv=kf, scoring="r2")
        pipe.fit(Xtr, ytr)
        pred = pipe.predict(Xte)
        rows.append({"model": name, "MAE": mean_absolute_error(yte, pred),
                     "RMSE": float(np.sqrt(mean_squared_error(yte, pred))),
                     "R2": r2_score(yte, pred), "cv_R2_mean": cv.mean(), "cv_R2_std": cv.std()})
        fitted[name] = pipe
    out["a2"] = rows
    reg_df = pd.DataFrame(rows).set_index("model")
    best = reg_df.drop(index="Dummy (mean)")["MAE"].idxmin()
    dummy_mae = reg_df.loc["Dummy (mean)", "MAE"]
    reliable = bool(reg_df.loc[best, "MAE"] < dummy_mae and reg_df.loc[best, "cv_R2_mean"] > 0)
    joblib.dump(fitted[best], MODELS / "kaggle_reg_best.joblib")
    out["a2_best"] = best
    out["path_reliable"] = reliable
    out["paths"] = sorted(df["actual_path_followed"].unique().tolist())
    return out


# --------------------------------------------------------------------------- Module B
def precision_recall_at_k(recs: dict, truth: dict, k=da.K):
    ps, rs = [], []
    for u, weak in truth.items():
        hits = len(set(recs.get(u, [])[:k]) & weak)
        ps.append(hits / k)
        rs.append(hits / len(weak))
    return float(np.mean(ps)), float(np.mean(rs))


def module_b():
    if not da.ASSIST_CSV.exists():
        if "--download" not in sys.argv:
            return {"available": False, "reason": f"{da.ASSIST_CSV} not found "
                    "(run `python src/train.py --download` to fetch it from the official link)"}
        print(f"Downloading {da.ASSIST_DOWNLOAD_URL} ...")
        da.download_assist()
    t = da.build_tables()
    stats, feats, labels = t["stats"], t["feats"], t["labels"]
    tr_users, te_users, popularity = t["tr_users"], t["te_users"], t["popularity"]

    pairs = feats.merge(labels, on=["user_id", "skill_name"], how="left")
    lab = pairs[pairs.labelled.fillna(False).astype(bool)]
    tr = lab[lab.user_id.isin(tr_users)]
    te = lab[lab.user_id.isin(te_users)]

    clf = Pipeline([("scale", StandardScaler()),
                    ("lr", LogisticRegression(class_weight="balanced", max_iter=2000))])
    clf.fit(tr[da.FEATURE_COLS], tr.weak)
    proba = clf.predict_proba(te[da.FEATURE_COLS])[:, 1]
    pred = (proba >= 0.5).astype(int)
    p, r, f, _ = precision_recall_fscore_support(te.weak, pred, average="binary", zero_division=0)
    clf_metrics = {"precision": p, "recall": r, "f1": f,
                   "accuracy": accuracy_score(te.weak, pred),
                   "auc": roc_auc_score(te.weak, proba),
                   "test_pairs": len(te), "test_positive_rate": float(te.weak.mean()),
                   "train_pairs": len(tr), "train_positive_rate": float(tr.weak.mean())}
    joblib.dump(clf, MODELS / "assist_weak_logreg.joblib")

    # Ranking evaluation on test learners with >= 1 real next-window weak spot
    weak = labels[labels.weak == 1]
    truth = {u: set(g.skill_name) for u, g in weak[weak.user_id.isin(te_users)].groupby("user_id")}

    pop_top = popularity.index[:da.K].tolist()

    te_feats = feats[feats.user_id.isin(te_users)].copy()
    te_feats["p_weak"] = clf.predict_proba(te_feats[da.FEATURE_COLS])[:, 1]
    te_feats["raw_acc"] = te_feats.n_correct / te_feats.attempts
    model_recs = {u: g.sort_values("p_weak", ascending=False).skill_name.tolist()
                  for u, g in te_feats.groupby("user_id")}
    low_recs = {u: g.sort_values(["raw_acc", "attempts"], ascending=[True, False]).skill_name.tolist()
                for u, g in te_feats.groupby("user_id")}
    pop_recs = {u: pop_top for u in te_users}

    rank_rows = []
    for name, recs in [("Popularity baseline", pop_recs),
                       ("Lowest own accuracy baseline", low_recs),
                       ("Logistic Regression (ours)", model_recs)]:
        pk, rk = precision_recall_at_k(recs, truth)
        rank_rows.append({"method": name, f"precision@{da.K}": pk, f"recall@{da.K}": rk})

    # How many real weak spots could never be ranked from history (skill unseen)?
    te_weak = weak[weak.user_id.isin(te_users)].merge(
        feats[["user_id", "skill_name"]], on=["user_id", "skill_name"], how="left", indicator=True)
    unseen = int((te_weak._merge == "left_only").sum())

    # Save test-learner attempts for the app (derived data: git-ignored)
    da.write_derived(t)

    return {"available": True, "cleaning": stats, "train_users": len(tr_users),
            "test_users": len(te_users), "test_users_with_weak": len(truth),
            "test_weak_spots": int(len(te_weak)), "test_weak_unseen_in_history": unseen,
            "classifier": clf_metrics, "ranking": rank_rows, "popularity_top": pop_top,
            "coefficients": dict(zip(da.FEATURE_COLS, clf.named_steps["lr"].coef_[0].round(4).tolist()))}


# --------------------------------------------------------------------------- report
def verdict_a1(a1):
    d = pd.DataFrame(a1).set_index("model")
    best_dummy = d.loc[["Dummy (most frequent)", "Dummy (stratified)"], "macro_f1"].max()
    best_dummy_cv = d.loc[["Dummy (most frequent)", "Dummy (stratified)"], "cv_macro_f1_mean"].max()
    # Our choice: "clearly beats" = higher test macro-F1 AND a CV macro-F1 gain
    # larger than one CV standard deviation of the model.
    lines = []
    for m in ["Logistic Regression", "Random Forest"]:
        cv_gain = d.loc[m, "cv_macro_f1_mean"] - best_dummy_cv
        beats = d.loc[m, "macro_f1"] > best_dummy and cv_gain > d.loc[m, "cv_macro_f1_std"]
        lines.append(f"- {m}: test macro-F1 {d.loc[m, 'macro_f1']:.3f} vs best dummy {best_dummy:.3f}; "
                     f"CV macro-F1 {d.loc[m, 'cv_macro_f1_mean']:.3f} vs best dummy {best_dummy_cv:.3f} "
                     f"(gain {cv_gain:+.3f}, CV std {d.loc[m, 'cv_macro_f1_std']:.3f}) "
                     f"-> {'clearly beats' if beats else 'does NOT clearly beat'} the baselines.")
    return lines


def write_report(a, b):
    L = ["# LearnAdapt AI — Results", "",
         "_Generated by `python src/train.py`. Do not edit by hand._", "",
         "## Module A — Kaggle learner profiles (one row per learner)", "",
         f"Rows: {a['rows']} (train {a['train_rows']}, test {a['test_rows']}); "
         "80/20 split stratified by learning_outcome, random_state=42 (our choice).", "",
         "### A1 — Classify learning_outcome (test set)", "",
         md_table(pd.DataFrame(a["a1"])), "", "Reading:"] + verdict_a1(a["a1"]) + [""]
    for name in ["Logistic Regression", "Random Forest"]:
        cm = pd.DataFrame(a["a1_confusion"][name], index=a["a1_labels"], columns=a["a1_labels"])
        cm.insert(0, "true \\ predicted", cm.index)
        L += [f"Confusion matrix — {name}:", "", md_table(cm), ""]
    reg = pd.DataFrame(a["a2"])
    L += ["### A2 — Predict final_assessment_score (test set)", "", md_table(reg), "",
          f"Best regressor by test MAE: **{a['a2_best']}**. Path recommendations are "
          f"**{'reliable' if a['path_reliable'] else 'LOW RELIABILITY'}** "
          "(rule, our choice: best regressor must beat the Dummy on test MAE AND have CV R² > 0).", ""]

    L += ["## Module B — ASSISTments quiz-by-quiz topic gaps", ""]
    if not b["available"]:
        L += [f"Not run: {b['reason']}", ""]
    else:
        s = b["cleaning"]
        L += ["### Cleaning (thresholds are our choice)", "",
              f"- Raw rows: {s['raw_rows']:,} ({s['raw_users']:,} learners)",
              f"- Dropped, no skill_name: {s['dropped_no_skill']:,}",
              f"- Dropped, scaffolding sub-questions (original = 0): {s['dropped_scaffolding']:,}",
              f"- Dropped, skills with < {da.MIN_SKILL_ATTEMPTS} attempts: {s['dropped_rare_skills']:,}",
              f"- Dropped, learners with < {da.MIN_USER_ATTEMPTS} attempts: {s['dropped_short_learners']:,}",
              f"- Kept: {s['rows']:,} attempts, {s['users']:,} learners, {s['skills']} skills", "",
              f"Learners split with GroupShuffleSplit (test_size=0.25, random_state=42): "
              f"{b['train_users']} train / {b['test_users']} test, no overlap.", "",
              "### Weak-spot classifier (test learner × skill pairs with ≥ 2 next-window attempts)", "",
              md_table(pd.DataFrame([b["classifier"]])), "",
              f"### Top-{da.K} recommendations vs real next-window weak spots", "",
              f"Evaluated on {b['test_users_with_weak']} test learners who have at least one real "
              f"weak spot in their next window ({b['test_weak_spots']} weak spots in total; "
              f"{b['test_weak_unseen_in_history']} of them are skills the learner never practised in "
              "history, so history-based methods cannot rank them).", "",
              md_table(pd.DataFrame(b["ranking"])), "",
              f"Popularity baseline top-{da.K}: {', '.join(b['popularity_top'])}", "",
              "Logistic regression coefficients (standardised features): "
              + ", ".join(f"{k} {v:+.3f}" for k, v in b["coefficients"].items()), ""]
    (REPORTS / "results.md").write_text("\n".join(L), encoding="utf-8")


def main():
    MODELS.mkdir(exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    print("Module A ...")
    a = module_a()
    print("Module B ...")
    b = module_b()
    (REPORTS / "results.json").write_text(json.dumps({"module_a": a, "module_b": b}, indent=2,
                                                     default=float), encoding="utf-8")
    write_report(a, b)
    print((REPORTS / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
