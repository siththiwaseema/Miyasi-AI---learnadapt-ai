"""LearnAdapt AI — Personalised Learning Path Prediction (Team මියැසිAI).

Run:  streamlit run app.py   (after: python src/train.py)
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import data_assist as da  # noqa: E402
import data_kaggle as dk  # noqa: E402

MODELS = ROOT / "models"
RESULTS_JSON = ROOT / "reports" / "results.json"
RESULTS_MD = ROOT / "reports" / "results.md"
KAGGLE_URL = "https://www.kaggle.com/datasets/ziya07/ai-powered-personalized-learning-dataset"
ASSIST_URL = ("https://sites.google.com/site/assistmentsdata/home/"
              "2009-2010-assistment-data/skill-builder-data-2009-2010")

st.set_page_config(page_title="LearnAdapt AI", page_icon="🎓", layout="wide")


# --------------------------------------------------------------------------- loaders
@st.cache_data
def load_results():
    if not RESULTS_JSON.exists():
        return None
    return json.loads(RESULTS_JSON.read_text(encoding="utf-8"))


@st.cache_data
def load_kaggle():
    return dk.load_kaggle()


@st.cache_resource
def load_model(name):
    p = MODELS / name
    return joblib.load(p) if p.exists() else None


@st.cache_data
def load_assist_derived():
    f = da.DERIVED_DIR
    need = [f / "assist_test_attempts.csv.gz", f / "assist_test_labels.csv.gz", f / "assist_popularity.csv"]
    if not all(p.exists() for p in need):
        return None
    att = pd.read_csv(need[0])
    lab = pd.read_csv(need[1])
    pop = pd.read_csv(need[2], index_col=0)["weak_count"]
    return att, lab, pop


res = load_results()
if res is None:
    st.error("reports/results.json not found. Run `python src/train.py` first.")
    st.stop()
A, B = res["module_a"], res["module_b"]
kdf = load_kaggle()
logreg = load_model("kaggle_clf_logreg.joblib")
reg_best = load_model("kaggle_reg_best.joblib")


# --------------------------------------------------------------------------- helpers
def clean_name(n: str) -> str:
    n = n.split("__", 1)[-1]
    for c in dk.CATEGORICAL:
        if n.startswith(c + "_"):
            return f"{c} = {n[len(c) + 1:]}"
    return n


def outcome_proba(X: pd.DataFrame) -> pd.DataFrame:
    p = logreg.predict_proba(X)[0]
    s = pd.Series(p, index=logreg.classes_).reindex(dk.OUTCOME_ORDER)
    return s.rename("probability").to_frame()


def fail_factors(X: pd.DataFrame, top=3) -> pd.DataFrame:
    """Coefficient × scaled value for the 'Fail' class of the logistic regression."""
    pre, lr = logreg.named_steps["pre"], logreg.named_steps["model"]
    z = pre.transform(X)
    z = z.toarray()[0] if hasattr(z, "toarray") else np.asarray(z)[0]
    coef = lr.coef_[list(lr.classes_).index("Fail")]
    contrib = pd.DataFrame({"feature": [clean_name(n) for n in pre.get_feature_names_out()],
                            "contribution to 'Fail'": coef * z})
    contrib = contrib[contrib["contribution to 'Fail'"] > 0]
    return contrib.sort_values("contribution to 'Fail'", ascending=False).head(top)


def path_scores(row: pd.Series) -> pd.DataFrame:
    out = []
    for path in A["paths"]:
        X = row[dk.FEATURES].to_frame().T.copy()
        X["actual_path_followed"] = path
        X[dk.NUMERIC] = X[dk.NUMERIC].astype(float)
        out.append({"path": path, "predicted final score": float(reg_best.predict(X)[0])})
    return pd.DataFrame(out)


def as_X(row: pd.Series) -> pd.DataFrame:
    X = row[dk.FEATURES].to_frame().T
    X[dk.NUMERIC] = X[dk.NUMERIC].astype(float)
    return X


def a_table(key):
    return pd.DataFrame(A[key]).set_index("model").round(3)


st.title("🎓 LearnAdapt AI — Personalised Learning Path Prediction")
st.caption("AI Innovation Hackathon 2026 · Team මියැසිAI (Siththi Waseema, Shazany Harid)")

tabs = st.tabs(["Overview", "Learner Profile & Path (A)", "Quiz-by-Quiz Gaps (B)",
                "What-if (A)", "Model Evaluation", "Data Insights", "Responsible AI & Limitations"])

# --------------------------------------------------------------------------- 1 Overview
with tabs[0]:
    st.header("Problem")
    st.write("Learners progress at different speeds, but most courses give everyone the same "
             "sequence of material. Teachers rarely have time to work out, learner by learner, "
             "what each person is strong at, where they are struggling and what to study next.")
    st.header("Solution")
    st.write("LearnAdapt AI has two separate modules:")
    st.markdown(
        "- **Module A — Learner profile & path** (Kaggle data, one row per learner): strengths and "
        "gaps vs other learners, predicted outcome probabilities, a comparison of the two learning "
        "paths, rule-based support actions, and the factors behind the prediction.\n"
        "- **Module B — Quiz-by-quiz topic gaps** (ASSISTments data, one row per attempt): from a "
        "learner's attempt history, rank the 3 skills most likely to be weak spots next, with a "
        "reason built from the learner's own numbers.")
    st.header("How it works")
    st.markdown(
        "1. `src/train.py` cleans both datasets, splits train/test without leakage, trains "
        "baselines and models, and writes `reports/results.md`.\n"
        "2. The models are saved in `models/`; this app only loads them (no retraining).\n"
        "3. Every number in the app is computed from the two datasets.")
    st.header("Datasets")
    sb = B.get("cleaning", {})
    st.markdown(
        f"| | Kaggle — AI-Powered Personalized Learning | ASSISTments 2009–2010 Skill Builder (corrected) |\n"
        f"|---|---|---|\n"
        f"| Source | [Kaggle (ziya07)]({KAGGLE_URL}) | [ASSISTments data site]({ASSIST_URL}) |\n"
        f"| Licence / terms | CC0: Public Domain | See terms on the data page; cite the page URL and Feng, Heffernan & Koedinger (2009) |\n"
        f"| Unit | one row per learner | one row per student-problem attempt |\n"
        f"| Rows used | {A['rows']:,} (train {A['train_rows']}, test {A['test_rows']}) | "
        f"{sb.get('rows', 0):,} attempts, {sb.get('users', 0):,} learners, {sb.get('skills', 0)} skills |\n"
        f"| Powers | Module A, What-if, Data Insights | Module B |")
    st.warning("**The two datasets describe different learners.** They are never joined or merged by "
               "ID. Module A and Module B are independent demonstrations on separate data.")

# --------------------------------------------------------------------------- 2 Module A
with tabs[1]:
    st.header("Learner Profile & Path — Module A (Kaggle)")
    st.caption("Only test-set learners are listed, so the model never saw their outcome during training.")
    sid = st.selectbox("Test learner", A["test_ids"], key="a_sid")
    row = kdf.set_index(dk.ID).loc[sid]
    X = as_X(row)

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Profile")
        st.dataframe(row[dk.FEATURES].astype(str).rename("value"))
    with c2:
        st.subheader("Strengths and gaps")
        st.caption("Percentile vs all 1,000 learners. Strength = top 25%, gap = bottom 25% (our choice). "
                   "For distraction_events, fewer is better.")
        prof = dk.percentile_profile(kdf, row)
        st.dataframe(prof, hide_index=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Predicted outcome (logistic regression)")
        st.bar_chart(outcome_proba(X))
        st.write(f"**Actual outcome:** {row[dk.TARGET_CLASS]} · actual final score "
                 f"{row[dk.TARGET_REG]}")
        st.info("The outcome classifier does not clearly beat a random baseline (see Model "
                "Evaluation). Treat these probabilities as illustrative, not as a judgement.")
    with c4:
        st.subheader("Path recommendation")
        ps = path_scores(row)
        best = ps.loc[ps["predicted final score"].idxmax(), "path"]
        st.dataframe(ps.round(2), hide_index=True)
        st.write(f"Recommended (higher predicted score): **{best}** · followed: "
                 f"{row['actual_path_followed']} · system recommended: {row['recommended_path']}")
        if A["path_reliable"]:
            st.success("Reliability: OK — the regressor beats the mean baseline.")
        else:
            st.error(f"⚠ LOW RELIABILITY — the best regressor ({A['a2_best']}) does not beat the "
                     "Dummy (mean) baseline on test MAE with CV R² > 0 (rule: our choice). "
                     "The difference between paths is not trustworthy.")

    c5, c6 = st.columns(2)
    with c5:
        st.subheader("Support actions (rule-based)")
        acts = dk.support_actions(prof, row["learning_style"])
        st.caption(f"Adapted to learning style: {row['learning_style']}. Rules are our choice.")
        if acts:
            for a in acts:
                st.write("• " + a)
        else:
            st.write("No gaps in the bottom 25% — keep the current plan.")
    with c6:
        st.subheader("Why? Top factors raising the chance of 'Fail'")
        st.caption("Logistic regression coefficient × scaled feature value.")
        ff = fail_factors(X)
        st.dataframe(ff.round(3), hide_index=True) if len(ff) else st.write("No factor raises 'Fail'.")

# --------------------------------------------------------------------------- 3 Module B
with tabs[2]:
    st.header("Quiz-by-Quiz Gaps — Module B (ASSISTments)")
    st.caption("Different learners from Module A. Only test learners (not used in training) are listed.")
    derived = load_assist_derived()
    clf_b = load_model("assist_weak_logreg.joblib")
    if derived is None or clf_b is None or not B.get("available"):
        st.warning("Module B data not found. Download the corrected ASSISTments file into "
                   "`data/raw/skill_builder_data_corrected.csv` and run `python src/train.py`.")
    else:
        att, lab, pop = derived
        uid = st.selectbox("Test learner (user_id)", sorted(att.user_id.unique()), key="b_uid")
        ua = att[att.user_id == uid]
        hist = ua[ua.is_history][["user_id", "skill_name", "correct"]]

        added_key = f"added_{uid}"
        st.session_state.setdefault(added_key, [])
        if st.session_state[added_key]:
            extra = pd.DataFrame(st.session_state[added_key])
            hist = pd.concat([hist, extra.assign(user_id=uid)], ignore_index=True)

        feats = da.skill_features(hist)
        scores = clf_b.predict_proba(feats[da.FEATURE_COLS])[:, 1]
        top = da.top_k(feats, scores)

        st.write(f"History: {len(hist)} attempts ({len(st.session_state[added_key])} added by you) · "
                 f"overall accuracy {hist.correct.mean():.0%} · next window: {int((~ua.is_history).sum())} attempts")
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Top-3 skills to practise next (model)")
            for _, r in top.iterrows():
                st.write(f"**{r.skill_name}** — P(weak) {r.p_weak:.0%}  \n{r.reason}")
        with c2:
            st.subheader("Baselines for comparison")
            raw = feats.assign(raw_acc=feats.n_correct / feats.attempts)
            low = raw.sort_values(["raw_acc", "attempts"], ascending=[True, False]).head(da.K)
            st.write("**Lowest own accuracy:** " + "; ".join(
                f"{s} ({a:.0%})" for s, a in zip(low.skill_name, low.raw_acc)))
            st.write("**Popularity (most often weak in training learners):** "
                     + "; ".join(pop.index[:da.K]))
            real = lab[(lab.user_id == uid) & (lab.weak == 1)]
            st.write("**Actual next-window weak spots** (< 60% with ≥ 2 attempts, our choice): "
                     + ("; ".join(f"{s} ({a:.0%} of {n})" for s, a, n in
                                  zip(real.skill_name, real.next_acc, real.next_attempts)) or "none"))

        st.subheader("History per skill")
        show = feats.assign(p_weak=scores).sort_values("p_weak", ascending=False)
        st.dataframe(show[["skill_name", "attempts", "n_correct", "smoothed_acc", "recent_acc",
                           "since_last", "p_weak"]].round(3), hide_index=True)

        st.subheader("Add new results and recompute")
        skills = sorted(pop.index.union(feats.skill_name))
        with st.form(f"add_{uid}"):
            s_sel = st.selectbox("Skill", skills)
            ok = st.radio("Result", ["Correct", "Incorrect"], horizontal=True)
            n_add = st.number_input("How many attempts with this result", 1, 20, 1)
            if st.form_submit_button("Add attempts"):
                st.session_state[added_key] += [{"skill_name": s_sel, "correct": int(ok == "Correct")}] * int(n_add)
                st.rerun()
        if st.button("Reset added results"):
            st.session_state[added_key] = []
            st.rerun()

# --------------------------------------------------------------------------- 4 What-if
with tabs[3]:
    st.header("What-if — Module A logistic regression")
    st.caption("Start from a test learner, change inputs, and watch the predicted probabilities change.")
    base_id = st.selectbox("Start from learner", A["test_ids"], key="w_sid")
    base = kdf.set_index(dk.ID).loc[base_id]
    vals = {}
    cols = st.columns(3)
    for i, c in enumerate(dk.NUMERIC):
        lo, hi = kdf[c].min(), kdf[c].max()
        with cols[i % 3]:
            if pd.api.types.is_integer_dtype(kdf[c]):
                vals[c] = st.slider(c, int(lo), int(hi), int(base[c]), key=f"w_{c}_{base_id}")
            else:
                vals[c] = st.slider(c, float(lo), float(hi), float(base[c]), key=f"w_{c}_{base_id}")
    for i, c in enumerate(dk.CATEGORICAL):
        opts = sorted(kdf[c].unique())
        with cols[i % 3]:
            vals[c] = st.selectbox(c, opts, index=opts.index(base[c]), key=f"w_{c}_{base_id}")
    Xw = pd.DataFrame([vals])[dk.FEATURES]
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Edited learner")
        st.bar_chart(outcome_proba(Xw))
    with c2:
        st.subheader("Original learner")
        st.bar_chart(outcome_proba(as_X(base)))
    st.write("Top factors raising 'Fail' for the edited learner:")
    st.dataframe(fail_factors(Xw).round(3), hide_index=True)
    st.info("Reminder: this model does not clearly beat the baselines, so changes here show how the "
            "model reacts, not what would really happen to a learner.")

# --------------------------------------------------------------------------- 5 Evaluation
with tabs[4]:
    st.header("Model Evaluation")
    st.subheader("Module A1 — learning_outcome classification (test set, 200 learners)")
    st.dataframe(a_table("a1"))
    cm_model = st.radio("Confusion matrix", ["Logistic Regression", "Random Forest"], horizontal=True)
    cm = pd.DataFrame(A["a1_confusion"][cm_model], index=[f"true {l}" for l in A["a1_labels"]],
                      columns=[f"pred {l}" for l in A["a1_labels"]])
    st.dataframe(cm)
    st.subheader("Module A2 — final_assessment_score regression (test set)")
    st.dataframe(a_table("a2"))
    st.subheader("Plain-language reading")
    a1 = a_table("a1")
    cv_spread = a1["cv_macro_f1_mean"].iloc[1:].max() - a1["cv_macro_f1_mean"].iloc[1:].min()
    st.markdown(
        f"- **Outcome classification:** a random guess that keeps the class mix (Dummy stratified) "
        f"gets test macro-F1 {a1.loc['Dummy (stratified)', 'macro_f1']:.3f}. Logistic regression gets "
        f"{a1.loc['Logistic Regression', 'macro_f1']:.3f} and random forest "
        f"{a1.loc['Random Forest', 'macro_f1']:.3f}. In 5-fold CV the three differ by only {cv_spread:.3f} "
        f"macro-F1, smaller than the CV standard deviation (~{a1['cv_macro_f1_std'].iloc[1:].mean():.3f}). "
        f"**The models do not clearly beat the baselines.**\n"
        f"- **Score regression:** no model beats predicting the average score (all R² ≤ 0). "
        f"So path recommendations are marked **low reliability**.\n"
        f"- Why: in this dataset `learning_outcome` hardly depends on the score — the mean final score "
        f"is about 80 for every outcome class (see Data Insights). The inputs carry little signal.")
    if B.get("available"):
        st.subheader("Module B — weak-spot prediction (ASSISTments test learners)")
        st.dataframe(pd.DataFrame([B["classifier"]]).round(3), hide_index=True)
        rk = pd.DataFrame(B["ranking"]).set_index("method").round(3)
        st.dataframe(rk)
        st.markdown(
            f"- Evaluated on {B['test_users_with_weak']} test learners with at least one real weak spot.\n"
            f"- Our model ({rk.iloc[2, 0]:.3f} precision@3) is clearly better than the popularity baseline "
            f"({rk.iloc[0, 0]:.3f}), but **about the same as the simple 'lowest own accuracy' baseline** "
            f"({rk.iloc[1, 0]:.3f}). Most of its signal comes from the learner's own accuracy.\n"
            f"- {B['test_weak_unseen_in_history']} of {B['test_weak_spots']} real weak spots are skills "
            "the learner never practised in history, so no history-based method can rank them.")
    with st.expander("Full reports/results.md"):
        st.markdown(RESULTS_MD.read_text(encoding="utf-8"))

# --------------------------------------------------------------------------- 6 Insights
with tabs[5]:
    st.header("Data Insights — Kaggle dataset (all 1,000 learners)")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Learning outcome counts")
        st.bar_chart(kdf[dk.TARGET_CLASS].value_counts().reindex(dk.OUTCOME_ORDER))
        st.subheader("Average final score by outcome")
        st.bar_chart(kdf.groupby(dk.TARGET_CLASS)[dk.TARGET_REG].mean().reindex(dk.OUTCOME_ORDER))
    with c2:
        st.subheader("Average final score by learning style")
        st.bar_chart(kdf.groupby("learning_style")[dk.TARGET_REG].mean())
        st.subheader("Average final score by difficulty")
        st.bar_chart(kdf.groupby("contextual_difficulty_level")[dk.TARGET_REG].mean())
    st.subheader("Correlations between numeric columns")
    num = kdf[dk.NUMERIC + ["path_efficiency_score", dk.TARGET_REG]]
    st.dataframe(num.corr().round(2))
    st.subheader("Quiz accuracy vs final score")
    st.scatter_chart(kdf, x="quiz_accuracy", y=dk.TARGET_REG, color=dk.TARGET_CLASS)

# --------------------------------------------------------------------------- 7 Responsible AI
with tabs[6]:
    st.header("Responsible AI")
    st.markdown(
        "- **Support, not labels.** Outputs are suggestions for a teacher or learner to discuss, "
        "never a grade or a judgement of ability. Predicted 'Fail' probabilities must not be used "
        "to restrict access to content.\n"
        "- **Privacy.** Both datasets are de-identified (anonymous IDs). The app stores nothing; "
        "results added in Module B live only in the browser session.\n"
        "- **Explainable.** Logistic regression factors (Module A) and reasons built from the "
        "learner's own numbers (Module B) are shown with every prediction. Support actions are "
        "plain, rule-based and labelled as such.\n"
        "- **Honest evaluation.** Every model is compared with baselines, and weak results are "
        "reported as weak.")
    st.header("Limitations")
    st.markdown(
        "- Module A data says it *emulates* real-world e-learning. Its outcome labels barely relate to "
        "the inputs or to the final score (e.g. STU019 scored 96 and is labelled Fail). Our models do "
        "not clearly beat baselines on it.\n"
        "- Module A path recommendations are low reliability: no regressor beats the mean baseline.\n"
        "- Only two paths exist in the data, and a learner followed only one, so the comparison is a "
        "model estimate, not a measured effect.\n"
        "- The two datasets are different learners; Module A and B cannot be combined for one person.\n"
        "- ASSISTments: problems where the learner used tutoring (hints) are marked incorrect, so "
        "'accuracy' mixes wrong answers and help-seeking.\n"
        f"- ASSISTments: {B.get('cleaning', {}).get('dropped_no_skill', 0):,} rows without a skill name "
        "were dropped. In the corrected file, two-skill problems keep a combined `skill_id` "
        "(e.g. `1_13`) but only one `skill_name`, which we use as given.\n"
        "- Attempt order uses `order_id` (our choice). It is described as non-chronological; we "
        "checked it agrees with the dataset's `opportunity` counter in 99.8% of learner-skill groups.\n"
        "- Module B's model is about equal to the simple 'lowest own accuracy' baseline, and skills "
        "never practised in history cannot be recommended by history-based methods.\n"
        "- All thresholds (25%/75% percentiles, 60% weak-spot accuracy, ≥ 20 attempts, ≥ 300 "
        "attempts per skill, 80/20 history split) are our choices, not validated standards.")
    st.header("Citations")
    st.markdown(
        f"- Ziya. *AI-Powered Personalized Learning Dataset*. Kaggle, CC0. {KAGGLE_URL}\n"
        f"- ASSISTments 2009–2010 Skill Builder data. {ASSIST_URL}\n"
        "- Feng, M., Heffernan, N.T., & Koedinger, K.R. (2009). Addressing the assessment challenge "
        "in an online system that tutors as it assesses. *User Modeling and User-Adapted "
        "Interaction*, 19(3), 243–266.")
