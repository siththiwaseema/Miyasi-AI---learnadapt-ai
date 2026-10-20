"""LearnAdapt AI — Personalised Learning Path Prediction (Team මියැසිAI).

Views: Home, Student, Teacher (plain language) and Developer insight (technical, in the sidebar).

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

st.set_page_config(page_title="LearnAdapt AI", page_icon="🎓", layout="wide",
                   initial_sidebar_state="collapsed")


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


@st.cache_resource(show_spinner="Loading quiz data, this takes about a minute the first time…")
def _build_assist_derived():
    """ASSISTments data is not in git (its terms forbid redistribution), so on a fresh
    deployment we fetch it from the official link once and build the test-learner tables."""
    return da.ensure_derived(download=True)


def load_assist_derived():
    # Exceptions are not cached by st.cache_resource, so a failed download is retried next run.
    try:
        return _build_assist_derived(), None
    except Exception as e:  # shown in the Module B tab instead of crashing the app
        return None, f"{type(e).__name__}: {e}"


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


# --------------------------------------------------------------------------- plain-language helpers
DEMO_A, DEMO_B = "STU935", 70733
NOT_A_JUDGEMENT = "💬 These are suggestions to discuss with your teacher, not a judgement."
FRIENDLY = {
    "age": "Age", "gender": "Gender", "education_level": "Education level",
    "learning_style": "Learning style", "previous_gpa": "Previous GPA",
    "completed_modules": "Modules completed", "avg_time_per_module": "Average time per module",
    "engagement_score": "Engagement", "distraction_events": "Focus (few distractions)",
    "quiz_accuracy": "Quiz accuracy", "feedback_score": "Feedback score",
    "contextual_difficulty_level": "Course difficulty", "recommended_path": "Suggested path",
    "actual_path_followed": "Path followed",
}


def how_to(text: str):
    st.info("**How to use this page** — " + text, icon="ℹ️")


def friendly_feature(n: str) -> str:
    """'learning_style = Visual' -> 'Learning style: Visual'."""
    if " = " in n:
        col, val = n.split(" = ", 1)
        return f"{FRIENDLY.get(col, col)}: {val}"
    return FRIENDLY.get(n, n)


def split_action(a: str):
    """'[completed_modules] Set a weekly ...' -> ('Modules completed', 'Set a weekly ...')."""
    metric, text = a[1:].split("] ", 1)
    return FRIENDLY.get(metric, metric), text


def pick_learner_a(key: str, label: str):
    ids = A["test_ids"]
    sid = st.selectbox(label, ids, index=ids.index(DEMO_A) if DEMO_A in ids else 0, key=key,
                       help=f"Tip: try {DEMO_A} for the demo.")
    return kdf.set_index(dk.ID).loc[sid]


def enter_own_id(label: str, key: str, valid_ids, demo, as_int: bool):
    """Students type their own ID (no list of other learners). Returns the ID or None."""
    typed = st.text_input(label, key=key, placeholder=f"e.g. {demo}",
                          help=f"Type your own ID. For the demo, try {demo}.").strip()
    if not typed:
        st.caption("Type your ID above and press Enter to see your results.")
        return None
    if as_int:
        if not typed.isdigit():
            st.warning("Your ID should be a number. Please check it and try again.")
            return None
        found = int(typed)
    else:
        found = typed.upper()
    if found not in set(valid_ids):
        st.warning("We couldn't find that ID. Please check it, or ask your teacher.")
        return None
    return found


def strength_gap_badges(prof: pd.DataFrame, with_numbers: bool):
    strengths = prof[prof.status == "Strength"]
    gaps = prof[prof.status == "Gap"]

    def line(r):
        name = FRIENDLY[r.metric]
        if not with_numbers:
            return name
        return f"{name} — higher than about {r['percentile (higher = better)']:.0f}% of learners"

    st.markdown("**🌟 Strengths**")
    st.markdown("  \n".join(f":green-badge[✓ {line(r)}]" for _, r in strengths.iterrows())
                or "No standout strengths yet — every learner has room to shine.")
    st.markdown("**🌱 Areas to grow**")
    st.markdown("  \n".join(f":orange-badge[↗ {line(r)}]" for _, r in gaps.iterrows())
                or "No big gaps — keep going!")


def module_b_data(friendly: bool):
    clf_b = load_model("assist_weak_logreg.joblib")
    derived, err = load_assist_derived() if clf_b is not None and B.get("available") else (None, None)
    if derived is None:
        if friendly:
            st.warning("Quiz data couldn't be loaded right now. Please try again in a minute.")
            if err:
                with st.expander("Technical details"):
                    st.code(err)
        return None
    return (*derived, clf_b)


def pick_learner_b(att, key: str, label: str):
    users = sorted(att.user_id.unique())
    return st.selectbox(label, users, index=users.index(DEMO_B) if DEMO_B in users else 0, key=key,
                        help=f"Tip: try {DEMO_B} for the demo.")


def learner_b(att, clf_b, uid):
    """History (+ results added in this session), features and model scores for one learner."""
    ua = att[att.user_id == uid]
    hist = ua[ua.is_history][["user_id", "skill_name", "correct"]]
    added_key = f"added_{uid}"
    st.session_state.setdefault(added_key, [])
    if st.session_state[added_key]:
        extra = pd.DataFrame(st.session_state[added_key])
        hist = pd.concat([hist, extra.assign(user_id=uid)], ignore_index=True)
    feats = da.skill_features(hist)
    scores = clf_b.predict_proba(feats[da.FEATURE_COLS])[:, 1]
    return ua, hist, feats, scores, da.top_k(feats, scores), added_key


def plain_reason(r, you: bool) -> str:
    who = "You got" if you else "Got"
    ago = "just now" if r.since_last == 0 else f"{int(r.since_last)} questions ago"
    return (f"{who} {int(r.n_correct)} of {int(r.attempts)} right so far · recent accuracy "
            f"{r.recent_acc:.0%} · last practised {ago}")


def topic_cards(top: pd.DataFrame, you: bool):
    for i, (_, r) in enumerate(top.iterrows(), start=1):
        with st.container(border=True):
            st.markdown(f"**Priority {i} · {r.skill_name}**")
            st.markdown(f":orange-badge[{r.p_weak:.0%} chance this topic needs more practice]")
            st.caption(plain_reason(r, you))


def history_table(feats: pd.DataFrame, scores):
    t = feats.assign(chance=scores).sort_values("chance", ascending=False)
    st.dataframe(pd.DataFrame({
        "Topic": t.skill_name,
        "Attempts": t.attempts.astype(int),
        "Correct": t.n_correct.astype(int),
        "Recent accuracy": t.recent_acc.map("{:.0%}".format),
        "Last practised": t.since_last.map(lambda s: "just now" if s == 0 else f"{int(s)} questions ago"),
        "Needs practice (chance)": t.chance.map("{:.0%}".format),
    }), hide_index=True)


def add_results(uid, feats, pop, prefix: str, you: bool):
    st.markdown("#### ➕ " + ("Just practised? Add your new results" if you else "Add new results"))
    st.caption("The suggestions update straight away. Nothing is saved after you close the page.")
    with st.form(f"{prefix}_add_{uid}"):
        topic = st.selectbox("Topic", sorted(pop.index.union(feats.skill_name)))
        ok = st.radio("Result", ["Right", "Wrong"], horizontal=True)
        n = st.number_input("How many questions", 1, 20, 1)
        if st.form_submit_button("Add results"):
            st.session_state[f"added_{uid}"] += [{"skill_name": topic, "correct": int(ok == "Right")}] * int(n)
            st.rerun()
    if st.button("Undo everything I added", key=f"{prefix}_reset_{uid}"):
        st.session_state[f"added_{uid}"] = []
        st.rerun()


def evaluator_expander(uid, feats, top, lab, pop):
    with st.expander("For evaluators: how good was this prediction?"):
        raw = feats.assign(raw_acc=feats.n_correct / feats.attempts)
        low = raw.sort_values(["raw_acc", "attempts"], ascending=[True, False]).head(da.K)
        real = lab[(lab.user_id == uid) & (lab.weak == 1)]
        hits = len(set(top.skill_name) & set(real.skill_name))
        st.markdown(
            "**Simple comparison method 1 — topics with the lowest accuracy so far:** "
            + "; ".join(f"{s} ({a:.0%})" for s, a in zip(low.skill_name, low.raw_acc)) + "  \n"
            "**Simple comparison method 2 — topics most learners found hard:** "
            + "; ".join(pop.index[:da.K]) + "  \n"
            "**What actually happened in this learner's later attempts** (topics below 60% with at "
            "least 2 tries — our choice): "
            + ("; ".join(f"{s} ({a:.0%} of {n})" for s, a, n in
                         zip(real.skill_name, real.next_acc, real.next_attempts)) or "none")
            + f"  \n**Our top 3 matched {hits} of these.** Overall results are in the "
              "🛠 Developer insight (in the sidebar).")


def reliability_note():
    if A["path_reliable"]:
        st.success("✓ Reliability OK: the score predictions beat simply guessing the average score.")
    else:
        st.error("⚠ LOW RELIABILITY — our score predictions are no better than guessing the average "
                 "score, so don't choose a path based on them.")


# --------------------------------------------------------------------------- home
def go(view: str):
    st.session_state["view"] = view
    st.session_state["nav"] = view


def home_view():
    st.subheader("Welcome! 👋")
    st.write("LearnAdapt AI looks at how a learner has been doing and suggests what to focus on "
             "next — always with a simple reason.")
    how_to("choose who you are below. You can switch at any time with the buttons at the top.")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("### 🎒 I'm a Student")
            st.write("See your strengths, the topics to practise next and tips that suit how you learn.")
            st.button("Start as a student", key="go_student", on_click=go, args=("Student",),
                      type="primary", width="stretch")
    with c2:
        with st.container(border=True):
            st.markdown("### 🧑‍🏫 I'm a Teacher")
            st.write("Pick a learner and get a short summary: strengths, gaps, top topics and "
                     "support ideas.")
            st.button("Start as a teacher", key="go_teacher", on_click=go, args=("Teacher",),
                      type="primary", width="stretch")
    st.caption("The learner profiles and the quiz histories come from two different public "
               "datasets about different learners, so they are shown separately.")
    st.caption("Looking for model details? Open the sidebar (› at the top left) → 🛠 Developer insight.")


# --------------------------------------------------------------------------- student
def student_view():
    st.subheader("🎒 Student")
    st.caption(NOT_A_JUDGEMENT)
    t1, t2 = st.tabs(["🌟 My strengths & tips", "📝 Topics to practise next"])
    with t1:
        how_to("type your learner ID. You'll see what you're doing well, what to grow next and "
               "tips that match your learning style.")
        sid = enter_own_id("Your learner ID", "s_sid", A["test_ids"], DEMO_A, as_int=False)
        if sid is not None:
            student_profile(kdf.set_index(dk.ID).loc[sid])
    with t2:
        how_to("type your quiz ID. You'll see the 3 topics most worth practising next and why. "
               "Practised something? Add your results at the bottom and the list updates.")
        data = module_b_data(friendly=True)
        if data is None:
            return
        att, lab, pop, clf_b = data
        uid = enter_own_id("Your quiz ID", "s_uid", att.user_id.unique(), DEMO_B, as_int=True)
        if uid is not None:
            student_topics(att, pop, clf_b, uid)


def student_profile(row):
    prof = dk.percentile_profile(kdf, row)
    with st.container(border=True):
        strength_gap_badges(prof, with_numbers=False)
    st.markdown(f"**💡 Tips for you** (you learn best the *{row['learning_style']}* way)")
    acts = dk.support_actions(prof, row["learning_style"])
    if acts:
        for a in acts:
            name, text = split_action(a)
            st.markdown(f"- **{name}:** {text}")
    else:
        st.markdown("- You're doing well across the board — keep up your current routine!")
    ps = path_scores(row)
    best = ps.loc[ps["predicted final score"].idxmax(), "path"]
    with st.container(border=True):
        st.markdown(f"**🧭 Suggested learning path:** :blue-badge[{best}]")
        if A["path_reliable"]:
            st.caption("Learners like you are predicted to get a higher final score on this path.")
        else:
            st.caption("Both paths lead to about the same predicted score for you, so either is a good choice."
			"Pick the one that suits you, and talk it over with your teacher.")


def student_topics(att, pop, clf_b, uid):
    _, hist, feats, scores, top, _ = learner_b(att, clf_b, uid)
    st.markdown(f"You've answered **{len(hist)}** questions so far and got "
                f"**{hist.correct.mean():.0%}** right. Here's where to focus next:")
    topic_cards(top, you=True)
    st.caption("Practising these a little each day usually helps more than one long session.")
    add_results(uid, feats, pop, "s", you=True)


# --------------------------------------------------------------------------- teacher
def teacher_view():
    st.subheader("🧑‍🏫 Teacher")
    st.caption(NOT_A_JUDGEMENT + " Learner profiles and quiz histories come from two different "
               "groups of learners, so pick a learner in each tab.")
    t1, t2, t3 = st.tabs(["📋 Learner summary", "📝 Quiz topics", "🔧 Try changes (what-if)"])
    with t1:
        how_to("pick a learner. The summary card shows strengths, gaps and support ideas. "
               "Open the sections below it for the model's guess and the reasons.")
        row = pick_learner_a("t_sid", "Learner")
        prof = dk.percentile_profile(kdf, row)
        X = as_X(row)
        with st.container(border=True):
            st.markdown(f"#### {row.name} · {row['education_level']} · learns best: "
                        f"{row['learning_style']}")
            strength_gap_badges(prof, with_numbers=True)
            st.markdown("**🛠 Suggested support** (simple rules we wrote, matched to learning style)")
            acts = dk.support_actions(prof, row["learning_style"])
            if acts:
                for a in acts:
                    name, text = split_action(a)
                    st.markdown(f"- **{name}:** {text}")
            else:
                st.markdown("- No big gaps — keep the current plan.")
        with st.expander("🔮 Model's guess at the course outcome"):
            p = outcome_proba(X)["probability"]
            st.dataframe(pd.DataFrame({"Outcome": p.index, "Chance": p.map("{:.0%}".format)}),
                         hide_index=True)
            st.write(f"Recorded outcome in the dataset: **{row[dk.TARGET_CLASS]}** "
                     f"(final score {row[dk.TARGET_REG]}).")
            st.caption("On this data the model is about as good as random guessing, so use this "
                       "only as a conversation starter — never as a label.")
        with st.expander("🧭 Which learning path?"):
            ps = path_scores(row)
            st.dataframe(pd.DataFrame({"Path": ps["path"],
                                       "Predicted final score": ps["predicted final score"].round(2)}),
                         hide_index=True)
            st.write(f"Path followed: {row['actual_path_followed']} · originally suggested: "
                     f"{row['recommended_path']}")
            reliability_note()
        with st.expander("❓ What pushed the model's guess towards 'Fail'"):
            ff = fail_factors(X)
            if len(ff):
                for i, f in enumerate(ff.feature, start=1):
                    st.markdown(f"{i}. {friendly_feature(f)}")
            else:
                st.write("Nothing in this learner's profile pushed the guess towards 'Fail'.")
            st.caption("Factors such as gender come from the dataset. They must never be used "
                       "to treat a learner differently.")
    with t2:
        how_to("pick a learner's quiz ID. The cards show the 3 topics most worth practising "
               "next and why. Add new results at the bottom to see the list update.")
        data = module_b_data(friendly=True)
        if data is not None:
            att, lab, pop, clf_b = data
            uid = pick_learner_b(att, "t_uid", "Learner quiz ID")
            ua, hist, feats, scores, top, added_key = learner_b(att, clf_b, uid)
            st.markdown(f"**{len(hist)}** questions answered so far "
                        f"({len(st.session_state[added_key])} added by you) · "
                        f"**{hist.correct.mean():.0%}** right overall · "
                        f"{int((~ua.is_history).sum())} later attempts kept aside for checking.")
            topic_cards(top, you=False)
            st.markdown("#### 📚 History per topic")
            history_table(feats, scores)
            evaluator_expander(uid, feats, top, lab, pop)
            add_results(uid, feats, pop, "t", you=False)
    with t3:
        how_to("start from a learner, change a few things (for example engagement) and see how "
               "the model's guess changes.")
        base = pick_learner_a("t_wsid", "Start from learner")
        vals = {}
        c1, c2 = st.columns(2)
        for i, c in enumerate(dk.NUMERIC):
            lo, hi = kdf[c].min(), kdf[c].max()
            with (c1 if i % 2 == 0 else c2):
                if pd.api.types.is_integer_dtype(kdf[c]):
                    vals[c] = st.slider(FRIENDLY[c], int(lo), int(hi), int(base[c]),
                                        key=f"tw_{c}_{base.name}")
                else:
                    vals[c] = st.slider(FRIENDLY[c], float(lo), float(hi), float(base[c]),
                                        key=f"tw_{c}_{base.name}")
        for i, c in enumerate(dk.CATEGORICAL):
            opts = sorted(kdf[c].unique())
            with (c1 if i % 2 == 0 else c2):
                vals[c] = st.selectbox(FRIENDLY[c], opts, index=opts.index(base[c]),
                                       key=f"tw_{c}_{base.name}")
        Xw = pd.DataFrame([vals])[dk.FEATURES]
        pw, pb = outcome_proba(Xw)["probability"], outcome_proba(as_X(base))["probability"]
        st.markdown("**Model's guess: before and after your changes**")
        st.dataframe(pd.DataFrame({"Outcome": pb.index, "Before": pb.map("{:.0%}".format),
                                   "After": pw.map("{:.0%}".format)}), hide_index=True)
        st.caption("This shows how the model reacts, not what would really happen — the model is "
                   "about as good as random guessing on this data.")


# --------------------------------------------------------------------------- developer insight
def developer_view():
    st.subheader("🛠 Developer insight")
    st.caption("Technical view: methods, metrics and baselines exactly as computed by `src/train.py`.")
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
        clf_b = load_model("assist_weak_logreg.joblib")
        derived, err = load_assist_derived() if clf_b is not None and B.get("available") else (None, None)
        if derived is None:
            st.warning("Module B data is not available. The app tries to download the corrected "
                       f"ASSISTments file from the [official link]({ASSIST_URL}) and build its tables; "
                       "locally you can also save it as `data/raw/skill_builder_data_corrected.csv` "
                       "and run `python src/train.py`.")
            if err:
                st.code(err)
        else:
            att, lab, pop = derived
            if att.user_id.nunique() != B["test_users"]:
                st.warning(f"Built {att.user_id.nunique()} test learners but training reported "
                           f"{B['test_users']}; the downloaded file may differ from the one used to train.")
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


# --------------------------------------------------------------------------- page
TOP_VIEWS = {"Home": "🏠 Home", "Student": "🎒 Student", "Teacher": "🧑‍🏫 Teacher"}
DEV_VIEW = "Developer insight"   # technical view, reached from the sidebar only


def on_top_nav():
    picked = st.session_state["nav"]
    if picked is None:  # clicking the selected button again: keep the current view
        if st.session_state["view"] in TOP_VIEWS:
            st.session_state["nav"] = st.session_state["view"]
    else:
        st.session_state["view"] = picked


def open_dev():
    st.session_state["view"] = DEV_VIEW
    st.session_state["nav"] = None


st.session_state.setdefault("view", "Home")
st.session_state.setdefault("nav", st.session_state["view"] if st.session_state["view"] in TOP_VIEWS else None)

with st.sidebar:
    st.markdown("### 🛠 Developer insight")
    st.caption("Methods, model results, simple comparison methods, data insights, responsible AI "
               "and limitations.")
    st.button("Open developer insight", key="go_dev", on_click=open_dev, width="stretch")

st.title("🎓 LearnAdapt AI")
st.caption("Personalised learning suggestions · AI Innovation Hackathon 2026 · Team මියැසිAI "
           "(Siththi Waseema, Shazany Harid)")
st.segmented_control("Choose a view", list(TOP_VIEWS), format_func=TOP_VIEWS.get, key="nav",
                     on_change=on_top_nav, label_visibility="collapsed")
{"Home": home_view, "Student": student_view, "Teacher": teacher_view,
 DEV_VIEW: developer_view}.get(st.session_state["view"], home_view)()
