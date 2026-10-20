# LearnAdapt AI — Personalised Learning Path Prediction

**AI Innovation Hackathon 2026 · Team මියැසිAI** — Siththi Waseema, Shazany Harid

LearnAdapt AI helps a teacher or learner see **what a learner is strong at, where they are
struggling, and what to study next**, with an explanation for every suggestion and an honest
comparison against simple baselines.

It has two **independent** modules built on two public datasets:

| | Module A — Learner profile & path | Module B — Quiz-by-quiz topic gaps |
|---|---|---|
| Data | Kaggle *AI-Powered Personalized Learning* (one row per learner) | ASSISTments 2009–2010 Skill Builder, corrected (one row per attempt) |
| Output | strengths/gaps, outcome probabilities, path comparison, support actions, "why" factors | top-3 skills to practise next, with reasons from the learner's own numbers |

> **The two datasets describe different learners.** They are never joined or merged by ID.

## Features (Streamlit tabs)

1. **Overview** — problem, solution, how it works, both datasets (source, licence, rows used).
2. **Learner Profile & Path (A)** — pick a *test* learner: profile, percentile strengths and gaps,
   predicted outcome probabilities next to the actual outcome, predicted final score on each of the
   two paths with a reliability badge, rule-based support actions adapted to learning style, and
   the top 3 logistic-regression factors raising the chance of "Fail".
3. **Quiz-by-Quiz Gaps (B)** — pick a *test* learner: per-skill history, top-3 next skills with
   reasons (e.g. "1/5 correct so far; recent accuracy 20%"), the two baseline lists, the real
   next-window weak spots, and an **add new results** form that appends attempts and recomputes.
4. **What-if (A)** — edit any input and watch the logistic-regression probabilities change.
5. **Model Evaluation** — every metric table, confusion matrices, and a plain-language reading.
6. **Data Insights** — outcome counts, average score by learning style / difficulty / outcome,
   correlations, quiz accuracy vs final score.
7. **Responsible AI & Limitations** — principles, every limitation we found, citations.

## Datasets

**A. Kaggle — AI-Powered Personalized Learning Dataset** (Ziya)
- <https://www.kaggle.com/datasets/ziya07/ai-powered-personalized-learning-dataset>
- Licence: **CC0: Public Domain**. Committed at `data/AI_Personalized_Learning.csv`.
- 1,000 rows × 18 columns, no missing values. The description says it *emulates* real-world
  e-learning scenarios.

**B. ASSISTments 2009–2010 Skill Builder — corrected file**
- <https://sites.google.com/site/assistmentsdata/home/2009-2010-assistment-data/skill-builder-data-2009-2010>
- We use the **corrected** file (`skill_builder_data_corrected.csv`, 346,860 rows), because the
  data page reports duplicated records in the original. **Not committed**: download it from the
  page above and save it as `data/raw/skill_builder_data_corrected.csv`.
- Please cite the data page URL above and:
  Feng, M., Heffernan, N.T., & Koedinger, K.R. (2009). Addressing the assessment challenge in an
  online system that tutors as it assesses. *User Modeling and User-Adapted Interaction*, 19(3), 243–266.

## Tech stack

Python 3.12 · pandas · NumPy · scikit-learn (with joblib) · Streamlit. Nothing else.

## Setup and run

```bash
git clone https://github.com/siththiwaseema/Miyasi-AI---learnadapt-ai
cd Miyasi-AI---learnadapt-ai
python -m venv .venv
# Windows: .venv\Scripts\activate      macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt

# Kaggle data is already in data/. For Module B, download the corrected ASSISTments file
# (link above) to data/raw/skill_builder_data_corrected.csv

python src/train.py        # trains everything, writes models/ and reports/results.md (~30 s)
streamlit run app.py
```

If the ASSISTments file is missing, `train.py` still trains Module A, and the Module B tab tells
you what to download.

## Project structure

```
app.py               Streamlit app (loads saved models; no retraining)
src/data_kaggle.py   load/split Kaggle data, percentiles, rule-based support actions
src/data_assist.py   load/clean ASSISTments, history/next split, features, labels
src/train.py         trains all models, saves models/ and reports/results.md + results.json
models/              saved joblib pipelines
reports/results.md   all metrics (generated)
docs/demo_script.md  5-minute demo
```

## Methodology

### Module A (Kaggle)
- **Inputs** (known before the outcome): age, gender, education_level, learning_style,
  previous_gpa, completed_modules, avg_time_per_module, engagement_score, distraction_events,
  quiz_accuracy, feedback_score, contextual_difficulty_level, recommended_path, actual_path_followed.
- **Never inputs** (leakage): path_efficiency_score, final_assessment_score, learning_outcome, student_id.
- **Split**: 80/20 train/test stratified by learning_outcome, random_state=42 (our choice).
- **Pipeline**: ColumnTransformer(StandardScaler on numeric, OneHotEncoder(handle_unknown="ignore")
  on categorical) + model. Preprocessing is fitted inside the pipeline, so CV folds never see test data.
- **A1 — learning_outcome**: DummyClassifier (most_frequent, stratified), LogisticRegression,
  RandomForestClassifier. Test accuracy, macro precision/recall/F1, confusion matrix, 5-fold
  stratified CV macro-F1 on the training set. "Clearly beats" = higher test macro-F1 **and** a CV
  gain larger than one CV standard deviation (our choice).
- **A2 — final_assessment_score**: DummyRegressor(mean), Ridge, RandomForestRegressor. MAE, RMSE,
  R², 5-fold CV R².
- **Path recommendation**: predict the score with actual_path_followed set to each path and suggest
  the higher one. Marked **low reliability** unless the best regressor beats the Dummy on test MAE
  **and** has CV R² > 0 (our choice).
- **Strengths/gaps**: percentile vs all 1,000 learners for previous_gpa, completed_modules,
  engagement_score, distraction_events (lower is better, so flipped), quiz_accuracy, feedback_score.
  Strength = top 25%, gap = bottom 25% (our choice).
- **Support actions**: one rule per gap, adapted to the learning styles found in the data
  (Visual, Auditory, Kinesthetic) — **rule-based, our choice**.
- **Explanation**: for logistic regression, coefficient × scaled value for the "Fail" class; top 3
  positive contributions.

### Module B (ASSISTments)
- **Coding checked in the file**: `correct` is 0/1; `original` is 0/1 (1 = main problem,
  0 = scaffolding sub-question).
- **Cleaning** (our choice): drop rows with no skill_name; keep main problems only (original = 1);
  keep skills with ≥ 300 attempts; keep learners with ≥ 20 attempts.
- **Attempt order** (our choice): `order_id`. The EduData notes call it non-chronological, so we
  checked it: within a learner and skill, the dataset's own `opportunity` counter increases with
  order_id in 99.8% of groups.
- **History / next window**: per learner, first 80% of attempts = history, last 20% = next window.
- **Label**: a skill is a weak spot if next-window accuracy < 60% with ≥ 2 attempts (our choice).
- **Features** per learner × skill, from history only: attempts, smoothed accuracy
  ((correct+1)/(attempts+2), our choice), accuracy over the last 5 attempts, attempts since the
  skill was last practised, learner's overall accuracy.
- **Split**: GroupShuffleSplit(test_size=0.25, random_state=42) on user_id — no learner in both sets.
- **Model**: StandardScaler + LogisticRegression(class_weight="balanced"), trained on learner × skill
  pairs that have ≥ 2 next-window attempts.
- **Baselines**: (1) popularity — skills most often weak across training learners; (2) each
  learner's own lowest history accuracy.
- **Metrics**: Precision@3 and Recall@3 on test learners with ≥ 1 real weak spot; classifier
  precision, recall, F1, accuracy, AUC at threshold 0.5.

## Results (copied from `reports/results.md`)

### Module A1 — learning_outcome (test set, 200 learners)

| model | accuracy | macro_precision | macro_recall | macro_f1 | cv_macro_f1_mean | cv_macro_f1_std |
|---|---|---|---|---|---|---|
| Dummy (most frequent) | 0.265 | 0.066 | 0.250 | 0.105 | 0.105 | 0.001 |
| Dummy (stratified) | 0.215 | 0.215 | 0.215 | 0.214 | 0.227 | 0.026 |
| Logistic Regression | 0.210 | 0.217 | 0.209 | 0.211 | 0.229 | 0.029 |
| Random Forest | 0.265 | 0.267 | 0.264 | 0.265 | 0.228 | 0.024 |

**Reading: neither model clearly beats the baselines.** Logistic regression CV gain +0.002
(CV std 0.029); random forest CV gain +0.001 (CV std 0.024). Random forest's higher test
macro-F1 (0.265) is not confirmed by cross-validation.

### Module A2 — final_assessment_score (test set)

| model | MAE | RMSE | R2 | cv_R2_mean | cv_R2_std |
|---|---|---|---|---|---|
| Dummy (mean) | 10.433 | 12.051 | -0.001 | -0.003 | 0.004 |
| Ridge | 10.537 | 12.166 | -0.020 | -0.045 | 0.030 |
| Random Forest | 10.796 | 12.495 | -0.076 | -0.027 | 0.017 |

**No model beats predicting the mean.** Path recommendations are therefore shown as
**LOW RELIABILITY** in the app.

### Module B — cleaning
- Raw rows: 346,860 (4,217 learners)
- Dropped, no skill_name: 72,270
- Dropped, scaffolding sub-questions (original = 0): 23,185
- Dropped, skills with < 300 attempts: 2,813
- Dropped, learners with < 20 attempts: 18,365
- Kept: 230,227 attempts, 2,021 learners, 75 skills; 1515 train / 506 test learners, no overlap.

### Module B — weak-spot classifier (1,162 test learner × skill pairs)

| precision | recall | f1 | accuracy | auc | test_positive_rate |
|---|---|---|---|---|---|
| 0.565 | 0.704 | 0.627 | 0.747 | 0.797 | 0.302 |

### Module B — top-3 recommendations (321 test learners with ≥ 1 real weak spot)

| method | precision@3 | recall@3 |
|---|---|---|
| Popularity baseline | 0.102 | 0.199 |
| Lowest own accuracy baseline | 0.187 | 0.372 |
| Logistic Regression (ours) | 0.188 | 0.369 |

**Reading:** our model is clearly better than popularity, but **about equal to the simple
"lowest own accuracy" baseline**. 248 of the 599 real weak spots are skills the learner never
practised in history, so no history-based method can rank them.

## Responsible AI
- **Support, not labels**: outputs are suggestions to discuss, never grades. Predicted "Fail"
  probabilities must not be used to restrict access to content.
- **Privacy**: both datasets use anonymous IDs; the app stores nothing, and results added in
  Module B live only in the browser session.
- **Explainable**: every prediction shows its factors (Module A) or the learner's own numbers
  (Module B). Support actions are plain rules, labelled as such.
- **Honest evaluation**: every model is compared with baselines; weak results are reported as weak.

## Limitations
- The Kaggle data *emulates* e-learning. learning_outcome barely relates to the inputs or to the
  score (mean final score ≈ 80 in every outcome class; e.g. STU019 scored 96 and is labelled Fail).
  Module A models do not clearly beat baselines.
- Path recommendations are low reliability. Each learner followed only one of the two paths, so
  the comparison is a model estimate, not a measured effect.
- The two datasets are different learners; Modules A and B cannot be combined for one person.
- ASSISTments marks questions where tutoring was used as incorrect, so "accuracy" mixes wrong
  answers and help-seeking.
- 72,270 ASSISTments rows had no skill name and were dropped. In the corrected file two-skill
  problems keep a combined `skill_id` (e.g. `1_13`) but only one `skill_name`; we use it as given.
- `order_id` is used as attempt order (checked against `opportunity`, 99.8% agreement), but it is
  described as non-chronological.
- Module B's model is roughly equal to the "lowest own accuracy" baseline.
- All thresholds are our choices, not validated standards.

## Stage 1 promises vs delivered

| Promise | Delivered | Where |
|---|---|---|
| Learner profile selection | ✅ | Tab 2 (Kaggle test learners), Tab 3 (ASSISTments test learners) |
| Learning history | ✅ | Tab 3: per-skill attempt history; Tab 2: profile |
| Strengths / gaps | ✅ | Tab 2: percentile-based (top/bottom 25%, our choice) |
| Recommendations with explanations | ✅ | Tab 2: path + rule-based actions + "why" factors; Tab 3: top-3 skills with reasons |
| Baseline comparison | ✅ | Dummy classifiers/regressor (A); popularity and lowest-accuracy (B) |
| Precision@K / Recall@K (Module B) | ✅ | K = 3; Tab 5 and results above |
| Classification metrics | ✅ | Accuracy, macro P/R/F1, confusion matrix, CV (A); P/R/F1/accuracy/AUC (B) |
| Train/test split without leakage | ✅ | Outcome columns excluded (A); history-only features + learner-grouped split (B) |
| Ethics | ✅ | Tab 7 and section above |
| Models beat baselines | ⚠ Partly | Module B beats popularity only; Module A does not beat baselines (reported honestly) |

## Team
**මියැසිAI** — Siththi Waseema, Shazany Harid
