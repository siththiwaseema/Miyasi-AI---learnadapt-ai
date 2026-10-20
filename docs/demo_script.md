# LearnAdapt AI — 5-minute demo script

All IDs and numbers below come from the running app (after `python src/train.py`, random_state=42).

## 0:00–0:40 · Overview tab
- "Learners move at different speeds, but courses give everyone the same sequence. LearnAdapt AI
  shows each learner's strengths, gaps and what to study next — with a reason every time."
- Point at the dataset table: Kaggle, 1,000 learners (CC0); ASSISTments corrected file,
  230,227 attempts from 2,021 learners after cleaning.
- Read the yellow box: **the two datasets are different learners and are never joined.**

## 0:40–1:50 · Learner Profile & Path (A)
- Select **STU935** (a test learner, Visual learner).
- Strengths and gaps: previous_gpa 3.5 → 75.9th percentile (**Strength**);
  completed_modules 3 → 6.4th percentile and engagement_score 68 → 21.8th percentile (**Gaps**).
- Support actions (rule-based, adapted to Visual style): "Set a weekly module target with small
  milestones; use diagrams, mind maps and colour-coded notes" and "Shorter, more interactive
  sessions to lift engagement; …".
- Predicted outcome: Fail 28.0%, Excellent 27.2%, Good 25.9%, Fair 18.8%. Actual: **Fail**, score 81.
  "Almost flat probabilities — the model is unsure, and the app says so."
- Path: M1→M2→M4 78.11 vs M1→M3→M5 78.08 → red **LOW RELIABILITY** badge. "Ridge does not beat
  predicting the average, so we don't pretend this 0.03-point difference means anything."
- Why: top factors raising "Fail" — learning_style = Visual (+0.159), avg_time_per_module (+0.081),
  gender = Male (+0.034).

## 1:50–3:00 · Quiz-by-Quiz Gaps (B)
- Select user **70733**: 453 history attempts, overall accuracy 74%, 114 attempts in the next window.
- Top-3 from the model:
  1. Solving for a variable — P(weak) 65% — "1/5 correct so far; recent accuracy 20%; last
     practised 90 attempts ago"
  2. Translations — 62% — "0/1 correct so far"
  3. Write Linear Equation from Graph — 62% — "0/1 correct so far"
- Compare with baselines: lowest own accuracy gives the same three skills; popularity gives
  Conversion of Fraction Decimals Percents, Equation Solving Two or Fewer Steps, Counting Methods.
- Be honest: this learner's real next-window weak spots were Calculations with Similar Figures,
  Circle Graph, Counting Methods, Order of Operations All, Pythagorean Theorem and Unit Rate — the
  model hit none of them here. Popularity got Counting Methods.
- Add new results: Skill = Solving for a variable, Correct, 3 attempts → **Add attempts**. It drops
  out of the top 3; Pythagorean Theorem (2/7, recent 40%) enters at 61%.

## 3:00–3:40 · What-if (A)
- Start from STU935. Raise engagement_score and completed_modules; watch the bar chart move.
- "This shows how the model reacts — not what would really happen, because the model is weak."

## 3:40–4:30 · Model Evaluation
- Module A1: Dummy (stratified) macro-F1 0.214; Logistic Regression 0.211; Random Forest 0.265 on
  test, but CV macro-F1 0.227 / 0.229 / 0.228 → **no clear win over baselines**.
- Module A2: Dummy MAE 10.433 beats Ridge 10.537 and Random Forest 10.796 → low reliability.
- Module B: AUC 0.797. Precision@3: popularity 0.102, lowest own accuracy 0.187, ours 0.188 —
  better than popularity, equal to the simple accuracy baseline.

## 4:30–5:00 · Data Insights + Responsible AI
- Data Insights: average final score by outcome is about 80 for every class — this is why
  Module A cannot predict outcomes.
- Responsible AI: support not labels, no data stored, explanations everywhere, every limitation
  listed, both datasets cited.
