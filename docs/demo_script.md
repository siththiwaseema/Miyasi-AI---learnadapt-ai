# LearnAdapt AI — 5-minute demo script

All IDs and numbers below come from the running app (after `python src/train.py`, random_state=42).
Before the demo, open **Student → Topics to practise next** once: on a fresh deployment the first
load downloads the quiz data ("Loading quiz data, this takes about a minute the first time…").

## 0:00–0:30 · Home
- "Learners move at different speeds, but courses give everyone the same sequence. LearnAdapt AI
  shows each learner what they're good at, what to grow next and what to practise — always with a
  simple reason."
- Point at the three choices: **I'm a Student**, **I'm a Teacher**, **For Judges & Evaluators**.
- Read the note: profiles and quiz histories come from two different public datasets about
  different learners, so they're shown separately.

## 0:30–1:45 · 🎒 Student
- Click **Start as a student**. Point at the note: "These are suggestions to discuss with your
  teacher, not a judgement."
- Students can't browse a list of learners — they type their own ID. Show that a wrong ID gives
  "We couldn't find that ID. Please check it, or ask your teacher."
- **My strengths & tips**: type **STU935** and press Enter:
  - Strength: green badge **Previous GPA**.
  - Areas to grow: orange badges **Modules completed** and **Engagement**.
  - Tips (Visual learner): "Set a weekly module target with small milestones; use diagrams, mind
    maps and colour-coded notes" and "Shorter, more interactive sessions to lift engagement; …".
  - "Which learning path suits you? Our tool can't tell this reliably yet" — we don't pretend.
- **Topics to practise next**: type quiz ID **70733** and press Enter: 453 questions, 74% right.
  - Priority 1 · Solving for a variable — 65% chance this topic needs more practice —
    "You got 1 of 5 right so far · recent accuracy 20% · last practised 90 questions ago".
  - Priority 2 · Translations — 62%. Priority 3 · Write Linear Equation from Graph — 62%.
  - Add results: Topic = Solving for a variable, Right, 3 questions → **Add results**. It drops off
    the list; Pythagorean Theorem enters ("2 of 7 right so far · recent accuracy 40%"), and the
    three cards now show 61% each.
  - Click **Undo everything I added**.

## 1:45–3:15 · 🧑‍🏫 Teacher
- Switch to **Teacher** at the top.
- Teachers pick from a list of learners (STU935 is pre-selected for the demo).
- **Learner summary**, STU935 (UG, learns best: Visual):
  - Strength: Previous GPA — higher than about 76% of learners.
  - Areas to grow: Modules completed — about 6%; Engagement — about 22%.
  - Suggested support: the two rule-based actions above.
  - Open **Model's guess**: Fail 28%, Excellent 27%, Good 26%, Fair 19%; recorded outcome Fail
    (final score 81). "Almost flat — the caption says it's about as good as random guessing, so
    it's a conversation starter, never a label."
  - Open **Which learning path?**: M1→M2→M4 78.11 vs M1→M3→M5 78.08, with the red
    **LOW RELIABILITY** box: no better than guessing the average score.
  - Open **What pushed the model's guess towards 'Fail'**: Learning style: Visual; Average time
    per module; Gender: Male — and the caution that factors like gender must never be used to
    treat a learner differently.
- **Quiz topics**, 70733: the same three priority cards, then the **History per topic** table
  (Topic, Attempts, Correct, Recent accuracy, Last practised).
  - Open **For evaluators: how good was this prediction?** Be honest: the later attempts showed
    trouble with Calculations with Similar Figures, Circle Graph, Counting Methods, Order of
    Operations All, Pythagorean Theorem and Unit Rate — **our top 3 matched 0 of these**.
- **Try changes (what-if)**: raise Engagement or Modules completed and compare the Before/After
  columns.

## 3:15–4:30 · ⚖️ Judges & Evaluators
- **Model Evaluation**:
  - Outcome classification: Dummy (stratified) macro-F1 0.214; Logistic Regression 0.211;
    Random Forest 0.265 on test, but CV macro-F1 0.227 / 0.229 / 0.228 → no clear win over baselines.
  - Score regression: Dummy MAE 10.433 beats Ridge 10.537 and Random Forest 10.796 → low reliability.
  - Quiz topics: AUC 0.797. Precision@3: popularity 0.102, lowest own accuracy 0.187, ours 0.188 —
    better than popularity, equal to the simple accuracy baseline.
- **Data Insights**: average final score is about 80 for every outcome class — this is why
  outcomes can't be predicted from this dataset.

## 4:30–5:00 · Responsible AI & Limitations
- Support not labels, no data stored, explanations everywhere, every limitation listed, both
  datasets cited, and the ASSISTments data is fetched from the official link rather than
  redistributed.
