# /sync-plan

Keep the project plans and memory in sync with the actual codebase state.
Run this after any training job completes or before starting a new experiment.

---

## What this does

1. Check what training processes are running
2. Find inference result files not yet in PLAN.md
3. Evaluate any unevaluated results
4. Update PLAN.md score table and next steps
5. Update the active phase plan doc with new results
6. Update memory/project_state.md
7. Move any newly obsolete scripts to scripts/archive/

---

## Step 1: Check running jobs

```bash
ps aux | grep -E "train_twotower|train_crossencoder|run_inference" | grep -v grep
```

For each running job, check its log file in /tmp/:
- Training: report current step, eval loss, ETA
- Inference: report sessions completed

Logs follow pattern `/tmp/<job_name>.log`. Check last 10 lines.

---

## Step 2: Find unevaluated result files

List all JSON files in exp/inference/devset/ and exp/inference/blind_a/.

For each devset file, check if its score appears in plan/PLAN.md.
Any file NOT listed in the score table is unevaluated.

Run evaluate_local.py on each unevaluated devset file:
```bash
python scripts/evaluate_local.py --pred exp/inference/devset/<file>.json
```

Record: nDCG@20, Hit@20, session count.

---

## Step 3: Update plan/PLAN.md

Update the score ladder and experiment timeline with any new results.
Keep the score ladder sorted descending by nDCG@20.
Mark the current best with a bold arrow.

Update the "Pending / Next Steps" section to reflect what actually needs to happen next
based on current state (what just finished, what is running, what hasn't been tried).

---

## Step 4: Update the active phase plan doc

Find the active phase doc in plan/ (the one marked "IN PROGRESS" or most recently created).
Add a Results table row for each new experiment.
Update the Decision Tree or Next Steps section.

If a phase just completed (all planned experiments done), mark it Complete and
start a new plan doc for the next phase if one doesn't exist yet.

New plan docs follow naming: plan/04_<name>.md, plan/05_<name>.md, etc.

---

## Step 5: Update memory

Update /Users/stealthmacmini/.claude/projects/-Users-stealthmacmini-Desktop-ReccysMusic/memory/project_state.md:
- Change "Best System" if a new result beats 0.1418
- Update "Active Training" section with current job status
- Add new rows to the Experiment Results table

---

## Step 6: Clean up scripts

Check if any scripts in scripts/ are now superseded by newer scripts.
A script is superseded if:
- A newer version exists (e.g. run_inference_twotower.py superseded by run_inference_twotower_v3.py)
- The approach it implements was abandoned per the plan docs
- It is a one-time diagnostic (inspect_, check_, test_)

Move superseded scripts to scripts/archive/. Do not delete.

---

## Output format

End with a brief status report:

```
SYNC COMPLETE — <date>

Running: <job> at step X/Y (ETA: ...)
New results: <n> files evaluated
Best: nDCG@20=<score> (<model>)
Plan updated: <which docs changed>
Scripts archived: <n> (total active: <n>)
```
