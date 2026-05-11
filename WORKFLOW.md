# Workflow

How we work on ReccysMusic. Read this before starting any task.

## File map

```
plan/
  PLAN.md                     Index: score ladder + active phase + pointers
  CURRENT_BEST_ITERATION.md   Live snapshot of best retrieval pool + best rescore
  0N_<phase>.md               At most ONE active phase plan
  BASELINE_WALKTHROUGH.md     Static reference for the dataset/competition
  archive/                    Concluded phase plans (kept for history, not for context)

scripts/inference/            Active inference + measurement scripts
scripts/train/                Training scripts
scripts/archive/              Superseded scripts

cache/                        BM25 index, embeddings (gitignored)
data/                         Training pairs (gitignored)
models/                       Fine-tuned models (gitignored)
exp/                          Inference outputs + analysis (gitignored)
```

## Plan lifecycle

Every non-trivial change goes through a phase plan. One active plan at a time.

1. **Open a phase**. Create `plan/0N_<short_name>.md` using the structure below.
   Bump the phase index. Add a one-line "active" entry to `plan/PLAN.md` linking
   to it.
2. **Work**. Commit/push regularly. Record only the iterations that affected the
   conclusion -- not every parameter you tried.
3. **Update CURRENT_BEST_ITERATION.md** the moment dev nDCG@20 beats the previous
   best. Old config moves to the "Previous bests" section as a one-liner.
4. **Close the phase**. Once the phase is concluded (won, lost, or superseded),
   move the plan file into `plan/archive/` and replace the active entry in
   `plan/PLAN.md` with a one-line summary + archive link.

## Phase plan template

```md
# Plan: <phase name>

## Goal
One sentence. Measurable.

## Current State
One paragraph. What's true today.

## Hypothesis
What you expect to happen and why.

## Steps
Numbered. Each step is one commit-sized unit of work.

## Validation
What number proves it worked. nDCG@20 dev preferred.

## Result
Filled in when closing the phase. Three lines max: what changed, what it scored,
what to do next.
```

## CURRENT_BEST_ITERATION.md update rule

Only updates when **dev nDCG@20 (1000 sessions) strictly beats the prior entry**.
A new entry contains:

- Date.
- Retrieval pool description (BM25, expansion sources, sizes).
- Rescore method (weights, formula reference).
- Dev nDCG@20, Hit@20.
- Blind submission file (if any).
- One-line reason this beat the prior best.

The previous best is collapsed to a one-line "Previous bests" entry.

## Archive rule

Archive a plan when **any** of these is true:

- Its hypothesis was falsified and you don't intend to revisit it.
- It was superseded by a later phase plan that subsumes its goal.
- Its result is now reflected in `CURRENT_BEST_ITERATION.md` or the score ladder
  in `PLAN.md`, and no active step depends on its details.

When archiving, condense the phase to one line in `PLAN.md` (date, headline
result, link). Do not delete -- history is the iteration path.

## Score ladder

`plan/PLAN.md` keeps a single chronological ladder of attempted systems by dev
nDCG@20. Add to the ladder only when a config has been evaluated on the full
1000-session dev set. Do not record partial-session runs in the ladder -- they
belong inside the phase plan.

## Inference + eval commands

```bash
# Dev inference
python scripts/inference/run_inference_fusion_recall_expansion.py \
    --tid <run_id> [flags] \
    --out_dir exp/inference/devset

# Evaluate
python scripts/inference/evaluate_local.py --pred exp/inference/devset/<run_id>.json

# Blind submission
python scripts/inference/run_inference_blind_fusion.py --tid <blind_id> [flags]
python scripts/inference/generate_responses_blind.py \
    --pred exp/inference/blind_a/<blind_id>.json
```

## Reading order for a new session

1. `plan/PLAN.md` -- where we are.
2. `plan/CURRENT_BEST_ITERATION.md` -- the system to beat.
3. The single active `plan/0N_*.md` -- what's in progress.
4. `plan/BASELINE_WALKTHROUGH.md` only if you need dataset internals.

Skip archived plans unless you are explicitly digging into the iteration path.
