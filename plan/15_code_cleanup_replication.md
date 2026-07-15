# Plan: Code cleanup for replicable 0.49 submission

## Goal

Main branch contains exactly the code needed to replicate the Blind B v2
system (composite 0.49) plus its training pipeline. Everything else moves to
`scripts/archive/`. Every claim in `paper_wikis/paper_draft.md` is verified
against the code; discrepancies are flagged for user review, not silently
fixed. No behavioral code changes.

## Current State

- Paper: `paper_wikis/paper_draft.md` (committed 1ec4c9e), account in
  `wiki/PAPER_SYSTEM_ACCOUNT.md`.
- v2 pipeline command recorded in `wiki/BLIND_B.md` (run_inference_fusion_
  recall_expansion.py with --anchor_v8d, s3cap LTR, SASRec buckets).
- `scripts/` has ~80 files across inference/train/analysis/build/kaggle;
  many belong to abandoned lines (cross-encoders, LLM rerankers, semantic-ID
  LLM, v3/v4b/v5 submission variants).

## Assumptions

- v2 = v1 track list (blind_b_v8d_s3cap_v1.json) + Opus pivot-aware responses,
  packaged as a zip. Response generation was manual (Claude subagents), so no
  response-generation script is load-bearing for replication of the ids.
- "Replicate" means: rebuild all artifacts (BM25 index, TT v8d data+model+index,
  cooccur table, CF-BPR, Qwen/CLAP indexes, semantic IDs, SASRec, LTR features
  +model) and rerun the v2 inference command.
- Archive = `git mv` into `scripts/archive/v5_pre_submission/` (new bucket),
  keeping history. No deletions of tracked code.

## Files To Read

- `scripts/inference/run_inference_fusion_recall_expansion.py` (v2 entry point,
  FEATURE_COLS, anchor builder, GPA logic, seed filtering)
- `scripts/train/build_twotower_v8d_data.py`, `train_twotower_lora.py`,
  `build_twotower_index.py`
- `scripts/train/train_ltr_lightgbm.py`
- `scripts/train/build_bm25_v2.py`, `build_cooccur_table.py`,
  `build_fusion_index.py`, `run_rqvae_talkplay.py`, `run_sasrac/run_sasrec_talkplay.py`,
  `build_sasrec_semantic_data.py`
- `scripts/inference/precompute_embeddings.py`, `evaluate_local.py`
- Whatever script packaged v2 (search exp/inference/blind_b + git log)

## Files To Modify

- Moves only: non-v2 scripts -> `scripts/archive/v5_pre_submission/`
- `paper_wikis/paper_draft.md` / `wiki/PAPER_SYSTEM_ACCOUNT.md` only after
  user reviews flagged discrepancies
- New: `REPLICATION.md` (or extend README) mapping paper section -> script ->
  artifact (if user wants it)

## Steps

1. Trace the v2 dependency graph: entry script imports, artifact paths in the
   v2 command, and the training scripts that produce each artifact.
2. Verify each paper claim against code (anchor format, LoRA config, epochs,
   batch, lr, pair count, hard-neg priority, GPA alignment, 67 features,
   feature groups, budgets per source, regex banks, dump session count).
3. Produce two lists: KEEP (with reason) and ARCHIVE (with reason), plus a
   DISCREPANCY list. Present to user before moving anything.
4. After user review: git mv archived scripts, remove __pycache__/logs from
   working tree (gitignore), commit.

## Validation

- `python -c "import ..."`-level sanity: kept inference script still runs
  --help after moves (no broken relative imports).
- Every artifact path in the v2 command maps to a KEEP script that builds it.
- git status clean; main pushed.

## Risks

- Some kept scripts may import helpers from files slated for archive; check
  imports before moving.
- The v2 response step is manual; paper says "decoupled" so this is fine, but
  the replication README must say so explicitly.

## Notes

- CLAUDE.md: no behavioral code edits. Discrepancies go to the user first.

## Status (2026-07-14)

Done:
- Audit complete: 9 discrepancies (D1-D9). D2-D6/D8 writeup fixes applied to
  paper_draft.md, CURRENT_BEST_ITERATION.md, wiki/PAPER_SYSTEM_ACCOUNT.md.
- D1 resolution: third_party/ gitignored; clone documented in README
  (eugeneyan/semantic-ids-llm @ b730d48) as Stage 2 prerequisite.
- D9 (new): pipeline hardcodes BM25_CACHE = cache/bm25/track_metadata (v1,
  built lazily by archived v0 script). build_bm25_v2.py builds _v2 which the
  0.49 pipeline never reads -> build_bm25_v2.py moved to ARCHIVE list; README
  documents the v1 index build inline.
- .gitignore: added third_party/, logs/, __pycache__/.
- README.md: full replication guide added (env, stage 0-3 builds, exact
  Blind B command, manual response/packaging step, dev eval, no-clone variant).
- Both scripts/*/WALKTHROUGH.md describe the archived v3/v13 fusion era ->
  move to archive with their scripts; README is the live replication doc.

Pending (blocked on intermittent classifier outage for mutating Bash):
- git mv batches to scripts/archive/v5_pre_submission/{inference,train,analysis}
  (+ kaggle/, build/, both WALKTHROUGH.md, build_bm25_v2.py).
- cp wiki/PAPER_SYSTEM_ACCOUNT.md -> paper_wikis/PAPER_SYSTEM_ACCOUNT.md.
- Regenerate paper_wikis/paper.docx (scripts/analysis/build_paper_docx.py).
- Sanity: entry script --help after moves; git status clean; commit + push.
