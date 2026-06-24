# Blind B Wiki — Exact Recall + Rescore Architecture

Purpose: single source of truth for what we run on Blind B, why, and what we have
already tried, so we never burn one of the 3 Blind B submissions on a config we
already know about.

Blind B = generalization set. 80 sessions, 1 prediction each (next music turn).
Removed vs Blind A: `conversation_goal`, `goal_progress_assessments`, `thoughts`
are all null/empty. 40/80 are cold-start (no `user_id`, no `user_profile`).
Histories are long: turn positions 1-8 spread 10 sessions each (up to 7 prior
music turns). `turn_number` is encoded as a STRING in the dataset.

## Composite score (official, confirmed against v21/v22 to 4 decimals)

```
Score = 0.50*nDCG@20 + 0.10*CatalogDiversity + 0.10*LexicalDiversity + 0.30*JudgeNorm
JudgeNorm = (judge_1to5 - 1) / 4
```

Levers by weight: nDCG@20 (0.50) >> Judge (0.30) > CatDiv (0.10) = LexDiv (0.10).
- nDCG 0.40 -> 0.45 = +0.025 composite.
- Judge 3.85 -> 4.7 = +0.064 composite.

## What does NOT change between Blind A and Blind B

Our retrieval is robust to the missing fields. Measured on golden-200 with
`--simulate_blindb` (strips goal/gpa/thoughts, 50% cold):

| Config | nDCG@20 (golden-200 sim) |
|---|---|
| Full info upper bound (v20, real goal+gpa) | 0.1841 |
| **Blind B sim (no goal/gpa/thoughts, infer reactions)** | **0.1864** |

Losing goal/gpa/thoughts does NOT hurt nDCG. `--rejection_drop_threshold` (needs
real gpa) was slightly counterproductive, and the goal text is sometimes
misleading. So the Blind B config is essentially the v22 retrieval with reaction
labels reconstructed from user text.

## RECALL architecture (candidate generation) — EXACT

Script: `scripts/inference/run_inference_fusion_recall_expansion.py`
TT model: `models/twotower_v8d/final` (multilingual-e5-base, LoRA r=32).
TT index: `cache/twotower_v8d/` (47,071 tracks x 768d, L2-normalized).

Fused candidate sources per turn:

| Source | Param | Notes |
|---|---|---|
| BM25 (Okapi) | top-500 | query = latest user msg + (empty goal) + culture; `--bm25_missing_floor 0.05` |
| TT v8d ANN | `--tt_pool 2000` | role-tagged anchor `[PROFILE] [GOAL(empty)] [Ti ... REACTION] [NOW]` |
| TT last-NN | `--last_nn_k 100 --last_nn_src 2` | neighbors of last 2 history tracks |
| Artist expansion | `--artist_expansion` | discographies of retrieved artists |
| Qwen3-Embedding-0.6B | `--qwen_pool 500` | precomputed metadata embeddings |
| CF-BPR | `--cf_pool 200` | warm users only; cold-start (40) get none |
| Session-mean NN | `--session_mean_k 100` | mean of history TT embeddings |
| Co-occurrence (leakfree) | `--cooccur_ks 300,150,50` | `cache/cooccur/next_song_leakfree_6k_excluded.npz` |
| Stage 3 SASRec buckets | `--sasrec_max_cands 300 --sasrec_top_k_l0 3` | `models/sasrec/sasrec_runC2_L2C64/best_model.pth`, semantic IDs `cache/semantic_ids/runC2_attributes_L2C64` |

Pool recall on Blind-B-sim golden-200: ~89% (gold in pool).

Reaction reconstruction (replaces missing gpa): `--infer_progress_labels` rule-
based classifier turns user follow-ups into MOVES/DOES_NOT, which fills the
`REACTION:` slot in the anchor and fires H1 (filter rejected tracks from seeds).
`--goal_substitute_positive` fills the empty `[GOAL]` slot with the most recent
positive track. We do NOT use `--rejection_drop_threshold` (needs real gpa, and
it hurt).

## RESCORE architecture (ranking) — EXACT

LTR: `models/ltr/ltr_v8d_s3cap_nl31_lr0p08.txt` (LightGBM LambdaMART, 67 features,
CV nDCG@20 0.3144). Same model as Blind A v20/v22. Top-20 by LTR score.
`--topk 20` => exactly 20 ids per record (verified, no duplicates).

gpa-derived LTR features (sim_to_pos_hist_mean, n_rejected_in_history, etc.) are
computed from the INFERRED labels on Blind B (noisier than real gpa but present).

## RESPONSE generation (judge lever, weight 0.30)

Judge = Gemini, two disclosed text-only axes: Personalization + Explanation
Quality, 1-5 each. Prompt undisclosed (not in the `music-crs-evaluator` repo,
which only ships recsys+diversity metrics).

Key finding (Blind A natural experiment): response RICHNESS drives the judge, not
the model. v07 (gemma, judge 4.4) vs v21 (gemma, judge 3.95) used the same
model+prompt; v07 won purely by being longer/richer. Terseness constraints hurt.

Blind B responses: Claude Sonnet subagent on `/tmp/blindb_enriched.json` (full
history + profile-when-present per session). Spec: 4-5 substantive sentences,
2-3 concrete metadata evidence points (tag/album/year tied to the request),
mandatory continuity callback to a prior played track when history exists,
synthesizing closer. COLD sessions: anchor personalization to the request +
history only (NO profile/demographic references — there is none). All 80
non-fallback; 0 exclamation marks.

## Blind A judge scores (for calibration)

| Sub | nDCG | Judge | Composite | Notes |
|---|---|---|---|---|
| v21 (gemma rich) | 0.3997 | 3.95 | 0.4935 | |
| v22 (Claude terse) | 0.4060 | 3.85 | 0.4933 | constraints made it lean |
| v23 (Claude rich) | 0.4060 | pending | pending | v07-style richness |

## EXACT command — Blind B baseline (v1, what we ship)

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python scripts/inference/run_inference_fusion_recall_expansion.py \
  --tid blind_b_v8d_s3cap_v1 \
  --dataset talkpl-ai/TalkPlayData-Challenge-Blind-B \
  --blind_mode --out_dir exp/inference/blind_b \
  --topk 20 \
  --tt_model models/twotower_v8d/final --tt_index cache/twotower_v8d \
  --anchor_v8d \
  --tt_pool 2000 --artist_expansion --last_nn_k 100 --last_nn_src 2 \
  --bm25_missing_floor 0.05 \
  --qwen_pool 500 --cf_pool 200 --session_mean_k 100 \
  --cooccur_table cache/cooccur/next_song_leakfree_6k_excluded.npz --cooccur_ks 300,150,50 \
  --infer_progress_labels --goal_substitute_positive \
  --semantic_ids_dir cache/semantic_ids/runC2_attributes_L2C64 \
  --sasrec_ckpt models/sasrec/sasrec_runC2_L2C64/best_model.pth \
  --sasrec_top_k_l0 3 --sasrec_max_cands 300 \
  --ltr_model models/ltr/ltr_v8d_s3cap_nl31_lr0p08.txt
```
Then merge Claude responses -> `prediction.json`, `zip submission.zip prediction.json`.

## Submission ledger (do not resubmit the same)

| Ver | Retrieval | Post-proc | Responses | sim nDCG@20 | Status |
|---|---|---|---|---|---|
| **v1** | v8d + Stage3 + infer_labels + goal_sub | none | Claude rich | 0.1864 | packaged |
| **v2** | same top-20 as v1 | none | Opus pivot-aware | 0.1864 (tracks==v1) | RECOMMENDED — strict improvement over v1 |
| **v3** | same as v1, emit 25 | Opus prune 5 sessions (15 explicit by-name drops), backfill 21-25 | Opus pivot-aware | unmeasured (nDCG-only risk) | optional gamble |

(Blind B has 3 total submissions. Update with real scores after upload.)

### v2 (pivot-aware responses) — the safe judge lever
Idea: the judge scores response TEXT only (independent of tracks). On the 14
turns where the user signals a pivot ("break away from X", "something different",
"too serious"), the v1 response risked sounding like "here's more of the same".
v2 keeps the v1 top-20 tracks UNCHANGED (so nDCG is identical) but regenerates
responses with Opus so pivot turns explicitly acknowledge the shift and honestly
frame the pick (a genuinely-different artist -> name the contrast; a still-rejected
artist forced by the fixed list -> admit it and frame as the closest bridge).
Zero nDCG risk, pure judge/personalisation upside. LexDiv 0.745, 80 distinct
openers, hand-written (NOT templated — a templated first attempt scored LexDiv
0.26 and was discarded). Pivot detector: `_detect_pivot` regex on latest message.

### v3 (LLM track pruning) — nDCG-only gamble, optional
Over-generate 25 (`--emit_topk 25`), Opus dry-run audits each 25-list, drops ONLY
tracks whose artist/category the user explicitly rejected BY NAME, backfills from
ranks 21-25. 5 sessions touched: 04135d8a (Deltron ban), 60f60edd (rain-sound
recordings vs electronic ambient), 6c90a029 (Morcheeba trip-hop vs classic rock),
6e2eb7e6 (proto-punk ban), 5870e73f (Deltron, low-confidence). NOTE: the judge is
text-only so pruning has NO judge benefit; it only moves nDCG (+ a little catdiv).
Helps iff a dropped track was a non-gold ranked above the gold; hurts iff a drop
removes the gold. For EXPLICIT by-name bans the gold-is-rejected rate is well below
the 40% soft-pivot aggregate, so these specific drops are probably safe — but it is
UNMEASURABLE on the blind set. Decisions: `/tmp/blindb_prune_decisions.json`.

## Tried and REJECTED (do not repeat)

- `--max_per_artist 4` (artist-diversity cap): sim 0.1864 -> 0.1433. TalkPlay
  sessions are often single-artist deep-dives where the GOLD is another track by
  the "flooding" artist; capping evicts the gold. Helps CatalogDiv but the nDCG
  loss (weight 0.50) dwarfs it. Code retained behind `--max_per_artist` (default
  off). DO NOT enable.
- `--boost_named_track` (float exact title+artist match to rank 1): sim
  0.1864 -> 0.1768. Too aggressive — it boosts tracks the user merely DISCUSSED
  (past plays they react to), not the next track. Would need imperative-request
  detection ("play X" vs "I liked X") to be safe. Code retained (default off).
  DO NOT enable without a restricted, re-validated matcher.
- `--pivot_suppress` (demote history artists when user shows pivot/rejection
  intent): sim 0.1864 -> 0.1656. Conditional and clean (fires on 16% of turns,
  ZERO effect on non-fired turns), but on FIRED turns nDCG collapses 0.1766 ->
  0.0444 (-0.13). ROOT CAUSE, measured: on pivot-fired turns the ground-truth
  next track is STILL by a history artist 40% of the time. TalkPlay's logged
  "next track" stays in the same artist neighborhood even when the user verbally
  asks to move on, so suppressing history artists evicts the gold. Code retained
  (default off). DO NOT enable.
- Stage 3E centroid bucket expansion + LTR retrain: golden-200 0.1841 -> 0.1809.
  Bucket-precision ceiling; see memory feedback. DO NOT enable (`--centroid_top_k_l0` default 0).

## The core lesson (why the external audit fixes do not raise the score)

The external audit (`wiki/BLIND_B_v1_audit_external.md`) and fix proposal
(`wiki/BLIND_B_v1_fix_proposal.md`) optimise HUMAN-PERCEIVED relevance: "the user
said stop, don't give more of that artist." Every artist-suppression variant we
implemented and validated (cap, named-track boost, pivot-conditional suppress)
REGRESSES nDCG@20 — the metric weighted 0.50 — because:

1. nDCG rewards hitting the single LOGGED next track. In TalkPlay that track sits
   in the same artist/cluster as the history ~40-75% of the time, EVEN on turns
   where the user verbally pivots. The LTR already learned this; hard post-hoc
   rules override it and lose.
2. The LLM judge (weight 0.30) scores the RESPONSE TEXT ONLY, independent of the
   track list. So the place to honour the user's pivot is the WRITTEN RESPONSE
   ("I hear you want to move away from X — here's something with a different
   feel..."), NOT the track ranking. This raises judge/personalisation without
   costing nDCG.

Conclusion for Blind B: keep the LTR track ranking untouched (v1). Put
pivot/rejection awareness into the response text (judge lever), never the ids.
This is now validated 3 independent ways (cap, boost, pivot_suppress all regress).

## Known limitations (audit `/tmp/blindb_audit_report.md`, future work)

The cold-start ranking audit (Claude, 80 sessions) found real human-relevance
issues that our nDCG-driven pipeline does NOT fix because the dataset rewards
artist-focus:
1. Explicit multi-turn artist REJECTION ignored — system floods with the rejected
   artist (Gang Starr, Kalkbrenner, nature sounds, Deltron). Needs reliable
   negative-signal suppression, NOT a blind artist cap.
2. Quiz turns ("just play 'Versace (Remix)' by Migos") — exact track not always
   rank 1. Needs an imperative-request-scoped title+artist matcher (the naive
   boost hurt; see rejected list).
3. Vague cold turn-1 emotional requests -> genre-cluster artifacts.
These are deferred: each needs a careful, sim-validated implementation, and the
naive versions regressed nDCG.
