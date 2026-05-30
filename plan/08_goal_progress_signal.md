# Plan: Leverage goal_progress_assessments at inference

## Goal

Use the per-turn `goal_progress_assessment` labels (MOVES_TOWARD_GOAL,
DOES_NOT_MOVE_TOWARD_GOAL, None) that are already present on prior turns of each
dev session to (a) avoid seeding retrieval from tracks the user has just
rejected, (b) modulate the static `goal` field in each query so it does not
mislead after a rejection streak, and (c) give the LTR ranker explicit
positive/negative-history similarity features.

Target: beat current best 0.1646 dev nDCG@20.

## Current State

- Inference (`scripts/inference/run_inference_fusion_recall_expansion.py`) reads
  session items but never touches `goal_progress_assessments`.
- NN seeds, TT session-mean, and the "last 2 track names" BM25 suffix are all
  drawn from raw `music_history[-k:]`, regardless of whether those prior tracks
  were rated negatively.
- Training builders already consume the column
  (`build_twotower_v6_data.py:118`, `build_crossencoder_v2_data.py:112`) to
  weight positives, so the field is known-good in `train`.

Observed example session (8 turns): t1=None, t2=MOVES_TOWARD, t3..t8 all
DOES_NOT_MOVE_TOWARD_GOAL. At turn 8 the current code seeds NN from t7 and t6,
both rejected. Every neighbor pulled in is similar to a track the user just
rejected. This pattern is the failure mode this plan targets.

## Hypothesis

H1: NN/session-mean/BM25-suffix seeded from a track rated DOES_NOT_MOVE pulls
candidates the user is more likely to reject. Filtering rejected tracks out of
the seed set should raise recall on turns deep in a rejection streak.

H2: Even when a candidate survives retrieval, similarity to prior rejected
tracks is informative for ranking. Adding `sim_to_negative_hist_mean` and
`artist_in_rejected_set` features should let LambdaMART downweight them.

H3: The static `goal` ("listener_goal") becomes misleading after a long
rejection streak — it described the user's pre-session expectation, not their
revealed preference. Two cheap, no-LLM interventions on the goal slot:

- **H3a (goal drop):** drop the static `goal` from all three queries when
  `n_consecutive_rejections >= rejection_drop_threshold` (default 3). Let
  `latest_user` and positive-history tracks carry the intent.
- **H3b (positive-anchor substitution):** when at least one prior
  MOVES_TOWARD_GOAL track exists, substitute its `name+artist` into the goal
  slot of all three queries in place of (or in addition to) the static goal
  string.

Explicitly NOT in scope: free-text query rewriting that encodes "avoid X" as
words. Bi-encoders handle negation poorly; rejection signal stays in the
filter/feature channel (H1/H2), not the text channel.

H1 lifts recall; H2 lifts ranking conditional on recall; H3 sharpens the
query semantics. They stack but partially overlap (H1 already removes
rejected tracks from the per-query history slices, so H3 mainly affects
the goal slot).

## Assumptions

- The column is present on the dev split (verify in Step 1).
- `None` on turn 1 is structural (no prior gold to assess). `None` on later
  turns is treated as neutral — keep, do not filter.
- DOES_NOT_MOVE rate is non-trivial across dev (verify in Step 1; if <5% of
  prior-turn labels, expected lift is small and we may stop at H2).
- The column is present on the blind/test split too. Verify before
  porting; if absent, this remains dev-only.

## Files To Read

- `scripts/inference/run_inference_fusion_recall_expansion.py` (lines around
  346-510 for history construction, NN seeds, session mean, BM25 query).
- `scripts/train/train_ltr_lightgbm.py` (feature loading shape, group ids).
- `plan/07_ranking_calibration.md` (FEATURE_COLS list, current LTR schema).

## Files To Modify

- `scripts/inference/run_inference_fusion_recall_expansion.py`
  - Build `progress_by_turn` lookup from `item.get("goal_progress_assessments")`.
  - Maintain parallel `music_history_labels: list[str|None]` alongside
    `music_history`.
  - Add `--use_goal_progress` master flag (default off, for clean ablation).
  - **H1 — seed filtering.** Replace seed sources for NN, session-mean, and
    BM25 last-N-track suffix and TT/Qwen/CLAP last-2-track suffix with a
    `positive_history()` helper that filters out `DOES_NOT_MOVE_TOWARD_GOAL`,
    with raw-history fallback when empty.
  - **H3a — goal drop.** Add `--rejection_drop_threshold` (int, default off
    when 0). When `n_consecutive_rejections` from the tail of
    `music_history_labels` is `>= threshold`, omit `goal` from the
    `tt_parts`, `bm25_parts`, and `sem_parts` constructions for that turn.
  - **H3b — positive-anchor substitution.** Add `--goal_substitute_positive`
    flag. When set and a most-recent MOVES_TOWARD_GOAL track exists,
    substitute its `name+artist` string into the goal slot of all three
    queries; otherwise keep the original `goal`. Mutually compatible with
    H3a: substitution takes precedence; if no positive track exists and the
    rejection streak threshold is met, drop the slot.
  - **H2 — LTR features.** Add to `FEATURE_COLS` and per-candidate
    computation:
    - `sim_to_pos_hist_mean` (TT cosine to mean of MOVES_TOWARD prior tracks)
    - `sim_to_neg_hist_mean` (TT cosine to mean of DOES_NOT_MOVE prior tracks)
    - `artist_in_rejected_set` (binary)
    - `n_rejected_in_history` (int, clipped at e.g. 10)
  - When `positive_history` is empty: zero out `sim_to_pos_hist_mean`; keep
    raw last-track seeding so retrieval still has anchors.

- `scripts/train/train_ltr_lightgbm.py`
  - No code change; just retrain on new feature matrix (one more column count).
  - Confirm group/label loading still works with extended feature width.

## Steps

1. **Verify data presence and distribution.**
   - dev: print fraction of sessions with non-empty `goal_progress_assessments`,
     fraction of prior-turn labels by class, count of sessions with
     `>=3 consecutive DOES_NOT_MOVE` turns.
   - blind/test (a sample if full split unavailable): confirm field present
     and non-empty.
   - Gate: if dev DOES_NOT_MOVE rate <2% of evaluation-relevant prior turns,
     stop and reconsider; skip H1, only do H2.

2. **Add `--use_goal_progress` plumbing without changing behavior.**
   - Build the lookup, attach labels to history, but do not change seeds yet.
   - Run dev with flag off and flag on (seeds unchanged) — must match exactly.

3. **Implement H1 (seed filtering).**
   - Switch NN seeds, session-mean, and BM25/TT/Qwen/CLAP last-N-track
     suffixes to `positive_history()` under the flag.
   - Dev run, LTR booster unchanged (apply existing 27-feat booster).
   - Compare to 0.1646 baseline. Bucket the delta by "session has rejection
     streak" vs not.

4. **Implement H3 (goal-slot modulation).**
   - Add `--rejection_drop_threshold` and `--goal_substitute_positive` flags.
   - Independent dev runs with H1 still active:
     - H3a alone: threshold=3, no substitution.
     - H3b alone: substitution on, threshold=0.
     - H3a + H3b combined.
   - Each run uses the existing 27-feat booster; we are only changing the
     query content, not the LTR features.
   - Sweep threshold ∈ {2, 3, 4} once a winning variant is identified.

5. **Implement H2 (features) and dump train features.**
   - Add four new features to `FEATURE_COLS`.
   - `--write_features` on train split with `--use_goal_progress` and the
     best H3 setting from Step 4 (so the booster trains on the same query
     distribution it will see at inference).

6. **Retrain LTR.**
   - LambdaMART with same hyperparameters as v3 (nl31 lr0.08) on the
     extended feature matrix.
   - Inspect gain table; expect `sim_to_neg_hist_mean` and
     `artist_in_rejected_set` to appear with meaningful importance.

7. **Eval H1+H2+H3 stacked on dev.**
   - `--ltr_model <new_booster>.txt --use_goal_progress` plus best H3 flags.
   - If dev nDCG@20 > 0.1646: update `CURRENT_BEST_ITERATION.md`, port to
     blind (after confirming column on blind/test in Step 1).

8. **Ablation table.**

   | Row | H1 (seed filter) | H3 (goal slot) | H2 (LTR feats / booster) | Purpose |
   |---|---|---|---|---|
   | A | off | off | off (27-feat v3) | 0.1646 baseline |
   | B | on  | off | off (27-feat v3) | H1 alone |
   | C | off | best | off (27-feat v3) | H3 alone |
   | D | on  | best | off (27-feat v3) | H1+H3, no new feats |
   | E | off | off | on (new booster) | H2 alone, raw seeds |
   | F | on  | best | on (new booster) | full stack |

   Decides which lever is doing the work and whether they are additive.

## Validation

- Step 1: prints land in `exp/analysis/goal_progress_stats.txt`. Also
  report the per-session distribution of `n_consecutive_rejections` to size
  H3a's surface area.
- Step 2: dev nDCG@20 with flag on = flag off (delta < 0.0001).
- Step 3, 4, 7: full 1000-session dev runs, no NaNs, pool size mean ≈ 1450.
- Step 4: each H3 variant prints (a) number of turns where `goal` was
  dropped, (b) number of turns where substitution fired, (c) dev nDCG@20.
- Step 6: LTR train log shows new features with non-zero gain; CV mean
  stable or improved vs LTR v3.
- Step 7 gate: dev nDCG@20 > 0.1646 to promote.

## Risks

- Column absent or sparse on blind/test: dev-only win, no leaderboard impact.
  Mitigation: verify in Step 1 before training.
- Filtering all recent history when every prior turn is DOES_NOT_MOVE leaves
  no anchor. Mitigation: explicit raw-history fallback in
  `positive_history()`; never return empty when raw history exists.
- `None`-label sessions: treating None as neutral may include the first track
  of a session that the user ends up rejecting at turn 2. Acceptable — we
  can't know retroactively, and turn 1 has no choice.
- Feature collinearity: `sim_to_pos_hist_mean` may duplicate
  `dist_to_recent_mean` (already in FEATURE_COLS). LambdaMART tolerates
  this, but check gain split.
- H3b risk: substituting one positive track's name+artist into the goal
  slot may bias retrieval too hard toward that single artist. Mitigation:
  consider name-only (drop artist) or top-2 positives averaged when more
  than one exists, and inspect retrieved-artist diversity in the eval log.
- H3a risk: dropping the static `goal` after K rejections discards useful
  high-level context (e.g. "workout playlist"). Threshold sweep in Step 4
  exists to find the point where the goal goes from helpful to harmful.
- Negation temptation: keep rejection signal out of any text concatenation.
  If a later iteration tries to append "avoid: X" strings to a query, this
  plan disallows it and explains why (bi-encoders fail on negation).
- Adding a feature changes the LTR feature width: a stale booster applied to
  new feature dump will silently misalign columns. Mitigation: assert
  `booster.num_feature() == FEATURE_COLS_USED` and gate by
  `--use_goal_progress` so the booster knows which schema to expect (or
  train two boosters: with/without these features).

## Notes

- Phase 8 (07_ranking_calibration.md) remains the active phase until this
  proposal clears Step 1 verification. Promote to active only after the
  data-presence gate passes.
- Score target: ≥ 0.166 dev nDCG@20 (modest +0.001 over 0.1646) to claim a
  real lift past noise. Anything < 0.165 = H1+H2 too weak to justify
  inference complexity; document and move on.
