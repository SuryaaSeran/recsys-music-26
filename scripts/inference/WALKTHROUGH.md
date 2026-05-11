# Inference Pipeline Walkthrough

Six scripts. Two are the production pipeline (dev + blind), three are the weight-tuning loop, one is the local evaluator.

```
run_inference_fusion.py            # Main: BM25 recall + 6-signal weighted rerank on dev set
run_inference_blind_fusion.py      # Same as above but for the blind submission format
generate_responses_blind.py        # Wrap each prediction with a Qwen-generated 2-sentence response
evaluate_local.py                  # Compute nDCG@{1,10,20} + Hit@K from a predictions JSON
precompute_embeddings.py           # One-off: cache all per-turn query embeddings + BM25 candidates
score_precomputed.py               # Fast grid search over fusion weights (uses precompute output)
```

The currently-best config:

```
--w_tt 0.32 --w_cf 0.10 --w_qwen_meta 0.40 --w_qwen_lyrics 0.08 --w_clap 0.05 --w_bm25 0.24 --bm25_norm
                              # plus optionally --w_attrs_hist 0.08
```

Dev nDCG@20 = 0.1519 without attrs_hist, 0.1529 with it (full 1000 sessions, 8000 turns).

---

## 0. Conversation schema (input contract)

Both the dev set and the blind set are conversations under `talkpl-ai/TalkPlayData-Challenge-{Dataset, Blind-A}`:

```python
session = {
  "session_id": "ba3da7b0-...",
  "user_id":    "32957cc8-...",
  "user_profile": {
    "preferred_musical_culture": "Western pop",
    ...
  },
  "conversation_goal": {
    "listener_goal": "discover new chill tracks for a quiet evening",
    ...
  },
  "conversations": [
    {"role": "user",      "content": "play me something chill", "turn_number": 1},
    {"role": "assistant", "content": "How about ...?",          "turn_number": 1},
    {"role": "music",     "content": "<track_uuid>",            "turn_number": 1},
    {"role": "user",      "content": "more like that",          "turn_number": 2},
    {"role": "music",     "content": "<track_uuid>",            "turn_number": 2},
    ...
  ]
}
```

Two important contracts:

1. **Dev format** has interleaved user / assistant / music turns. Each `music` turn is a *target*: the system at turn N has access to everything before turn N's music message and must predict the gold `track_uuid`.
2. **Blind format** (`Blind-A`) has only one prediction point per session: `conversations[-1]` is the user's request, everything before is history, **no music turn at the end**. Output for blind is one prediction per session, not per turn.

Output contract (`predictions.json`):

```json
[
  {
    "session_id": "...",
    "user_id": "...",
    "turn_number": 1,
    "predicted_track_ids": ["uuid1", ..., "uuid20"],
    "predicted_response": "Try \"X\" by Y — it's a great fit for ..."
  },
  ...
]
```

Order of `predicted_track_ids` matters — nDCG penalizes lower ranks logarithmically.

---

## 1. `run_inference_fusion.py`  — the dev pipeline

**The one script that ties everything together.** Reads the dev set, runs BM25 recall + multi-signal rerank for each music turn, and writes the predictions JSON.

### Inputs

- `talkpl-ai/TalkPlayData-Challenge-Dataset` (`test` split, 1000 sessions, 8000 music turns)
- `cache/bm25/track_metadata/` — BM25 model + parallel `track_ids.json` (47,071 tracks)
- `cache/twotower_v3/` — fine-tuned dense index (47,071 × 384, normalized)
- `cache/qwen3_meta/`, `cache/qwen3_attr/`, `cache/qwen3_lyrics/`, `cache/clap/`, `cache/cf_bpr/` — pre-computed track embeddings (54,476 each)
- `cache/user_cf_bpr.json` — CF-BPR user vectors
- Models loaded at startup:
  - `models/twotower_v3/final` — local fine-tuned bi-encoder
  - `Qwen/Qwen3-Embedding-0.6B` — for query encoding (track side already cached)
  - `laion_clap.CLAP_Module(amodel="HTSAT-tiny")` — for query encoding (audio is text-paired)

### Output

`exp/inference/devset/<tid>.json` — list of `{session_id, user_id, turn_number, predicted_track_ids[20], predicted_response}`.

### Per-turn flow (the heart of the system)

For each music turn we do this:

```python
# 1. Three query variants — each tuned for the signal that consumes it
latest_user = text_history[-1] if text_history else ""

tt_parts = [latest_user, goal, culture]
for tid in music_history[-2:]:
    tt_parts.append(f"{name} {artist}")           # name+artist only — matches training format
tt_query = " ".join(p for p in tt_parts if p)

bm25_parts = [goal, culture]
for tid in music_history[-args.hist_turns:]:      # default 4
    bm25_parts.append(get_track_text(tid))         # name+artist+tags — BM25 likes long
bm25_parts.extend(text_history[-args.text_turns:])
bm25_query = " ".join(p for p in bm25_parts if p)

cleaned = clean_query(latest_user) or latest_user  # strip "can you", "I'd like", etc
sem_parts = [cleaned, goal, culture]
for tid in music_history[-args.sem_hist:]:         # default 2
    sem_parts.append(f"{name} {artist}")
semantic_query = " ".join(p for p in sem_parts if p)

# 2. BM25 recall → top-500 (over-fetch by 3x|seen| then filter)
retrieve_k = args.bm25_pool + len(seen) * 3
raw_tids, raw_scores = retrieve_bm25(bm25_query, retrieve_k)
filtered = [(t, s) for t, s in zip(raw_tids, raw_scores) if t not in seen][:args.bm25_pool]
cands, bm25_scores = unzip(filtered)

# 3. Encode three query embeddings (CF user vector is just a lookup)
tt_emb   = tt_model.encode(tt_query,                normalize_embeddings=True)
qwen_emb = qwen_model.encode(QWEN_INSTR + sem_query, normalize_embeddings=True)
clap_emb = clap_model.get_text_embedding([sem_query])[0]; clap_emb /= norm(clap_emb)
user_emb = user_cf.get(user_id)                     # None if cold-start

# 4. Compute full-index dot products once per turn (single matmul each)
tt_all   = tt_embs        @ tt_emb
qm_all   = qwen_meta_embs @ qwen_emb
ql_all   = qwen_lyrics_embs @ qwen_emb if w_qwen_lyrics > 0 else None
clap_all = clap_embs      @ clap_emb
cf_all   = (cf_track_embs @ user_emb) if user_emb is not None else None

# 5. BM25 signal: NORMALIZED variant (the +0.0016 winning change)
if args.bm25_norm:
    bm25_rr = {tid: s / max(bm25_scores[0], 1e-8) for tid, s in zip(cands, bm25_scores)}
else:
    bm25_rr = {tid: 1.0 / (r + 1) for r, tid in enumerate(cands)}

# 6. Score each of 500 candidates, take top-20
total = sum of (w_signal * cosine_signal[tid]) for each signal
top_idx = np.argsort(total)[::-1][:20]
predicted_track_ids = [cands[i] for i in top_idx]
```

### Feature engineering — the three-query split

We use **three different query strings** because each signal has different needs:

| Query | Optimized for | Why this shape |
|---|---|---|
| `tt_query` (compact, ~100 tokens) | TwoTower MNRL training matched this exact format | Anything else and the encoder is out-of-distribution; long queries hit the 256-token wall. |
| `bm25_query` (long, ~500+ tokens) | BM25 is a bag of words — more terms = better recall | No truncation issue (BM25 doesn't have one); tags help match documents with the same tags. |
| `semantic_query` (cleaned, ~50-150 tokens) | Qwen3 + CLAP are trained on natural-language retrieval prompts | Conversational filler ("can you please") drowns out content; cleaning improved Qwen3 cosines. |

The `clean_query` regex strips: `can you, could you, would you, please, i want, i'd like, i need, i'm looking for, recommend, suggest, play me, find me, show me, give me, how about, what about, i feel like, i'm in the mood for, do you have, do you know`. It's not exhaustive, just empirically-derived from looking at the dataset.

### The fusion weights

Default values are tuned (best so far):

```
w_tt = 0.32      w_cf = 0.10      w_qwen_meta = 0.40
w_qwen_lyrics = 0.08              w_clap = 0.05
w_bm25 = 0.24    bm25_norm = True
```

How they were found: `score_precomputed.py --grid_search` — vectorized eval over the 8000 dev turns, ~30 seconds per 12k-config sweep. See §6 for how grid search works.

### The "BM25 norm" change explained

For each turn, BM25 returns 500 candidates with raw scores — typically ~30 for rank 1, ~15 for rank 50, ~10 for rank 500. The original signal we used was **reciprocal rank**: `1 / (rank + 1)`. That gives:

| BM25 rank | reciprocal rank | normalized score |
|---|---|---|
| 1 | 1.000 | 1.000 |
| 5 | 0.167 | ~0.85 |
| 10 | 0.091 | ~0.75 |
| 50 | 0.020 | ~0.50 |

RR collapses to ~0 by rank 10, so the BM25 component is essentially "rank-1 wins, everyone else gets nothing". With normalized score, ranks 1-50 all carry meaningful signal. **This single change is +0.0016 on the full dev set** (0.1473 → 0.1489), and it allowed the optimal `w_bm25` to roughly double (0.13 → 0.24) since the signal is no longer concentrated.

### Dense pool expansion (deliberately not used)

The `--cf_pool`, `--qwen_pool`, `--tt_pool` flags can add candidates outside the BM25 top-500 by retrieving the dense top-K and unioning into the candidate set. **This consistently hurts.** At pool=200, dense-only candidates lack BM25 rank signal and so can't compete with BM25-pool candidates that have both signals. Confirmed across CF, Qwen3, and TT pools. The flags are kept for ablation but `pool=0` is the production setting.

### Pitfalls

- **Cold-start users get `user_emb = None`** — this is expected for users not in train. CF score is then 0 for all candidates, so `w_cf * 0 = 0` and CF effectively drops out. Don't crash — `if user_emb is not None` guards.
- **The candidate scoring loop is `for i, tid in enumerate(cands)`** — it's *not* fully vectorized because we need the per-tid `id2idx` lookup against each signal's track list (different orderings for cf/clap/qwen3/tt). At 500 cands × 8000 turns = 4M iterations, this is the main bottleneck (~2.1s per session).
- **`text_history` includes assistant turns**, not just user. If you change this, BM25 recall changes silently — the dataset has heavy assistant filler that BM25 happily indexes against.
- **`seen` excludes already-played tracks** — without this, the model would happily re-recommend the same track every turn (BM25 score doesn't drop on repeat).
- **`retrieve_k = bm25_pool + 3*|seen|`** — over-fetch to make sure we still get 500 candidates after filtering seen tracks. 3x is empirical; if your sessions get longer than ~150 tracks you'd want to bump it.

### Opportunities

- **Vectorize the scoring loop.** Build the `bm25_to_X` lookup arrays once (already done in `score_precomputed.py`) and replace the per-tid loop with a single fancy-index gather. ~5x speedup per turn.
- **Per-query weight adjustment.** Currently weights are global. A simple classifier (artist-mention-detected? mood-only? specific-album?) could pick from 3-4 weight presets — likely worth +0.002.
- **Audio-side CLAP.** We use CLAP's text encoder against the pre-computed audio embeddings. If you had short clips of user-described audio (whistled melody, hummed tune) you could use CLAP's audio encoder symmetrically.
- **Re-rank cascade.** Take the top 50 from the fusion score and re-rank them with a small cross-encoder — but the v3 cross-encoder attempts hurt nDCG (see `archive/v3_crossencoder/`). You'd need a much better-tuned CE to make this work.

---

## 2. `run_inference_blind_fusion.py`  — same pipeline, blind format

Identical scoring to `run_inference_fusion.py`. The only difference is iteration shape:

```python
# Dev: iterate music turns *within* each session, accumulate history as you go
for turn in conversations:
    if turn["role"] == "music":
        # ... predict ...
        music_history.append(turn["content"])
    else:
        text_history.append(turn["content"])

# Blind: only one prediction per session at conversations[-1]
user_query = conversations[-1]["content"]
history_convs = conversations[:-1]
music_history = [t["content"] for t in history_convs if t["role"] == "music"]
text_history  = [t["content"] for t in history_convs if t["role"] in ("user", "assistant")]
```

Number of predictions: 80 (vs 8000 for dev). Runtime is ~30 seconds, dominated by model loading.

The script supports `--w_qwen_lyrics` and `--bm25_norm` exactly like the dev script. Defaults are conservative; you should pass the explicit best-config flags:

```bash
python scripts/inference/run_inference_blind_fusion.py \
    --tid blind_a_fusion_v13_tuned \
    --w_tt 0.32 --w_cf 0.10 --w_qwen_meta 0.40 --w_qwen_lyrics 0.08 \
    --w_clap 0.05 --w_bm25 0.24 --bm25_norm
```

### Pitfalls

- **`turn_number` of the prediction is `conversations[-1]["turn_number"]`**, not 1. Some early submissions hardcoded 1 and got rejected — the grader matches on `(session_id, turn_number)`.
- **Cold-start frequency is much higher in blind** (50%+ users not in train) than dev. CF effectively drops out for half the predictions. Don't expect CF to help blind as much as dev.
- **One-shot context.** No history accumulates within a blind session because each session has only one prediction point. Anything you add to `music_history` post-prediction would be wasted.

### Opportunities

- The blind set is small enough (80 predictions) that you can afford much heavier computation per query — e.g., a cross-encoder on the top-50 — without runtime concerns. Worth experimenting with there even if it's too slow for full dev runs.

---

## 3. `generate_responses_blind.py`  — Qwen response wrapper

The competition has two graded components: (a) `predicted_track_ids` (nDCG@20) and (b) `predicted_response` (LLM-as-Judge). The retrieval scripts emit a stub like `'I recommend "X" by Y based on your request.'`. This script replaces the stub with a 2-3 sentence Qwen-generated response.

### Inputs

- A predictions JSON from `run_inference_blind_fusion.py`
- `models/qwen_sid_patched` — local Qwen model loaded via `mlx_lm` (M-series-friendly inference)
- The metadata dict (for track names + tags shown to the LLM)

### Output

`<original_path>_qwen.json` — same predictions, but with `predicted_response` filled in by Qwen.

### Prompt template (~200 tokens)

```python
messages = [
  {"role": "system", "content":
   "You are a friendly music recommendation assistant. Give a brief (2-3 sentence) "
   "recommendation that references the user's request and explains why the top track fits."},
  ...history_convs[-4:]...,                               # interleaved user/assistant/music
  {"role": "user", "content": user_query},
  {"role": "user", "content":
   f"Based on the request, here are my recommendations:\n{recs_text}\n\n"
   f"Please give a brief recommendation response about the top track."}
]
```

`recs_text` is the top-3 picks formatted as `"Name" by Artist (tags: a, b, c, d, e)`. Music turns in history are shown as `'I recommend "Name" by Artist.'` so the model sees a coherent multi-turn shape.

### Pitfalls

- **`max_tokens=120`** by default. Qwen sometimes runs over and trails into incoherence; 120 is empirical for "complete a 2-3 sentence rec".
- **The model can hallucinate** about a track ("a fusion of various genres"). It's working from `name + artist + 5 tags`, which is thin. You'll see plausible-sounding but generic responses.
- **MLX is M-series specific.** On non-Apple hardware, swap `from mlx_lm import load, generate` for `transformers` and bump batch size for throughput.
- **`models/qwen_sid_patched`** is a local checkpoint. Original origin: a fine-tune attempt that didn't help retrieval (semantic IDs era, archived). It still works fine as a generic chat model.
- **Generation throughput is ~1.7 it/s** (see `/tmp/gen_v6_resp.log`). 80 sessions = ~45 seconds. Full 8000-turn dev would be ~80 minutes — don't generate responses for dev unless you're testing the response code.

### Opportunities

- **Better metadata in prompt.** Include album, year, popularity. Currently the model sees 5 tags and a name; a richer card would let it write more specific copy.
- **Stronger system prompt.** The current one is generic. A few-shot version with examples of "good" responses would tighten output.
- **Rule-based fallback for known patterns.** If the user explicitly named an artist, the response should always reference that name; templating that constraint outside the LLM is more reliable than hoping it follows instructions.

---

## 4. `evaluate_local.py`  — local nDCG calculator

A small standalone scoring script. No models loaded.

### Inputs

- A predictions JSON file
- The HF dev set (to read gold tracks)

### Output (stdout)

```
Predictions file: exp/inference/devset/fusion_v9_norm_full.json
Evaluated sessions: 1000
Total prediction points: 8000
nDCG@1:  0.0510  (Hit@1:  408/8000 = 5.1%)
nDCG@10: 0.1287  (Hit@10: 1845/8000 = 23.1%)
nDCG@20: 0.1489  (Hit@20: 2486/8000 = 31.1%)
```

### Math

For each music turn:
- Find the rank `r` of the gold track in `predicted_track_ids` (1-indexed). If not present, `r = None`.
- nDCG@K contribution = `1 / log2(r + 1)` if `r ≤ K` else 0.
- Hit@K = 1 if `r ≤ K` else 0.
- Normalize by 1 (single relevant track per turn, so the ideal DCG@K is `1 / log2(2) = 1.0`).

Average over all turns.

### Pitfalls

- **Lookup is `(session_id, turn_number)`.** If a prediction's `turn_number` is wrong, the turn is silently scored as "no prediction" → ndcg contribution 0. Always print the `Total prediction points` count to make sure it matches `Evaluated sessions × turns`.
- **It re-downloads the dataset on every call.** With HF caching this is fast on the second run but the first run takes a minute.
- **The `--sessions` flag truncates to first N**, but you should always run on the full set before claiming an improvement. 200-session estimates over-promise by ~0.005 absolute (see project memory).

### Opportunities

- Add Recall@20 (whether gold is in pool, regardless of rank). Useful for separating recall failures from rerank failures.
- Add per-session breakdown: which sessions/users are systematically getting it right vs. systematically wrong? Identifies whether to invest in cold-start handling, long-history handling, etc.

---

## 5. `precompute_embeddings.py`  — one-off cache for fast tuning

Eliminates the model-loading + per-turn-encoding cost for every weight-tuning experiment. Once this runs, weight changes evaluate in ~30 seconds via `score_precomputed.py`.

### Inputs

Same models and caches as `run_inference_fusion.py` (loads everything once).

### Outputs (in `cache/dev_embeddings/`)

| File | Shape | What it stores |
|---|---|---|
| `turns.json` | list of dicts | One entry per turn: `{session_id, user_id, turn_number, gold, has_cf}` |
| `bm25_cands.npy` | `(N, 500)` int32 | BM25 track indices (into `bm25/track_metadata/track_ids.json`); `-1` for missing |
| `bm25_scores.npy` | `(N, 500)` float32 | Raw BM25 scores |
| `tt_embs.npy` | `(N, 384)` float32 | Per-turn TwoTower query embedding |
| `qwen_embs.npy` | `(N, 1024)` float32 | Per-turn Qwen3 query embedding |
| `clap_embs.npy` | `(N, 512)` float32 | Per-turn CLAP query embedding |
| `cf_user.npy` | `(N, 128)` float32 | Per-turn CF user vector (zeros for cold-start) |
| `attrs_hist.npy` | `(N, 1024)` float32 | Avg of last 4 played tracks' qwen3_attr embeddings (zeros if no history) |

`N` = total music turns across all 1000 dev sessions ≈ 8000.

### Why precompute these specifically

The query embeddings (tt, qwen, clap, cf, attrs_hist) are what depend on conversation context — they change every turn but are *deterministic* given the conversation. Precomputing them once avoids re-running 3 large transformer models 8000 times for every weight experiment.

The BM25 candidates and raw scores are also turn-dependent and slow to recompute (BM25 retrieval over 47k tracks). Caching them too means *any* fusion weight scheme can be evaluated without ever touching a model.

### Feature engineering of the cache

Note `attrs_hist` (style-history signal): for each turn, average the qwen3_attr embeddings of the last 4 played tracks, then re-normalize. This captures "what kind of music the user has been hearing in this session". Used only with `--w_attrs_hist > 0`.

### Pitfalls

- **2-hour run.** Dominated by Qwen3-Embedding-0.6B at ~1 query/second on M-series. Don't run twice if you don't have to.
- **Re-running silently overwrites.** Your tuning history will be wrong if you re-precomputed mid-experiment with a different model loaded.
- **Don't forget to update this script if you change the live inference's query construction.** If `run_inference_fusion.py` builds `bm25_query` differently, the precomputed `bm25_cands.npy` becomes stale and your grid search optimizes for the wrong distribution.

### Opportunities

- **Save lyrics query embeddings separately** if you ever want to use a *different* instruction for lyrics retrieval ("Given a music mood description, retrieve songs with matching lyrical themes"). Currently we use the same `qwen_emb` for both meta and lyrics scoring.
- **Save the full top-50 of each dense signal** so the grid search can also explore dense pool expansion. Currently the cache locks you into BM25-pool-only scoring.

---

## 6. `score_precomputed.py`  — vectorized weight tuning

The fast inner loop. Loads the precomputed cache, computes all six signals as `(N, 500)` matrices once, then evaluates any weight combination as a single matrix sum + argmax.

### Inputs

- `cache/dev_embeddings/` (from §5)
- All track-side embedding caches (`cache/twotower_v3/`, `cache/qwen3_meta/`, `cache/qwen3_lyrics/`, `cache/qwen3_attr/`, `cache/clap/`, `cache/cf_bpr/`)
- BM25 track ID list

### What it does

1. **Build `bm25_to_X` lookups** — a `(47071,)` int32 array mapping each BM25 track index to that signal's track index. `-1` for tracks not in the signal's space (e.g., the 7,405 "extra" tracks in qwen3_meta but not in BM25). Computed once.
2. **For each of (tt, qwen, lyrics, clap, cf, attrs_hist):** gather candidate track embeddings and dot with the per-turn query embedding. Result: `(N, 500)` cosine matrix per signal. Computed once. Total memory: ~250MB.
3. **Build the BM25 signal both ways** — RR (`1 / (rank+1)`) and normalized (`score / score[0]`). Both are `(N, 500)`. Computed once.
4. **Locate the gold's column position** within each turn's BM25 candidate list (vectorized with `argmax(matches, axis=1)`).
5. **`evaluate(weights)`** — single weighted sum over the precomputed matrices:
   ```python
   total = w_tt*tt_cos + w_qm*qm_cos + w_lyrics*lyrics_cos + w_clap*clap_cos
         + w_cf*cf_cos + w_ah*ah_cos + w_bm25*bm25_sig
   gold_score = total[arange(N), gold_col]
   ranks = (total > gold_score[:, None]).sum(axis=1) + 1
   ndcg = (1 / log2(ranks + 1) where rank ≤ 20 else 0).mean()
   ```
   ~5 ms per evaluation. A 12,000-config grid search is ~60s.

### Best result so far

```bash
python scripts/inference/score_precomputed.py --grid_search

# Top config (after refining around the boundary):
nDCG@20 = 0.1529  Hit@20 = 0.316
tt=0.32  cf=0.08  qm=0.32  lyrics=0.10  ah=0.08  clap=0.03  bm25=0.20
```

vs default v6 (0.1473) → +0.0056 just from grid tuning + lyrics + attrs_hist + normalized BM25.

### Single-config invocation (used to verify reproducibility)

```bash
python scripts/inference/score_precomputed.py \
    --w_tt 0.35 --w_cf 0.12 --w_qwen_meta 0.30 --w_clap 0.10 --w_bm25 0.13 --bm25_norm
# nDCG@20: 0.1489  Hit@20: 0.311      <- exactly matches fusion_v9_norm_full.json
```

This exact match confirmed the grid search is reliable: any improvement seen in the search transfers 1:1 to a real `run_inference_fusion.py` run with the same flags.

### Pitfalls

- **The grid search is only over 12,000 configs in a small box.** It's not global — extending the box is sometimes necessary (the first sweep found w_bm25 at the upper boundary 0.18; refining shifted the optimum to 0.24).
- **Local minima look flat at the top.** 30+ configs scored within 0.0005 of the best in the last sweep. Pick a "stable" config (one near several others) rather than the absolute argmax.
- **`evaluate` ignores `bm25_norm=False` after the grid search shifted to "norm always wins"** — the second sweep hardcoded `True`. If you ever want to verify the RR variant again, restore the outer loop.
- **The cache must match the live inference.** Any change to query construction (text history length, semantic-query cleaning, etc.) means precompute is stale; the grid search optimum is then overfit to a query distribution you no longer use.
- **No bootstrap CIs.** A 0.0001 gap between two configs is noise; the script doesn't tell you that. Trust gaps ≥0.001.

### Opportunities

- **Coordinate descent** instead of grid: starting from current best, sweep one weight at a time over ~10 values. Finds the ridge much faster than a 7-D box search.
- **Add per-turn weight gating.** The grid search optimizes a global weight; in practice the optimal w_cf is much higher for warm-start users than cold. A two-segment evaluator (warm vs cold) would let you tune them separately.
- **L1 sparsity penalty.** Encourage small weights for signals that don't help — would simplify the score and could match the production code's defaults more cleanly.

---

## End-to-end inference run

```bash
# A. Verify cache is in place (no models needed)
python scripts/inference/score_precomputed.py \
    --w_tt 0.32 --w_cf 0.10 --w_qwen_meta 0.40 --w_qwen_lyrics 0.08 \
    --w_clap 0.05 --w_bm25 0.24 --bm25_norm
# Expected: nDCG@20 ≈ 0.1519  (or 0.1529 with --w_attrs_hist 0.08 if added in code)

# B. Generate dev predictions for record-keeping
python scripts/inference/run_inference_fusion.py --tid fusion_v13_tuned \
    --w_tt 0.32 --w_cf 0.10 --w_qwen_meta 0.40 --w_qwen_lyrics 0.08 \
    --w_clap 0.05 --w_bm25 0.24 --bm25_norm

# C. Score
python scripts/inference/evaluate_local.py \
    --pred exp/inference/devset/fusion_v13_tuned.json

# D. Run blind set with same config
python scripts/inference/run_inference_blind_fusion.py --tid blind_a_fusion_v13_tuned \
    --w_tt 0.32 --w_cf 0.10 --w_qwen_meta 0.40 --w_qwen_lyrics 0.08 \
    --w_clap 0.05 --w_bm25 0.24 --bm25_norm

# E. Generate Qwen responses on top of the blind predictions
python scripts/inference/generate_responses_blind.py \
    --pred exp/inference/blind_a/blind_a_fusion_v13_tuned.json
# → exp/inference/blind_a/blind_a_fusion_v13_tuned_qwen.json
```

That last file is the submission.

---

## Cross-cutting pitfalls

- **Memory pressure on M-series.** Loading all five precomputed track indexes simultaneously (CF, CLAP, Qwen3 meta/attr/lyrics) plus all three live models (TT, Qwen3, CLAP) takes ~6-8 GB. Running two inference processes in parallel triggered 7.9 GB of swap during testing — don't.
- **Reciprocal-rank vs normalized-score is a property of one signal but it changes the optimal weights of *all* signals.** The grid had to be rerun after the BM25 change.
- **The 41% of dev turns where gold is outside the BM25 top-500 cap recall at 0.588 nDCG.** Currently we're at 0.152 = 25% of ceiling. Reranking is far from saturated, but recall is the long-term ceiling.

## Cross-cutting opportunities

1. **Cache-aware coordinate descent** instead of grid search would find the ridge faster.
2. **Per-turn classifier** for query type (artist-mention / mood / album / "more like this") — pick weights from a small set of presets. Likely the biggest available gain without retraining.
3. **Use the 7,405 "extra" tracks** in qwen3_meta/attr/lyrics that BM25 doesn't index. They're currently invisible to recall; even using only their lyrics embeddings to seed the candidate pool when the user's query has emotive content could pull in tracks BM25 misses.
4. **Add a popularity prior** as a 7th signal at low weight (e.g., 0.02 × popularity / 100). Gold tracks are systematically more popular (mean 43 vs 36); a tiny prior should help marginal cases.
5. **Cross-encoder rerank on top-50** — but only if you can train one that doesn't collapse like the v3 attempts. The Qwen3 model is good for this if you have GPU budget.
