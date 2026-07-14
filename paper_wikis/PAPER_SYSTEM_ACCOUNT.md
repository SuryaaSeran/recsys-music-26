# A CPU-Scale Conversational Music Recommender: Full Technical Account of the v2 Blind B System

Status: paper source document. Every number traces to a ledger entry, a log, or a
benchmark run (`scripts/analysis/bench_cpu_components.py`,
`exp/analysis/bench_cpu_components.json`). Companion references:
`plan/PLAN.md` (score ladder), `plan/CURRENT_BEST_ITERATION.md` (architecture
snapshot), `wiki/BLIND_B.md` (submission ledger), `plan/archive/*` (phase
histories).

---

## 0. Abstract and contributions

We describe a conversational music recommender that ranks tracks and writes the
assistant turn for the TalkPlayData Challenge (ACM RecSys 2026), and that trains
and serves in its entirety on a single consumer desktop (Apple M4 Mac mini, 16 GB
unified memory, no discrete GPU). The system fuses nine cheap retrievers into a
recall pool, arbitrates them with a gradient-boosted learning-to-rank model, and
generates the response with a decoupled LLM stage. Its best result on the blind
generalization set (Blind B) is composite **0.49**. The account below is written
to double as paper source, and its structure follows the challenge topics.

**Contributions, mapped to the challenge's topics of interest:**

1. **Multi-turn preference elicitation and dialogue-aware recommendation
   (Sections 3.6, 3.7, 8).** A role-tagged dialogue anchor encodes the running
   conversation as structured turns with explicit `REACTION` slots, so the
   retriever conditions on *what the user accepted and rejected*, not just the
   latest message. We show that the feedback signal in this data is off by one
   turn, that 43% of logged "positive" tracks were in fact user-rejected, and
   that modeling rejection (as hard negatives in training and as seed-filtering
   plus session-state features at inference) is the single highest-ROI work in
   the project. On the blind set, where the ground-truth feedback channel is
   removed, we reconstruct it from the dialogue text with a light rule-based
   reaction classifier and lose nothing.

2. **Joint modeling of ranked retrieval and response generation
   (Sections 1, 7, and Section 9 result 1).** The competition scores ranking and text response in
   one composite, but its judge reads response text only. We treat the two
   objectives jointly-but-decoupled: the ranker optimizes logged-next-track
   nDCG, the response stage optimizes the text-only judge, and neither is
   allowed to overwrite the other. The central empirical finding is that this
   separation is *required*: every attempt to let response-side "user empathy"
   edit the track list regressed ranking (four independent confirmations).

3. **Efficient and scalable architectures for large-scale music recommendation
   (Section 10).** Every design choice trades a heavyweight component for a
   CPU-cheap one: brute-force exact similarity over a 47K x 768 matrix instead
   of an ANN index, one LoRA-adapted 279M encoder used once per turn instead of
   per-candidate neural scoring, a 0.6 MB gradient-boosted ranker instead of a
   cross-encoder. We report measured CPU latencies, memory, and end-to-end wall
   times, and give a leaner eight-source variant (Section 10.6) that drops the
   sequential-ID stack for near-zero quality cost and materially less setup.

Additional topics the account speaks to: **personalization from profile,
listening history and conversational context** (profile slot, session-mean and
CF sources, graceful cold-start; Sections 4, 8); **diversity and exploration**
(catalog/lexical-diversity composite levers driven by the response stage and by
never collapsing the pool to one artist; Sections 1, 7); and **analysis of
conversational-recommendation behavior and failure cases** (Section 9: TalkPlay
sessions are artist deep-dives, so the logged next track hides inside the very
clusters a human curator would prune, which is why hand rules lose).

---

## 1. Task and evaluation

**Competition:** TalkPlayData Challenge (ACM RecSys 2026). A conversational music
recommendation system must, at each turn of a multi-turn dialogue, (a) rank 20
tracks from a 47,071-track catalog such that the single logged next track ranks
high, and (b) generate the assistant's natural-language response.

**Data:**

| Split | Size | Labels | Notes |
|---|---|---|---|
| train | 15,199 sessions | gold track per music turn + `goal_progress_assessment` | model training |
| dev (test split) | 1,000 sessions / 8,000 turns | gold tracks | local evaluation loop |
| Blind A | 80 sessions, 1 prediction each | none (leaderboard) | full metadata present |
| Blind B | 80 sessions, 1 prediction each | none (leaderboard) | generalization set: `conversation_goal`, `goal_progress_assessments`, `thoughts` all removed; 40/80 sessions cold-start (no `user_id`, no `user_profile`) |

**Metric (composite, confirmed to 4 decimals against our own submissions):**

```
Score = 0.50 * nDCG@20 + 0.10 * CatalogDiversity + 0.10 * LexicalDiversity + 0.30 * JudgeNorm
JudgeNorm = (judge_1to5 - 1) / 4
```

With one relevant item per turn, nDCG@20 = 1/log2(rank+1) if the gold track is in
the top 20, else 0. The judge is Gemini, scoring two disclosed text-only axes
(Personalization, Explanation Quality) on the response text alone — it never sees
the track list. This asymmetry shapes the entire late-stage strategy (Section 8).

**Final result:** Blind B composite **0.49** (submission v2), our best on the
generalization set. The system behind it is the subject of this document.

---

## 2. System overview

Three stages, all running on a single consumer desktop (Apple M4 Mac mini, 16 GB
unified memory, no discrete GPU):

```
                 RECALL (9 fused candidate sources, ~3,100 candidates/turn)
conversation --> BM25 | two-tower ANN | last-track NN | artist expansion |
                 Qwen3-embedding NN | CF-BPR | session-mean NN | co-occurrence |
                 SASRec semantic-bucket expansion
                          |
                 RESCORE (LightGBM LambdaMART, 67 features, top-20 by score)
                          |
                 RESPOND (LLM-written response per session; text-only judge lever)
```

- **Recall** is a union of independent retrievers. No single retriever exceeds
  ~59% gold recall at 500 candidates; the fused pool reaches **~89%** on the
  Blind-B-simulated holdout.
- **Rescore** is a learning-to-rank gradient-boosted tree model that replaces
  hand-tuned score fusion. Its strongest feature is *multi-source agreement* —
  how many independent retrievers found the candidate.
- **Respond** is decoupled: the judge reads only text, so response quality is an
  additive composite lever that cannot hurt ranking.

The only trained neural component is a LoRA fine-tune of a 279M-parameter text
embedding model. Everything else is either a classical index (BM25), precomputed
embeddings shipped with the competition data (Qwen3, CLAP, CF-BPR), a count
table (co-occurrence), or a gradient-boosted tree ensemble (LightGBM).

---

## 3. Development narrative: how each ceiling was found and broken

The dev-set score ladder, in order of discovery (full 1,000-session nDCG@20):

```
0.0861  BM25, corpus = name + artist + album
0.0960  + tag_list in corpus
0.1313  + seen-track exclusion
0.1418  + fine-tuned two-tower (compact query), linear fusion w=0.7
0.1533  + multi-source pool expansion (artist, TT@1000, last-NN)
0.1609  + LightGBM LambdaMART reranker (replaces linear fusion)
0.1646  LTR with 27 source-aware features (Phase A)
0.1653  + popularity/year features + regularization (Phase B)
0.1684  + tt_pool=2000, 39 features (Phase D)
0.1729  two-tower v8c: e5-base 512-token anchors, gpa-corrected labels
0.1748  two-tower v8d: role-tagged structured anchor
0.1864  + gpa-aware inference + entity/album/era features (60-feat Tier 1)
```

Blind A nDCG@20 track record: 0.3701 → 0.3990 → **0.3997**. Blind A composite
record: 0.2771 (bad responses) → 0.4837 → **0.4935** (rich responses). Blind B
composite: **0.49** (v2, the system described here), 0.48 (v3, a regression that
taught us the most important lesson in Section 9).

### 3.1 Lexical baseline and the first two lessons (0.086 → 0.131)

Okapi BM25 over `track_name + artist_name + album_name` scores 0.0861. Two
changes, both nearly free, lift it 53%:

1. **Index the tag list.** Tags carry the genre/mood/era vocabulary that
   conversational requests actually use ("something mellow and acoustic").
   +0.010.
2. **Exclude already-played tracks.** The query quotes history, so BM25 returns
   previously recommended tracks near the top. Filtering them re-ranks new
   relevant tracks up: 0.0960 → 0.1313 (+37% relative). The single largest
   cheap win in the project.

Query construction that survived to the final system: latest user message + goal
+ culture + last 4 played tracks (name/artist/tags) + last 4 text turns. Using
all 8 turns hurts — old turns dilute the IDF mass of current-request terms.

### 3.2 Zero-shot dense retrieval does not work here

Every off-the-shelf dense option lost to BM25:

| Approach | nDCG@20 |
|---|---|
| all-MiniLM-L6-v2, zero-shot | 0.0654 |
| BM25 + MiniLM RRF | 0.0775 |
| Competition-provided Qwen3 track embeddings, zero-shot query encoding | inconsistent, below BM25 |

The failure is a query/document distribution gap: conversational queries ("I
want something upbeat for a run, like that last one but faster") do not embed
near track-metadata text in general-purpose embedding spaces. Conclusion that
held for the rest of the project: **dense retrieval on this task requires
fine-tuning on (conversation, gold-track-text) pairs.**

### 3.3 The two-tower and the truncation trap (0.1418)

First fine-tune: all-MiniLM-L6-v2 (22M params, 256-token limit), MNRL/InfoNCE
loss with in-batch negatives, anchors = full conversation context. It barely
beat BM25 — diagnosis showed **81% of anchors exceeded 256 tokens** and were
right-truncated, frequently cutting off the user's actual request.

Fix (v3): a compact anchor `{latest_user_turn} {goal} {culture}
{last_2_track_name_artist}` — median 101 tokens, request always first. Fused as
`score = 0.7*cosine + 0.3*bm25_reciprocal_rank`: **0.1418**.

Two instructive failures followed:

- **v4, BM25 hard negatives:** regressed at every fusion weight. The BM25 top-5
  negatives are lexically close but not embedding-space competitive; training
  against them teaches superficial distinctions.
- **v5, triplet loss:** collapsed outright (0.0525). MNRL with in-batch
  negatives is the stable choice at this data scale.

A rank-audit on turns where the gold was in the pool exposed the real ceiling:
the *linear fusion formula*. A BM25 rank-1 candidate with cosine 0.35 scores
0.7·0.35 + 0.3·1.0 = 0.545; a gold track at BM25 rank 50 with cosine 0.60 scores
0.7·0.60 + 0.3·(1/51) = 0.426. The rank-1 lexical prior is unbeatable no matter
how good the embedding gets. Perfect re-ranking within the pool would have
scored 0.588 — we were at 24% of the pool ceiling. **The problem was scoring,
not recall.**

### 3.4 Pool recall: necessary but not sufficient (0.1518, flat)

Systematic recall audit (gold-in-pool rate, 1,000 dev sessions):

| Pool | Recall |
|---|---|
| BM25@500 | 59.0% |
| + two-tower @500 | 72.7% |
| + two-tower @1000 | 78.6% |
| + artist expansion (full discographies of retrieved artists) | +6.1 pts |
| BM25@500 + artist + TT@1000 | **80.6%** |

Counter-intuitive findings: "improving" the BM25 query or index (field weights,
stopwords, stemming) *reduced* recall — more text dilutes IDF. Artist expansion
is the best recall-per-cost source in the system: a dictionary lookup, no model.

But nDCG stayed flat at 0.1518. Expansion candidates enter with `bm25_score=0`
and the linear fusion penalizes them structurally; rescued gold tracks could not
out-score BM25 rank-1 incumbents. Recall and scoring had to be fixed together.

### 3.5 The pivotal move: learned ranking replaces linear fusion (0.1609 → 0.1684)

We replaced the hand-tuned linear fusion with **LightGBM LambdaMART** (nDCG@20
objective) over per-candidate features dumped by running the full retrieval
pipeline on held-out training sessions. This is the single most important
architectural decision in the system:

- It **decouples recall from precision.** Any source can add candidates without
  a hand-assigned weight; the booster learns when to trust each signal.
- It surfaced the strongest feature we have: **`n_sources` — the number of
  independent retrieval sources that returned the candidate** (gain 3x the
  runner-up). Multi-source agreement is a de-facto ensemble vote and transfers
  across distributions better than any single score.
- It absorbs heterogeneous feature types: cosines, reciprocal ranks, count
  tables, metadata, session-state counters.

Alternatives tested and rejected at this stage (details in Section 9):
cross-encoder rerankers, LLM rerankers, a neural ListNet MLP — all lost to
LambdaMART, mostly for data-shape reasons (1 positive per ~3,100 candidates).

Feature engineering and regularization (lambda_l2, min_sum_hessian, path_smooth
— needed because sparse features like popularity=0 invite overfitting) walked
dev up: 0.1646 (27 feat) → 0.1653 (29 feat) → 0.1684 (39 feat, tt_pool=2000).

### 3.6 Dialogue-aware encoding: two-tower v8 family (0.1684 → 0.1748)

*(Topic: multi-turn preference elicitation and dialogue-aware recommendation.)*

MiniLM's 256-token window forced the compact anchor, which discards most of the
conversation. Switch to **`intfloat/multilingual-e5-base`** (XLM-RoBERTa, 12
layers, 768-dim, 512 tokens, 279M params), fine-tuned with **LoRA** (r=32,
alpha=64, on q/k/v projections = 1.8M trainable params, 0.64% of the model).
LoRA is not a nicety — full fine-tuning of a 279M model is impossible on a
16 GB unified-memory machine (Adam states alone ~3 GB; larger models OOM the
Metal backend outright). LoRA + gradient checkpointing trains at ~16 s/step
(effective batch 64) on the M4.

The anchor is built by a **tokenizer-aware greedy builder**: a fixed core
(profile, goal, latest request) is always included; history turns are inserted
most-recent-first, each only if it fits the 510-token budget, with per-turn
fallbacks (full → short → minimal). The user's request can never be truncated.
Only ~5% of anchors hit the budget at all.

The final anchor format (v8d) is *role-tagged*, giving the encoder explicit
dialogue structure:

```
query: [PROFILE] {age} · {country} · {gender} · {culture} · {language}
[GOAL] {listener_goal} ({specificity})
[T1] USER: {q1} | REC: {track – artist} | ASST: {r1} | REACTION: liked/rejected
...
[NOW] USER: {current request}
```

Documents: `passage: {name} by {artist} | Album: {album} | Tags: {tags} | {year}`.
The `query:`/`passage:` prefixes match E5 pretraining and must be identical
between training data, index build, and inference.

### 3.7 Label hygiene: the highest ROI work in the project (0.1748 → 0.1864)

Three data-quality discoveries, each worth more than a typical architecture
change:

**(a) The feedback label is off by one turn.** `goal_progress_assessment` (gpa)
at turn T is the listener's verdict on the recommendation made at turn **T-1**
(action at t, feedback recorded at t+1). Every gpa-conditioned pipeline before
2026-06-05 weighted gold tracks with a label that judged the *previous* turn's
track. All builders now index `turn_number - 1`.

**(b) 43% of training positives were contaminated.** In the LTR training dump,
43.2% of turns carry gold tracks labeled `DOES_NOT_MOVE_TOWARD_GOAL` — tracks
the logged system played and the user rejected. Training them as positives
teaches the ranker to rank rejected tracks highly. Fix: `--skip_no_progress`
drops these turns from LTR training entirely (a rejected gold is not a negative
for the *retrieval* problem — it is an unusable group). For the two-tower,
`--drop_rejected` removes those turns from positive pairs and recycles the
rejected tracks as **session-level hard negatives** — same intent context,
explicitly refused. Hard-negative priority order per anchor: confirmed session
rejections (up to 2) > BM25 top hard negs (up to 2, only for specific-goal
sessions) > artist-repeat distractors (1) > in-batch negatives. A "MOVES
protection" rule forbids ever sampling a user-accepted track as a negative.

**(c) Co-occurrence leakage.** The next-track co-occurrence table was initially
built on sessions overlapping the LTR feature dump. Gold tracks had
co-occurrence hits in 81.9% of leaky turns vs 28.1% of clean turns — a 3x
inflated feature that would collapse on blind data. Rebuilt with all 6,000
feature-dump sessions excluded (`next_song_leakfree_6k_excluded.npz`).
Diagnostic now standard: split any interaction-derived feature's gold-hit rate
by table-membership before trusting it.

On top of clean labels, gpa-awareness moved into **inference** (H1/H2/H3):

- **H1 — seed filtering:** rejected tracks are removed from the history seeds
  that drive NN expansion, session-mean, and the BM25 query (else, after a run
  of rejections, every seed is a track the user just refused).
- **H2 — session-state features:** candidate similarity to the mean embedding
  of accepted vs rejected history tracks, artist-in-rejected-set flag, counts
  of accepted/rejected turns, consecutive-rejection counter.
- **H3 — goal-slot modulation:** `--goal_substitute_positive` replaces the
  static goal text in the anchor with the most recent *accepted* track ("more
  like the last thing that worked").

Plus Tier-1 features: album-continuity (same album as last track, album in
recent window), era/genre/mood keyword extraction from the request matched
against candidate tags and release year, within-artist popularity/transition
ranks, and semantic-ID bucket-match counts. Result: **0.1864 dev, 0.3990
Blind A** (60 features). Adding SASRec bucket expansion (Section 4, source 9)
nudged Blind A to **0.3997** (67 features, the config Blind B v2 uses).

---

## 4. RECALL architecture, exact

Script: `scripts/inference/run_inference_fusion_recall_expansion.py`. Per turn,
nine sources are fused into one deduplicated candidate pool (~3,100 candidates
mean):

| # | Source | Size | What it contributes |
|---|---|---|---|
| 1 | BM25 (Okapi, `bm25s`) | top-500 | lexical match on name/artist/album/tags; query = latest message + goal + culture + last-4 tracks + last-4 turns; missing-vocab turns smoothed by score floor 0.05 |
| 2 | Two-tower v8d ANN | top-2000 | fine-tuned conversational-intent match; brute-force cosine over 47,071 x 768 L2-normalized matrix |
| 3 | Last-track NN | 100 x last 2 tracks | "more like what just played" in TT space |
| 4 | Artist expansion | full discographies (popularity-capped) | dictionary lookup for every artist retrieved or mentioned; the cheapest high-recall source |
| 5 | Qwen3-Embedding-0.6B NN | top-500 | second opinion in a different embedding space; track side precomputed (competition-provided), query encoded by the frozen 0.6B model |
| 6 | CF-BPR | top-200 | collaborative signal, warm users only (cold-start sessions get none — degrades gracefully) |
| 7 | Session-mean NN | top-100 | NN of the mean TT embedding of all history tracks; session-level taste centroid |
| 8 | Co-occurrence | 300/150/50 | tracks that followed the last 1/2/3 history tracks in training sessions (leak-free table) |
| 9 | SASRec semantic-bucket expansion (optional; see 10.6) | cap 300 | RQ-VAE semantic IDs (2-level, 64 codes/level) over track attribute embeddings; a small SASRec predicts the next L0 bucket from the session's bucket sequence; members of the top-3 predicted buckets join the pool. Auxiliary only; the eight-source config without it reaches parity |

Pool recall: **87.4%** on dev (8,000 turns), **~89%** on the Blind-B-simulated
golden-200 holdout. The residual ~5% is "truly unreachable" — the gold is in no
source's top-5000.

Design rule learned the hard way: **add candidates, never re-weight by source
by hand.** Every candidate carries per-source features and origin flags; the
rescorer decides. Sources with individually mediocre precision (co-occurrence,
buckets) still help because membership feeds `n_sources`.

## 5. RESCORE architecture, exact

**Model:** LightGBM LambdaMART, `lambdarank` with nDCG@20 objective. Production
booster for the Blind B v2 system: `ltr_v8d_s3cap_nl31_lr0p08.txt` — **67
features**, 31 leaves, lr 0.08, lambda_l2 0.1, min_sum_hessian 0.1, path_smooth
1.0, feature/bagging fraction 0.8, truncation level 30, 5-fold session-level CV
with early stopping (75 rounds). CV nDCG@20 = 0.3144. The model file is ~0.5 MB
of text.

**Feature groups (67):**

| Group | Features | Note |
|---|---|---|
| A. Retrieval scores | tt_cos, tt_rank_sig, qwen meta/lyrics cos, CLAP audio cos, cf_cos, bm25 reciprocal-rank, cooccur rank signal | the raw votes |
| B. Multi-source agreement | n_sources, log1p(n_sources), n_sources/7 | **57% of total gain** |
| C. Track metadata | popularity percentile, years-since-release, tag overlap with query | |
| D. Session state (gpa/H2) | sim to accepted-history mean, sim to rejected-history mean, artist-in-rejected-set, rejected/accepted turn counts, consecutive rejections, CF distance to last/mean history | |
| E. Intent extraction (Tier 1) | era/genre/mood/instrument keywords in request, candidate era/genre match, album continuity (same-album-as-last, album-in-recent-window), within-artist popularity/transition ranks, semantic-bucket match counts | |
| F. Stage-3 bucket | SASRec bucket-origin and rank features | +0.0007 Blind A |

**Training data construction** is the pipeline run in dump mode: the full
9-source retrieval executes on 6,000 held-out training sessions and writes one
row per (turn, candidate) — 24,718 groups, 77.7M rows, 18 GB NPZ, positive rate
0.00035. LambdaMART is the only ranker family we found that thrives on this
shape: it computes pairwise gradients within each group (one positive vs ~3,100
negatives), so extreme imbalance is handled natively, no sampling tricks.

**Session-level discipline everywhere:** the 6,000 dump sessions are excluded
from two-tower training; the co-occurrence table excludes the dump sessions;
train/valid splits are at session level; the golden-200 dev holdout used for
model selection is never trained on.

## 6. Two-tower training, exact

- Base: `intfloat/multilingual-e5-base` (279M). LoRA r=32, alpha=64, dropout
  0.05, on q/k/v. 1.8M trainable params.
- Data: ~74K anchor/positive pairs from 15,199 train sessions (minus the 6K LTR
  dump sessions), v8d role-tagged anchors, gpa-corrected positive weighting
  (rejected-turn positives dropped with p=0.7), hard negatives as in Section
  3.7.
- Loss: MultipleNegativesRankingLoss (InfoNCE with in-batch negatives) plus 2-3
  explicit hard negatives per row.
- Schedule: 3 epochs, batch 8 x grad-accum 4 (effective 32), lr 1e-4, warmup
  200, gradient checkpointing.
- Output: a PEFT LoRA adapter served on top of the base `multilingual-e5-base`.
  The shipped artifact `models/twotower_v8d/final` is the adapter plus tokenizer
  (24.2 MB on disk: 7.1 MB `adapter_model.safetensors` + 17 MB tokenizer); the
  base encoder (~1.1 GB f32) is loaded from the Hugging Face cache at runtime and
  the adapter applied. Track index = one forward pass over the catalog ->
  47,071 x 768 float32 matrix (146.5 MB), L2-normalized, brute-force matmul at
  query time. (The adapter can optionally be merged into the base weights to ship
  a single standalone encoder; we serve the unmerged adapter.)

## 7. Response generation and joint modeling with retrieval (the 0.30 lever)

*(Topic: joint modeling of ranked retrieval and response generation.)*

The composite scores retrieval and response together, but the judge reads only
response text. We therefore model the two objectives jointly-but-decoupled: the
same per-turn context feeds both stages, the ranker is trained on
logged-next-track relevance, and the response stage is optimized independently
for the text-only judge. The coupling is one-directional and deliberately weak
(the response may *describe* the ranked list but never *reorder* it), which
Section 9 (result 1) shows is not a simplification but a requirement.

The judge reads only response text. Empirically it dominated our composite
movement more than retrieval:

| Blind A sub | nDCG@20 | Judge | Composite |
|---|---|---|---|
| v04 (generic responses) | 0.3709 | 1.10 | 0.2771 |
| v07 (rich responses) | 0.3164 | 4.40 | 0.4837 |
| v21 (best retrieval + rich) | 0.3997 | 3.95 | 0.4935 |

A +0.05 nDCG gain moves composite +0.025; moving the judge from 3.85 to 4.7
moves it +0.064. Findings that transferred to Blind B:

- **Richness drives the judge, not the model.** Same model, same prompt: longer
  responses with concrete evidence (specific tag + album + year tied to the
  request) score higher. Terseness constraints measurably hurt.
- The winning response recipe: 4-5 substantive sentences; 2-3 concrete metadata
  evidence points; explicit callback to a prior played track when history
  exists; profile/culture callback when a profile exists (never for cold
  sessions); a synthesizing closer; no templated openers (distinct openings
  across the 80 sessions protect LexicalDiversity).
- **Pivot-awareness belongs in the text, not the ranking.** When the user says
  "enough of this artist," the response acknowledges the pivot — but the track
  list stays untouched (Section 9 explains why).

## 8. Blind B: preference elicitation without a ground-truth feedback channel

*(Topics: multi-turn preference elicitation; personalization under missing
profile/context.)*

Blind B removes exactly the signals the gpa-aware machinery consumes: no goal,
no progress labels, no thoughts, and half the sessions have no user profile.
This is the sharpest test of dialogue-aware modeling in the competition: the
system must elicit and track preferences from the raw conversation alone.

**Step 1 — simulate before submitting.** `--simulate_blindb` strips those
fields from labeled dev sessions (and blanks profiles on half, matching the
40/80 cold split), so any Blind B config can be measured locally. On the
golden-200 holdout:

| Config | nDCG@20 (sim) |
|---|---|
| Full information (real goal + gpa) | 0.1841 |
| Blind-B condition (stripped, reactions inferred) | **0.1864** |

Losing goal/gpa/thoughts costs nothing. The goal text is sometimes actively
misleading; the learned features carry the signal. This robustness is not
accidental — it falls out of multi-source agreement dominating the rescorer and
of the gpa features being *reconstructable*:

**Step 2 — reconstruct the missing feedback channel.**
`--infer_progress_labels` classifies each user follow-up with two small regex
banks (acceptance: "love it", "perfect", "more like this"; rejection: "not
what", "something different", "too slow"). The inferred label fills the
`REACTION:` slot of the anchor and drives H1 seed filtering and all H2 session
features — noisier than real gpa, but present. `--goal_substitute_positive`
fills the empty `[GOAL]` slot with the last accepted track. The
consecutive-rejection goal-drop (H3b) is disabled — it needs real labels and
hurt with inferred ones.

**Step 3 — cold-start degradation is structural, not special-cased.** Cold
sessions simply have: empty `[PROFILE]` slot, no CF source (source 6 returns
nothing), profile-free responses. Everything else operates unchanged.

**The submitted v2:**

- Retrieval: the exact command in `wiki/BLIND_B.md` — all 9 sources, inferred
  reactions, goal substitution, 67-feature s3cap LTR, top-20 emitted, no
  post-processing of any kind.
- Responses: LLM-written per session against the full conversation + profile
  context, following the Section 7 recipe, with honest-mismatch framing (when
  the exact requested track is not rank 1, the response says so and explains
  the pick) and pivot acknowledgment in text.
- Result: **composite 0.49**, our best Blind B score. The follow-up v3, which
  let an LLM prune "obviously wrong" tracks from the top-20, scored **0.48** —
  a live confirmation of the central negative result:

## 9. Negative results and failure-case analysis (each one load-bearing)

*(Topic: analysis of conversational-recommendation behavior and failure cases.)*

1. **Never edit the LTR top-20 post-hoc.** Four independent confirmations:
   per-artist cap (sim 0.1864 -> 0.1433), exact-title boost (-> 0.1768),
   pivot-conditional artist suppression (-> 0.1656; on fired turns nDCG
   collapsed 0.1766 -> 0.0444), and the real v3 submission (0.49 -> 0.48).
   Root cause, measured: even on turns where the user verbally pivots, the
   logged next track is still by a history artist ~40% of the time. TalkPlay
   sessions are artist deep-dives; the gold hides inside the very clusters a
   human curator would prune. The booster already knows this; hand rules
   override it and lose. Human-perceived relevance and logged-next-track nDCG
   are different objectives — serve the first in text, the second in ranking.
2. **Zero-shot rerankers are poison here.** Pre-trained cross-encoder: 0.1228
   dev (vs 0.1684 base). Qwen3-8B reranker on Blind A: nDCG -0.068 and lexical
   diversity collapsed 0.5909 -> 0.0125 (near-identical picks across sessions).
   A fine-tuned CE only broke even. The engineered-feature LambdaMART already
   extracts the signal a reranker would add, at ~10,000x less compute.
3. **Neural listwise LTR fails at this pool shape.** ListNet MLP: softmax over
   ~3,100 candidates with 1 positive gives gradient ~0.0006 at the positive;
   loss pinned at log(pool) forever. Pairwise tree boosting is the right tool.
4. **Semantic-ID bucket recall has a hard ceiling.** SASRec next-bucket
   hit@3 = 42.1%, centroid-match 47.3% (target was >90% for a primary recall
   channel). Bucket expansion is kept only as a capped auxiliary source
   (+0.0007 Blind A); a bucket-centroid variant with retrained LTR regressed
   (0.1841 -> 0.1809). Sequence-of-buckets does not determine next-track
   bucket; intent does.
5. **Hand-tuned improvements to BM25 hurt.** Field weighting, stopword removal,
   stemming, query reformulation by a small LLM — all flat or negative. IDF
   dilution beats cleverness at this corpus size.
6. **More context is not free.** Full 8-turn histories in the BM25 query and
   pre-trained dense retrieval both underperform focused 4-turn context.

## 10. Efficient and scalable architecture: the entire system on one CPU-class machine

*(Topic: efficient and scalable architectures for large-scale music
recommendation.)*

**Hardware for all results in this document:** one Apple M4 Mac mini, 16 GB
unified memory, macOS. No CUDA device, no cloud training. The two-tower
fine-tune used the Metal backend (consumer unified memory, LoRA-sized);
everything else — every index build, the LTR training, and all inference
components — is CPU-native (numpy, bm25s, LightGBM). The benchmark script
(`scripts/analysis/bench_cpu_components.py`) forces `device=cpu` for every
neural component to make the CPU claim exact.

### 10.1 Why this architecture is CPU-cheap by construction

- **Catalog-scale brute force beats ANN indexes.** 47,071 x 768 float32 is
  ~147 MB; a full exact similarity scan is one matmul (~72 MFLOPs) — low
  milliseconds on any modern CPU (measured 2.2 ms median, Section 10.2). No
  FAISS, no HNSW build, no recall loss.
- **The heaviest trained component is a 279M encoder used once per turn.** One
  512-token forward pass per turn per encoder (e5-base + Qwen3-0.6B). No
  per-candidate neural scoring anywhere — the per-candidate work is a tree
  ensemble on 67 floats.
- **The rescorer is ~0.5 MB of trees.** Scoring 3,100 candidates is
  single-digit milliseconds. Training it is a LightGBM job, not a GPU job.
- **Track-side embeddings are precomputed once** (index build = one pass over
  47K passages) or shipped with the dataset (Qwen3, CLAP, CF-BPR).
- **LoRA makes the one neural fine-tune fit in 16 GB.** 1.8M trainable params;
  optimizer state ~40 MB vs ~3 GB for full fine-tuning, which does not fit.

### 10.2 Measured component latencies (CPU-forced)

Measured on the M4 (10 cores), `device=cpu` forced, median over 20 repeats,
OMP_NUM_THREADS=4 except where noted. Sources:
`scripts/analysis/bench_cpu_components.py` (encoders, ANN, sizes),
`scripts/analysis/bench_bm25_ltr.py` (BM25, LTR), consolidated into
`exp/analysis/bench_cpu_components.json`.
<!-- BENCH:BEGIN -->

| Component | Median | p90 |
|---|---|---|
| TT query encode (e5-base, ~400-token anchor) | 46.8 ms | 47.8 ms |
| Exact ANN, 47,071 x 768, top-2000 | 2.2 ms | 2.3 ms |
| Qwen3-0.6B query encode (single-thread, OMP=1) | 601 ms | — |
| Qwen3 exact ANN, 54,476 x 1024, top-500 | 3.2 ms | 3.3 ms |
| BM25 retrieve top-500 | 0.5 ms | 0.6 ms |
| LTR score 3,100 candidates x 67 features | 3.4 ms | 3.7 ms |
| TT passage encode, batch-32 (index-build unit) | 193 ms | 194 ms |

Memory (measured RSS): 280 MB for the BM25 + LightGBM ranking stage in
isolation; **1.85 GB** with both text encoders (e5-base + Qwen3-0.6B) and the two
largest track indexes resident; the full pipeline with every index loaded is
~2.3 GB, comfortably inside 16 GB.

Two engineering notes the benchmark surfaced, both relevant to a CPU deployment:
(1) importing `torch`/`transformers` in the same process as `bm25s` + `lightgbm`
triggered a native-threadpool deadlock (0% CPU) on this machine, so the ranking
stage is benchmarked (and is cleanest to serve) in a process that does not import
the deep-learning stack; (2) the Qwen3-0.6B custom modeling code deadlocks on CPU
encode under `OMP_NUM_THREADS>1` and must be run single-threaded, which is also
its slowest-per-turn component (601 ms) and the strongest argument for the
eight-source lean config (Section 10.6), which can drop it entirely.
<!-- BENCH:END -->

### 10.3 Measured pipeline wall times (production logs)

| Job | Wall time | Per unit |
|---|---|---|
| Golden-200 eval (200 sessions, full 9-source pipeline + LTR) | 7 min 14 s | 2.17 s/session |
| Full dev eval (1,000 sessions, 8,000 turns) | ~35-40 min | ~0.26 s/turn amortized |
| Blind submission inference (80 sessions) | ~3 min | |
| LTR feature dump (6,000 sessions, the LTR training data) | ~3-4 h | one-off per feature-set change |
| Two-tower LoRA fine-tune (74K pairs, 3 epochs) | ~10 h | ~16 s/step, effective batch 64, one-off |
| TT index build (47,071 tracks) | tens of minutes | one-off per model |
| LightGBM training (77.7M rows x 67 feat, 5-fold CV) | ~1-2 h | one-off |

The iteration loop that actually mattered — change a feature, re-dump if pool
composition changed, retrain LTR, golden-200 check, full-dev gate — turns
around in under half a day on this machine, which is what made ~30 recorded
ladder iterations feasible.

### 10.4 Artifact inventory (measured, `bench_cpu_components.py`)

Project-shipped artifacts on disk:

| Artifact | Size |
|---|---|
| Two-tower LoRA adapter + tokenizer (served, `models/twotower_v8d/final`) | 24.2 MB |
| Two-tower track index (47,071 x 768 f32) | 146.5 MB |
| BM25 index | 21.9 MB |
| Qwen3 metadata index (lyrics index the same; attributes comparable) | 225.3 MB |
| CLAP audio index | 113.7 MB |
| Co-occurrence table (leak-free, 6k-excluded) | 1.7 MB |
| SASRec checkpoint | 22.2 MB |
| Semantic-ID tables | 8.0 MB |
| LTR booster (67-feat s3cap / 60-feat tier1) | 0.6 MB / 0.4 MB |
| LTR training dump (transient, deleted after training) | 18 GB |

Runtime dependencies pulled from the Hugging Face cache (not shipped in-repo):

| Base model | Params | Approx size (f32) | Used for |
|---|---|---|---|
| `intfloat/multilingual-e5-base` | 279M | ~1.1 GB | two-tower query + index encoding (adapter applied) |
| `Qwen/Qwen3-Embedding-0.6B` | 0.6B | ~2.3 GB | source-5 query encoding (track side precomputed) |

Serving-footprint reality check: the *repo* artifacts are only ~0.6 GB, but the
honest memory number is dominated by the two base encoders loaded on CPU plus the
indexes. The benchmark's measured peak process RSS after loading every component
is reported in Section 10.2; that is the number to cite for RAM, not the on-disk
total. The eight-source lean config (Section 10.6) drops SASRec (22.2 MB) and the
semantic-ID tables (8.0 MB) and can drop the Qwen3-0.6B runtime dependency if
source 5 is also removed — the single largest RAM saving available.

### 10.5 What this implies for deployment

Per-turn neural cost is dominated by the two encoder forward passes: the
Qwen3-0.6B query encode (601 ms single-thread) and the e5-base query encode
(47 ms). Everything else per turn is single-digit milliseconds — both exact-ANN
scans (2.2 ms and 3.2 ms), BM25 (0.5 ms), and ranking 3,100 candidates (3.4 ms).
This is the concrete case for the eight-source lean config (Section 10.6):
dropping source 5 removes the single most expensive per-turn component and its
~0.7 GB of resident model, taking per-turn latency to well under 100 ms of
neural work. At the measured ~2 s/session (8 turns of history processing
included), a single commodity machine serves interactive conversational
recommendation over a ~50K-item catalog with no accelerator, and the whole
train/eval/iterate loop — including the only neural fine-tune — runs on the same
box. Scaling the catalog 10x moves the exact-ANN scan from ~147 MB to ~1.5 GB
and from ~2 ms to tens of milliseconds before an ANN index is even worth
considering.

### 10.6 A leaner eight-source configuration (drop the sequential-ID stack)

The ninth source, SASRec semantic-bucket expansion (Section 4, source 9), is the
most expensive component to *build* and the weakest to *keep*. It requires an
entire offline pipeline that nothing else in the system needs: train a two-level
RQ-VAE (64 codes/level) over track attribute embeddings to assign semantic IDs,
then train a SASRec sequence model to predict the next bucket from the session's
bucket sequence. At serve time it adds a SASRec forward pass and a bucket lookup
per turn, plus seven bucket-origin features to the ranker (the 67-feature s3cap
booster vs the 60-feature tier1 booster).

Its measured value is marginal. Bucket next-hit ceilings are low (hit@3 = 42.1%,
centroid-match 47.3%; Section 9.4), so buckets are only a capped auxiliary source
worth +0.0007 nDCG on Blind A. On the Blind-B-simulated golden-200 holdout, the
**eight-source configuration** (no SASRec, no bucket features, 60-feature tier1
LTR `ltr_v8d_tier1_nl31_lr0p08.txt`) scores **0.1894** vs **0.1864** for the
shipped nine-source s3cap config — a small, reproducible improvement on the
simulator, i.e. dropping the stack is at worst neutral and possibly slightly
positive. (This lean config was fully packaged as a Blind B candidate but not
spent on the live leaderboard; its evidence is the golden-200 simulator, not a
leaderboard score. The two submissions that were spent, v2 at 0.49 and v3 at
0.48, both used the nine-source config.)

What the eight-source variant removes:

| Removed | Kind | Saving |
|---|---|---|
| RQ-VAE codebook training | offline pipeline | one training run + third-party dependency |
| SASRec sequence-model training | offline pipeline | one training run |
| Semantic-ID assignment over catalog | offline pass | eliminated |
| SASRec forward pass + bucket lookup per turn | per-turn inference | small but nonzero |
| SASRec checkpoint + semantic-ID tables on disk | artifacts | 22.2 MB + 8.0 MB |
| 7 bucket-origin features | ranker input | 67 -> 60 features |

Why recall barely moves: every source in the pool feeds the `n_sources`
agreement feature, so removing a low-precision auxiliary source that rarely
uniquely-covers the gold leaves the ~89% pool recall essentially intact. The
recall the buckets contributed was almost entirely redundant with the other
eight sources.

Recommendation for a from-scratch reimplementation: **start with the eight
sources.** It reaches parity on the simulator, drops two model-training pipelines
and a third-party dependency, and shortens the critical build path to: fine-tune
one LoRA encoder, precompute one track index, build a BM25 index, load the
shipped Qwen3/CLAP/CF embeddings, build the leak-free co-occurrence table, dump
features, train one LightGBM ranker. The sequential-ID stack can be added later
as an ablation if a bucket-recall breakthrough (Section 9, result 4) ever raises its
ceiling.

## 11. Condensed lessons

1. **Fuse many weak retrievers; let a tree ranker arbitrate.** Multi-source
   agreement was the strongest single signal in the system.
2. **Pool recall is necessary, never sufficient.** Every recall gain was
   worthless until the scorer could exploit it.
3. **Label hygiene beats architecture.** The off-by-one feedback label, the
   43% contaminated positives, and the co-occurrence leakage each moved the
   score more than most model changes.
4. **Optimize the metric you are scored on, not the one you feel.** Post-hoc
   "obvious" track-list cleanups regressed nDCG every single time; the honest
   place for user-empathy is the response text, which the judge reads.
5. **Simulate the blind condition before spending a submission.** The
   `--simulate_blindb` harness predicted that losing goal/gpa/profile costs
   nothing, and it was right.
6. **Consumer hardware is not a limitation for this problem class.** Brute
   force at 50K-catalog scale, LoRA for the one fine-tune, trees for ranking —
   nothing in the loop needs a GPU cluster, and the fast iteration this enables
   is itself a competitive advantage.
