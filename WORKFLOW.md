# Music CRS — Complete Workflow

## What This System Does

Given a multi-turn conversation between a user and a music assistant, predict which 20 tracks the assistant should recommend at each turn. Evaluated by nDCG@20 (does the actual gold track appear in our top-20, and how high is it ranked?).

---

## Datasets

| Dataset | HuggingFace ID | What It Contains |
|---|---|---|
| Track Metadata | `talkpl-ai/TalkPlayData-Challenge-Track-Metadata` | 47,071 tracks: name, artist, album, tags, release date |
| Conversations | `talkpl-ai/TalkPlayData-Challenge-Dataset` | 1,000 test sessions, each with up to 8 turns |
| Track Embeddings | `talkpl-ai/TalkPlayData-Challenge-Track-Embeddings` | 46,424 tracks × 4480-dim vector each (multi-modal: audio+image+CF+text concatenated) |
| User Embeddings | `talkpl-ai/TalkPlayData-Challenge-User-Embeddings` | CF-BPR + other embeddings per user |
| Blind A | `talkpl-ai/TalkPlayData-Challenge-Blind-A` | 80 sessions for leaderboard submission |

Track metadata has two splits: `all_tracks` (47,071) and `test_tracks` (7,405, a subset of all_tracks). All gold tracks in the test set are in `all_tracks`.

---

## Conversation Structure

Each session has up to 8 turns. Each turn has three entries:
- `role: user` — the user's natural language request
- `role: music` — the track ID the system recommended (this is what we predict)
- `role: assistant` — the system's text response

```
Turn 1 user:      "Play me something upbeat to work out to"
Turn 1 music:     "abc123-track-id"   ← we predict this
Turn 1 assistant: "I recommend X by Y..."
Turn 2 user:      "I liked that, give me something similar"
Turn 2 music:     "def456-track-id"   ← we predict this
...
```

---

## Evaluation

```bash
python scripts/evaluate_local.py --pred exp/inference/devset/PREDICTIONS.json
```

Computes nDCG@{1,10,20} and Hit@{1,10,20}. Primary metric: **nDCG@20**.

The gold track for each turn is the `role: music` entry. If our top-20 list contains the gold track at rank R, the score is `1/log2(R+1)`. If not in top-20, score is 0.

---

## Complete Pipeline (Best System)

### Step 1 — Build the BM25 Index

```bash
# First run only — auto-caches to cache/bm25/track_metadata/
python scripts/run_inference_bm25.py --sessions 1 --tid warmup
```

The BM25 index stores one document per track:
```
track_name: Fluorescent Adolescent
artist_name: Arctic Monkeys
album_name: Suck It and See
release_date: 2011
tag_list: indie rock, british, punk, catchy, alternative...
```

**Why include tag_list:** Users ask things like "I want something melancholic and acoustic". Without tags, BM25 can only match exact track/artist names. Tags like `["melancholic", "acoustic", "folk"]` let BM25 match on mood and genre vocabulary. This alone lifted nDCG@20 from ~0.077 (name+artist+album only) to 0.0861 (+12%).

### Step 2 — Build the Retrieval Query

For each turn we want to predict, we build a single text string and run BM25 search against the track corpus.

**Query structure:**
```
{listener_goal}
{preferred_musical_culture}
{track_name_1} {artist_1} {tags_1}   ← previous recommended track (with full metadata)
{track_name_2} {artist_2} {tags_2}   ← previous recommended track (with full metadata)
... (last 4 recommended tracks)
{user_text_turn_1}                   ← previous user/assistant text (last 4 turns)
{user_text_turn_2}
...
{current_user_query}                 ← the actual request this turn
```

**Why include previous track metadata (name+artist+tags):**
The conversation builds context. If the user previously got "Fluorescent Adolescent" and liked it, including its tags (`indie rock, british, punk, catchy`) in the query helps BM25 find other tracks with similar genre/mood. The competition baseline only included the raw track ID or plain name — adding full tags raised nDCG@20 from 0.0861 to 0.0960 (+11%).

**Why include goal and culture:**
`listener_goal` describes the user's high-level intent (e.g., "play one specific popular song from a genre"). `preferred_musical_culture` (e.g., "Anglo-American Rock") gives genre context that helps BM25 find tracks in the right space.

### Step 3 — Retrieve BM25 Candidates and Exclude Seen Tracks

```python
# Retrieve extra candidates to compensate for filtering
retrieve_k = topk + len(seen_tracks) * 3

tokens = bm25s.tokenize([query.lower()])
results = bm25_model.retrieve(tokens, k=retrieve_k)
candidates = [track_ids[int(i)] for i in results.documents[0]]

# Remove tracks already recommended in this conversation
seen = set(previously_recommended_track_ids)
candidates = [t for t in candidates if t not in seen]

top_20 = candidates[:20]
```

**Why exclude seen tracks — the biggest single improvement (+37%):**
When the conversation references a previously recommended track ("Yes, Fluorescent Adolescent is great! Can you give me something similar?"), the BM25 query contains that track's name. BM25 then retrieves that same track as its top result (since it perfectly matches). This pushes the actual gold track down.

Before exclusion: BM25 returns [Fluorescent Adolescent, gold_track, ...]  → gold at rank 2 → low nDCG
After exclusion: BM25 returns [Fluorescent Adolescent, gold_track, ...]  → filter out seen → gold at rank 1 → high nDCG

This lifted nDCG@20 from 0.0960 to 0.1313 (+37%) and nDCG@1 from 0.9% to 4.4%.

---

## Running Inference

### Dev/Test Set

```bash
source .venv/bin/activate

# Best system: BM25 + tag expansion + seen exclusion
python scripts/run_inference_bm25_tagexpand.py \
  --sessions 0 \          # 0 = all sessions
  --tid my_run \
  --out_dir exp/inference/devset

# Evaluate
python scripts/evaluate_local.py --pred exp/inference/devset/my_run.json
```

### Blind A (Leaderboard Submission)

```bash
# Step 1: Generate retrieval predictions
python scripts/run_inference_blind.py \
  --dataset talkpl-ai/TalkPlayData-Challenge-Blind-A \
  --tid blind_a_v2 \
  --out_dir exp/inference/blind_a

# Step 2: Replace template responses with Qwen-generated responses
python scripts/generate_responses_blind.py \
  --pred exp/inference/blind_a/blind_a_v2.json
# Output: exp/inference/blind_a/blind_a_v2_qwen.json
```

**Difference between blind and dev scripts:**
- Dev set: iterate over turns that have a `role: music` entry (ground truth available)
- Blind set: predict for `conversations[-1]` (last turn, no ground truth yet)

---

## What Was Tried and Rejected

### CF-BPR Reranking (HURT: -4%)
The challenge provides 128-dim collaborative filtering embeddings for users and tracks. Idea: after BM25, rerank candidates by user-track CF cosine similarity.

Why it hurts: CF captures long-term user taste (similar users liked similar tracks). But the gold track was chosen for a specific conversational context ("give me something like Arctic Monkeys"). CF pushes "popular with similar users" tracks up, which aren't necessarily the contextually correct choice.

### Dense Reranking with all-MiniLM-L6-v2 (HURT: -12%)
Idea: encode the query with a sentence transformer, retrieve tracks whose text embeddings are most similar.

Why it hurts: The query is conversational ("I want something upbeat") but track metadata is flat text ("Levitating Dua Lipa pop dance"). These don't embed close together in a general-purpose sentence embedding model. A domain-specific fine-tuned model would be needed.

### CF Candidate Expansion (No effect)
Idea: add top-50 CF-similar tracks to the BM25 pool, rank all by BM25. Adds recall without hurting precision. Result: no improvement. The gold tracks not found by BM25 are also not ranked highly by CF.

### Qwen 0.5B Query Reformulation (Slightly hurt: -1%)
Idea: use Qwen to extract artist names, genres, moods from the conversation and add them as BM25 query terms. The small model (0.5B parameters) doesn't reliably extract useful terms — it generates noise that slightly hurts BM25 quality.

---

## Score Progression (Complete)

| System | nDCG@20 | Hit@20 | Notes |
|---|---|---|---|
| BM25, name+artist+album only | 0.0861 | 21.9% | baseline |
| + tag_list in corpus + query expansion | 0.0960 | 25.9% | +11% |
| + exclude already-recommended tracks | 0.1313 | 27.4% | +37%, prev best |
| + CF-BPR reranking | 0.1262 | 27.6% | WORSE |
| + all-MiniLM dense (pretrained, no fine-tune) | 0.0775 | 17.5% | WORSE |
| + Qwen3 metadata slice (no fine-tune) | <0.1313 | — | WORSE |
| Two-tower v1 (long query 500tok, 1ep, w=0.3, pool=200) | 0.1364 | 29.0% | +3.8% |
| Two-tower v3 (compact query 101tok, 2ep, w=0.7, pool=200) | 0.1406 | 29.4% | +7.1% |
| **Two-tower v3, w=0.7, pool=500** | **0.1418** | **29.8%** | **+8.0%, current best** |
| Two-tower v3b (3rd epoch from v3) | 0.1416 | 29.5% | converged, no gain |
| v3 + Semantic clusters K=500 c=3 w=0.7 (hybrid) | 0.1403 | 29.4% | pool recall 76.6% but no nDCG gain |
| v3 + Semantic clusters K=500 c=3 RRF | 0.1396 | 30.1% | RRF more hits but lower precision |
| v3 + Dense top-500 + BM25 top-500 RRF | 0.1401 | 30.2% | same issue: more hits, lower nDCG |
| v3 + pool=1000 | 0.1417 | 29.8% | plateaued, pool size not the bottleneck |
| Two-tower v4 (hard neg fine-tune, 1ep, lr=1e-5) | TBD | TBD | training in progress |

BM25 recall ceiling (measured on 100 sessions):
- Hit@20: 27%
- Hit@100: 44%
- Hit@500: 59%  ← two-tower reranks within this

Pool recall (measured on 100 sessions, K=500 clusters):
- BM25 top-500: 60.0%
- BM25+clusters merged: 76.6% (+16.6% from clusters)
- Note: clusters add recall but NOT nDCG — cluster-only gold tracks rank below top-20

Key finding: the bottleneck is RERANKER QUALITY, not recall. Within the 58.8% pool where gold exists, we rank it in top-20 only ~50% of the time. Dense recall expansion adds hits at positions 15-20 only, with minimal nDCG impact.

---

## Two-Tower System (Current Best)

### What Was Tried and Why It Worked

**Critical bug fix — query truncation:** Full BM25 query is 500+ tokens median. all-MiniLM-L6-v2 has a 256-token limit. 81% of queries were truncated before the user's actual request (which comes last). Fix: compact query puts user request FIRST:
```
{latest_user_turn} {goal} {culture} {last_2_track_name} {last_2_track_artist}
```
Median 101 tokens. This alone was responsible for most of the v3 vs v1 improvement.

**Two separate queries:** BM25 still uses the full long query (good for recall). Dense encoder uses compact query (fits in 256 tokens, user request always visible).

**Training:**
- Base model: sentence-transformers/all-MiniLM-L6-v2 (384-dim, 256-token limit)
- Loss: MultipleNegativesRankingLoss (in-batch negatives, batch=32 → 31 negatives/example)
- Data: 115,520 (compact_query, gold_track_text) pairs from train split
- Epochs: 2 (converged, 3rd epoch no gain)
- Script: `scripts/train_twotower.py --data_dir data/twotower_v3 --epochs 2 --batch_size 32`

**Inference:**
- BM25 retrieves pool=500 candidates (full query)
- Dense encoder scores those candidates (compact query)
- Combined score: 0.7 × cosine + 0.3 × BM25_reciprocal_rank
- Script: `scripts/run_inference_twotower_v3.py`

### What Was Tried and Failed

- **Long query (v1/v2)**: 81% truncation, model never sees user request → only +3.8% vs BM25
- **Dense pool expansion (v1/v2)**: adding dense top-100 to BM25-200 gave no improvement
- **Qwen3 metadata slice**: zero-shot, no fine-tuning → worse than BM25
- **3rd training epoch**: converged, no gain
- **Semantic cluster recall (K-means, K=200/300/500)**: adds 16.6% pool recall but hurts nDCG (cluster-only gold tracks score below BM25 candidates in all scoring schemes: weighted sum, RRF)
- **Full dense recall (brute-force cosine over 47k) + BM25 RRF**: Hit@20 +0.4% but nDCG@20 -0.17% — same issue as cluster recall
- **BM25 pool=1000**: identical to pool=500, confirms pool size not the bottleneck
- **Pure dense (no BM25)**: 0.1163 nDCG@20 — dense alone is much worse than BM25

---

## Next: Hard Negative Training (Plan v4)

### Motivation

When the gold track is in the BM25 pool, the two-tower model ranks it in top-20 only ~50% of the time. The model uses in-batch negatives (random tracks from other queries), which are easy negatives. The gold track loses to BM25-rank-1 tracks that lexically match the query but are NOT the gold track.

Hard negatives (BM25 top-K non-gold tracks) are lexically similar to the gold track. Training with these forces the model to learn finer semantic distinctions, improving precision at top ranks.

### Implementation

Training data already has `negative_1` through `negative_5` (BM25 top-5 non-gold tracks per query, from `data/twotower_v3/`).

Changes to `train_twotower.py`:
- Added `--hard_neg` flag
- When enabled, loads `negative_1` from training data as an explicit `negative` column
- `MultipleNegativesRankingLoss` uses the negative as a hard negative alongside in-batch negatives

Training command:
```bash
python scripts/train_twotower.py \
    --data_dir data/twotower_v3 \
    --out_dir models/twotower_v4 \
    --base_model models/twotower_v3/final \
    --epochs 1 --batch_size 32 --lr 1e-5 --warmup_steps 100 --hard_neg
```

### Expected Outcome

Hard negatives are standard in dense retrieval (DPR, ANCE). Expected: +1-3% nDCG@20.
The model should learn to push the gold track above lexically similar non-gold tracks.

---

## Previous Attempt: Semantic Cluster Recall (Plan v4 - Abandoned)

### Motivation

BM25 Hit@500 = 58.8%. The two-tower only reranks within the BM25 pool — it can't recover the 41.2% of gold tracks that BM25 misses entirely. A parallel dense recall path is needed.

### Approach

Build K-means clusters (K=200/300/500) from the fine-tuned v3 track embeddings.
These clusters are text-predictable (same encoder trained for music queries).

At query time:
1. Encode compact query → cosine sim vs cluster centroids → top-5 clusters (~750 tracks)
2. Merge with BM25 top-500 (~1000-1200 unique candidates)
3. Rerank combined pool with v3 model → top-20

Expected: pool recall goes from 58.8% to 65-75%, translating to +2-5% nDCG@20.

Scripts created (results above):
- `scripts/build_semantic_codebook.py`
- `scripts/run_inference_hybrid_recall.py`

---

## Key Scripts

| Script | Purpose |
|---|---|
| `scripts/run_inference_bm25_tagexpand.py` | BM25-only inference (0.1313) |
| `scripts/run_inference_twotower_v3.py` | Best system: BM25+two-tower (0.1418) |
| `scripts/train_twotower.py` | Two-tower fine-tuning |
| `scripts/build_twotower_data.py` | Build (compact_query, gold_track) training pairs |
| `scripts/build_twotower_index.py` | Encode all tracks with fine-tuned model |
| `scripts/run_inference_blind.py` | Blind set inference |
| `scripts/generate_responses_blind.py` | Qwen responses for blind set |
| `scripts/evaluate_local.py` | Local nDCG@20 evaluation |
| `scripts/eval_semantic_candidate_recall.py` | Diagnostic: codebook recall ceiling |
| `scripts/build_dense_index.py` | pretrained all-MiniLM index (not used in best system) |
| `scripts/build_semantic_codebook.py` | K-means codebook from v3 track embeddings |
| `scripts/run_inference_hybrid_recall.py` | BM25+cluster hybrid (tried, no nDCG gain) |
| `scripts/run_inference_dense_bm25_rrf.py` | Dense top-K + BM25 RRF (tried, no gain) |
| `scripts/run_inference_blind_twotower.py` | Blind set with two-tower v3 (current best) |
