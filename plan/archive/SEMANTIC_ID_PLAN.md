# Semantic ID Implementation Plan
## Building on the Baseline → a Competitive Music-CRS Entry

---

## The Core Idea in Plain Language

The baseline works like this:

```
User says: "something melancholic like Nils Frahm"
    ↓
BM25 searches: "Nils Frahm" in track_name/artist_name fields
    ↓
Returns: top-20 track IDs by keyword overlap
    ↓
Llama reads: metadata of track #1 only → writes a response
```

The problem: retrieval and response are completely disconnected. The LLM only ever sees one track. It can't reason across the full ranked list. It can't learn from what the user said in previous turns. And BM25 has no understanding of "melancholic".

Semantic IDs fix all of this by giving the LLM a new vocabulary where every word is a catalog track:

```
User says: "something melancholic like Nils Frahm"
    ↓
Fine-tuned LLM reads: user profile + listening history (as ID tokens) + dialogue
    ↓
LLM generates: "<12> <303>  <45> <289>  <7> <412> ..." (20 track tokens)
    ↓
Codebook lookup: each 2-token pair → real track_id (instant, no search index)
    ↓
Same LLM continues: writes a grounded response about the tracks it just "retrieved"
```

One model. One forward pass. Retrieval and generation unified.

---

## What Already Exists (Current State)

```
music-crs-baselines/
├── mcrs/retrieval_modules/
│   ├── bm25.py                   ✅ works
│   ├── bert.py                   ✅ works
│   ├── semantic_ids.py           ⚠️  exists but needs fine-tuned model to be useful
│   └── constrained_decoding.py   ✅ complete
├── mcrs/lm_modules/llama.py      ⚠️  expand_vocabulary_for_semantic_ids() added
│                                     load_finetuned_lora() added
│                                     but no trained model yet
├── mcrs/db_item/music_catalog.py ✅  id_to_metadata(use_semantic_id=True) implemented
└── run_inference_semantic_ids.py ✅  inference pipeline complete

src/
├── quantize/build_semantic_ids.py  ✅ complete
├── model/music_crs_model.py        ✅ complete
├── train/dataset.py                ✅ complete
├── train/dpo_dataset.py            ✅ complete
├── train/train.py                  ✅ complete
├── infer/constrained_decoding.py   ✅ complete
├── infer/mmr.py                    ✅ complete
└── infer/run_inference.py          ✅ complete
```

**What's missing:** The trained model. Everything else is scaffolding — we need to actually run the pipeline on real data to produce a fine-tuned checkpoint that makes `semantic_ids.py` useful.

---

## Phase Overview

```
Phase 0  Get baseline score             (1 day)   → know your floor
Phase 1  Data download + codebook       (1 day)   → prerequisite for everything
Phase 2  Hybrid retrieval               (1 day)   → immediate nDCG gain, no training
Phase 3  SFT on training data           (2-3 days)→ the main model
Phase 4  DPO sharpening                 (1 day)   → ranking + diversity gains
Phase 5  Full inference + MMR           (0.5 day) → final pipeline
Phase 6  Blind A submission             (deadline)
Phase 7  Cold-start hardening           (for Blind B)
```

Each phase is independently runnable and produces a measurable dev set score. Never skip Phase 0 — you need a baseline number before you can claim improvement.

---

## Phase 0 — Establish Baseline Score (Day 1, ~4 hours)

**Goal:** Run the official baseline, get a Dev set score, understand the floor.

**Why first:** Every claim of improvement is relative to this number. If you don't have it, you can't tell if your changes help.

### Steps

```bash
cd music-crs-baselines

# 1. Install deps
pip install -e ".[dev]"

# 2. Run BM25 baseline on devset
python run_inference_devset.py --tid llama1b_bm25_devset --batch_size 16

# 3. Run BERT baseline on devset
python run_inference_devset.py --tid llama1b_bert_devset --batch_size 8

# 4. Evaluate both
cd ../music-crs-evaluator
python evaluate_devset.py --predictions ../music-crs-baselines/exp/inference/devset/llama1b_bm25_devset.json
python evaluate_devset.py --predictions ../music-crs-baselines/exp/inference/devset/llama1b_bert_devset.json
```

### What to record
Write down these numbers — they are your reference for every future experiment:

| Metric | BM25 score | BERT score |
|---|---|---|
| nDCG@20 | | |
| nDCG@10 | | |
| nDCG@1  | | |
| Catalog diversity | | |
| Distinct-2 | | |

**Expected:** BM25 will beat BERT on sessions where users mention artist names. BERT will beat BM25 on vibe/mood queries. Both will have weak LLM-as-Judge scores because only 1 track is passed to the LLM.

---

## Phase 1 — Data Download + Codebook Build (Day 1-2, ~6 hours)

**Goal:** Download all datasets, build the semantic ID codebook. This is the foundation that Phase 3 depends on.

### 1a. Download TalkPlayData

```python
# Run this once — downloads train/dev/test splits + metadata + embeddings
from datasets import load_dataset

# Conversation data (training examples)
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
ds.save_to_disk("./data/TalkPlayData-Challenge-Dataset")

# Track metadata (track_name, artist, album, tags, etc.)
meta = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
meta.save_to_disk("./data/TalkPlayData-Challenge-Track-Metadata")

# Pre-computed multimodal embeddings (audio + lyrics + CF) — this is the key asset
embeds = load_dataset("talkpl-ai/TalkPlayData-2-Track-Embeddings")
embeds.save_to_disk("./data/TalkPlayData-2-Track-Embeddings")

# User metadata + embeddings
user_meta = load_dataset("talkpl-ai/TalkPlayData-Challenge-User-Metadata")
user_meta.save_to_disk("./data/TalkPlayData-Challenge-User-Metadata")
```

### 1b. Extract embeddings to numpy

The HF dataset stores embeddings as lists. Extract them to `.npy` for fast numpy loading:

```python
import numpy as np
from datasets import load_from_disk

ds = load_from_disk("./data/TalkPlayData-2-Track-Embeddings")["all_tracks"]
track_ids  = ds["track_id"]
embeddings = np.array(ds["embedding"], dtype=np.float32)

np.save("./data/track_embeddings.npy", embeddings)
with open("./data/track_ids.txt", "w") as f:
    f.write("\n".join(track_ids))

print(f"Saved {len(track_ids):,} embeddings of dim {embeddings.shape[1]}")
```

### 1c. Build the semantic ID codebook

```bash
python src/quantize/build_semantic_ids.py --config config/train.yaml
# → data/codebook.pkl (takes ~20 min on CPU for 1M tracks)
```

**What this does inside `build_semantic_ids.py`:**

```
1M track embeddings (shape: [1_000_000, D])
        │
        ▼ L2 normalize
        │
        ▼ MiniBatchKMeans (256 clusters) — Level 1
        │
        ├── codes1[i] = which of the 256 coarse clusters track i belongs to
        │                 e.g. track "Clair de Lune" → cluster 12 (classical/piano cluster)
        │
        ▼ compute residuals (embedding - cluster_center)
        │
        ▼ MiniBatchKMeans (256 clusters) — Level 2
        │
        └── codes2[i] = which fine cluster the residual belongs to
                         e.g. "Clair de Lune" → fine cluster 47

Final ID: track → ("<12>", "<303>")  [fine token is offset by 256]
```

**Verify the codebook makes sense:**

```python
import pickle, numpy as np
with open("data/codebook.pkl", "rb") as f:
    cb = pickle.load(f)

# Check: similar tracks should share coarse codes
# e.g. all jazz tracks should cluster together
print(f"Unique (coarse, fine) pairs used: {len(cb['valid_pairs'])}")
print(f"Avg tracks per bucket: {1_000_000 / len(cb['valid_pairs']):.1f}")
# Good result: 5-20 tracks per bucket on average
# Bad result: >100 per bucket → too few clusters, increase n_coarse/n_fine
```

**Why 256×256 and not 512×512?**

- 256 coarse × 256 fine = 65,536 possible pairs, but only ~50,000-60,000 will actually be populated for 1M tracks
- Average bucket size: ~15-20 tracks per (coarse, fine) pair — small enough that the top-1 track in a bucket is almost always correct
- Total new tokens added to the LLM: 512 (256 coarse + 256 fine) — small enough not to hurt the base model
- Increasing to 512×512 would give finer granularity but needs more training data to learn all 1024 new tokens

### What the codebook stores

```python
codebook = {
    "km1": MiniBatchKMeans,          # level-1 KMeans model (for encoding new tracks)
    "km2": MiniBatchKMeans,          # level-2 KMeans model
    "track_to_codes": {              # track_id → (coarse_int, fine_int)
        "T00001": (12, 47),
        "T00002": (12, 203),
        ...
    },
    "codes_to_tracks": {             # (coarse_int, fine_int) → [track_id, ...]
        (12, 47): ["T00001", "T00892"],   # multiple tracks can share a bucket
        ...
    },
    "valid_coarse": {0, 1, ..., 255},    # set of coarse codes that have ≥1 track
    "valid_pairs": {(12,47), (12,203), ...},  # all populated (c, f) pairs
}
```

---

## Phase 2 — Hybrid Retrieval (No Training Required) (Day 2, ~4 hours)

**Goal:** Replace BM25-only with BM25+BERT hybrid via Reciprocal Rank Fusion. Immediate nDCG gain, zero training required. This is your quick win while the model trains.

**Expected gain: +2 to +5 nDCG@20 points over the best single retriever.**

### Why RRF works

BM25 returns a ranked list. BERT returns a ranked list. They disagree on order. RRF combines them by giving each track a score based on its rank in each list, not its raw score (so you don't need to normalise across different score scales):

```
RRF score(track) = 1/(60 + rank_in_bm25) + 1/(60 + rank_in_bert)
```

The constant 60 smooths out the difference between rank 1 and rank 2 being astronomically large in raw retrieval scores. Tracks that appear in the top of **both** lists get the highest combined score.

### Build it

Create `music-crs-baselines/mcrs/retrieval_modules/hybrid.py`:

```python
from .bm25 import BM25_MODEL
from .bert import BERT_MODEL

class HybridRetrieval:
    """
    BM25 + BERT fused via Reciprocal Rank Fusion (RRF).
    
    BM25 wins on named-entity queries: "Nils Frahm", "jazz from the 70s"
    BERT wins on semantic queries: "melancholic piano", "upbeat driving music"
    RRF takes the best of both: tracks that rank highly in EITHER retriever
    bubble to the top.
    
    The k=60 constant is standard from the RRF paper (Cormack et al., 2009).
    Empirically: k=60 works well, but you can tune it on the Dev set.
    """
    
    def __init__(self, dataset_name, split_types, corpus_types, cache_dir, k=60,
                 bm25_weight=1.0, bert_weight=1.0):
        print("Loading BM25 retriever...")
        self.bm25 = BM25_MODEL(dataset_name, split_types, corpus_types, cache_dir)
        print("Loading BERT retriever...")
        self.bert = BERT_MODEL(dataset_name, split_types, corpus_types, cache_dir)
        self.k = k
        self.bm25_weight = bm25_weight
        self.bert_weight = bert_weight

    def _rrf_merge(self, bm25_results: list[str], bert_results: list[str]) -> list[str]:
        scores: dict[str, float] = {}
        for rank, tid in enumerate(bm25_results):
            scores[tid] = scores.get(tid, 0.0) + self.bm25_weight / (self.k + rank + 1)
        for rank, tid in enumerate(bert_results):
            scores[tid] = scores.get(tid, 0.0) + self.bert_weight / (self.k + rank + 1)
        return sorted(scores.keys(), key=lambda t: scores[t], reverse=True)

    def text_to_item_retrieval(self, query: str, topk: int = 20) -> list[str]:
        # Retrieve more candidates than needed before merging
        bm25_results = self.bm25.text_to_item_retrieval(query, topk=100)
        bert_results = self.bert.text_to_item_retrieval(query, topk=100)
        return self._rrf_merge(bm25_results, bert_results)[:topk]

    def batch_text_to_item_retrieval(self, queries: list[str], topk: int = 20) -> list[list[str]]:
        bm25_batch = self.bm25.batch_text_to_item_retrieval(queries, topk=100)
        bert_batch = self.bert.batch_text_to_item_retrieval(queries, topk=100)
        return [
            self._rrf_merge(bm25_batch[i], bert_batch[i])[:topk]
            for i in range(len(queries))
        ]
```

Register in `__init__.py`:
```python
elif retrieval_type == "hybrid":
    from .hybrid import HybridRetrieval
    return HybridRetrieval(dataset_name, track_split_types, corpus_types, cache_dir)
```

Add config `config/llama1b_hybrid_devset.yaml` (copy bm25 config, change `retrieval_type: "hybrid"`).

Run and compare to Phase 0 scores. **This is your Phase 2 checkpoint.**

### Optional: replace BERT with provided multimodal embeddings

The baseline's BERT module re-encodes metadata text with `bert-base-uncased`. But the organizers provide pre-computed **multimodal track embeddings** (audio + lyrics + CF) at `talkpl-ai/TalkPlayData-2-Track-Embeddings`. These are far better.

To use them: in `bert.py`, replace the `build_index()` method to load from `data/track_embeddings.npy` instead of running the BERT encoder. This alone gives a significant quality jump on semantic queries, for zero training cost.

---

## Phase 3 — Supervised Fine-Tuning (Days 3-5, ~2-3 days of GPU time)

**Goal:** Fine-tune Llama-3.2-1B-Instruct to jointly do retrieval (emit semantic ID tokens) and response generation (emit explanatory text) in a single forward pass.

**This is the most important phase.** Everything else is incremental. SFT is where the base model learns the task.

### What the training data looks like

Each training example comes from one user turn in the Train split. Format:

```
<|system|>
You are a conversational music recommender. Given the user profile, listening 
history, and conversation, recommend exactly 20 tracks as semantic ID pairs 
(one per line), then write a natural language response.

[USER PROFILE] age_group: 25-34, gender: F, country_name: United States
[HISTORY] <12> <303> | <45> <289> | <7> <412> | ...  ← last 20 listened tracks as ID tokens
[DIALOGUE]
  user: I want something melancholic but beautiful, like Nils Frahm
  assistant: <12> <303> [prev recommended track ID]
  user: Maybe something with more piano?
<|assistant|>
<12> <303>        ← track 1 (ground truth, from Train annotations)
<45> <289>        ← track 2
<7>  <412>        ← track 3
...               ← 20 lines total
[RESPONSE] These tracks share Nils Frahm's introspective piano style...
```

**Input:** everything before `<|assistant|>`
**Target:** the 20 ID lines + `[RESPONSE]` text

The loss is computed only on the target tokens (the 20 IDs + response). Input tokens are masked with -100.

### Why encoding history as ID tokens matters

The baseline encodes user profile as text but ignores listening history entirely. With Semantic IDs, the listening history becomes a sequence of tokens the model already understands:

- `<12> <303>` is a specific region of music space — if many history tracks share prefix `<12>`, the model learns the user likes that cluster (classical/piano)
- The LLM can attend to these tokens the same way it attends to text — no separate user encoder needed

### Two-loss training

The model is trained with **two cross-entropy loss terms simultaneously**:

```python
# Both losses computed in the same forward pass
logits = model(input_ids)

# Loss A: on the 20 semantic ID tokens (trains retrieval)
retrieval_loss = cross_entropy(logits[id_positions], id_token_targets)

# Loss B: on the response text tokens (trains generation quality)
response_loss = cross_entropy(logits[response_positions], response_token_targets)

total_loss = cfg.retrieval_loss_weight * retrieval_loss + cfg.response_loss_weight * response_loss
```

Both `retrieval_loss_weight` and `response_loss_weight` default to 1.0. You can tune these — upweighting retrieval loss pushes nDCG higher; upweighting response loss pushes Distinct-2 and LLM-as-Judge higher.

### LoRA: why freeze the base model

Llama-3.2-1B-Instruct already knows English. We don't want to overwrite that — we want to add to it. LoRA adds small trainable adapter matrices to the attention layers while keeping the base weights frozen:

```
Original attention weight W: [d, d]    frozen, no grad
LoRA adapter:  W_A [d, r] @ W_B [r, d]  trainable, r=16 (rank)

Effective weight: W + alpha/r * W_A @ W_B
```

With rank=16 and 4 attention layers targeted, we have ~5M trainable parameters on a 1B model — about 0.5%. This fits in ~8GB GPU memory and trains in hours rather than days.

**Additionally:** the 512 new ID token embeddings are trainable from scratch. They start random and learn meaning from the training data — `<12>` learns to represent "music that sounds like cluster 12".

### The 10% English mix-in — why it matters

If you train only on music recommendation data, the model will catastrophically forget how to write English. After a few hundred steps, responses become garbled. The fix: mix in 10% of general instruction-tuning data (e.g. a small slice of Alpaca or FLAN). This prevents forgetting while barely affecting music task performance.

```python
# In dataset.py:
# 90% of batches: music CRS examples
# 10% of batches: plain English instruction-following examples
```

### Training recipe

```bash
# SFT phase (2-3 days on 1x A100 80GB, or ~8 hours on 4x A100)
python src/train/train.py --config config/train.yaml --stage sft

# Monitor: watch eval_loss on Dev set — stop when it plateaus
# Expected: eval_loss drops from ~3.5 → ~1.8 over 3 epochs
# Checkpoint saved every 500 steps to exp/checkpoints/sft/
```

**Key hyperparameters to watch:**

| Parameter | Default | If nDCG low → | If response quality low → |
|---|---|---|---|
| `retrieval_loss_weight` | 1.0 | increase to 1.5 | decrease to 0.7 |
| `response_loss_weight` | 1.0 | decrease to 0.7 | increase to 1.5 |
| `learning_rate` | 2e-4 | lower to 1e-4 | same |
| `lora_r` | 16 | increase to 32 | same |
| `english_mix_ratio` | 0.10 | decrease to 0.05 | increase to 0.15 |

### Evaluating SFT checkpoint

```bash
# Run inference with SFT checkpoint (no DPO yet)
python music-crs-baselines/run_inference_semantic_ids.py \
    --tid llama1b_semantic_ids_devset --batch_size 4

# Evaluate
python music-crs-evaluator/evaluate_devset.py \
    --predictions music-crs-baselines/exp/inference/devset/llama1b_semantic_ids_devset.json
```

**Expected gains over BM25 baseline:**
- nDCG@20: +8 to +15 points (the model has seen the training data; BM25 hasn't)
- LLM-as-Judge: +significant (response is grounded in the same tracks the model retrieved)
- Distinct-2: +moderate (model generates varied explanations, not templates)

---

## Phase 4 — DPO Sharpening (Day 5-6, ~1 day)

**Goal:** Make the ranking sharper and the responses more varied using Direct Preference Optimization.

**DPO teaches the model which output is better for the same input, without explicit reward modelling.** It's more stable than RLHF and requires only preference pairs, not scores.

### Two types of preference pairs

**Type A — Retrieval pairs** (sharpens nDCG):

```
Prompt: [same user + dialogue]
Chosen:  <12> <303>  <- ground truth track 1
         <45> <289>  <- ground truth track 2
         ...
         [RESPONSE] ...

Rejected: <88> <142>  <- random non-GT track
          <201> <7>   <- random non-GT track
          ...
          [RESPONSE] ...
```

The model learns: "given this conversation, these 20 IDs are better than those 20 IDs."

**Type B — Response pairs** (sharpens Distinct-2 and LLM-as-Judge):

```
Prompt: [same user + dialogue + same 20 IDs]
Chosen:  [RESPONSE] These tracks share Nils Frahm's introspective piano style,
                    particularly in Lambert's use of sparse arrangements...

Rejected: [RESPONSE] Here are some tracks you might enjoy.
```

The model learns: "for these specific tracks, this explanation is better than that generic template."

### DPO training

```bash
python src/train/train.py --config config/train.yaml --stage dpo
# Runs on top of the SFT checkpoint
# 1 epoch, smaller LR (5e-5), ~4-8 hours on 1x A100
# → exp/checkpoints/dpo_final/
```

**Expected gains over SFT checkpoint:**
- nDCG@20: +1 to +3 points
- Distinct-2: +5 to +10 points (biggest gain — DPO explicitly penalises generic responses)
- Catalog diversity: neutral (handled by MMR in Phase 5)

---

## Phase 5 — Full Inference Pipeline + MMR (Day 6, ~4 hours)

**Goal:** Wire everything together, tune MMR lambda, produce the final Dev submission.

### Full pipeline

```
User profile + listening history (as ID tokens) + 8-turn dialogue
        │
        ▼
Fine-tuned Llama-3.2-1B (SFT + DPO, expanded vocab)
        │
        ▼ Constrained beam search (prefix trie, no hallucinations)
        │
        ▼ 20 semantic ID pairs decoded → 20 track_ids
        │
        ▼ MMR reranking (lambda tuned on Dev)
        │
        ├── predicted_track_ids: [top-20]
        └── predicted_response: "These tracks share..."  (from same forward pass)
```

### Tune MMR lambda

```bash
# Generate dev predictions with lambda=1.0 (pure relevance, no diversity)
python music-crs-baselines/run_inference_semantic_ids.py --tid llama1b_semantic_ids_devset

# Sweep lambda
python src/utils/tune_mmr.py \
    --predictions music-crs-baselines/exp/inference/devset/llama1b_semantic_ids_devset.json \
    --ground_truth data/ground_truth_dev.json \
    --embeddings data/track_embeddings.npy \
    --track_ids data/track_ids.txt
```

The sweep prints a table like:

```
  lambda    nDCG@20    Coverage    Combined
     1.0     0.3241      0.0124      0.2304   ← pure relevance, terrible diversity
     0.7     0.3198      0.0891      0.2906
     0.5     0.3112      0.1834      0.3229   ← best combined
     0.3     0.2841      0.2901      0.2857
     0.0     0.1022      0.4912      0.2188   ← pure diversity, terrible relevance
```

Pick the lambda that maximises the combined score. Set it in `config/llama1b_semantic_ids_devset.yaml`.

### Final evaluation

```bash
python music-crs-evaluator/evaluate_devset.py \
    --predictions music-crs-baselines/exp/inference/devset/llama1b_semantic_ids_devset.json
```

Compare all metrics against Phase 0 baseline scores. Record the delta.

---

## Phase 6 — Blind A Submission (By Deadline)

```bash
# Rerun inference on blind_a split
python music-crs-baselines/run_inference_semantic_ids.py \
    --tid llama1b_semantic_ids_blindset_A --batch_size 4

# Package submission (check evaluator README for exact format)
# Submit to: https://www.codabench.org/competitions/15786/
```

**Important:** Blind A uses the same warm-start users as Dev. Blind B will stress cold-start (no listening history). Keep your BM25 fallback path active in `run_inference_semantic_ids.py` for users with no listening history.

---

## Phase 7 — Cold-Start Hardening (Before Blind B)

Blind B explicitly contains cold-start sessions — users with no listening history. The Semantic ID system's listening history input becomes `"(none)"` and the model may degrade.

Two fixes:

**Fix A — Text-only fallback path:**
If `listening_history` is empty, route to the Hybrid retriever (BM25+BERT) instead of Semantic ID generation. The hybrid doesn't need user embeddings.

```python
# In run_inference_semantic_ids.py
if not listening_history:
    track_ids = music_crs.hybrid_retrieval.text_to_item_retrieval(query, topk=20)
else:
    track_ids = music_crs.retrieval.text_to_item_retrieval(query, topk=20)
```

**Fix B — Cold-start training examples:**
TalkPlayData has a `test_cold` split. Include cold-start examples (empty history) in SFT training by zeroing out the `[HISTORY]` section for some training examples (10-20% randomly). The model learns that `[HISTORY] (none)` means "use only dialogue context".

---

## Expected Score Progression

| Phase | System | nDCG@20 | Coverage | Distinct-2 | LLM-Judge |
|---|---|---|---|---|---|
| 0 | BM25 baseline | ~0.18 | ~0.01 | ~0.35 | ~0.40 |
| 0 | BERT baseline | ~0.20 | ~0.01 | ~0.35 | ~0.40 |
| 2 | Hybrid RRF | ~0.24 | ~0.02 | ~0.35 | ~0.40 |
| 3 | + SFT | ~0.32 | ~0.05 | ~0.55 | ~0.65 |
| 4 | + DPO | ~0.34 | ~0.05 | ~0.65 | ~0.70 |
| 5 | + MMR tuned | ~0.33 | ~0.18 | ~0.65 | ~0.70 |

*Numbers are estimates based on analogous results from Text2Tracks and TalkPlay-Tools papers. Your actual numbers will differ — this is a rough expected trajectory.*

---

## Decision Points

At each phase, you may need to make a judgment call:

**After Phase 2:** If Hybrid barely beats BM25, your bottleneck is retrieval quality, not fusion strategy. Move to Phase 3 immediately rather than optimizing RRF weights.

**After Phase 3 SFT:** If nDCG is good but LLM-as-Judge is poor, increase `response_loss_weight` and retrain. If nDCG is poor but responses are good, increase `retrieval_loss_weight`.

**After Phase 4 DPO:** If Distinct-2 is still low, your DPO rejection responses are too similar to chosen responses. Make them more generic (shorter, more templated) to create a stronger contrast signal.

**After Phase 5 MMR:** If catalog diversity is still low despite MMR, the model is generating the same ~100 tracks every session. This is a training data coverage issue — check if the training split has diverse ground-truth tracks across sessions.

---

## File Map: What to Touch in Each Phase

| Phase | Files to create/modify |
|---|---|
| 0 | run the existing files, nothing to change |
| 1 | `scripts/download_data.py` (new), `config/train.yaml` (update paths) |
| 2 | `mcrs/retrieval_modules/hybrid.py` (new), `mcrs/retrieval_modules/__init__.py`, `config/llama1b_hybrid_devset.yaml` |
| 3 | `src/train/train.py`, `src/train/dataset.py`, `config/train.yaml` — all exist, just run |
| 4 | `src/train/dpo_dataset.py`, `src/train/train.py --stage dpo` — exists, just run |
| 5 | `run_inference_semantic_ids.py` (exists), `src/utils/tune_mmr.py` (exists), update `mmr_lambda` in config |
| 6 | Add blindset_A config, run inference on blind split |
| 7 | Modify `run_inference_semantic_ids.py` fallback logic, retrain with cold-start examples |

---

*Written May 2026. All code referenced exists in `music-crs-baselines/` and `src/`.*
