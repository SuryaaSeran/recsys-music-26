# RecSys Challenge 2026 — Conversational Music Recommendation (Music-CRS)

> **ACM RecSys 2026** · Minneapolis, Minnesota, USA · September 28 – October 2, 2026  
> Challenge website: [recsyschallenge.com/2026](https://www.recsyschallenge.com/2026/)  
> Organised by the [NLP4MusA](https://sites.google.com/view/nlp4musa-2026) group (lead: Seungheon Doh)

---

## Table of Contents

1. [Task Overview](#task-overview)
2. [Dataset — TalkPlayData](#dataset--talkplaydata)
3. [What Your System Must Produce](#what-your-system-must-produce)
4. [Evaluation](#evaluation)
5. [Baseline Code](#baseline-code)
6. [Key Papers & Literature](#key-papers--literature)
7. [Recommended Architecture](#recommended-architecture)
8. [Adapting "Speak Spotify" for Music-CRS](#adapting-speak-spotify-for-music-crs)
9. [Timeline](#timeline)

---

## Task Overview

The challenge asks you to build a **Conversational Music Recommender System (Music-CRS)**: a system that, given a multi-turn dialogue between a user and a recommender, must simultaneously:

1. **Rank** the top-20 most relevant tracks from a ~1 million track catalog.
2. **Generate** a natural-language response that explains and justifies the recommendations.

This is a hybrid retrieval + generation task. You cannot win by optimising ranking alone — the system's language responses are independently scored by an LLM judge.

### The core challenge

Standard music recommenders work from implicit signals (plays, skips). Here, user intent arrives as **free-form conversation** — across up to 8 turns — mixing constraints, mood descriptions, artist references, and exploratory requests. The system must understand nuanced natural-language preference and translate it into precise catalog retrieval, while producing coherent, personalised, explanatory dialogue at every turn.

---

## Dataset — TalkPlayData

The dataset is **TalkPlayData-2**, a large-scale synthetic conversational music dataset built using an agentic simulation pipeline where a Listener LLM and a RecSys LLM converse under information asymmetry and explicit conversation goals.

### What's included

| Component | Details |
|---|---|
| **Conversations** | Multi-turn dialogues, ~8 turns per session average |
| **Music catalog** | ~1 million tracks |
| **Track metadata** | Title, artist, album, genre, tags, year, language |
| **User profiles** | Demographic/preference context |
| **Listening histories** | Per-user past consumption |
| **Pre-computed embeddings** | Multimodal track embeddings (audio + lyrics + metadata + CF) and user CF embeddings — provided out of the box |

### Dataset splits

| Split | Purpose |
|---|---|
| **Train** | Model training + development |
| **Development** | Local evaluation before blind submission |
| **Blind A** | Interim leaderboard (released April 10, 2026) |
| **Blind B** | Final leaderboard — includes cold-start stress test |

> **Important:** Blind B explicitly tests cold-start scenarios. A retrieval path that does not rely on user CF embeddings is essential as a fallback.

### Why synthetic data matters

Because the conversations are LLM-generated under explicit rules (conversation goals, information asymmetry, turn structure), there is **exploitable structure**: goals are consistent across a session; turn-1 utterances are broader than turn-8; user profile fields gate certain preferences. A small exploratory data analysis (EDA) on Train before modelling will pay dividends.

**Hugging Face dataset:** [talkpl-ai/TalkPlayData-2](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2)  
**Generation pipeline repo:** [github.com/talkpl-ai/talkplaydata-2](https://github.com/talkpl-ai/talkplaydata-2)

---

## What Your System Must Produce

For **every turn** in every conversation, your system outputs a JSON object with two fields:

```json
{
  "track_ids": ["id_001", "id_002", ..., "id_020"],  // ranked list of top-20 catalog IDs
  "response": "Here are some tracks that match your mood — starting with Nils Frahm whose sparse piano ..."
}
```

- `track_ids` must be exactly 20 items, ranked by predicted relevance (best first).
- `response` is a free-text natural-language reply to the user's last utterance.

Input available at each turn:
- Full dialogue history up to the current turn
- User profile
- User listening history (with embeddings)
- Track catalog + metadata + pre-computed embeddings

---

## Evaluation

Submissions are scored across **four dimensions**. The primary leaderboard metric is nDCG@20, but the other three dimensions materially affect ranking and cannot be ignored.

### 1. Retrieval Quality — nDCG@20 (primary)

**What it measures:** How highly your system ranks the ground-truth tracks within the top-20 results.

**Formula:** Normalised Discounted Cumulative Gain at cutoff 20, evaluated at every turn, then macro-averaged over all sessions and turns.

```
nDCG@20 = DCG@20 / IDCG@20

DCG@20 = sum over k=1..20 of [ relevance(k) / log2(k+1) ]
```

- nDCG@{1, 10, 20} are all reported; @20 is the tiebreaker.
- Macro-averaging means every session (regardless of length) counts equally — don't sacrifice any user.

**Evaluator script:** `evaluate_devset.py` in [nlp4musa/music-crs-evaluator](https://github.com/nlp4musa/music-crs-evaluator)

### 2. Catalog Diversity

**What it measures:** How broadly your system recommends across the full catalog, rather than always returning the same popular tracks.

Computed **globally** across the entire prediction file (not per session), using catalog coverage — what fraction of the catalog is touched by any recommendation across all turns.

**Why it matters:** A system that consistently recommends the same 500 tracks will score zero here even with perfect nDCG. Diversity and relevance must be balanced, e.g., with MMR or DPP-based re-ranking.

### 3. Lexical Diversity — Distinct-2

**What it measures:** How varied the language in your generated responses is, using the **Distinct-2** metric (the ratio of unique bigrams to total bigrams across all responses).

**Formula:**
```
Distinct-2 = |unique bigrams across all responses| / |total bigrams across all responses|
```

A system that generates near-identical boilerplate responses for every turn scores near zero. Distinct-2 rewards genuine language variation — different sentence structures, different musical vocabulary, different justification angles.

### 4. LLM-as-Judge — Personalisation & Explanation Quality

**What it measures:** A Gemini LLM judge reads each (dialogue turn, recommended tracks, response) triple and scores two sub-dimensions:
- **Personalisation:** Does the response reflect the specific user's preferences and history?
- **Explanation quality:** Is the recommendation justification grounded, coherent, and musically meaningful?

**Why it matters:** This is where a Chain-of-Thought approach pays off. Systems that generate internally reasoned responses (even if the CoT is hidden) tend to score higher on explanation quality than templated responses.

### Scoring summary

| Metric | Scope | Notes |
|---|---|---|
| nDCG@20 | Per turn → macro-avg | **Primary leaderboard metric** |
| nDCG@1, nDCG@10 | Per turn → macro-avg | Reported, not primary |
| Catalog diversity | Global | Cover the long tail |
| Distinct-2 | Global | Vary your language |
| LLM-as-Judge (personalisation) | Per turn | Gemini judge |
| LLM-as-Judge (explanation) | Per turn | Gemini judge |

---

## Baseline Code

Two official repos from the NLP4MusA organizers:

### [`nlp4musa/music-crs-baselines`](https://github.com/nlp4musa/music-crs-baselines)

The baseline systems. Contains runnable end-to-end implementations of reference approaches on TalkPlayData-2, intended as a starting floor to beat. Key components:

- **Dense retrieval baseline** — encodes dialogue turns + user profile into a query vector; retrieves top-20 via ANN search against the provided multimodal track embeddings. Uses the pre-computed embeddings directly — no retraining required to run this.
- **BM25 sparse retrieval baseline** — indexes track metadata and tags; queries with extracted keywords from the dialogue turn. Useful as a complementary signal; sparse and dense are highly complementary.
- **CLAP multimodal baseline** — uses the CLAP audio-text encoder to bridge natural-language descriptions to audio-grounded track representations. Strong on mood/vibe queries that don't name artists or genres.
- **Response generation** — each baseline appends a simple template or small-LLM response generator on top of the retrieved tracks. This is intentionally weak — the LLM-as-Judge dimension is where participants can gain the most over the baseline.

The baselines establish the **floor** for nDCG@20. A well-tuned two-stage system (dense retrieval → LLM re-ranker) should comfortably beat them.

### [`nlp4musa/music-crs-evaluator`](https://github.com/nlp4musa/music-crs-evaluator)

The official evaluation harness. Run this locally against your Dev set predictions before any blind submission.

```
music-crs-evaluator/
├── requirements.txt           # Python deps (install first)
├── make_ground_truth.py       # Generates ground-truth files from the raw dataset
├── evaluate_devset.py         # Main evaluation script — run this on your predictions
└── metrics/
    ├── ndcg.py                # nDCG@{1,10,20} implementation
    ├── retrieval.py           # Retrieval metric utilities
    ├── catalog_diversity.py   # Catalog coverage computation
    └── lexical_diversity.py   # Distinct-2 computation
```

**Usage:**
```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Generate ground truth from Train/Dev split
python make_ground_truth.py --data_path ./TalkPlayData-2 --split dev

# 3. Evaluate your predictions file
python evaluate_devset.py \
    --predictions ./my_predictions.json \
    --ground_truth ./ground_truth_dev.json
```

**Prediction file format:**
```json
[
  {
    "session_id": "session_001",
    "turn_id": 3,
    "track_ids": ["t_00412", "t_08821", ...],  // top-20, ranked
    "response": "Based on your love of late-night ambient music ..."
  },
  ...
]
```

The evaluator prints a breakdown per metric and writes a summary JSON. **Treat the Dev score as your primary development signal** — Blind A/B are only revealed after submission.

---

## Key Papers & Literature

### Must-read (organizer's own work — this IS the task)

| Paper | What it covers |
|---|---|
| [TalkPlay (arXiv:2502.13713)](https://arxiv.org/abs/2502.13713) | Unifies dialogue + retrieval + ranking as next-token prediction; multimodal music tokenizer (audio, lyrics, CF, tags). End-to-end architecture. |
| [TalkPlayData-2 (arXiv:2509.09685)](https://arxiv.org/abs/2509.09685) | Describes the agentic synthetic-data pipeline — Listener LLM ↔ RecSys LLM, conversation goals, information asymmetry, 8 turns, cold-start splits. Explains why the data looks the way it does. |
| [TalkPlay-Tools (arXiv:2510.01698)](https://arxiv.org/abs/2510.01698) | LLM as orchestrator with tool calls: Boolean SQL filter + BM25 + dense embedding + generative semantic-ID retrieval, fused. The closest published method to a strong baseline on this exact data family. |

### LLM-based CRS architectures

| Paper | Key idea |
|---|---|
| [RecLLM (arXiv:2305.07961)](https://arxiv.org/abs/2305.07961) | Unifies dialogue management + retrieval + ranking + explanation in one LLM. CoT rationale → user-facing explanation. |
| [ReFICR](https://arxiv.org/abs/2406.02543) | Decomposes CRS into 5 subtasks; trains GRITLM with QLoRA + contrastive + LM losses. One-model-many-heads blueprint. |
| [ECPO (ACL 2025 Findings)](https://aclanthology.org/2025.findings-acl.307.pdf) | Expectation-Confirmation Preference Optimization for multi-turn CRS. Models how user satisfaction evolves across turns. |
| [UniCRS](https://arxiv.org/abs/2206.09363) | Earlier unified prompting baseline; frequently the prior SOTA reference in CRS papers. |

### Generative retrieval at scale (1M tracks)

| Paper | Key idea |
|---|---|
| [Text2Tracks (arXiv:2503.24193)](https://arxiv.org/abs/2503.24193) | Flan-T5 emits semantic IDs (3 tokens/track from RQ-KMeans over CF embeddings). 127% better than bi-encoder. Directly applicable to this challenge's pre-computed embeddings. |
| [Semantic IDs: Joint Generative Search & Rec (arXiv:2508.10478)](https://arxiv.org/abs/2508.10478) | Shared/prefix-shared codebooks; one model handles search + recommendation; confirms RQ-KMeans > RQ-VAE. |
| [Teaching LLMs to Speak Spotify (blog, Nov 2025)](https://research.atspotify.com/2025/11/teaching-large-language-models-to-speak-spotify-how-semantic-ids-enable) | Production-scale synthesis: ~1B decoder-only LLM, 4 tasks (rec + search + playlist gen + user understanding), multimodal R-LFQ codebooks, vLLM serving + Redis URI store. The architectural blueprint for a top-tier entry. |
| [Generative Rec with Semantic IDs: Practitioner's Handbook (arXiv:2507.22224)](https://arxiv.org/abs/2507.22224) | Comprehensive survey of RQ-VAE / RQ-KMeans / LFQ choices, training recipes, eval pitfalls. Read before choosing your quantizer. |

### Multimodal music embeddings & retrieval

| Paper | Key idea |
|---|---|
| [JAM — Deezer (arXiv:2507.15826)](https://arxiv.org/abs/2507.15826) | TransE-style user/query/item translation in shared latent space; cross-attention + sparse MoE over audio, lyrics, CF. Lightweight; strong "no LLM fine-tuning" alternative. |
| [Talking to Your Recs (CEUR-WS Vol-3787)](https://ceur-ws.org/Vol-3787/paper6.pdf) | Contrastive enhancement of pre-trained text embeddings with audio/image/CF signals. Useful trick when given pre-computed embeddings like in this challenge. |
| [CLAP](https://arxiv.org/abs/2211.06687) | Audio-text contrastive retrieval; reasonable to use as a frozen retriever. Included in the official baselines. |

### LLM as re-ranker & explainer

| Paper | Key idea |
|---|---|
| [LLM as Explainable Re-Ranker (arXiv:2512.03439)](https://arxiv.org/abs/2512.03439) | SFT + DPO two-stage; bootstrapped position de-biasing. Re-ranks top-K dense output and produces justification text in the same pass. Directly targets the LLM-as-Judge dimension. |
| [Microsoft RecAI](https://github.com/microsoft/RecAI) | Mature toolkit: InteRecAgent + RecExplainer + RecLM-Evaluator. Useful for orchestration/explainer split. |

---

## Recommended Architecture

A pragmatic stack that maps cleanly onto all four scoring dimensions:

```
User profile + listening history + 8-turn dialogue
        │
        ▼
┌─────────────────────────────────────────┐
│         Stage 1: Candidate Retrieval    │
│                                         │
│  ┌──────────┐  ┌────────┐  ┌─────────┐ │
│  │  Dense   │  │  BM25  │  │Semantic │ │
│  │ (embed.) │  │(sparse)│  │ID (T5)  │ │
│  └────┬─────┘  └───┬────┘  └────┬────┘ │
│       └────────────┴────────────┘       │
│                    │ top ~500 candidates│
└────────────────────┼────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│         Stage 2: LLM Re-Ranker          │
│                                         │
│  Input: dialogue + user + candidates    │
│  Output: ranked top-20 IDs + response   │
│  Training: SFT on train, DPO on pairs   │
│  Base model: 1B-class open-weight LLM   │
│  (LoRA fine-tune; freeze base weights)  │
└─────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│         Stage 3: Diversity Post-proc    │
│                                         │
│  MMR or DPP re-ranking on top-20        │
│  Tuned on Dev set                       │
│  ~free points on catalog diversity      │
└─────────────────────────────────────────┘
```

**Key design decisions:**

- **Semantic IDs:** Run RQ-KMeans on the provided multimodal track embeddings → 2–3 token IDs per track. This makes 1M-track generative retrieval tractable with a small T5/T5-base backbone (Text2Tracks recipe).
- **Joint generation:** Have the re-ranker emit both the ranked list and the natural-language response in one forward pass (RecLLM/Speak Spotify style). This is the highest-leverage move for the LLM-as-Judge dimension.
- **Cold-start fallback:** Keep a text/audio-only retrieval path (CLAP or BM25) that doesn't need user CF embeddings. Blind B stresses this explicitly.
- **Diversity:** Apply MMR post-processing after re-ranking; tune the lambda hyperparameter on Dev. This is essentially free nDCG-neutral diversity gain.
- **Distinct-2:** Vary response templates, use temperature sampling, include a lexical-variety penalty term in DPO data construction.

---

## Adapting "Speak Spotify" for Music-CRS

Spotify's November 2025 paper ["Teaching LLMs to Speak Spotify"](https://research.atspotify.com/2025/11/teaching-large-language-models-to-speak-spotify-how-semantic-ids-enable) is the closest published architecture to what a top-tier Music-CRS entry needs. It proves that giving an LLM a new vocabulary where each "word" = a catalog track lets it retrieve, rank, and explain in a single forward pass. Every component maps directly onto this challenge.

The key insight vs the official baselines: the baselines do retrieval and response generation as **two separate steps** (retrieve → template). The Speak Spotify approach collapses them into one model — the response is automatically grounded in the specific tracks retrieved, which is exactly what the Gemini LLM judge rewards.

### Step 1 — Quantize embeddings into Semantic IDs (~20 min, free)

**What Spotify does:** Trains multimodal embeddings from audio + co-listening signals, then applies Residual Lookup-Free Quantization (R-LFQ) to map each track to a short token sequence like `<4><17><92>`.

**What you do:** The challenge already ships pre-computed multimodal track embeddings — skip the embedding training entirely. Run **RQ-KMeans** on those embeddings:

```python
from sklearn.cluster import MiniBatchKMeans
import numpy as np

# Load provided embeddings: shape (1_000_000, D)
embeddings = np.load("track_embeddings.npy")

# Level 1: 256 coarse clusters
km1 = MiniBatchKMeans(n_clusters=256, random_state=42).fit(embeddings)
codes1 = km1.predict(embeddings)

# Level 2: 256 fine clusters on residuals
residuals = embeddings - km1.cluster_centers_[codes1]
km2 = MiniBatchKMeans(n_clusters=256, random_state=42).fit(residuals)
codes2 = km2.predict(residuals)

# Each track gets a 2-token ID, e.g. track_id → ("<12>", "<47>")
semantic_ids = {track_id: (f"<{c1}>", f"<{c2}>") for track_id, c1, c2
                in zip(track_ids, codes1, codes2)}
```

**Why it matters:** Instead of generating "Nils Frahm – Says" (8+ tokens, hallucination-prone), the model generates `<12><47>` (2 tokens, unambiguous). Beam search over 20 tracks = 40 tokens total vs 160+. Faster, no entity resolution needed.

### Step 2 — Expand the LLM vocabulary (LoRA, ~5M trainable params)

**What Spotify does:** Expands a ~1B open-weight decoder-only LLM's tokenizer with all codebook entries as new special tokens. Freezes all base model weights; trains only the new token embedding rows.

**What you do:** Take **Llama-3.2-1B-Instruct** (already referenced in the official baseline config `lm_type`). Add the 512 new ID tokens, freeze the base model, apply LoRA on attention layers, and train only LoRA adapters + new token embeddings:

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")

# Add 512 new special tokens (256 coarse + 256 fine)
new_tokens = [f"<{i}>" for i in range(512)]
tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B-Instruct")
model.resize_token_embeddings(len(tokenizer))

# Freeze base model; only new embeddings + LoRA are trainable
lora_config = LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj","v_proj"])
model = get_peft_model(model, lora_config)
```

> **Critical:** Mix ~10% plain English instruction-tuning data during training to prevent catastrophic forgetting. Without this the model loses fluency and the LLM-as-Judge score collapses.

### Step 3 — Joint retrieval + response training (SFT → DPO)

**What Spotify does:** Multi-task SFT over 4 tasks (rec, search, playlist gen, user understanding). Reports +22% vs single-task.

**What you do:** Two objectives trained in the same forward pass:

**Input format** (one training example per dialogue turn):
```
[SYSTEM] You are a music recommender. Given the user profile, listening history, and conversation, 
recommend 20 tracks as semantic IDs then explain your choices.

[USER PROFILE] age=28, gender=F, country=US
[HISTORY] <12><47> <8><201> <55><13> ...   ← listening history as ID tokens
[DIALOGUE]
  User: I want something melancholic but beautiful, like Nils Frahm
  System: <previous response>
  User: Maybe something with more piano?

[ASSISTANT] <12><47> <33><91> <12><209> ... <44><7>
These tracks share Nils Frahm's introspective piano style. Starting with Lambert whose 
sparse arrangements...
```

**Loss A (retrieval):** Cross-entropy on the 20 ID tokens against ground-truth track IDs from the Train split.

**Loss B (response):** Cross-entropy on the response text tokens against ground-truth responses.

**DPO on top:**
- Retrieval pairs: top-5 ground-truth tracks = positive, tracks ranked 50+ = negative
- Response pairs: specific/grounded response = positive, generic template = negative
- DPO sharpens both ranking and Distinct-2 simultaneously

### Step 4 — Constrained beam search at inference

**What Spotify does:** Diversified beam search (beam=60, diversity penalty=0.25). Redis key-value store for URI lookup.

**What you do:**

```python
# Build a prefix trie of all valid ID sequences — prevents hallucinating non-catalog tracks
from transformers import LogitsProcessor

class SemanticIDConstraint(LogitsProcessor):
    def __init__(self, valid_prefixes, tokenizer):
        self.valid_prefixes = valid_prefixes  # set of (tok1,) and (tok1, tok2) tuples
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        # Mask any token that would produce an invalid ID prefix
        current_prefix = tuple(input_ids[0, -1:].tolist())
        valid_next = {seq[len(current_prefix)] for seq in self.valid_prefixes
                      if seq[:len(current_prefix)] == current_prefix}
        mask = torch.full_like(scores, float('-inf'))
        mask[:, list(valid_next)] = 0
        return scores + mask

# Inference: one forward pass → 20 IDs + response text
outputs = model.generate(
    input_ids,
    num_beams=40,
    num_beam_groups=20,         # one group per track slot
    diversity_penalty=0.3,
    max_new_tokens=60,          # 40 ID tokens + 20 for response start
    logits_processor=[SemanticIDConstraint(valid_prefixes, tokenizer)]
)

# Map decoded IDs back to track_ids via codebook dict (plain Python, instant)
track_ids = [codebook_dict[(tok1, tok2)] for tok1, tok2 in decoded_id_pairs]
```

Post-decode, apply **MMR re-ranking** (lambda=0.5, tuned on Dev) to reorder the 20 tracks for catalog diversity — this is free points with zero impact on the retrieval model.

### What each step buys you on the leaderboard

| Step | nDCG@20 | Catalog Diversity | Distinct-2 | LLM-as-Judge |
|---|---|---|---|---|
| Semantic ID quantization | ↑ (faster search, less hallucination) | neutral | neutral | neutral |
| Vocab expand + LoRA | ↑↑ (joint training signal) | neutral | neutral | neutral |
| Joint SFT | ↑↑ | neutral | ↑ | ↑↑ |
| DPO on pairs | ↑ | neutral | ↑↑ | ↑ |
| Constrained decoding | ↑ (no invalid IDs) | neutral | neutral | neutral |
| MMR post-proc | neutral | ↑↑ | neutral | neutral |
| 10% English mix | neutral | neutral | ↑ | ↑↑ |

The biggest single win is **Joint SFT** — it simultaneously improves nDCG (retrieval loss), Distinct-2 (varied response outputs), and LLM-as-Judge (grounded explanations). Everything else is incremental on top.

---

## Timeline

| Date | Milestone |
|---|---|
| April 10, 2026 | Dataset + Blind A released; baselines published |
| June 30, 2026 | Submission deadline |
| September 28 – October 2, 2026 | RecSys 2026 conference (Minneapolis); results presented |

---

## Quick-start Checklist

- [ ] Download TalkPlayData-2 from [Hugging Face](https://huggingface.co/datasets/talkpl-ai/TalkPlayData-2)
- [ ] Clone and run the [music-crs-evaluator](https://github.com/nlp4musa/music-crs-evaluator) on a dummy prediction file to verify your environment
- [ ] Clone [music-crs-baselines](https://github.com/nlp4musa/music-crs-baselines) and reproduce the dense retrieval baseline score on Dev — this is your floor
- [ ] Run EDA on Train conversations: plot turn-length distribution, goal types, user profile field usage
- [ ] Quantize the provided track embeddings with RQ-KMeans (sklearn MiniBatchKMeans, 2 levels × 256 codes) — generates your semantic ID vocabulary
- [ ] Fine-tune a Flan-T5-base on (dialogue → semantic ID list) as a fast first-pass retriever (Text2Tracks recipe)
- [ ] Swap in an LLM re-ranker that also writes the response; evaluate Distinct-2 and LLM-judge score on Dev
- [ ] Add MMR post-processing; tune lambda for catalog diversity vs. nDCG tradeoff on Dev
- [ ] Stress-test on cold-start users (no listening history); confirm fallback path works

---

*Last updated: May 2026 — added Speak Spotify adaptation guide with code*
# recsys-music-26
