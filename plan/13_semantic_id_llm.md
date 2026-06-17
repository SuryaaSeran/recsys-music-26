# Plan 13: Semantic ID LLM for Conversational Bucket Prediction

## What eugeneyan's repo does

Two-stage LLM fine-tune on Qwen3-8B (Amazon video games):
1. **Vocab extension** (`finetune_qwen3_8b_vocab.py`): add `<|sid_0|>`..`<|sid_N|>` tokens, train
   only the new embeddings for 1000 steps so the model learns what each bucket ID "means".
2. **Full fine-tune** (`finetune_qwen3_8b_full.py`): train all parameters on a mix of:
   - seq → next bucket (SID-to-SID)
   - title/description → SID (text-to-ID)
   - SID → title/description (ID-to-text)
   - multi-turn conversations (SID history → next SID, with natural language steering)

The result is a model that is "bilingual" — it can reason in natural language AND in bucket IDs
in the same pass. At inference: user text → model outputs `<|sid_start|><|sid_7|><|sid_end|>`.

## Our adaptation for TalkPlay

Key differences from the video-games use case:
- 64 L0 buckets (not 256), 2 levels (not 4)
- Conversation structure: [PROFILE] [GOAL] [T1..Tn] [NOW] — not just a purchase sequence
- We have listener thoughts, gpa labels, goal specificity
- Blind A turns are music recommendation turns, not e-commerce

---

## Phase 1: Bucket Descriptions (immediately actionable)

For each of the 64 L0 buckets in `runF_v8e_L2C64`, generate a 2-3 sentence natural language
description of what the bucket contains — genres, moods, eras, styles.

### Generate bucket member lists

```python
# For each L0 code 0..63, get up to 20 representative tracks with metadata
import numpy as np, json
from collections import defaultdict
from datasets import load_dataset, concatenate_datasets

codes = np.load('cache/semantic_ids/runF_v8e_L2C64/semantic_ids.npy')
tids  = np.load('cache/semantic_ids/runF_v8e_L2C64/track_ids.npy', allow_pickle=True)
meta_ds = load_dataset('talkpl-ai/TalkPlayData-Challenge-Track-Metadata')
# ... build bucket_members[l0] = [(name, artist, tags, year), ...]
```

### Prompt for each bucket

```
I have a music catalog. The following tracks have been grouped into a single semantic cluster:

1. "Hurt" by Nine Inch Nails | Tags: industrial, rock, 90s | 1994
2. "Black" by Pearl Jam | Tags: grunge, alternative rock, 90s | 1991
3. "Creep" by Radiohead | Tags: alternative rock, britpop | 1992
... (20 tracks)

Write a 2-3 sentence description of what musical style, mood, era, or genre defines this cluster.
Be specific. Mention genre, mood, tempo, era if applicable.
```

Output: 64 bucket descriptions saved to `cache/semantic_ids/runF_v8e_L2C64/bucket_descriptions.json`

---

## Phase 2: Training Data Construction

Three types of examples, all using the same tokeniser format as the TT v8e anchor:

### Type A — Query → Bucket (primary task, most examples)

Each music turn in the TalkPlay TRAIN split becomes:

```
[SYSTEM] You are a music bucket predictor. Given a conversation context,
output the semantic bucket ID for the next recommended track.
Output only the bucket token.

[USER]
[PROFILE] 25-34 · UK · female · Anglo-American Rock · English
[GOAL] discover new artists, intense/dramatic  (LH)
[T1] USER: Play something intense | REC: Hurt – NIN | REACTION: liked | LISTENER: Great dark energy
[NOW] USER: More like that but maybe something newer?
Predict bucket:

[ASSISTANT] <|sid_7|>
```

Gold bucket = L0 code of the gold track's semantic ID.

### Type B — Bucket → Description (bidirectional)

```
[USER] What musical style does bucket <|sid_7|> represent?
[ASSISTANT] Dark, introspective alternative and grunge rock from the 1990s,
characterized by raw emotional intensity and distorted guitar work.
```

### Type C — Description → Bucket (bidirectional)

```
[USER] Which bucket best describes: dark emotional alternative rock from the 90s?
[ASSISTANT] <|sid_7|>
```

### Type D — Multi-turn query refinement

```
[USER] [T1..Tn history] [NOW] More like that
[ASSISTANT] <|sid_7|>
[USER] But more recent, maybe 2010s?
[ASSISTANT] <|sid_12|>
```

### Mix ratio

| Type | Count | Purpose |
|---|---:|---|
| A (query→bucket) | ~25K | Core retrieval task |
| B (bucket→desc) | 64 × 5 = 320 | Semantic grounding |
| C (desc→bucket) | 64 × 5 = 320 | Reverse lookup |
| D (multi-turn) | ~2K | Conversational steering |

---

## Phase 3: Model Training

### Base model options for MPS (Apple Silicon)

| Model | Params | MPS fit | Expected quality |
|---|---:|---|---|
| Qwen3-0.6B | 0.6B | ✓ (easy) | Good for ID prediction |
| Qwen3-1.5B | 1.5B | ✓ (fits) | Better reasoning |
| Qwen3-4B | 4B | ✓ (tight, needs grad_ckpt) | Best for MPS |

Qwen3-1.5B is the sweet spot: fits comfortably on 16GB MPS, and has enough capacity
to learn the tripartite mapping (query → ID, ID → text, text → ID).

### Vocabulary extension

Add tokens `<|sid_0|>` to `<|sid_63|>` + `<|sid_start|>` + `<|sid_end|>` = 66 new tokens.

Stage 1: Train only new token embeddings (freeze all other params) for 500 steps on
Types B + C (bucket↔description). This gives the tokens meaningful positions in embedding space.

Stage 2: LoRA fine-tune (r=16, alpha=32 on q,k,v,o) on all types A+B+C+D for 2-3 epochs.

### Loss

Standard causal LM cross-entropy. For Type A, mask loss on everything before `Predict bucket:`.
For Types B/C/D, compute loss on the full response.

---

## Phase 4: Inference Integration

At each turn, run the LLM on the conversation context:

```python
prompt = build_prompt(profile, goal, history_blocks, current_query)
output = model.generate(prompt, max_new_tokens=8)
bucket_id = parse_sid_token(output)  # extract <|sid_X|>
```

Then expand all tracks from that bucket + top-1 alternative bucket (for diversity).

This replaces SASRec Stage 3 or runs in parallel with it. The LLM prediction is
goal-aware and conversation-aware, which SASRec is not.

**Key advantage**: the model sees [GOAL], [LISTENER:], and [REACTION:] — the full
conversational signal — not just track IDs. It can respond to "something newer" or
"more upbeat" without retraining.

---

## Execution Plan

### Step 0 — Right now (no API needed)
- Write `scripts/train/generate_bucket_descriptions.py`:
  - Loads track metadata per L0 bucket
  - Builds prompt per bucket
  - Saves member list + prompt to `cache/semantic_ids/runF_v8e_L2C64/bucket_members.json`
  - Waits for LLM API to generate descriptions

### Step 1 — Needs LLM API (Gemini refill or Anthropic key)
- Run description generation for 64 buckets (~5 min at 5s/bucket)
- Save to `cache/semantic_ids/runF_v8e_L2C64/bucket_descriptions.json`

### Step 2 — Build training data
- Write `scripts/train/build_semantic_id_llm_data.py`
- Output: `data/semantic_id_llm/train.jsonl` + `valid.jsonl`

### Step 3 — Vocab extension + LoRA fine-tune
- Write `scripts/train/finetune_sid_qwen3.py`
- Stage 1: embed init (~30 min on MPS)
- Stage 2: LoRA fine-tune (~3h on MPS)

### Step 4 — Inference integration
- Write `scripts/inference/semantic_id_llm_retrieval.py`
- Slot into `run_inference_fusion_recall_expansion.py` alongside SASRec Stage 3

---

## Files to create

```
scripts/train/generate_bucket_descriptions.py   # step 0
scripts/train/build_semantic_id_llm_data.py     # step 2
scripts/train/finetune_sid_qwen3.py             # step 3
scripts/inference/semantic_id_llm_retrieval.py  # step 4
cache/semantic_ids/runF_v8e_L2C64/
  bucket_members.json      # tracks per bucket (step 0)
  bucket_descriptions.json # LLM descriptions (step 1)
data/semantic_id_llm/
  train.jsonl              # fine-tuning data (step 2)
  valid.jsonl
models/sid_qwen3/          # trained model (step 3)
```
