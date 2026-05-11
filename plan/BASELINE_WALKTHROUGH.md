# Baseline Code Walkthrough + Semantic ID Integration Guide

This document walks through every file in `music-crs-baselines/`, explains what each piece does and why, then shows exactly where and how to integrate Semantic IDs to go from the baseline to a competitive system.

---

## Table of Contents

1. [How the Baseline Runs — Big Picture](#1-how-the-baseline-runs--big-picture)
2. [Entry Point: `run_inference_devset.py`](#2-entry-point-run_inference_devsetpy)
3. [The Orchestrator: `mcrs/crs_baseline.py`](#3-the-orchestrator-mcrscrs_baselinepy)
4. [Retrieval Module A: `mcrs/retrieval_modules/bm25.py`](#4-retrieval-module-a-mcrsretrieval_modulesbm25py)
5. [Retrieval Module B: `mcrs/retrieval_modules/bert.py`](#5-retrieval-module-b-mcrsretrieval_modulesbertpy)
6. [Language Model: `mcrs/lm_modules/llama.py`](#6-language-model-mcrslm_modulesllamapy)
7. [Databases: `db_item` and `db_user`](#7-databases-db_item-and-db_user)
8. [System Prompts](#8-system-prompts)
9. [Config Files](#9-config-files)
10. [The Fundamental Problem with the Baseline](#10-the-fundamental-problem-with-the-baseline)
11. [Semantic ID Integration — Where & How](#11-semantic-id-integration--where--how)
12. [Integration Checklist](#12-integration-checklist)

---

## 1. How the Baseline Runs — Big Picture

```
run_inference_devset.py
        │
        │  loads config/{tid}.yaml
        │  loads TalkPlayData-Challenge-Dataset from HuggingFace
        │
        ▼
  CRS_BASELINE (crs_baseline.py)
        │
        ├── lm      = LLAMA_MODEL          ← generates the text response
        ├── retrieval = BM25_MODEL         ← finds candidate tracks
        │             or BERT_MODEL
        ├── item_db = MusicCatalogDB       ← track metadata lookup
        └── user_db = UserProfileDB        ← user profile lookup
        │
        │  For each session × 8 turns:
        │
        ▼
  chat() or batch_chat()
        │
        ├── Stage 1: retrieval.text_to_item_retrieval(full_dialogue) → top-20 track IDs
        │
        └── Stage 2: lm.response_generation(system_prompt, history, top_1_track_metadata)
                                                                          ↑
                                                          ONLY top-1 track goes to LLM!
```

The key thing to notice immediately: **retrieval and response generation are completely separate**. The LLM never sees more than the top-1 retrieved track. It has no way to reason about the full ranked list, and the retrieval model has no way to learn from the LLM's understanding of user intent. This is the core weakness our Semantic ID integration fixes.

---

## 2. Entry Point: `run_inference_devset.py`

**What it does:** Loops over every session in the test dataset, processes all 8 turns per session in batches, and writes results to `exp/inference/devset/{tid}.json`.

### Key function: `chat_history_parser()`

```python
def chat_history_parser(conversations, music_crs, target_turn_number):
```

This parses conversation history up to (but not including) the target turn:

```python
df_history = df_conversation[df_conversation['turn_number'] < target_turn_number]
```

For each past turn, if the role is `"music"` (i.e. the system recommended a track), it converts the raw `track_id` into human-readable metadata:

```python
if turn_data['role'] == "music":
    current_role = "assistant"
    current_content = music_crs.item_db.id_to_metadata(turn_data['content'])
    # → "track_id: T123, track_name: say something, artist_name: a great big world, ..."
```

**Why this matters for us:** The baseline converts track IDs → text for the LLM's context. With Semantic IDs, we will instead keep them as `<12> <47>` tokens — the LLM speaks the same language as the retrieval system.

### Main loop

```python
for target_turn_number in range(1, 9):   # 8 turns per session
    chat_history, user_query = chat_history_parser(...)
    batch_data.append({'user_query': user_query, 'user_id': user_id, 'session_memory': chat_history})
```

All turns across all sessions are flattened into one big list, then processed in batches of `batch_size` (default 16). Results are written as:

```json
{
  "session_id": "...",
  "user_id": "...",
  "turn_number": 3,
  "predicted_track_ids": ["t_001", "t_002", ..., "t_020"],
  "predicted_response": "Here are some tracks..."
}
```

---

## 3. The Orchestrator: `mcrs/crs_baseline.py`

This is the central class that wires everything together.

### `__init__` — what gets loaded at startup

```python
self.lm        = load_lm_module(...)        # Llama-3.2-1B-Instruct
self.retrieval = load_retrieval_module(...)  # BM25 or BERT index
self.item_db   = MusicCatalogDB(...)         # track_id → metadata dict
self.user_db   = UserProfileDB(...)          # user_id → profile dict
self.role_prompt = {                         # 3 text prompt templates
    "role_play": ...,
    "personalization": ...,
    "response_generation": ...,
}
```

### `_get_system_prompt(user_id)` — building the system prompt

```python
system_prompt = role_play_prompt + response_generation_prompt
if user_id:
    user_profile_str = self.user_db.id_to_profile_str(user_id)
    system_prompt += personalization_prompt + '\n' + user_profile_str
```

The user profile (age group, gender, country) is appended to the system prompt. This is the **only personalization** in the baseline — there is no listening history, no embedding-based user modelling.

**Our integration adds:** listening history encoded as Semantic ID token sequences, passed directly in the prompt.

### `chat()` — the single-turn inference method

```python
def chat(self, user_query, user_id=None):
    self.session_memory.append({"role": "user", "content": user_query})

    # Stage 1: Build retrieval query from full dialogue history
    retrieval_input = "\n".join([f"{m['role']}: {m['content']}" for m in self.session_memory])
    retrieval_items = self.retrieval.text_to_item_retrieval(retrieval_input, topk=20)

    # Stage 2: Look up only the TOP-1 track's metadata
    recommend_item = self.item_db.id_to_metadata(retrieval_items[0])

    # Stage 3: LLM generates a response about just that one track
    response = self.lm.response_generation(system_prompt, self.session_memory, recommend_item)
```

**Three critical weaknesses here:**
1. The LLM only sees `retrieval_items[0]` — the top-1 track. If retrieval is wrong, the response is wrong.
2. The retrieval query is just raw concatenated text — no semantic understanding of what changed turn-by-turn.
3. The LLM cannot influence retrieval. If it "understands" the user wants jazz but BM25 returns pop, there's no correction mechanism.

### `batch_chat()` — batched version

Same logic but processes multiple turns in parallel. Uses `batch_text_to_item_retrieval()` and `batch_response_generation()` for GPU efficiency. The fallback path (sequential) handles retrieval/LM modules that don't implement batch methods.

---

## 4. Retrieval Module A: `mcrs/retrieval_modules/bm25.py`

### What it does

Builds a BM25 keyword index over track metadata fields (`track_name`, `artist_name`, `album_name`, `release_date`) and retrieves the top-k tracks whose metadata best matches the user's text query by keyword overlap.

### Index building

```python
corpus = []
for track_id in track_ids:
    metadata_str = self._stringify_metadata(metadata_dict[track_id])
    corpus.append(metadata_str)
    # e.g. "track_name: clair de lune\nartist_name: debussy\nalbum_name: ..."

corpus_tokens = bm25s.tokenize(corpus)
retriever = bm25s.BM25()
retriever.index(corpus_tokens)
retriever.save(f"{cache_dir}/bm25/{corpus_name}")
```

The index is cached to disk — subsequent runs load directly from cache, skipping the build step.

### Retrieval

```python
def text_to_item_retrieval(self, query: str, topk: int) -> list[str]:
    query_tokens = bm25s.tokenize([query.lower()])
    doc_scores = self.bm25_model.retrieve(query_tokens, k=topk, return_as="tuple")
    return [self.track_ids[item['id']] for item in doc_scores.documents[0]]
```

Query = the entire dialogue history concatenated. This works well when the user mentions specific artist names or track titles. It fails when the user says things like "something chill for a rainy afternoon" — there's no "chill" or "rainy" in track metadata.

**BM25 strength:** Fast, no GPU needed, perfect recall on named-entity queries (artist/title).
**BM25 weakness:** Zero semantic understanding of mood, vibe, tempo, or genre descriptions.

---

## 5. Retrieval Module B: `mcrs/retrieval_modules/bert.py`

### What it does

Encodes all track metadata strings using BERT (mean-pooled, L2-normalized) into a dense embedding matrix. At query time, encodes the dialogue history the same way and retrieves by cosine similarity.

### Index building

```python
# For each track: stringify metadata → tokenize → BERT encode → mean pool → L2 normalize
outputs = self.model(**batch)
pooled = self._mean_pool(outputs.last_hidden_state, batch["attention_mask"])
pooled = F.normalize(pooled, p=2, dim=1)
# Save as [N_tracks, 768] matrix
torch.save(embedding_mat, os.path.join(self.index_dir, "embeddings.pt"))
```

### Retrieval

```python
query_emb = self._mean_pool(...)          # encode query → [1, 768]
scores = torch.matmul(self.embeddings, query_emb)   # [N, 768] @ [768] → [N]
top_indices = torch.topk(scores, k=topk).indices
```

**BERT strength:** Handles semantic similarity better than BM25 — "upbeat driving music" can match tracks tagged "energetic pop".
**BERT weakness:** `bert-base-uncased` is not music-domain-trained. The embeddings only capture text semantics of the *metadata*, not the actual audio character of the track.

**What the organizers provide instead:** Pre-computed multimodal track embeddings (audio + lyrics + CF) from `talkpl-ai/TalkPlayData-2-Track-Embeddings`. These are far better than re-encoding metadata with BERT. The BERT baseline literally ignores these.

---

## 6. Language Model: `mcrs/lm_modules/llama.py`

### What it does

Wraps Llama-3.2-1B-Instruct to generate the conversational response given the system prompt, chat history, and the top-1 recommended track metadata.

### `_format_chat_history()`

```python
def _format_chat_history(self, sys_prompt, chat_history, recommend_item):
    chat_data = [{"role": "system", "content": sys_prompt}]
    chat_data += chat_history           # previous turns
    chat_data += [{"role": "assistant", "content": recommend_item}]  # ← injects top-1 track
    chat_template = self.tokenizer.apply_chat_template(chat_data, ...)
```

The recommended item is injected as a fake "assistant" turn **before** the actual generation. So the LLM sees the track metadata as if it had already "said" it, then continues from there to explain/justify it.

This is a clever prompting trick — the LLM is never asked to retrieve, only to comment on what was already retrieved. But it means the response quality is entirely dependent on retrieval quality.

### `response_generation()` — single turn

```python
outputs = self.lm.generate(input_ids, attention_mask=attention_mask, max_new_tokens=512)
generated_text = self.tokenizer.batch_decode(outputs[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
```

Greedy decoding, 512 tokens max. No constraints, no beam search.

### `batch_response_generation()` — batched

Same but processes multiple turns at once with left-side padding (important: tokenizer is set to `padding_side="left"` in `__init__`). `max_new_tokens=64` for batch mode (much shorter than single mode's 512) to keep batch inference fast.

**This is a subtle problem:** 64 tokens is often too short for a good music justification, which hurts the LLM-as-Judge score. Increasing this improves quality but slows batch inference.

---

## 7. Databases: `db_item` and `db_user`

### `MusicCatalogDB` — track metadata

```python
def id_to_metadata(self, track_id: str) -> str:
    metadata = self.metadata_dict[track_id]
    entity_str = f"track_id: {track_id}"
    for corpus_type in self.corpus_types:
        entity_str += f", {corpus_type}: {', '.join(metadata[corpus_type]).lower()}"
    return entity_str
    # → "track_id: T123, track_name: clair de lune, artist_name: debussy, ..."
```

Note: `use_semantic_id=False` parameter exists but is **not implemented** — it's a placeholder the organizers left for exactly our integration.

### `UserProfileDB` — user profiles

```python
def id_to_profile_str(self, user_id: str) -> str:
    return "\n".join([f"{key}: {user_profile[key]}" for key in self.default_columns])
    # → "user_id: U42\nage_group: 25-34\ngender: F\ncountry_name: United States"
```

Only 4 fields. No listening history. No embedding. This is the entire user model in the baseline.

---

## 8. System Prompts

Three text files loaded at startup:

**`roleplay.txt`**
```
You are an expert music recommendation assistant. Your task is to understand
user preferences and provide personalized music recommendations.
```

**`response_generation.txt`**
```
Based on the user query and the recommended track from tool calling results,
provide a brief response that:
1. MUST base your response on the previously recommended track...
2. If the recommended track doesn't match the user's query, apologize...
3. Share key details including title, artist, and relevant musical information...
```

**`personalization.txt`** — adds the formatted user profile below the above.

The response generation prompt explicitly tells the LLM to apologize if the track doesn't match. This is an honest acknowledgment of how often BM25/BERT retrieval fails on this task.

---

## 9. Config Files

```yaml
# config/llama1b_bm25_devset.yaml
lm_type: "meta-llama/Llama-3.2-1B-Instruct"
retrieval_type: "bm25"
test_dataset_name: "talkpl-ai/TalkPlayData-Challenge-Dataset"
item_db_name: "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
user_db_name: "talkpl-ai/TalkPlayData-Challenge-User-Metadata"
track_split_types: ["all_tracks"]
corpus_types: ["track_name", "artist_name", "album_name", "release_date"]
device: "cuda"
attn_implementation: "flash_attention_2"
```

Four config variants ship: BM25 vs BERT × devset vs blindset_A. The only differences are `retrieval_type` and `test_dataset_name`.

---

## 10. The Fundamental Problem with the Baseline

Here is a summary of every weakness, ordered by impact on the leaderboard:

| Weakness | Impact | Metric hurt |
|---|---|---|
| LLM only sees top-1 track | Response ungrounded in full ranking | LLM-as-Judge ↓↓ |
| Retrieval & generation are disconnected | No joint optimisation | nDCG@20 ↓↓ |
| BM25 fails on mood/vibe queries | ~50% of queries are semantic | nDCG@20 ↓↓ |
| BERT ignores provided multimodal embeddings | Suboptimal track vectors | nDCG@20 ↓ |
| User model = 4 text fields only | No listening history used | LLM-as-Judge ↓ (personalisation) |
| `max_new_tokens=64` in batch mode | Responses too short to explain well | Distinct-2 ↓, LLM-as-Judge ↓ |
| No diversity post-processing | Popular tracks dominate every session | Catalog diversity ↓↓ |
| No fine-tuning on training data | Model has never seen this task | All metrics ↓ |

---

## 11. Semantic ID Integration — Where & How

We integrate Semantic IDs **surgically into the baseline codebase** — minimal changes, maximum impact. The baseline's modular design makes this clean.

### Overview of changes

```
music-crs-baselines/
├── config/
│   └── llama1b_semantic_ids_devset.yaml    ← NEW config
├── mcrs/
│   ├── crs_baseline.py                     ← ADD: semantic_id mode, MMR, history as IDs
│   ├── db_item/
│   │   └── music_catalog.py                ← EXTEND: id_to_metadata() semantic ID path
│   ├── lm_modules/
│   │   └── llama.py                        ← ADD: vocab expansion + constrained generate
│   └── retrieval_modules/
│       └── semantic_ids.py                 ← NEW: generative retrieval module
└── scripts/
    └── build_semantic_ids.py               ← NEW: run once to build codebook
```

### Change 1 — New retrieval module: `mcrs/retrieval_modules/semantic_ids.py`

This replaces BM25/BERT with generative retrieval. The LLM directly emits track IDs as tokens — no separate index needed.

```python
# mcrs/retrieval_modules/semantic_ids.py

class SemanticIDRetrieval:
    """
    Generative retrieval: fine-tuned LLM emits semantic ID token sequences.
    Implements the same interface as BM25_MODEL and BERT_MODEL so it
    slots into CRS_BASELINE with zero other changes.
    """
    retrieval_type = "semantic_ids"

    def __init__(self, codebook_path: str, model, tokenizer, cfg: dict):
        import pickle
        with open(codebook_path, "rb") as f:
            self.codebook = pickle.load(f)
        self.model = model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self._build_constraint_processor()

    def _build_constraint_processor(self):
        from mcrs.retrieval_modules.constrained_decoding import build_constraint_processor
        self.logits_processor = build_constraint_processor(
            codebook=self.codebook,
            tokenizer=self.tokenizer,
        )

    def text_to_item_retrieval(self, query: str, topk: int = 20) -> list[str]:
        return self.batch_text_to_item_retrieval([query], topk)[0]

    def batch_text_to_item_retrieval(self, queries: list[str], topk: int = 20) -> list[list[str]]:
        from transformers import LogitsProcessorList
        from mcrs.retrieval_modules.decode_ids import decode_id_tokens

        inputs = self.tokenizer(
            queries, return_tensors="pt", padding=True, truncation=True, max_length=1800
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                num_beams=self.cfg.get("beam_size", 40),
                num_beam_groups=self.cfg.get("num_beam_groups", 20),
                diversity_penalty=self.cfg.get("diversity_penalty", 0.3),
                max_new_tokens=topk * 3,   # 2 tokens per track + separator
                logits_processor=LogitsProcessorList([self.logits_processor]),
                pad_token_id=self.tokenizer.eos_token_id,
            )

        results = []
        for i, out in enumerate(outputs):
            gen_ids = out[inputs["input_ids"].shape[1]:]
            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=False)
            track_ids = decode_id_tokens(gen_text, self.codebook)
            # Pad if fewer than topk decoded
            if len(track_ids) < topk:
                fallback = list(self.codebook["track_to_codes"].keys())
                track_ids += fallback[:topk - len(track_ids)]
            results.append(track_ids[:topk])
        return results
```

### Change 2 — Extend `MusicCatalogDB.id_to_metadata()` to use Semantic IDs

The organizers already added `use_semantic_id=False` as a parameter. We implement it:

```python
# In mcrs/db_item/music_catalog.py

def id_to_metadata(self, track_id: str, use_semantic_id: bool = False) -> str:
    metadata = self.metadata_dict[track_id]
    
    if use_semantic_id and hasattr(self, 'track_to_codes'):
        # Return semantic ID tokens instead of text
        # Used when injecting recommended tracks back into the LLM context
        c1, c2 = self.track_to_codes[track_id]
        return f"<{c1}> <{c2 + 256}>"   # coarse token + fine token (offset)
    
    # Original text path (keep for backwards compat)
    track_id_str = metadata['track_id']
    entity_str = f"track_id: {track_id_str}"
    for corpus_type in self.corpus_types:
        corpus_type_value = ", ".join(metadata[corpus_type]).lower()
        entity_str += f", {corpus_type}: {corpus_type_value}"
    return entity_str

def load_codebook(self, codebook_path: str):
    """Call once after __init__ to enable semantic ID mode."""
    import pickle
    with open(codebook_path, "rb") as f:
        codebook = pickle.load(f)
    self.track_to_codes = codebook["track_to_codes"]
```

### Change 3 — Extend `LLAMA_MODEL` to expand vocabulary

```python
# In mcrs/lm_modules/llama.py — add to __init__ or as a new classmethod

def expand_vocabulary_for_semantic_ids(self, n_tokens: int = 512):
    """Add semantic ID tokens to the tokenizer and resize model embeddings."""
    from peft import LoraConfig, get_peft_model, TaskType

    new_tokens = [f"<{i}>" for i in range(n_tokens)]
    added = self.tokenizer.add_special_tokens({"additional_special_tokens": new_tokens})
    self.lm.resize_token_embeddings(len(self.tokenizer))
    print(f"Added {added} semantic ID tokens. Vocab size: {len(self.tokenizer)}")

def load_finetuned_lora(self, checkpoint_path: str):
    """Load a LoRA checkpoint trained on Music-CRS."""
    from peft import PeftModel
    self.lm = PeftModel.from_pretrained(self.lm, checkpoint_path)
    self.lm.eval()
    print(f"Loaded LoRA checkpoint from {checkpoint_path}")
```

### Change 4 — Extend `CRS_BASELINE.chat()` to pass all 20 tracks + use listening history

This is the most important change. Instead of passing only top-1 track to the LLM, pass all 20:

```python
# In mcrs/crs_baseline.py — modified chat() method

def chat(self, user_query: str, user_id: Optional[str] = None,
         listening_history: list[str] = None,
         use_semantic_ids: bool = False) -> dict:

    self.session_memory.append({"role": "user", "content": user_query})
    system_prompt = self._get_system_prompt(user_id)

    # Build retrieval query — optionally include listening history as ID tokens
    retrieval_input = "\n".join([
        f"{m['role']}: {m['content']}" for m in self.session_memory
    ])
    if listening_history and use_semantic_ids:
        history_ids = self._format_history_as_ids(listening_history)
        retrieval_input = f"[HISTORY] {history_ids}\n" + retrieval_input

    # Stage 1: retrieve top-20
    retrieval_items = self.retrieval.text_to_item_retrieval(retrieval_input, topk=20)

    # Stage 2: MMR reranking for diversity
    if hasattr(self, 'mmr_reranker') and self.mmr_reranker:
        retrieval_items = self.mmr_reranker.rerank(retrieval_items, retrieval_input)

    # Stage 3: Format ALL top-20 for the LLM (not just top-1)
    if use_semantic_ids:
        # Pass as ID tokens — compact, no hallucination risk
        recommend_context = " | ".join([
            self.item_db.id_to_metadata(tid, use_semantic_id=True)
            for tid in retrieval_items[:20]
        ])
    else:
        # Original: only top-1 as text
        recommend_context = self.item_db.id_to_metadata(retrieval_items[0])

    # Stage 4: generate response
    response = self.lm.response_generation(system_prompt, self.session_memory, recommend_context)

    return {
        "user_id": user_id,
        "user_query": user_query,
        "retrieval_items": retrieval_items,
        "recommend_item": self.item_db.id_to_metadata(retrieval_items[0]),
        "response": response,
    }

def _format_history_as_ids(self, track_ids: list[str]) -> str:
    """Convert listening history track IDs into semantic ID token strings."""
    tokens = []
    for tid in track_ids[-20:]:   # last 20 tracks
        if hasattr(self.item_db, 'track_to_codes') and tid in self.item_db.track_to_codes:
            c1, c2 = self.item_db.track_to_codes[tid]
            tokens.append(f"<{c1}> <{c2 + 256}>")
    return " | ".join(tokens) if tokens else "(none)"
```

### Change 5 — New config file

```yaml
# config/llama1b_semantic_ids_devset.yaml
lm_type: "meta-llama/Llama-3.2-1B-Instruct"
retrieval_type: "semantic_ids"
test_dataset_name: "talkpl-ai/TalkPlayData-Challenge-Dataset"
item_db_name: "talkpl-ai/TalkPlayData-Challenge-Track-Metadata"
user_db_name: "talkpl-ai/TalkPlayData-Challenge-User-Metadata"
track_split_types: ["all_tracks"]
user_split_types: ["all_users"]
corpus_types: ["track_name", "artist_name", "album_name", "release_date"]
cache_dir: "./cache"
device: "cuda"
attn_implementation: "flash_attention_2"

# Semantic ID specific
codebook_path: "./cache/codebook.pkl"
lora_checkpoint: "./exp/checkpoints/dpo_final"
use_semantic_ids: true
beam_size: 40
num_beam_groups: 20
diversity_penalty: 0.3
mmr_lambda: 0.5
```

---

## 12. Integration Checklist

Do these in order. Each step is independently runnable and testable:

- [ ] **Step 0 — Get a baseline score first**
  ```bash
  cd music-crs-baselines
  python run_inference_devset.py --tid llama1b_bm25_devset --batch_size 8
  # → exp/inference/devset/llama1b_bm25_devset.json
  # Evaluate this with music-crs-evaluator. This is your floor.
  ```

- [ ] **Step 1 — Switch from BERT to provided multimodal embeddings**
  The organizers provide `talkpl-ai/TalkPlayData-2-Track-Embeddings`. Load these instead of computing BERT embeddings on metadata. Expected gain: +3–5 nDCG points. Code change: modify `bert.py` to load the pre-computed `.npy` instead of running the BERT encoder.

- [ ] **Step 2 — Build the semantic ID codebook**
  ```bash
  python ../src/quantize/build_semantic_ids.py --config ../config/train.yaml
  # → cache/codebook.pkl
  ```

- [ ] **Step 3 — Fine-tune the LLM on training data**
  ```bash
  python ../src/train/train.py --config ../config/train.yaml --stage sft
  # → exp/checkpoints/sft_final/
  ```

- [ ] **Step 4 — Run DPO**
  ```bash
  python ../src/train/train.py --config ../config/train.yaml --stage dpo
  # → exp/checkpoints/dpo_final/
  ```

- [ ] **Step 5 — Run inference with Semantic IDs**
  ```bash
  python run_inference_devset.py --tid llama1b_semantic_ids_devset --batch_size 4
  ```

- [ ] **Step 6 — Tune MMR lambda on dev set**
  ```bash
  python ../src/utils/tune_mmr.py \
    --predictions exp/inference/devset/llama1b_semantic_ids_devset.json \
    --ground_truth ../data/ground_truth_dev.json
  # Update mmr_lambda in config
  ```

- [ ] **Step 7 — Evaluate and compare to baseline**
  ```bash
  python ../music-crs-evaluator/evaluate_devset.py \
    --predictions exp/inference/devset/llama1b_semantic_ids_devset.json
  ```

---

*This document reflects the actual code in `music-crs-baselines/` as of May 2026.*
