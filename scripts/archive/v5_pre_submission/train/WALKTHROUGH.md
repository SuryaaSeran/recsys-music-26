# Train Pipeline Walkthrough

This directory contains everything needed to (re)build the assets used by the inference pipeline. Run the four scripts below in order. Total time on an M-series Mac: ~5 hours, mostly spent fine-tuning the bi-encoder.

```
build_fusion_index.py        # 1. Cache all precomputed embeddings (CF, CLAP, Qwen3 *3) + user CF vectors
build_twotower_data.py       # 2. Mine training pairs (anchor / gold / hard negatives) from train sessions
train_twotower.py            # 3. Fine-tune all-MiniLM-L6-v2 with MNRL on those pairs
build_twotower_index.py      # 4. Encode every track with the trained model -> dense index
```

The fusion inference reads from these caches; nothing else is trained from scratch in this project.

---

## 0. Datasets and what we use from each

| HF dataset | Split | Used for |
|---|---|---|
| `talkpl-ai/TalkPlayData-Challenge-Dataset` | `train` | Mining `(anchor, positive, negatives)` pairs (`build_twotower_data.py`) |
| `talkpl-ai/TalkPlayData-Challenge-Dataset` | `test` (= dev) | Eval at inference time |
| `talkpl-ai/TalkPlayData-Challenge-Track-Metadata` | `all_tracks` + `test_tracks` | Building track text for BM25 + dense indexing |
| `talkpl-ai/TalkPlayData-Challenge-Track-Embeddings` | `all_tracks` + `test_tracks` | Pre-computed CF, CLAP, Qwen3 (meta/attr/lyrics) — never re-trained |
| `talkpl-ai/TalkPlayData-Challenge-User-Embeddings` | `train` + `test_warm` + `test_cold` | CF-BPR user vectors (cold-start users get None) |

`metadata_dict[track_id]` schema (after concatenating `all_tracks` and `test_tracks`):

```python
{
  "track_id":     "97f5eeec-...",
  "track_name":   ["With Rainy Eyes"],         # always wrapped in a list
  "artist_name":  ["Emancipator"],
  "album_name":   ["Soon It Will Be Cold Enough"],
  "tag_list":     ["relaxing", "experimental", "Instrumental", "piano", ...],
  "popularity":   39.0,
  "release_date": "2006-12-06",
  "duration":     300920,                       # ms
  "ISRC":         ["TCABY1497179"],
  "artist_id":    [...], "album_id": [...],
}
```

The list-wrapping is a TalkPlay convention — always index `[0]` defensively, with `(row.get("track_name") or [""])[0]` to survive missing values.

---

## 1. `build_fusion_index.py`  — cache pre-computed embeddings

**Why this exists.** The challenge ships pre-computed track and user embeddings in HuggingFace datasets. Hitting HF on every inference run is slow and turns inference into a network-dependent process. This script downloads them once and caches as `.npy` arrays for fast `np.load` later.

### Inputs

- `talkpl-ai/TalkPlayData-Challenge-Track-Embeddings` (HF) — five columns per track:
  - `cf-bpr` — 128-dim collaborative filtering BPR
  - `audio-laion_clap` — 512-dim LAION CLAP audio embedding
  - `metadata-qwen3_embedding_0.6b` — 1024-dim Qwen3 over track metadata text
  - `attributes-qwen3_embedding_0.6b` — 1024-dim Qwen3 over attributes
  - `lyrics-qwen3_embedding_0.6b` — 1024-dim Qwen3 over lyrics
- `talkpl-ai/TalkPlayData-Challenge-User-Embeddings` — `cf-bpr` per user

### Outputs

```
cache/cf_bpr/track_embeddings.npy        (54476, 128)   float32, L2-normalized
cache/clap/track_embeddings.npy          (54476, 512)
cache/qwen3_meta/track_embeddings.npy    (54476, 1024)
cache/qwen3_attr/track_embeddings.npy    (54476, 1024)
cache/qwen3_lyrics/track_embeddings.npy  (54476, 1024)
cache/<name>/track_ids.json              # parallel list of UUIDs in row order
cache/user_cf_bpr.json                   # {user_id: [128-dim vector]}
```

### Key code

```python
def safe_emb(rows, key, dim):
    out = np.zeros((len(rows), dim), dtype=np.float32)
    for i, row in enumerate(rows):
        v = row[key]
        if v and len(v) == dim:
            out[i] = v          # leave zeros for missing/short
    return out

def normalize(x):
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(n, 1e-8)
```

Two non-obvious decisions:

- **Missing embeddings become zeros.** A zero vector has `cos(., q) = 0` for any `q`, which is the right behaviour: a track without a CLAP/lyrics/CF embedding effectively skips that signal at inference instead of crashing or producing garbage.
- **L2-normalize at write time.** Inference does dot products — pre-normalizing means `a @ b == cos(a, b)`, no per-query normalization in the hot loop.

### Pitfalls

- **The 7,405-track gap.** Track-metadata has 47,071 tracks but track-embeddings has 54,476. The extras are present in `cache/qwen3_meta` etc. but **not** in the BM25 index (which is built only from metadata-listed tracks). Inference handles this with `bm25_to_X` lookup tables (`-1` for "track not in this signal's index"), but if you change the BM25 build you must keep the lookups in sync.
- **Don't filter to BM25-known tracks here.** Future work might recall outside the BM25 pool, and you'd want the full 54k index.
- **Re-running over-writes cache silently.** If a HF dataset version changes, you'll get a quietly-shifted index that still loads fine. Keep a hash or version file if you ever pin reproducibility hard.

### Opportunities

- The challenge gives you **five** track signals but inference uses meta + attr + lyrics + CLAP + CF. There's no signal we're throwing away, but you could derive new signals from these (e.g., `meta + lyrics` averaged is essentially a free signal).
- Popularity is in metadata but not embedded — gold tracks average pop=43 vs all=36, so a `popularity / 100` scalar could be added as a sixth signal.

---

## 2. `build_twotower_data.py`  — mine (anchor, positive, negatives) triples

**Why this exists.** The bi-encoder learns from contrastive pairs. The challenge gives us paired conversations and tracks, but no "negative" labels — so we mine BM25 hard negatives ourselves.

### Inputs

- `talkpl-ai/TalkPlayData-Challenge-Dataset` (`train` split, ~30k sessions)
- `cache/bm25/track_metadata/` — BM25 index built externally over track texts
- The metadata dict from §0

### Outputs

```
data/twotower/train.jsonl       # ~115k examples (one per music turn in train sessions, after filtering)
data/twotower/valid.jsonl       # ~6k examples (held-out 5% of sessions, by --valid_frac)
```

### Schema of one example

```json
{
  "anchor":     "play me something chill I like emancipator chill listening Western pop With Rainy Eyes Emancipator",
  "positive":   "Bonobo Black Sands downtempo electronic atmospheric trip-hop instrumental",
  "negative_1": "Tycho Awake ambient electronic instrumental chillwave",
  "negative_2": "...",
  "negative_3": "...",
  "negative_4": "...",
  "negative_5": "..."
}
```

### Feature engineering — the critical part

The `anchor` is **NOT** the full conversation. It's a *compact* query, deliberately:

```python
latest_user = text_in_history[-1] if text_in_history else ""
parts = [latest_user, goal, culture]
for tid in music_in_history[-2:]:
    parts.append(f"{name} {artist}")    # name+artist only, no tags
anchor = " ".join(p for p in parts if p)
```

**Reasoning** (this is the single most important lesson learned in the project):

The all-MiniLM-L6-v2 base model has a 256-token soft limit (positional embeddings up to 512, but it was trained on shorter inputs). An earlier version concatenated the full conversation + every prior track's name + artist + **tags** + the last 4 user turns. Median token length: 530, with 81% of queries truncated. The model was effectively ranking based on whatever survived truncation — usually a noisy slice.

Switching to the compact format above brought median tokens to ~101, training loss dropped by ~30%, and dev nDCG@20 went from 0.1364 (v1, long query) to 0.1418 (v3, compact query). **Order matters**: latest user request first because it's the strongest signal and we want it to survive truncation if it ever happens.

The `positive` text is the gold track's `name + artist + tags`. Tags are included on the document side because (a) document-side text isn't user-controlled and so isn't biased toward truncation, (b) tags carry the genre/mood signal that BM25 also relies on, keeping the document representation rich.

The `negative_K` columns come from BM25 top-K (K=5) on the anchor, excluding the gold and excluding tracks already played in this session. These are **hard negatives**: tracks BM25 thinks are relevant for the same query but aren't gold. Including them in the file gives downstream code the option to use them; v3 (best) trains with MNRL only and ignores them. v4 used them with MNRL+hard, v5 used them with TripletLoss.

### Pitfalls

- **BM25 hard negatives are noisy positives.** The TalkPlay corpus has many tracks that are genuinely relevant for a given conversation — the dataset just picks one as gold. Training the model to push these "negatives" away made v4 worse, not better (0.1418 → 0.1364). If you use them, expect to need careful weighting or curriculum.
- **Empty texts.** Some tracks have no name/artist/tags (data quality). The `if not gold_text.strip(): music_in_history.append(...); continue` line skips those turns rather than emitting empty positives.
- **`valid` is held-out at the *session* level**, not the example level — this is what you want, otherwise the same conversation context bleeds into both splits.
- **Random seeds matter** because of the session-level split. `--seed 42` is fixed; if you change it, the v3 model's eval loss numbers won't be comparable.

### Opportunities

- Sample more than 5 hard negatives and use harder mining (rank 50-100 instead of 1-5) — current top-5 BM25 is *too* hard (often the real positive).
- Multi-positive: pick a few tracks the user listened to in the same session as additional positives — turns are not the only signal.
- Use the existing pairs to build (query, query) pairs for retrieval-aware query encoding, not just track-side encoding.

---

## 3. `train_twotower.py`  — fine-tune the bi-encoder

**Why this exists.** Out-of-the-box `all-MiniLM-L6-v2` is trained on web NLI/STS — it has no idea what TalkPlay-style requests look like. We fine-tune it for ~one pass over the mined pairs.

### Inputs

- `data/twotower/train.jsonl` and `valid.jsonl` from step 2
- Base model `sentence-transformers/all-MiniLM-L6-v2` (22M params, 384-dim, 256 tokens effective)

### Output

- `models/twotower_v3/final/` — full SentenceTransformer save (config, tokenizer, weights)

### Architecture

| | |
|---|---|
| Tokenizer | BERT WordPiece, 30522 vocab |
| Encoder | 6-layer BERT, hidden 384, intermediate 1536, 22M params |
| Pooler | mean-pooling (default for sentence-transformers MiniLM) |
| Output | 384-dim, normalized at encode time |

### Best config (v3)

```bash
python scripts/train/train_twotower.py \
    --data_dir data/twotower \
    --out_dir models/twotower_v3 \
    --epochs 2 --batch_size 32 --lr 2e-5 --warmup_steps 200
```

### Loss

Default: `MultipleNegativesRankingLoss(model)` (MNRL). For each training row, the positive is the gold; the **other 31 positives in the batch act as in-batch negatives**. This is why batch size matters — at batch=32 you get 31 implicit negatives per anchor.

The flags `--hard_neg` and `--triplet` enable two variants:

- `--hard_neg`: adds `negative_1` from the JSONL as an *explicit* negative (in addition to in-batch). v4 used this for one extra epoch on top of v3 — got 0.1364, worse than v3's 0.1418.
- `--triplet`: switches to `TripletLoss(margin=0.5)`, requires `--hard_neg`. v5 used this and **collapsed** to 0.0525 nDCG — the margin pushed all examples to the same point in space.

### MPS-specific care

```python
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")
device = "mps" if torch.backends.mps.is_available() else "cpu"
fp16=False, bf16=False                  # MPS doesn't support these reliably
dataloader_num_workers=0                # MPS has issues with multi-worker dataloaders
```

These were all needed to keep training from OOMing or hanging on M-series.

### Pitfalls

- **In-batch negatives can include duplicates.** If two rows in the same batch happen to share a positive, MNRL will treat one as a negative for the other. Probability is low at batch=32 across 115k examples but worth knowing.
- **Eval loss isn't nDCG.** Loss going down doesn't always mean retrieval going up. Always run `evaluate_local.py` on a held-out dev set after training.
- **Don't trust 200-session estimates.** v6 (a config change) showed +0.0057 on 200 sessions but only +0.0001 on the full 1000. Validate retraining on the full set or use the precomputed grid search.
- **Hard negatives hurt at this dataset size.** See v4. The training pairs are noisy enough that the model's already discriminating fine; pushing real positives away makes it worse.
- **TripletLoss margins above ~0.3 are unstable** for bi-encoders trained from a generic MiniLM start — v5 collapsed at 0.5.
- **Seed isn't set on the trainer.** Same data + same hyperparams will produce slightly different models. Fine for ablation but not bit-reproducible.

### Opportunities

- **Bigger base model.** all-mpnet-base-v2 has 768-dim and a 384-token effective window; a single epoch from there might beat v3.
- **Curriculum on negatives.** Start with random negatives (in-batch only), gradually mix in BM25-rank-50 negatives, then BM25-rank-5. Avoid the v4 collapse.
- **Two-encoder asymmetry.** Currently the same MiniLM encodes both the anchor and the document. Split into a query encoder and a document encoder (still tied at the start, then separately fine-tuned) — common in DPR-style training.
- **Multi-task loss.** Add an auxiliary objective: predict whether the artist of the gold track appears in `music_in_history` — would teach the encoder to use session context properly.

---

## 4. `build_twotower_index.py`  — encode the entire catalog

**Why this exists.** Inference scores cosine similarity against every track. We encode all 47,071 tracks once, save as `.npy`, and then dot-product against query embeddings at inference time (a single 47071×384 matmul, milliseconds on CPU).

### Inputs

- `models/twotower_v3/final/` from step 3
- The metadata dict from §0 (only the 47,071 metadata-listed tracks)

### Output

```
cache/twotower_v3/track_embeddings.npy    (47071, 384)   float32, L2-normalized
cache/twotower_v3/track_ids.json          # parallel list of UUIDs
```

### Track text used at encode time

```python
def get_track_text(tid):
    name   = (row.get("track_name")  or [""])[0]
    artist = (row.get("artist_name") or [""])[0]
    tags   = " ".join(row.get("tag_list") or [])
    return f"{name} {artist} {tags}".strip()
```

This **must match** the document side of training (`positive` field in the JSONL) — same name+artist+tags. If you change one without the other, the encoder's seen one distribution at training and a different one at indexing, and cosine similarity becomes meaningless.

### Pitfalls

- **`batch_size=512` is aggressive.** Drop to 128 if MPS OOMs on a smaller machine.
- **Track count must be 47,071 exactly.** That's how many tracks are in the metadata dataset. If you ever encode against the 54,476-track embeddings dataset's track list, you'll get phantom indices the BM25 pool can never refer to.
- **Re-encoding silently overwrites.** If you swap to a v4/v5 model and re-run with the same `--out_dir`, your old index is gone. Use `--out_dir cache/twotower_v4` to keep them side by side.

### Opportunities

- **Add album_name to track text.** The BM25 index already uses album terms, but the dense encoder doesn't see them. This is a one-line change with potential upside for "more from that album" queries.
- **Multi-vector tracks.** Encode `name+artist`, `tags`, and `album` separately and store all three; let the inference combine them. More flexible than the single concatenated string.
- **Quantize to int8.** 47k×384 is small, but if you ever scale to a larger catalog, int8 quantization is essentially free in retrieval quality and cuts memory 4x.

---

## End-to-end run

```bash
# 1. Cache the precomputed embeddings (~5 min, network-bound)
python scripts/train/build_fusion_index.py

# 2. Mine training data (~30 min, BM25-bound)
python scripts/train/build_twotower_data.py --hard_negs 5

# 3. Fine-tune (~3 hours on M-series Mac at batch=32, 2 epochs)
python scripts/train/train_twotower.py \
    --data_dir data/twotower --out_dir models/twotower_v3 \
    --epochs 2 --batch_size 32 --lr 2e-5 --warmup_steps 200

# 4. Encode the catalog (~5 min)
python scripts/train/build_twotower_index.py \
    --model models/twotower_v3/final --out_dir cache/twotower_v3
```

After this, everything `scripts/inference/` needs is on disk. See `scripts/inference/WALKTHROUGH.md` for what happens next.

---

## Cross-cutting pitfalls

- **The BM25 index itself is not built by any script in this repo** — `cache/bm25/track_metadata/` is assumed to exist. If you ever need to rebuild it, it's a `bm25s.BM25.fit(tokens)` over `name + artist + album + tags + release_date`, then `bm25_model.save(CACHE_PATH)`.
- **MPS quirks** (no fp16, no multi-worker DataLoaders, occasional kernel hangs) account for several of the `os.environ` and arg defaults. On CUDA you'd want fp16 + 4 workers and ~3x faster training.
- **All caches assume relative paths from repo root.** Every `np.load("cache/...")` will fail if you `cd` into a subdirectory. The scripts are designed to be run from `/Users/.../ReccysMusic/`.

## Cross-cutting opportunities

1. **Re-train `models/twotower_v3` with album_name added** to both `build_twotower_data.py`'s `get_track_text` and `build_twotower_index.py`'s. Same change in both places, ~3 hours of training, possible +0.001-0.003 nDCG.
2. **Mine pairs from the dev set turns we currently fail** — i.e., where gold is at BM25 rank 50+ but final rank > 20. These are the model's actual failure modes; targeted training on them would shift the cosine separation right where we need it.
3. **Add a popularity scalar at index time** so it's available as a free signal in the fusion (currently popularity is in metadata but unused in any embedding).
