# Phase 1 & 2: BM25 Baseline + Pre-trained Dense

**Status: Complete**
**Period:** Competition start
**Outcome:** BM25 + tag expansion + seen exclusion = 0.1313 nDCG@20. This became the floor.

---

## What Was Tried

### BM25 variants

All run with `scripts/archive/run_inference_bm25.py` or `scripts/archive/run_inference_bm25_tagexpand.py`.

| Config | nDCG@20 | Hit@20 | Result file |
|---|---|---|---|
| name + artist + album only | 0.0861 | 21.9% | `devset/bm25_full.json` |
| + tag_list in query | 0.0960 | 25.9% | `devset/bm25_tagexpand_v1.json` |
| + exclude seen tracks | 0.1313 | 27.4% | `devset/bm25_tag_exclseen.json` |
| Full history in query | ~0.096 | ~25% | `devset/bm25_tag_allhist.json` |
| No history, query text only | worse | — | `devset/bm25_tag_nohist.json` |

**Key finding:** Seen-track exclusion alone added +0.035 nDCG@20 (+3.5 points). Tracks
previously recommended are common in BM25 results because the query mentions them.
Excluding them pushes new relevant tracks up.

### CF Reranking

Script: `scripts/archive/run_inference_cf_expand.py`, `scripts/archive/run_inference_hybrid.py`

| Config | nDCG@20 | Hit@20 | Sessions | Result file |
|---|---|---|---|---|
| BM25 + CF rerank (w=0.6) | 0.1196 | 31.8% | 50 | `devset/hybrid_cf06.json` |

**Finding:** CF improved Hit@20 substantially (+4%) but hurt nDCG@20 (-0.012).
CF signals track popularity/similarity clusters but not conversational relevance.
CF reranking was abandoned.

### Qwen Query Reformulation

Script: `scripts/archive/run_inference_qwen_query.py`

Result: No meaningful improvement. Qwen entity extraction added latency with no nDCG gain.
The BM25 query already contains the user's natural language; reformulation does not help.

---

## Pre-trained Dense Retrieval

### all-MiniLM-L6-v2 (pre-trained, no fine-tuning)

Script: `scripts/archive/run_inference_dense_hybrid.py`, `scripts/build_dense_index.py`

| Config | nDCG@20 | Result file |
|---|---|---|
| Dense only | 0.0654 | `devset/dense_only.json` |
| BM25+Dense RRF | 0.0775 | `devset/rrf_d200_b500.json` |

**Finding:** Pre-trained all-MiniLM-L6-v2 performs worse than BM25 alone on this task.
Music retrieval queries are conversational and specific; the model has not seen this format.

### Qwen3 Embeddings (challenge-provided)

Script: `scripts/archive/run_inference_qwen3_dense.py`

| Config | nDCG@20 | Sessions | Notes |
|---|---|---|---|
| Qwen3 dense probe w=0.5 | 0.1509 | 50 | Unreliable small sample |
| Qwen3 dense probe w=0.7 | 0.0975 | 100 | |

**Finding:** The challenge provides pre-computed multimodal embeddings (audio+lyrics+CF+text),
but they do not align with conversational query encoding. Encoding the query with Qwen3 and
comparing against track embeddings yielded inconsistent results. Fine-tuning on query-track
pairs is necessary.

---

## What the BM25 Query Looks Like

```
{goal} {culture} {track_1_name} {track_1_artist} {track_1_tags} ... {user_turn_1} {user_turn_2} ...
```

Uses last 4 recommended tracks (metadata) + last 4 user text turns.
Full query construction is in `scripts/archive/run_inference_bm25_tagexpand.py` → `build_query()`.

---

## Lessons

1. Tag list in track metadata is the most useful BM25 field — genre, mood, era tags.
2. Exclude seen tracks before returning top-20. This is the single biggest win in Phase 1.
3. History length matters: last 4 tracks + last 4 turns is optimal. Full history adds noise.
4. Pre-trained dense models need fine-tuning to be useful here. Do not waste time with zero-shot dense retrieval.
