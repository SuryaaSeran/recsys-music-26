# Plan: Stage A recall pivot — generative cf-bpr -> fusion (BM25 + dense)

## Context
The generative cf-bpr semantic-ID retriever underperformed and overfit. Pivoted Stage A
recall to classic fusion (BM25 + dense content), which more than doubled recall untrained.

## What we learned (2026-06-06)
- **Generative cf-bpr fails as a recall engine.** Trained Llama-3.2-1B to emit the gold's
  cf-bpr 4-tuple. Dev recall@200 ~0.20 and it **overfit** (epoch4 0.20 -> epoch9 0.10 as
  train tgt_acc 0.45->0.94; the repo trainer reuses the same 15199 examples each epoch).
  Root issue: conversation text -> *collaborative* cf code is the wrong mapping (cf encodes
  co-listen structure, not stated intent), and cf 4-tuples are near-unique so the generator
  must be exact.
- **Content beats collaborative; lexical beats both.** Dense conversation->track recall
  (untrained, Qwen3-Embedding-0.6B), n=300:
  metadata @200=0.30, attributes @200=0.28, lyrics @200=0.09. cf has no text encoder.
  BM25 over track text (name/artist/album/tags/year) @200=0.38 — strongest single signal,
  because users name artists/songs explicitly.
- **Fusion wins.** RRF(BM25 + dense metadata + dense attributes), n=1000 dev:

  | recall@ | 20 | 50 | 100 | 200 | 500 | 1000 |
  |---|---|---|---|---|---|---|
  | fusion | 0.227 | 0.325 | 0.388 | **0.441** | 0.526 | 0.600 |

  vs generative cf-bpr ~0.20@200. Untrained, frozen Qwen3 + BM25.
- Still below the OLD repo's 0.808@~1468 (which used a **trained** two-tower + more sources).
  That gap is the recall headroom.

## Decision
Stage A = **fusion recall** (BM25 + dense Qwen3, RRF). Generative semantic IDs dropped for
recall (may return later as a candidate source or for reranking features, not the engine).

## Implemented
- `src/recall/fusion.py` — `FusionRetriever` (BM25 over `Track.text()` + dense Qwen3
  metadata/attributes, RRF), `render_query` (culture + goal + last-3 turns).
- `scripts/eval_recall.py` — dev recall@K.
- Diagnostics: `scripts/dense_recall.py`, `scripts/fusion_recall.py`.

## Roadmap to higher recall (the lever)
1. **Trained dense retriever** (conversation encoder <-> track encoder, two-tower). Frozen
   Qwen3 metadata dense is only 0.31@200; a fine-tuned tower is the biggest lever toward
   the old repo's 0.5-0.8. Train on (context -> gold track) pairs.
2. **More sources** in the fusion: cf-bpr ANN for warm tracks, artist/album co-occurrence,
   CLAP audio, last-played-track NN. Tune RRF weights.
3. **Pool size** is a Stage-B tradeoff: recall@200=0.44 vs @500=0.53 vs @1000=0.60. Pick by
   how many candidates Qwen3-Reranker-4B can rerank per turn.
4. **Query construction**: weight the last user turn, entity extraction for BM25.

## Note on gpa
When using goal_progress_assessment to weight/filter, it is off by one (gpa_T judges the
rec at T-1); see the original repo's `plan/PLAN.md` "Data correction". Not needed for pure
recall (every turn has a gold).
