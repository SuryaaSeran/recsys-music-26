# Plan: Recall Improvement (target >80% pool recall @500)

## Goal

Get BM25@500 union dense@K pool recall above 80%. Current best (v3 TT): 72.7%.
Pool recall is the ceiling for any reranker -- if gold isn't in the pool, no scoring fixes it.

## Current State (from exp/analysis/recall_audit_summary.txt, v3 indexes)

```
BM25 only @500                     0.590
BM25 + TT v3 @500                  0.727
BM25 + TT v3 @1000                 0.786
BM25 + TT + QM @500                0.742
BM25 + TT + QM + QL @500           0.752
BM25 + ALL signals @500            0.781
BM25 + ALL signals @1000           0.846   <- already past 80% if we expand
```

So 80% is reachable just by expanding pool size. The harder goal is hitting 80% at @500 to keep reranker work bounded.

## Current BM25 setup

- Corpus: `track_name + artist_name + album_name + release_date + tag_list` (all lowercased, joined)
- Tokenizer: default `bm25s.tokenize` (lowercase, basic split, no stemming, no stopwords)
- Params: k1=1.5, b=0.75 (lucene defaults)
- Query (inference): `goal + culture + last_4_track_texts + last_4_user_assistant_msgs`
- Pool size: BM25@500, deduped against seen tracks

## Hypotheses (ranked by expected impact)

1. **v6 TT recall is stronger than v3**. Re-audit with v6 → expect BM25+TT@500 > 0.74.
2. **BM25 query has too much conversational noise**. Stripping fillers + stopwords from user turns should sharpen lexical match.
3. **Pool fusion (not just signal fusion)**. Currently BM25 fixes the pool at 500, dense signals only rerank inside. Add TT@K and QM@K as additional pool members → pool recall jumps to ~0.78 from the audit.
4. **BM25 corpus weighting**. Artist + track_name should be weighted higher than tag_list (47k tracks share popular tags like "rock"). Either duplicate the field in the corpus string or use field-specific BM25.
5. **BM25 tokenization improvements**. Stopwords + light stemming (Porter) should improve tag and album-title matches.
6. **Multi-query BM25**. Run two BM25 queries (entity-focused vs mood-focused) and union top-K from each.

## Approach: measure → improve → re-measure

For each change, run audit_recall.py (or equivalent on 1000 sessions) and report pool recall@500.

## Steps

1. Re-run audit with v6 TT index (baseline for v6).
2. Add BM25@500 + TT-v6@500 + QM@500 + CF@500 union pool measurement.
3. If <80%: improve BM25 query (strip stopwords/fillers).
4. If <80%: rebuild BM25 corpus with field weighting.
5. If <80%: BM25 tokenization (stopwords + stemming).
6. If <80%: multi-query BM25 (entity + mood).
7. If <80%: pool expansion to @750 or @1000.

## Validation

After each change: report pool recall@500 on full 1000 dev sessions.
Final: pool recall@500 > 0.80.

## Risks

- Bigger pool slows downstream scoring (TT cosine, fusion) linearly. @1000 is 2x cost.
- Multi-query BM25 doubles BM25 cost.
- Tokenization changes require rebuilding the index (~2 minutes).

## Iteration Log

| Iteration | Change | BM25@500 | BM25+TT@500 | BM25+artist+TT@1000 |
|---|---|---|---|---|
| 0 (baseline) | v6 TT, original query+index | 59.0% | 72.9% | — |
| 1 | Improved BM25 query (all_text, clean, multi) | 58.7% | 73.1% | — |
| 2 | BM25 v2 index (field weights + stopwords) | 57.1% | 71.1% | — |
| 3 | BM25@750 pool | 62.9% | 74.9% | — |
| 4 | Artist expansion | 65.1% | 75.5% | — |
| **5 (final)** | BM25@500 + artist + TT@1000 | 59.0% | — | **80.6%** |

Key findings:
- BM25 query improvements and index changes hurt (more text = diluted IDF)
- Artist expansion is high-value: +6.1% for free via dict lookup
- Pool expansion (TT@1000) is the critical lever to cross 80%
- BM25@500 + artist + TT@1000 → 80.6% ✓ TARGET MET

## Result: pool recall ≥80% achieved, but nDCG unchanged

- Artist expansion + TT@1000 integrated into `run_inference_fusion.py`
- Full 1000-session eval with 80.6% pool recall: nDCG@20 = **0.1518** (no gain)

**Root cause:** expansion candidates enter with `bm25_score=0.0`. The BM25 RR signal
(w_bm25=0.10) penalizes them relative to BM25 top-500 tracks. Even gold tracks rescued
by artist expansion cannot score into the top-20 against BM25 rank-1 competitors.

**Conclusion:** higher pool recall is necessary but not sufficient. The scoring formula
`score = w_tt*cosine + w_bm25*RR + ...` is broken for candidates outside BM25.

**Next step:** cross-encoder reranker that ignores BM25 rank entirely and scores on
(query, track text) interaction. CE is the right tool -- it was abandoned after v1
overfit. Retrain CE on v6 data (richer track text) or use CE to rerank artist-expanded pool.
