# Competitor Systems (RecSys Challenge 2026, Music CRS)

Notes from reading published competitor code. Read-only analysis; sources are
public GitHub repos cloned locally. Comparisons are against our Blind B v2
system (composite 0.49): 9-source recall + 67-feature LightGBM LambdaMART +
manual pivot-aware responses. See `plan/CURRENT_BEST_ITERATION.md` and
`wiki/BLIND_B.md` for our side.

---

## 1. volart — artvolgin/music-crs-recsys2026

Source: https://github.com/artvolgin/music-crs-recsys2026 (read 2026-07-16).
3-stage retrieve -> rerank -> respond pipeline, ~5,972 lines of Python in `src/`.

### Stage 1: retrieval (top-500 pool)

RRF fusion over 5 lanes:

1. Dense: OpenAI `text-embedding-3-large` over track name/artist/album vs a
   `gpt-4o-mini`-rewritten one-line query.
2. BM25 (`bm25s`) over the same fields + tags.
3. Entity filter: `gpt-4o-mini` extracts named artists/tracks/albums/era;
   exact catalog matches are promoted (up to 60 filter slots, ranked by
   dense score).
4. LLM-description dense lane: OpenAI embeddings over `gpt-4o-mini`-generated
   track descriptions (subgenre/era/mood).
5. Track co-occurrence computed only over provided challenge-train dialogues.

Production entry point: `retrieve_batch_hybrid_with_entity_filter`
(`src/retrieval/retrieve.py`), called from `src/orchestration/pipeline.py`.

### Stage 2: rerank (500 -> 20)

LightGBM LambdaMART, model version v9, 69 features
(`src/reranker/lambdamart/features.py`). Hyperparams: `objective=lambdarank`,
`learning_rate=0.05`, `num_leaves=63`, early stopping 50. Trained on 5,000
provided-train sessions with a deterministic md5-based 1,500-session held-out
subset excluded, seed=7.

Feature families: retrieval scores/ranks (6), candidate intrinsics (8),
conversation context (3), cross-history "inertia" (6), category/specificity/
profile one-hots (~22; categories A-K, specificity LL/LH/HL/HH same as ours),
listener-goal x candidate + era stability (10), mood-class overlap via a
hand-built lexicon (4), train gold-frequency priors (3), co-occurrence (3),
MOVES-rate priors (3; Bayesian-smoothed `(moves+2)/(moves+does_not+4)`,
computed on 8,704 sessions disjoint from reranker training).

### Stage 3: response (tracks untouched)

`gemini-3.1-flash-lite-preview` drafts from top-3 tracks, then three text-only
passes (refine -> harden -> lexdiv), each gated on an independent editorial
critic (`gpt-4o-mini`, generic rubric, explicitly not the competition judge,
stated as an anti-manipulation measure). The lexdiv pass de-templates repeated
stock phrases via LLM (same LexDiv collapse problem we hit; they solved it
with automated de-templating, we solved it with hand-written responses).

### Provided-embeddings finding (key)

They built infrastructure to use the provided TalkPlayData-2 embeddings but
none of it shipped:

- `src/utils/isrc_index.py` joins the challenge catalog (UUIDs) to
  TalkPlayData-2 rows (Spotify IDs) via ISRC codes, 96.1% coverage, and can
  load `metadata/attributes/lyrics-qwen3_embedding_0.6b`, `audio-laion_clap`,
  `image-siglip2`, `cf-bpr` matrices.
- `retrieve.py` has experimental lanes over these (dense top-k per field,
  CF-BPR mean-of-played-vectors, weighted RRF with a category router boosting
  CLAP x2 for audio-cue categories) with no production caller.
- A BM25 v2 with TP2 tag enrichment (~80 tags/track vs 10-15) exists, off by
  default, no production caller (parallel to our own unused BM25 v2, D9).
- Reranker features contain no provided-embedding signals; their dense cosine
  feature is OpenAI.

Net: they evaluated the provided embeddings, abandoned them, and paid for
OpenAI embeddings. We built on them (attributes-qwen3 feeds our Qwen lane and
the RQ-VAE semantic IDs behind SASRec; cf-bpr is one of our 9 recall sources).

### Progress-labels finding (key divergence)

Their feature-history docstring documents v7 (graded {0,1,2} labels from
`goal_progress_assessment`) as a mathematical no-op for nDCG with one positive
per query. They also found the 13.7k progress-labeled set "hard to train on
directly" (regressed val nDCG@20) and used it only for aggregate frequency
priors. We reached the opposite exploitation: drop rejected-gold LTR groups
(`--skip_no_progress`), mine TT hard negatives from rejections, and modulate
inference queries (H1/H3), all of which gained for us. Same signal, different
mechanism: they tried it as a label grade (gradient no-op), we used it for
group filtering, negative mining, and query construction.

### Compliance / reproducibility

- `COMPLIANCE.md`: explicit non-use disclosure (LFM-2B, Last.fm getSimilar,
  track2vec, Spotify-MPD, external demographic/temporal priors); documents
  "reverted exploit features" that exist in git history but not the final
  model. Written for organizer top-entry validation.
- `PYTHONHASHSEED=0` pinned for byte-identical track lists; reply text
  disclosed as LLM-sampled, non-deterministic. Submitted `prediction.json`
  shipped in-repo.
- ~4.7 GB artifacts via a Hugging Face dataset, including API replay caches
  keyed on (session_id, turn) so Stage 1/2 + base reply reproduce with zero
  API spend.

### Contrast with ours

| Axis | volart | ours |
|---|---|---|
| Recall | 5 lanes, RRF | 9 sources, fusion + LTR rescoring |
| Dense encoder | OpenAI text-embedding-3-large (frozen, paid API) | multilingual-e5-base + LoRA fine-tune (TT v8d, local) |
| Provided embeddings | Evaluated, not shipped | Core (qwen3 attrs, cf-bpr) |
| Semantic IDs / SASRec | None | RQ-VAE buckets + SASRec expansion lane |
| Reranker | LGBM LambdaMART, 69 feat, leaves=63, lr=0.05 | LGBM LambdaMART, 67 feat, leaves=31, lr=0.08 |
| Progress labels | No-op as label grades; priors only | Group filtering + hard negs + query modulation |
| Responses | Automated Gemini + 3 critic-gated passes | Manual, pivot-aware (Opus), out of pipeline |
| External API dependency | OpenAI + Gemini for retrieval and response | None for retrieval |
| Determinism | PYTHONHASHSEED=0, replay caches | Local artifacts, no documented hash-seed pin |

---

## 2. niwatori — ryowk/recsys2026-niwatori

Source: https://github.com/ryowk/recsys2026-niwatori (read 2026-07-16). Top-5
entry. Blind-B Codabench score ≈ 0.59 (overall), local 5-fold nDCG@20 =
0.2743 on 129,592 public labeled rows. Config name
`blind_b_safe_combined_tpd1_parts_cooc500_t200_cv5`. Full method writeup in
their `docs/method.md`.

```
Blind-B input
  -> retriever/union            14 candidate sources, ordered_unique merge
  -> reranker (LightGBM)        LambdaRank, 200 trees, 176 features -> top 20
  -> responder (Qwen3.6-27B)    10 seeded generations -> lexical-diversity pick
```

### Stage 1: retrieval (14-source union, no global cap)

Concatenated in source order with `ordered_unique` (not RRF — plain dedup
concat, per-source scores/ranks kept as reranker features). Sources: `bm25`
(5-field, cap 200), `tfidf` (track metadata only, cap 200), `twotower`
(LoRA on Qwen3-Embedding-0.6B query tower vs. official track embeddings,
cap 200), `history_artist`/`history_album` (artists/albums in listening
history, src top500), `last_artist`/`last_album` (continuation from the last
music turn, src top500), `exact_album_artist`/`exact_title` (surface string
match, src top500), `tag_intent` (genre/mood/descriptor BM25, cap 100),
`cooc_track` (co-occurrence with history, cap 500), `transition_track`
(Markov next-track counts from last track, src top500), `cooc_album` (cap
200), `cooc_artist_name` (cap 100).

Mean candidate pool: 853 for public-labeled rows, 669 for Blind-B.
`recall@20 = 0.4102`, `@50 = 0.5129`, `@100 = 0.5693`, `@200 = 0.6129`.

**TalkPlayData-1 usage**: not an independent source — its co-occurrence/
transition counts are added into `cooc_track`/`transition_track` after
mapping TPD1 Spotify IDs to the challenge catalog via an ISRC/Spotify-map
built from TalkPlayData-2 (unmapped tracks dropped). They audited for
train/TPD1 duplicate leakage (full-sequence and full-text exact overlap = 0)
before treating it as safe external data added identically to every fold.

**Two-tower retriever** (their closest analogue to our TT-v8d): Qwen3-
Embedding-0.6B LoRA query tower vs. a track tower built from metadata/audio/
image/CF/attribute/lyrics/popularity features (i.e. they fuse the *provided*
embeddings into the track-tower features — unlike volart, they do use the
challenge embeddings here). Standalone recall (`oof5_top500_bsafe`, fold0):
`recall@20=0.2770, recall@100=0.5067, recall@500=0.6515, mrr@500=0.0721`.
They tried mixing TPD1 pairs into training (`sample200k`) and TPD1-only
pretraining — both regressed standalone recall vs. public-only training, so
neither shipped in the final 5-fold sweep; documented as a negative result
in their retriever README, same "tried it, logged the regression, moved on"
discipline as our own ablation notes.

**Blind-B-safe design (structural, not incidental)**: Blind B withholds
`conversation_goal`, `goal_progress_assessments`, and `thoughts` for some
users. Their entire pipeline is built to never depend on these fields in
query construction or ranking — not just "skip if missing" but fixed-width
feature vectors where blind-B-safe columns are neutralized as *values*
(kept as columns so the shipped model's 176-feature layout always matches,
checked at load time) rather than dropped as columns. This is a materially
different design response than ours: we use `goal_progress_assessment`
directly where available and fall back to `--infer_progress_labels` (text-
derived reactions) for Blind B, which has no ground truth. They instead
architected the whole reranker to be blind to the field's presence/absence
at the schema level.

### Stage 2: rerank (full union pool, no truncation before reranking)

LightGBM LambdaRank, 176 features. Hyperparams: `n_estimators=200,
num_leaves=63, learning_rate=0.04, subsample=0.85, colsample_bytree=0.85,
min_child_samples=20, seed=20260520`. Trained on positive-target rows only
(gold present in candidates), fit on all public labeled rows for the shipped
model.

Feature groups: track/user/turn basics; history-consistency (artist/album/
track match, last-music match, tag overlap); query-metadata similarity
(TF-IDF, intent tags, dense cosine on Qwen3-Embedding vectors — this is
where the provided embeddings feed the reranker directly); metadata
extensions (ISRC year/country, duration buckets, age-release-year
consistency, seasonal tags); hierarchical popularity (within-artist/album
popularity/counts); tag-chain features (history-tag Jaccard/cosine, PPMI
graph neighbor overlap); per-source presence/rank/score transforms
including the TPD1-blended `score__challenge`/`score__tpd1`/
`score__transition_probability` parts.

### Stage 3: response (Qwen3.6-27B, ensemble-selected, not critic-gated)

Top-3 ranked tracks -> Qwen/Qwen3.6-27B (bf16, single 80GB GPU), thinking
mode explicitly disabled (`enable_thinking: false`) for direct response.
`max_new_tokens=200, temperature=0.7, top_p=0.9`. 10 seeded generations per
row; the response maximizing corpus-level lexical diversity (distinct-1 +
distinct-2, greedy selection over 30 seeded random-order trials) is chosen —
this is their answer to the same LexDiv-collapse problem volart solves with
an LLM critic pass and we solve by hand-writing responses. No separate
critic model; diversity is a deterministic post-hoc statistic over the
sampled candidates, not an LLM judgment call.

Their config-history comments (in `responder/qwen36_27b/README.md`) log a
full response-prompt sweep with per-submission Blind-A composite/LLM-judge
scores across ~10 prompt/tone variants (calm_curator, warm_friend,
music_critic, discovery_dj, etc.) — evidence of the same kind of prompt-
tone ablation work we did manually for our pivot-aware Opus responses, but
run as a coded, re-runnable sweep against a fixed ranking.

### Reproducibility (their strongest point vs. both other systems)

- Ships a `weights/reranker_lgbm.txt` + dense caches + union artifact
  (~0.3GB total) that make ranking **deterministic and bit-exact**: 
  `verify_blind_b_ranking.py --strict` reproduces all 80 Blind-B rows
  identically on the pinned `uv.lock` environment.
  Responder text is disclosed as GPU-sampling-nondeterministic (seeded, but
  CUDA kernels aren't bit-stable); track IDs are unaffected.
- Three explicit reproduction tiers documented in the README: (1) load-only
  inference (no GPU needed for ranking, ~10 min CPU + a GPU pass for the
  responder), (2) 5-fold CV validation from public sources, (3) full
  train-from-scratch (~half a day, needs the two-tower GPU training).
- `uv.lock` pins exact library versions (torch 2.11.0, transformers 5.7.0,
  lightgbm 4.6.0, etc.) — a stronger determinism guarantee than either
  volart's `PYTHONHASHSEED=0` note or our repo's current state (we don't
  ship a pinned lockfile or a bit-exact verification script).
- Compliance notes are short and specific rather than a separate document:
  `track_emb.test_tracks` (target-side track set) never used; TF-IDF fits
  catalog-only; retrievers use OOF fitting so train-row features never leak
  the row's own fold.

### Contrast with ours

| Axis | niwatori | ours |
|---|---|---|
| Recall | 14 sources, ordered_unique concat (no cap on total pool) | 9 sources, fused + LTR-rescored |
| Provided embeddings | Used directly (Qwen3-Embedding-0.6B in two-tower track tower + dense cosine reranker feature) | Core (qwen3 attrs feed our Qwen lane + RQ-VAE semantic IDs) |
| Dense encoder | Qwen3-Embedding-0.6B LoRA query tower (local fine-tune) | multilingual-e5-base LoRA (TT-v8d, local fine-tune) — same "fine-tune the query side, keep track side frozen from provided/official embeddings" philosophy |
| External sequence data | TalkPlayData-1 blended into cooc/transition counts, audited for leakage | Not used (not mentioned in our pipeline) |
| Reranker | LGBM LambdaRank, 176 feat, no candidate truncation before rerank, leaves=63, lr=0.04, trees=200 | LGBM LambdaMART, 67 feat, leaves=31, lr=0.08, early_stop=75 |
| Progress-aware fields | Structurally excluded (blind-B-safe by design, fixed-width neutralization) | Used directly when available; text-inferred fallback for Blind B |
| Semantic IDs / SASRec | None | RQ-VAE buckets + SASRec expansion lane |
| Responses | Automated Qwen3.6-27B, 10-way lexdiv selection, no critic model | Manual, pivot-aware (Opus), out of pipeline |
| Reproducibility | Bit-exact ranking verification script + pinned lockfile | Local artifacts, no lockfile or bit-exact check |
| Reported score | Blind-B Codabench ≈ 0.59 overall | Blind B composite 0.49 (v2, submitted) |

### Cross-cutting observations (all three systems)

- **Recall fusion strategy differs across all three**: volart uses RRF,
  niwatori uses plain ordered-unique concatenation with per-source features
  left for the reranker to weigh, we use a custom fusion + rescoring step.
  Both competitors lean harder on the reranker to sort out source quality
  than on the fusion step itself.
- **Every system independently rediscovered TF-IDF/BM25 + a supervised
  dense retriever + co-occurrence as a solid 3-legged retrieval base** —
  the design converges even though implementations differ substantially.
- **Both competitors treat the LLM response-diversity problem (LexDiv
  collapse) as worth solving in code** (critic-gated passes for volart,
  distinct-n-optimal selection for niwatori); we solved it by hand-writing
  responses rather than automating a fix — worth flagging since niwatori's
  Blind-B score (≈0.59) is notably higher than ours (0.49) and both
  competitors have a coded, scalable response stage while ours doesn't.
- **niwatori is the only one of the three with a bit-exact reproducibility
  verification script and a pinned dependency lockfile** — a gap worth
  considering for our own repo if organizer-facing validation matters here,
  though this was not requested and no code changes have been made based on
  this observation.
