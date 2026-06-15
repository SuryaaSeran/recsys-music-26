# Feature + Metadata Audit (2026-06-07, post-v17)

Canonical reference for what features the live ranker uses, what data they depend on, and what's left on the table. Run this through before designing the next push.

## Cache freshness — VERIFIED CURRENT

Forced fresh pull of all three HF datasets after clearing `~/.cache/huggingface/datasets/talkpl-ai*` (caches were last touched 2026-05-01, ~5 weeks old).

| Dataset | Rows | Schema |
|---|---|---|
| `TalkPlayData-Challenge-Track-Metadata` | all_tracks=47,071 / test_tracks=7,405 | 11 fields (see below) |
| `TalkPlayData-Challenge-Dataset` | train=15,199 / test=1,000 | session_id, user_id, session_date, user_profile, conversation_goal, conversations, goal_progress_assessments |
| `TalkPlayData-Challenge-Blind-A` | test=80 | same as Dataset |

**Drift check**: track_ids in every cache (BM25, TT v8c/v8d, Qwen3 metas, CLAP, CF-BPR, co-occurrence) match the fresh `all_tracks` set exactly (47,071 / 47,071). `test_tracks` is a strict subset of `all_tracks` — no separate set to index.

**Conclusion**: no rebuild required. All cached artifacts are consistent with the live dataset.

## Track metadata schema (11 fields)

```
['track_id', 'ISRC', 'track_name', 'artist_name', 'album_name', 'tag_list',
 'popularity', 'release_date', 'duration', 'artist_id', 'album_id']
```

### Used by the pipeline (6 fields)

| Field | Type | Consumed by |
|---|---|---|
| track_id | str | primary key everywhere |
| track_name | list[str] | BM25 (3× weight), `get_track_text`, response template |
| artist_name | list[str] | BM25 (2× weight), artist expansion, within-artist ranking, rejection-set matching |
| album_name | list[str] | BM25 (1×), `get_track_text` |
| tag_list | list[str] | BM25 (top-20 tags 1×), `get_track_text`, `tag_overlap_count`, `query_track_tag_sim` |
| popularity | float | `popularity` feature, `popularity_pctile`, `within_artist_pop_rank`, artist-catalog ranking |
| release_date | str (YYYY-MM-DD) | `track_year`, `years_since_release`, BM25 (year extracted) |

### NOT used — opportunities

| Field | Coverage | Cardinality | Feature idea |
|---|---|---|---|
| **duration** | 47,071/47,071 (100%) | int ms, median 227s, p95 384s | `duration_sec`, `duration_z`, `is_short_track` — weak signal alone but may improve goal-style match (e.g. workout goals → upbeat ≈ shorter tracks; "background" goals → instrumentals tend longer) |
| **album_id** | list, 77,933 strings → 30,874 unique albums | mean 2.5 tracks/album | `same_album_as_last_track` (binary), `same_album_as_any_history` — strong signal for sequential-listening goals; could deliver clean lift |
| **artist_id** | list, 58,544 → 11,138 unique artists | +2,163 more unique than `artist_name` strings (8,975) | Replace `artist_name`-string matching everywhere with `artist_id`-based: cleaner rejection-set, cleaner within-artist grouping, cleaner artist_expansion (currently collapses different artists with same name) |
| **ISRC** | 45,737/47,071 (97%) | list, 45,736 unique | Cross-catalog dedup only; not a per-track feature on its own |

## The 50 features actually scored (LTR v8d booster, ranked by gain)

Gains and split counts from `lgb.Booster(model_file='models/ltr/ltr_v8d_nl31_lr0p08.txt').feature_importance()`.

| Rank | Feature | Gain | Splits | Source |
|---:|---|---:|---:|---|
| 1 | n_sources | 1,057,158 | 129 | pool-union count |
| 2 | cf_cos | 122,698 | 451 | `cache/cf_bpr` |
| 3 | n_sources_norm | 79,589 | 110 | derived |
| 4 | log1p_n_sources | 51,833 | 9 | derived |
| 5 | tt_cos | 47,003 | 287 | `cache/twotower_v8d` |
| 6 | tt_rank_sig | 41,783 | 208 | TT pool rank |
| 7 | artist_sig | 32,202 | 211 | artist-expansion rank |
| 8 | bm25_signal | 31,710 | 195 | `cache/bm25` |
| 9 | popularity_pctile | 27,782 | 228 | metadata.popularity |
| 10 | nn_sig | 20,703 | 137 | TT last-track NN |
| 11 | within_artist_pop_rank | 19,861 | 257 | metadata.popularity grouped by `artist_name` (lowercased) |
| 12 | dist_to_last | 19,547 | 136 | TT cosine |
| 13 | popularity | 16,603 | 147 | metadata.popularity raw |
| 14 | dist_to_recent_mean | 14,577 | 158 | TT cosine |
| 15 | qm_rank_sig | 12,305 | 162 | `cache/qwen3_meta` |
| 16 | cf_dist_to_recent_mean | 10,442 | 133 | `cache/cf_bpr` |
| 17 | pool_size | 9,671 | 178 | len(cands) |
| 18 | query_track_tag_sim | 9,268 | 170 | tag overlap with latest user msg |
| 19 | cf_dist_to_last | 5,669 | 102 | `cache/cf_bpr` |
| 20 | collab_rank_sig | 5,650 | 66 | `cache/cooccur/leakfree_6k` |
| 21 | ql_cos | 5,579 | 169 | `cache/qwen3_lyrics` |
| 22 | mean_nn_rank_sig | 5,450 | 79 | session-mean NN |
| 23 | tag_overlap_count | 4,475 | 123 | tag overlap with BM25 query |
| 24 | turn_number | 4,451 | 87 | session position |
| 25 | collab_score | 4,442 | 75 | cooccur weight |
| 26 | qm_cos | 4,361 | 104 | qwen-meta cosine |
| 27 | track_year | 3,049 | 94 | release_date |
| 28 | turns_toward_goal | 2,285 | 63 | gpa (re-keyed) |
| 29 | clap_cos | 1,900 | 81 | `cache/clap` |
| 30 | years_since_release | 1,886 | 59 | 2026 − track_year |
| 31 | nn_source_count | 1,803 | 27 | how many history tracks NN'd this candidate |
| 32 | collab_origin | 1,453 | 11 | cooccur binary |
| 33 | collab_source_count | 1,452 | 34 | cooccur sources |
| 34 | within_artist_trans_rank | 1,293 | 38 | per-artist transition rank |
| 35 | query_len_tokens | 1,284 | 48 | |
| 36 | consecutive_rejections_tail | 1,011 | 28 | gpa (re-keyed) |
| 37 | mean_nn_origin | 896 | 13 | |
| 38 | artist_origin | 630 | 8 | |
| 39 | bm25_top1 | 625 | 16 | |
| 40 | goal_category | 456 | 18 | one-hot of A-K |
| 41 | history_len | 452 | 10 | |
| 42 | tt_origin | 417 | 7 | |
| 43 | user_has_followup | 227 | 7 | regex |
| 44 | user_has_negation | 79 | 4 | regex |
| 45 | qm_origin | 47 | 2 | |
| 46 | bm25_origin | 19 | 1 | |
| **47** | **sim_to_pos_hist_mean** | **0** | **0** | requires `--use_goal_progress` (off) |
| **48** | **sim_to_neg_hist_mean** | **0** | **0** | requires `--use_goal_progress` (off) |
| **49** | **artist_in_rejected_set** | **0** | **0** | requires `--use_goal_progress` (off) |
| **50** | **n_rejected_in_history** | **0** | **0** | requires `--use_goal_progress` (off) |

## Headline insights

1. **n_sources dominates by 8.6×** (gain 1,057k vs #2 cf_cos at 123k). The strongest single signal is "how many retrieval sources agreed on this candidate." Every move that improves source diversity directly helps the top feature.

2. **cf_cos at #2** is surprisingly strong (gain 123k, 451 splits — most splits of any feature). Collaborative-filtering similarity beats every dense-text encoder feature. CF-BPR is doing heavy lifting.

3. **4 features are pure dead weight in current booster**: the H2 history features (`sim_to_pos_hist_mean`, `sim_to_neg_hist_mean`, `artist_in_rejected_set`, `n_rejected_in_history`). They're always zero because `--use_goal_progress` isn't enabled at inference. **Activating that flag unlocks ~+0.04 retrieval lift** based on v10's v8b+H1H3 result (0.30 → 0.37 on blind A).

4. **TT signals are middle-of-pack** (tt_cos #5, tt_rank_sig #6). After the v8d encoder upgrade, the model is leaning more on multi-source agreement than any single dense vector.

5. **Goal_category (one-hot of A-K) is near-bottom** at gain 456 — the goal-text content (consumed by TT encoder) carries more signal than the categorical bucket.

6. **artist_name string-matching is leaving signal on the table.** Current `within_artist_pop_rank` (gain 19,861, #11) and `artist_origin` group tracks by lowercased `artist_name` string. The fresh metadata shows `artist_id` has 11,138 unique values vs 8,975 unique names — meaning ~2,163 "different artists with same name" are being collapsed today. Switching to `artist_id`-based grouping is a structural cleanup.

## Cache → feature dependency matrix

| Feature group | Cache | Rebuild trigger |
|---|---|---|
| tt_cos, tt_rank_sig, tt_origin, nn_sig, dist_to_last, dist_to_recent_mean, mean_nn_rank_sig, sim_to_pos/neg_hist_mean | `cache/twotower_v8d/track_embeddings.npy` | new TT model OR catalog drift |
| qm_cos, qm_rank_sig, qm_origin | `cache/qwen3_meta/` | catalog drift |
| ql_cos | `cache/qwen3_lyrics/` | catalog drift |
| clap_cos | `cache/clap/` | catalog drift |
| cf_cos, cf_dist_to_last, cf_dist_to_recent_mean | `cache/cf_bpr/` + `cache/user_cf_bpr.json` | session changes |
| bm25_signal, bm25_origin, bm25_top1 | `cache/bm25/track_metadata/` | metadata changes |
| artist_sig, artist_origin, within_artist_* | metadata.artist_name + popularity | metadata changes |
| collab_*, dist_to_* (recent mean) | `cache/cooccur/next_song_leakfree_6k_excluded.npz` | sessions changes |
| popularity, popularity_pctile, track_year, years_since_release, tag_overlap_count, query_track_tag_sim | live metadata read | catalog changes |
| turn_number, history_len, goal_category, query_len_tokens, n_sources*, log1p_n_sources, turns_toward_goal, consecutive_rejections_tail | session JSON + gpa | per-turn |

## Recommendations for next iteration

Ranked by expected nDCG impact vs effort:

1. **Enable `--use_goal_progress` at inference** (turns on H1+H3 + activates the 4 zero-importance H2 features). Zero code change — flag flip. Re-dump LTR with the flag to give the H2 features non-zero values during training. **Estimated lift on blind A: +0.04** (v8b → v8b+H1H3 was +0.07).

2. **Replace `artist_name`-string matching with `artist_id`-list matching** in:
   - `within_artist_*` grouping (currently uses lowercased name)
   - rejection-set tracking
   - artist expansion pool
   Adds 2,163 properly-disambiguated artists. Small code change. **Estimated lift: +0.005-0.01** (cleaner signal on artist-collision edge cases).

3. **Add `album_id`-based features**: `same_album_as_last_history`, `n_history_in_same_album`. Strong intuition fit for sequential-album sessions. **Estimated lift: +0.005-0.015**.

4. **Add `duration` feature(s)**: weak signal, but free. **Estimated lift: +0.002-0.005**.

5. **Consider pruning the 4 zero-importance H2 features** from `FEATURE_COLS` if we don't enable `--use_goal_progress` — they're wasting LGBM training time on dead columns. Lower priority since they cost nothing once trained.

## Files referenced

- `scripts/inference/run_inference_fusion_recall_expansion.py:476-516` — `FEATURE_COLS` list
- `scripts/inference/run_inference_fusion_recall_expansion.py:1219-1272` — `feat[i] = (...)` tuple
- `scripts/train/build_bm25_v2.py` — BM25 field weighting
- `scripts/train/build_cooccur_table.py` — co-occurrence
- `models/ltr/ltr_v8d_nl31_lr0p08.txt` — current best booster
