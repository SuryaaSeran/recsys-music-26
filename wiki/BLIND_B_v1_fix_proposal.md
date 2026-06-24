# Blind B v1 Fix Proposal

Targets the top 5 fixable issues from the ranking audit. Ordered by expected impact and implementation cost.

---

## Fix 1: Tighten `boost_named_track` (addresses P2, P3, P5)

**Problem:** Users name a specific track+artist and it either ranks #2 (Versace Remix) or is absent from top-20 (Fallen, Czar Refaeli). The existing `boost_named_track` logic (lines 1992-2014 of `run_inference_fusion_recall_expansion.py`) requires both track_name and artist_name to appear as word-boundary matches in user text. This fails when:
- The user names only the track, not the artist (common in later turns where artist context is implicit).
- The track name is <4 chars (filtered out by min-length guard).
- The match text normalization strips characters that matter (e.g., accented characters, parenthetical remix tags).

**Fix:**
1. **Relax to track-only matching when artist is already in conversation context.** If the artist appeared in any prior turn (played history or user messages), allow a track-title-only match to trigger the boost. This handles "play Fallen" when Sarah McLachlan is already the conversation topic.
2. **Match against parenthetical variants.** Normalize "(Remix)" / "(feat. X)" / "(Remastered)" as optional suffixes. "Versace Remix" should match "Versace (Remix)".
3. **Expand search window from top-400 to full candidate pool for title matches.** The current top-400 cutoff means if the correct track scores low in LTR (because the model over-indexed on the wrong artist cluster), the boost never sees it. For explicit title requests, scan the full pool. This is O(N) string matching on ~5000 candidates, negligible cost.
4. **If the named track is not in the candidate pool at all, log it.** Track corpus coverage gaps (like Czar Refaeli) should be surfaced, not silently ignored.

**Expected impact:** Fixes P3 (Versace Remix at rank 2 instead of 1) directly. Partially addresses P2 (if "Fallen" is in the corpus but scored low) and P5 (if "Czar Refaeli" is in the corpus).

**Cost:** ~30 lines changed in `boost_named_track`. No model retrain. No new features.

---

## Fix 2: Enforce `max_per_artist` and lower the cap (addresses P1, P6, P7, P8, P10)

**Problem:** Sessions show 9 Kodak Black, 17 Kalkbrenner, 19 Gang Starr, 14 nature-sound tracks in the top 20. The `max_per_artist` flag exists (lines 2016-2042) but is either set too high or disabled for the Blind B run.

**Fix:**
1. **Set `max_per_artist=3` as default for all runs.** The audit shows that even "good" sessions (Parov Stelar, Myrkur, Kyle Dixon) would still work with 3 per artist since the user is exploring one artist's catalog and 3 tracks gives enough coverage.
2. **For cold-start sessions (no listening history), lower to `max_per_artist=2`.** Cold-start flooding is the worst failure mode. 2 per artist forces diversity.
3. **Apply the cap to "effective artist" clusters, not just exact artist_id.** "Paul Kalkbrenner" and "Fritz Kalkbrenner" are different artist_ids but the same sonic cluster. Similarly, "Nature Sounds," "FX Makers," "Outside Broadcast Recordings," and "Wp Sounds" are all rain recordings from different pseudo-artists. Group by artist family where possible (same last name, or same semantic cluster in embedding space with cosine > 0.95).

**Expected impact:** Directly fixes the flooding in P1, P6, P7, P8, P10. Forces the ranker to surface diverse candidates that currently get buried.

**Cost:** Fix (1) and (2) are config changes. Fix (3) requires a small artist-clustering step (precomputable offline). No model retrain.

---

## Fix 3: Hard-suppress previously rejected artists on explicit pivot (addresses P6, P9, P10, P11)

**Problem:** The pipeline has rejection-aware LTR features (`sim_to_neg_hist_mean`, `artist_in_rejected_set`, `n_rejected_in_history`, `consecutive_rejections_tail`) but the LGBMRanker model does not weight them aggressively enough to overcome the strong positive signal from conversation context. When the user says "stop giving me Gang Starr," the word "Gang Starr" appears many times in context, creating a strong retrieval signal that the rejection features cannot overcome.

**Fix:**
1. **Post-LTR hard filter: if the user explicitly names an artist with a rejection verb in the current turn, drop that artist from the candidate pool entirely.** Detection: look for patterns like "[artist] + {stop, no more, break away, tired of, don't want, not X}" in the latest user message. This is a simple regex/keyword check on extracted entities.
2. **When `consecutive_rejections_tail >= 3`, apply a score penalty multiplier (e.g., 0.5x) to the dominant artist in the history.** This is softer than the hard filter but catches cases where the user has not explicitly named the artist to reject but has rejected its tracks three times running.
3. **Exclude rejected artists from the artist-expansion source list.** Currently `artist_expansion` (lines ~1100-1140) adds tracks from all artists mentioned in conversation. If an artist was in a rejected turn, skip it in expansion. This prevents the rejected artist from even entering the candidate pool via this path.

**Expected impact:** Directly fixes P6, P9, P10, P11. These are the most user-hostile failures in the audit (user says "stop" and gets more of the same thing).

**Cost:** Fix (1) is ~20 lines of post-LTR filtering. Fix (2) is a config-tunable multiplier. Fix (3) is a small change to the artist-expansion loop. No model retrain.

---

## Fix 4: Suppress nature/rain/meditation audio unless explicitly requested (addresses P8)

**Problem:** Session `60f60edd` is the most extreme failure: 14 of 20 slots filled with rain/thunder/nature recordings when the user asked for electronic ambient. These tracks come from pseudo-artists ("Nature Sounds," "FX Makers," "Outside Broadcast Recordings") that share the tag "ambient" or "relaxing" with legitimate electronic ambient music.

**Fix:**
1. **Tag-based category blocklist.** Create a small blocklist of artist_ids or a tag pattern (e.g., tracks where ALL tags are from {"New Age", "relaxing", "nature", "rain", "meditation", "sleep sounds"} and the artist name contains "Sounds", "Recordings", "FX"). Suppress these from candidate pool unless the user query contains explicit nature/rain keywords.
2. **Alternatively, use the CLAP audio embedding.** Nature sounds and electronic ambient have very different audio profiles. If CLAP similarity to a known rain recording is > 0.9, flag the track and require explicit user intent to include it.

**Expected impact:** Fixes P8 completely. Low risk of false positives since the blocklist targets non-music audio.

**Cost:** ~15 lines. A small curated list of ~20 pseudo-artist IDs or a tag regex. No model retrain.

---

## Fix 5: Decay conversation-context weight as rejection count increases (addresses all artist-lock issues)

**Problem:** The BM25 and semantic queries both incorporate the full dialogue history. When the user has been discussing one artist for 5 turns (even to reject it), that artist's name dominates the query text. The retrieval stage pulls in that artist heavily, and even if the LTR model demotes it, the candidate pool is already contaminated.

**Fix:**
1. **Weight recent turns higher, rejected turns lower in BM25 query construction.** Currently all last-N dialogue turns are concatenated equally. Instead, omit or downweight turns where the system recommendation was rejected. If turn t-2 was rejected, exclude its system response (which contains the rejected track names) from the BM25 query.
2. **For the semantic query, use only the latest user message + goal when `consecutive_rejections_tail >= 2`.** This resets the semantic query to the user's current intent rather than the accumulated conversation topic.
3. **In entity extraction, distinguish "mentioned to request" vs "mentioned to reject."** An artist name following "not," "stop," "no more," "break away from" should be excluded from the entity BM25 query, not included.

**Expected impact:** Reduces the root cause of artist lock-in at the retrieval stage, before LTR even runs. Complements Fix 3 (which operates post-LTR).

**Cost:** Medium. Requires changes to query construction logic (~50 lines). No model retrain, but may shift score distributions enough to warrant re-tuning LTR weights on the validation set.

---

## What This Does NOT Fix

| Issue | Why not addressed |
|-------|-------------------|
| Exact lyric search (P4: "Your name is a strong and mighty tower") | Requires lyric-level indexing, a new data source. Not a quick fix. |
| Tempo/key matching (sessions `46e8aa14`, `bf27c872`) | Requires audio-analysis metadata not currently in the pipeline. Out of scope. |
| Cover art / visual attribute queries (P1 partial) | Requires album art embeddings or a visual metadata index. Separate project. |
| Cold-start genre bias (P12: worship songs for "happy music") | Requires popularity-aware genre diversification at retrieval. Fixable but needs careful tuning to avoid homogenizing results. Defer to next iteration. |

---

## Implementation Order

| Priority | Fix | Effort | Impact |
|----------|-----|--------|--------|
| 1 | Fix 2: Lower `max_per_artist` cap | Config change | Fixes 5+ sessions |
| 2 | Fix 4: Nature-sound blocklist | 15 lines | Fixes P8 completely |
| 3 | Fix 1: Tighten `boost_named_track` | 30 lines | Fixes P3, partially P2/P5 |
| 4 | Fix 3: Hard-suppress rejected artists | 20 lines | Fixes P6, P9, P10, P11 |
| 5 | Fix 5: Decay context on rejections | 50 lines | Root-cause fix for artist lock-in |

Fixes 1-4 are zero-retrain, testable on the existing Blind B sessions within a day. Fix 5 needs validation-set re-tuning but no new model training.
