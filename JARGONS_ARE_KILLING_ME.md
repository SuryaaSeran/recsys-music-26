# Clarification: How BM25 and TT Signals Work

## BM25

### Query construction

Built from four parts concatenated as plain text:

1. `goal` -- the session-level listener goal
2. `culture` -- the user's preferred musical culture
3. Last N played track texts (each formatted as "Track Name Artist Name", via `get_track_text`)
4. Last N user/assistant text turns

Example (turn 2 of session `ba3da7b0`):

```
play one specific song that is known for its high popularity within its genre or era.
Anglo-American Rock
Heart-Shaped Box Nirvana
Play the highly popular grunge track from the 90s, 'Heart-Shaped Box' by Nirvana.
Absolutely! Pulling up "Heart-Shaped Box" by Nirvana for you right now. Great choice!
Perfect! That was the popular song I was looking for. Now, what are some other highly popular alternative rock tracks...
```

### Signal value

BM25 retrieves `--bm25_pool` tracks and scores them. The signal per candidate:

- **In BM25 pool**: `raw_score / max_score` (normalised to [0, 1])
- **Not in BM25 pool** (entered via artist/TT/NN expansion): `bm25_missing_floor` (default 0.05)

The floor is inside `w_bm25`, not an additive constant. A non-BM25 candidate gets
`w_bm25 * 0.05` from this term, not zero.

---

## Two-Tower (TT)

### Track encoding (offline, done once)

Each track is encoded as a single string:

```
<track_name> by <artist_name> | Album: <album_name> | Tags: <tag1> <tag2> ... | <year>
```

Example:

```
Heart-Shaped Box by Nirvana | Album: In Utero | Tags: grunge alternative rock 90s | 1993
```

Fields: `track_name`, `artist_name`, `album_name`, up to 12 tags from `tag_list`, year
from `release_date`. The string is passed through the fine-tuned TT document encoder
(SentenceTransformer) with L2 normalisation and stored in `tt_embs`.

### Query encoding (per turn, online)

Built from:

1. Latest user message (cleaned)
2. `goal`
3. `culture`
4. Last N played tracks as "Name - Artist" strings

Example (same turn 2):

```
Perfect! That was the popular song I was looking for. Now, what are some other highly popular alternative rock tracks...
play one specific song that is known for its high popularity within its genre or era.
Anglo-American Rock
Heart-Shaped Box - Nirvana
```

This is encoded with the same model's query encoder (L2 normalised).

### Signal value

Dot product of the query vector against the candidate's pre-encoded track vector.
Because both are L2-normalised this equals cosine similarity, in [-1, 1].

### TT rank signal (separate feature)

For candidates that entered the pool via TT expansion, their 0-based rank in the
top-K sorted by TT cosine is also used as a feature:

```
tt_rank_sig = 1 / log2(tt_rank + 2)
```

This is 0 for candidates not in the TT pool (BM25-only or artist/NN candidates).

---

## Qwen Meta and Qwen Lyrics

### Track side (offline, pre-computed by dataset authors)

Both embeddings come from the HuggingFace dataset
`talkpl-ai/TalkPlayData-Challenge-Track-Embeddings`, encoded with
`Qwen3-Embedding-0.6B` (1024-dim). Two separate fields per track:

- `metadata-qwen3_embedding_0.6b` -- track metadata (name, artist, album, tags, etc.)
- `lyrics-qwen3_embedding_0.6b` -- track lyrics text

The exact text fed to the model is baked into the pre-computed vectors; it is not in
this repo. Vectors are loaded as-is, L2-normalised, and stored in `cache/qwen3_meta/`
and `cache/qwen3_lyrics/`.

### Query side (online, per turn)

A single query embedding is computed with the same model using an instruction prefix:

```
Instruct: Given a music listener's request, retrieve relevant music tracks
Query: <semantic_query>
```

where `semantic_query` = cleaned latest user message + goal + culture + last N played
tracks as "Name - Artist". This is the same query used for CLAP.

The instruction prefix is query-side only -- the track embeddings were encoded without
it (standard Qwen3-Embedding usage pattern).

### Signal values

The same query vector is dotted against both matrices independently:

```python
qm_all = qwen_meta_embs   @ qwen_emb   # cosine vs. metadata embedding
ql_all = qwen_lyrics_embs @ qwen_emb   # cosine vs. lyrics embedding
```

They enter the score as separate terms with independent weights `w_qwen_meta` and
`w_qwen_lyrics`.

---

## CLAP

### Track side (offline)

Audio embeddings from LAION CLAP (`HTSAT-tiny`, 512-dim), pre-computed by the dataset
authors and loaded from `cache/clap/track_embeddings.npy`. These are audio-derived
vectors -- the model processes the actual audio signal, not text. L2-normalised.

### Query side (online, per turn)

CLAP's text encoder is run on `semantic_query` (same string as Qwen, but without an
instruction prefix) using `clap_model.get_text_embedding(...)`. Manually L2-normalised.

### Signal value

Dot product of the text query vector against the track's audio embedding:

```python
clap_all = clap_embs @ clap_emb
```

This is a cross-modal similarity: text query vs. audio-derived track vector. It
captures acoustic/timbral properties that metadata text does not encode.

---

## CF (Collaborative Filtering)

### Track and user embeddings (offline)

Both come from the dataset (`talkpl-ai/TalkPlayData-Challenge-Track-Embeddings` and
`talkpl-ai/TalkPlayData-Challenge-User-Embeddings`). The model is BPR (Bayesian
Personalised Ranking), 128-dim. Track vectors are in `cache/cf_bpr/`, user vectors in
`cache/user_cf_bpr.json`. Both are L2-normalised at load time.

### Signal value

For a known user, the signal is the dot product of the user's CF vector against each
track's CF vector:

```python
cf_all = cf_track_embs @ user_emb
```

This is a pure popularity/interaction signal -- it has no query text involvement. For
cold-start users (not in the CF index), `user_emb` is `None` and the CF term is 0 for
all candidates.

CF also participates in pool expansion: the top `--cf_pool` tracks by CF score are
added to the candidate pool before scoring.

---

## Artist signal

### How artists are identified

At startup, a dict `artist_to_tids` is built from track metadata: for every track,
each artist name is lowercased and mapped to that track's ID. Each artist's list is
capped at `--artist_cap` (default 50) tracks in metadata order.

At query time, two sources of artist names are collected:

1. Text history: every user/assistant turn is scanned for verbatim artist name matches
   (longest-match-first to avoid substring collisions). Match source = `"user_text"`.
2. Play history: the `artist_name` field of each recently played track. Match source =
   `"played_track_artist"`.

All tracks from matched artists' catalogs (up to the cap) are added to the pool.

### Example (session `ba3da7b0`, turn 4)

By turn 4 the conversation looks like:

```
[user] "Play Heart-Shaped Box by Nirvana"
[music] Heart-Shaped Box (played)
[assistant] ...
[user] "What other popular alternative rock tracks do you recommend?"
[music] Fluorescent Adolescent by Arctic Monkeys (played)
[assistant] ...
[user] "Yes, great! What else from Arctic Monkeys?"
[music] D Is For Dangerous by Arctic Monkeys (played)
[assistant] ...
[user] "Another solid track. Can you recommend another highly popular alternative rock track?"
```

Artist names collected at turn 5:

- From **text history**: the user said "Arctic Monkeys" explicitly -- verbatim match
  against the catalog dict -> `mentioned["arctic monkeys"] = "user_text"`
- From **play history**: Heart-Shaped Box has `artist_name = ["Nirvana"]`, Fluorescent
  Adolescent and D Is For Dangerous both have `artist_name = ["Arctic Monkeys"]` ->
  `mentioned["nirvana"] = "played_track_artist"`, `mentioned["arctic monkeys"]` already
  set so not overwritten.

Catalog expansion:

- `artist_to_tids["arctic monkeys"]` -- up to 50 Arctic Monkeys tracks added to pool,
  ordered as they appear in the metadata. e.g. "R U Mine?" at index 0, "Do I Wanna
  Know?" at index 1, etc.
- `artist_to_tids["nirvana"]` -- up to 50 Nirvana tracks added.

Signal for "R U Mine?" by Arctic Monkeys: `artist_rank = 0`,
`artist_sig = 1 / log2(0 + 2) = 1.0` (maximum).

Signal for the 10th Arctic Monkeys track in the list: `artist_rank = 9`,
`artist_sig = 1 / log2(9 + 2) = 1 / log2(11) ≈ 0.29`.

Already-played tracks are in `seen` and skipped before being added to the pool, so
Heart-Shaped Box, Fluorescent Adolescent, and D Is For Dangerous are never candidates.

### Signal value

`artist_rank` = 0-based position of the candidate within the matched artist's catalog
list (minimum across artists if the candidate belongs to multiple matched artists).

```
artist_sig = 1 / log2(artist_rank + 2)
```

0 for candidates with no artist match. Rank 0 (first track in catalog) gives the
maximum signal of `1 / log2(2) = 1.0`.

---

## Last-track NN signal

### How neighbors are found

For each of the last `--last_nn_src` (default 2) played tracks, the TT embedding of
that source track is dotted against all track embeddings to get cosine similarities.
The top `--last_nn_k` (default 100) neighbors (excluding the source track itself) are
added to the pool. This is done in TT embedding space -- purely based on learned
two-tower similarity, not metadata text.

### Signal value

`nn_rank` = 0-based neighbor rank of the candidate (minimum across all source tracks
if it was a neighbor of more than one recent track).

```
nn_sig = 1 / log2(nn_rank + 2)
```

0 for candidates that are not a neighbor of any recent track. Rank 0 gives
`1 / log2(2) = 1.0`.

---

## Final score

```
score = w_bm25        * bm25_signal
      + w_tt          * tt_cosine
      + w_qwen_meta   * qm_cosine
      + w_qwen_lyrics * ql_cosine
      + w_clap        * clap_cosine
      + w_cf          * cf_cosine
      + w_tt_rank     * tt_rank_sig
      + w_artist      * artist_sig
      + w_nn          * nn_sig
      + w_bm25_origin * (1 if "bm25" in sources else 0)
```
