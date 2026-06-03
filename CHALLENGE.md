# TalkPlayData Challenge: Music Conversational Recommendation

Authoritative spec for the new-from-scratch approach. Everything below is verified
against the actual HuggingFace datasets and the official evaluator schema.

## What the challenge is

ACM RecSys 2026 Music CRS (Conversational Recommendation System), the
TalkPlayData Challenge. A user and an assistant hold a multi-turn conversation
about music. At each `music` turn the system must recommend the next track to
play. The system also generates a natural-language response explaining the pick.

It is two jobs in one:
1. **Retrieval / ranking**: pick the right track id from a 47,071-track catalog.
2. **Response generation**: write the assistant text (only scored on the blind set).

## Goal

For every `music` turn in a session, output an ordered list of up to 20 candidate
`track_id`s (best first), plus a response string. Maximise ranking quality of the
single ground-truth track and, on the blind set, response quality.

## Evaluation metrics

| Metric | Scope | How computed | Role |
|---|---|---|---|
| **nDCG@20** | dev + blind | Standard nDCG vs the single gold track per turn, macro-averaged over turn positions then sessions. With one relevant item: `1/log2(rank+1)` if gold in top 20 else 0. | **Primary** |
| Hit@20 | dev | gold appears in top 20 | secondary |
| Catalog Diversity | both | unique recommended track ids / 47071 | complementary |
| Lexical Diversity | both | distinct-2 over all `predicted_response` strings | complementary |
| **LLM-as-Judge** | **blind only** | Gemini judges response on Personalization + Explanation Quality. Prompt undisclosed. | blind response quality |

Official composite (blind) is dominated by the LLM judge. Dev has no judge, so the
local loop optimizes nDCG@20 only.

Evaluator: <https://github.com/nlp4musa/music-crs-evaluator.git>

## Datasets (all on HuggingFace, already cached locally)

Cache root: `~/.cache/huggingface/hub/`

| Dataset | Splits | Use |
|---|---|---|
| `talkpl-ai/TalkPlayData-Challenge-Dataset` | `train` (sessions w/ gold), `test` (1000 dev sessions, 8000 turns, gold present) | train + local eval |
| `talkpl-ai/TalkPlayData-Challenge-Blind-A` | `test` (80 sessions, no gold) | leaderboard |
| `talkpl-ai/TalkPlayData-Challenge-Track-Metadata` | `all_tracks` (47071), `test_tracks` | catalog |
| `talkpl-ai/TalkPlayData-Challenge-Track-Embeddings` | `all_tracks`, `test_tracks` | precomputed track vectors |
| `talkpl-ai/TalkPlayData-Challenge-User-Embeddings` | train + warm/cold test users | precomputed user CF vectors |

## INPUT: a session

One row per session. Columns (verified):

```
session_id                 str    e.g. "ba3da7b0-1e81-4d2a-90fa-65ee1f4d7348"
user_id                    str
session_date               str    "2020-01-18"
user_profile               dict   see below
conversation_goal          dict   see below
conversations              list[dict]   the turns, see below
goal_progress_assessments  list[dict]   per-turn label (train/dev only meaningful)
```

`user_profile`:
```json
{
  "age": 36, "age_group": "30s",
  "country_code": "MX", "country_name": "Mexico",
  "gender": "male", "preferred_language": "English",
  "preferred_musical_culture": "Anglo-American Rock",
  "user_id": "...", "user_split": "test_warm"   // test_warm | test_cold | train
}
```

`conversation_goal`:
```json
{
  "category": "J",
  "listener_goal": "play one specific song that is known for its high popularity within its genre or era.",
  "specificity": "HH"
}
```

`conversations` is an ordered list of turn dicts. Each turn:
```json
{ "role": "user" | "music" | "assistant",
  "content": "...",
  "thought": "...",
  "turn_number": 1 }
```
- `role: "user"`   -> user's message text.
- `role: "music"`  -> `content` is the **gold track_id** played that turn. This is
  the prediction target. `thought` explains why (present in train/dev, hidden in blind).
- `role: "assistant"` -> assistant's natural-language response text.

A turn_number groups a (user -> music -> assistant) triple. Dev sessions run
turn_number 1..8. The model predicts the `music` content for each turn given all
prior turns.

`goal_progress_assessments`: list of `{turn_number, goal_progress_assessment}` where
the label is `MOVES_TOWARD_GOAL` | `DOES_NOT_MOVE_TOWARD_GOAL` | `None`.

### Blind A specifics (IMPORTANT: predict the LAST turn only)

Same schema, but the conversation is **truncated at a `user` message** and you
complete it. Verified across all 80 sessions:

- Every session ends on a `user` turn. You predict **exactly one** `music` turn per
  session: the next track after the last user message. **80 predictions total**, not
  8-per-session like dev.
- History length **varies per session**. The terminating `turn_number` ranges 1..8
  (counts: t1=20, t2=15, t3=10, t4=5, t5=8, t6=9, t7=8, t8=5). So the number of
  already-completed music turns = `last_turn_number - 1` (0 to 7 prior gold tracks
  of context).
- No gold ids, no assistant text to learn from. The `predicted_response` **must** be
  filled (LLM-judged).

Consequence for modeling/selection: dev macro-averages over all 8 turn positions,
but blind only ever tests one **terminal** turn per session, with a known
position mix skewed toward early turns (35/80 end at t1-t2). Weight dev analysis by
that distribution rather than treating all positions equally.

## CATALOG: track metadata (`all_tracks`, 47071 rows)

```
track_id      str
ISRC          array[str]
track_name    array[str]
artist_name   array[str]
album_name    array[str]
tag_list      array[str]   free-form tags
popularity    float
release_date  str          "2006-12-06"
duration      int          ms
artist_id     array[str]
album_id      array[str]
```
(name/artist/album/ids come wrapped in 1-element arrays.)

## PRECOMPUTED TRACK EMBEDDINGS

Source: <https://huggingface.co/datasets/talkpl-ai/TalkPlayData-Challenge-Track-Embeddings>
Parquet `all_tracks` (sharded 4 files) + `test_tracks`. One row per track_id with
six **separate** modality vectors. **Keep them separate, do not concatenate** — each
is its own space; let the model decide how to combine them.

| Column | Dim | Source | Present / 47071 |
|---|---:|---|---:|
| `audio-laion_clap` | 512 | LAION CLAP audio | 46579 |
| `image-siglip2` | 768 | SigLIP2 cover image | 46485 |
| `cf-bpr` | 128 | collaborative filtering BPR | 46455 |
| `attributes-qwen3_embedding_0.6b` | 1024 | Qwen3 over attributes | 46579 |
| `lyrics-qwen3_embedding_0.6b` | 1024 | Qwen3 over lyrics | 46579 |
| `metadata-qwen3_embedding_0.6b` | 1024 | Qwen3 over metadata | 46579 |

Every track_id has a row, but individual modalities have gaps (empty array). Treat a
missing modality as absent (zero-fill + presence mask), not zero-signal. User
embeddings dataset provides matching cf-bpr user vectors for warm users; cold users
have none.

Parser: `src/tracks.py` (`load_catalog`, `load_track_embeddings`). The loader returns
per-modality matrices aligned to one track_id list plus a presence mask per modality.

## OUTPUT / SUBMISSION

A JSON array, one entry per predicted music turn:

```json
[
  {
    "session_id":          "69137__2020-02-08",
    "user_id":             "69137",
    "turn_number":         1,
    "predicted_track_ids": ["id1", "id2", "..."],
    "predicted_response":  "Here are some songs you might enjoy."
  }
]
```

| Field | Type | Notes |
|---|---|---|
| `session_id` | str | as provided |
| `user_id` | str | as provided |
| `turn_number` | int | the music turn predicted |
| `predicted_track_ids` | list[str] | up to 20, best first, unique, valid catalog ids |
| `predicted_response` | str | dev: `""` ok. **Blind: must be filled** (judged) |

Submission: write `prediction.json` (exact name), `zip submission.zip prediction.json`,
upload on the portal.

| Phase | Records | Submissions/day | Total |
|---|---:|---:|---:|
| Blind A | 80 | 10 | 500 |
| Blind B (final) | TBD | 1 | 10 |

## Quick load snippet

```python
import pandas as pd
H = "~/.cache/huggingface/hub"
import glob, os
def pq(ds, name):
    p = glob.glob(os.path.expanduser(f"{H}/datasets--talkpl-ai--{ds}/snapshots/*/data/{name}*.parquet"))
    return pd.concat([pd.read_parquet(x) for x in sorted(p)], ignore_index=True)

dev   = pq("TalkPlayData-Challenge-Dataset", "test")        # 1000 sessions, gold present
train = pq("TalkPlayData-Challenge-Dataset", "train")
blind = pq("TalkPlayData-Challenge-Blind-A", "test")        # 80 sessions, no gold
meta  = pq("TalkPlayData-Challenge-Track-Metadata", "all_tracks")   # 47071 tracks
emb   = pq("TalkPlayData-Challenge-Track-Embeddings", "all_tracks") # 6 modality vectors
```
