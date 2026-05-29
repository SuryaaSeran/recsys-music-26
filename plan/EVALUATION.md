# Evaluation Reference

Canonical reference for the metrics, prediction JSON schema, and submission
rules. Pulled from the official evaluator and the competition page; keep in
sync if anything upstream changes.

## Evaluator code

- Official upstream: <https://github.com/nlp4musa/music-crs-evaluator.git>
- Mirrored locally at `music-crs-evaluator/` (gitignored, cloned alongside
  this repo). Run with `python music-crs-evaluator/evaluate_devset.py
  --pred exp/inference/<dataset>/<tid>.json`.
- Our local mirror: `scripts/inference/evaluate_local.py`. Same per-turn
  nDCG / catalog div / lex div numbers; faster to invoke.

## Metrics

| Dimension | What it measures | How it is computed | Role |
|---|---|---|---|
| **nDCG@20** | Ranking quality of the recommended tracks | Standard nDCG against the single ground-truth track per turn, macro-averaged per turn-position then across positions | **Primary** recommendation metric |
| **Catalog Diversity** | How broadly the system covers the music catalog | Unique recommended track ids across all predictions ÷ catalog size (47,071) | Complementary diversity indicator |
| **Lexical Diversity** | How varied the generated language is | Distinct-2 (unique bigrams ÷ total bigrams across all `predicted_response` strings) | Complementary response-generation indicator |
| **LLM-as-Judge** | Quality of generated explanation | Blind-set responses judged by a Gemini model on Personalization and Explanation Quality. Prompt undisclosed by organisers. | **Blind-set only**, response-quality evaluation |

The judge is purely text-only and independent of ranking — a good response
can earn judge points even when `predicted_track_ids` is unchanged.

## Required prediction JSON schema

Save under `exp/inference/<dataset>/<tid>.json` (already the convention).
The competition wants the file inside the submission zip named exactly
`prediction.json`.

```json
[
  {
    "session_id":           "69137__2020-02-08",
    "user_id":              "69137",
    "turn_number":          1,
    "predicted_track_ids":  ["...", "...", "...", "..."],
    "predicted_response":   "Here are some songs you might enjoy."
  }
]
```

| Field | Type | Required | Notes |
|---|---|---|---|
| `session_id` | string | yes | `{user_id}__{date}` for dev, opaque UUID for Blind A |
| `user_id` | string | yes | as provided by the dataset |
| `turn_number` | int | yes | 1-8; the music turn being predicted |
| `predicted_track_ids` | list[string] | yes | up to 20, ordered most-relevant first, unique within an entry, must be valid catalog ids |
| `predicted_response` | string | yes | dev: `""` is acceptable; **Blind: must be filled** — judged for personalisation + quality |

## Submission rules

| Phase | Records | Submissions / day | Total |
|---|---:|---:|---:|
| Blind A | 80 | 10 | 500 |
| Blind B (final) | TBD | 1 | 10 |

Submission steps:
1. Generate `prediction.json` in the required schema for the eval set.
2. `zip submission.zip prediction.json` (file must be named exactly that).
3. Upload `submission.zip` on the competition portal.

## Datasets

- Dev: `talkpl-ai/TalkPlayData-Challenge-Dataset`, split `test` (1000
  sessions, 8000 music turns).
- Blind A: `talkpl-ai/TalkPlayData-Challenge-Blind-A`, split `test` (80
  sessions, 80 predictions — one per session, predicting the next music
  turn after `conversations[-1]`).
- Catalog: `talkpl-ai/TalkPlayData-Challenge-Track-Metadata`, splits
  `all_tracks` + `test_tracks` (47,071 tracks total).
