# Training and Evaluation Data

## Datasets (all from HuggingFace, competition-provided)

| Dataset | What it contains |
|---|---|
| `talkpl-ai/TalkPlayData-Challenge-Dataset` | Conversation sessions: train split + test split (dev) + blind set |
| `talkpl-ai/TalkPlayData-Challenge-Track-Metadata` | Track catalog: name, artist, album, tags, release date, popularity |
| `talkpl-ai/TalkPlayData-Challenge-Track-Embeddings` | Pre-computed track vectors: CF-BPR, CLAP audio, Qwen3 metadata, Qwen3 attributes, Qwen3 lyrics |
| `talkpl-ai/TalkPlayData-Challenge-User-Embeddings` | Pre-computed user CF-BPR vectors (train users + warm/cold test users) |
| `talkpl-ai/TalkPlayData-Challenge-Blind-A` | Blind set A: sessions without gold labels, used for leaderboard submission |

---

## Training data (Two-Tower model)

The TT model is the only component we train. Everything else (BM25, Qwen, CLAP, CF)
uses pre-built indexes or pre-computed embeddings from the competition dataset.

### Source

The `train` split of `TalkPlayData-Challenge-Dataset`. This is a set of multi-turn
conversations where each `[music]` turn carries a gold track ID -- the track the system
was supposed to recommend.

### How examples are built (v6, current)

Script: `scripts/train/build_twotower_v6_data.py`

Each `[music]` turn becomes one training example with three parts:

**Anchor (query)**

Concatenation of:
- Latest user message
- Goal: `<listener_goal>`
- Type: `<goal_category> <goal_specificity>`
- `<preferred_musical_culture>`
- `<age_group> <country>`
- Last 2 played tracks as "Name Artist"

Example:
```
Yes, that's a great one! Arctic Monkeys always deliver. What other highly popular alternative rock tracks...
Goal: play one specific song that is known for its high popularity within its genre or era.
Type: specific_track discovery
Anglo-American Rock
25-34 United Kingdom
Heart-Shaped Box Nirvana Fluorescent Adolescent Arctic Monkeys
```

**Positive (gold track text)**

```
<track_name> by <artist_name> | Album: <album_name> | Tags: <tag1> <tag2> ... | <year>
```

Example:
```
D Is For Dangerous by Arctic Monkeys | Album: Favourite Worst Nightmare | Tags: indie rock alternative 2000s | 2007
```

**Negatives (5 per example, mixed)**

| Type | Count | How selected |
|---|---|---|
| BM25 hard | up to 2 | Top-100 BM25 results for the anchor, excluding gold and already-played |
| Rejected-track | up to 1 | Random track from prior turns rated `DOES_NOT_MOVE_TOWARD_GOAL` |
| Random | fill to 5 | Randomly sampled from full catalog, excluding seen tracks |

**Positive weighting**

Each turn's gold track has a `goal_progress_assessment` label:
- `MOVES_TOWARD_GOAL` -> weight 1.0 (kept)
- `None` (unlabelled) -> weight 1.0 (kept)
- `DOES_NOT_MOVE_TOWARD_GOAL` -> weight 0.4 (60% randomly dropped before training)

MNRL does not support per-sample weights, so we approximate by probabilistic dropping.

> **CORRECTION (2026-06-05): this weighting is off by one turn.** `goal_progress_assessment`
> at turn T is the listener's verdict on the recommendation made at turn **T-1**, not turn T
> (action at t, feedback recorded at t+1). So the weight for the turn-T gold should come from
> `gpa_{T+1}`, not `gpa_T`. As written above we weighted each gold by the label that judges
> the *previous* turn's track. gpa resolves only for R_1..R_7 (R_8 has no gpa_9); turn 1's
> gpa is `None` = no prior rec. gpa is also goal-relative (progress toward the session goal),
> not turn-level satisfaction. See `plan/PLAN.md` "Data correction" for the full note.

### Train/valid split

95% of sessions -> train, 5% -> validation. Split is at session level so no turn
leakage between splits.

### Loss function

`MultipleNegativesRankingLoss` (MNRL) from `sentence-transformers`. In-batch negatives
are added on top of the explicit hard/random negatives in each row.

### Base model

`sentence-transformers/all-MiniLM-L6-v2`, fine-tuned for 1 epoch.
Current production model: `models/twotower_v6/final`.

---

## Evaluation data

### Dev evaluation (local, primary loop)

**Source**: `test` split of `TalkPlayData-Challenge-Dataset`.

This is 1000 sessions (8000 turns). Gold labels are present -- each `[music]` turn
has the correct track ID. We evaluate against this after every meaningful change.

**Metric**: nDCG@20 (primary), also nDCG@1, nDCG@10, Hit@20.

With only one relevant item per turn, nDCG@20 simplifies to:

```
nDCG@20 = 1 / log2(rank + 1)   if gold in top 20
         = 0                    otherwise
```

Script: `scripts/inference/evaluate_local.py`

```bash
python scripts/inference/evaluate_local.py --pred exp/inference/devset/<run>.json
```

**Rule**: dev nDCG@20 must be a full-1000-session run to count as a new best.
Faster iteration runs (200 sessions, `--sessions 200`) are used for direction
signals only.

### Blind A (leaderboard)

**Source**: `talkpl-ai/TalkPlayData-Challenge-Blind-A`. No gold labels.
Submitted to the competition leaderboard. Used only to confirm that dev gains
transfer to the official metric.

Script: `scripts/inference/run_inference_blind_fusion.py`

### What each eval surface tells you

| Surface | Sessions | Gold labels | Speed | Use |
|---|---|---|---|---|
| Dev 200-session | 200 | Yes | ~8 min | Direction signal during sweeps |
| Dev 1000-session | 1000 | Yes | ~40 min | Confirm a new best; update `CURRENT_BEST_ITERATION.md` |
| Blind A | full blind set | No | ~40 min | Leaderboard confirmation; run sparingly |

---

## What we do NOT train

| Component | Why not trained |
|---|---|
| BM25 | Lexical index; no parameters |
| Qwen3-Embedding | Pre-computed track vectors from dataset; query embedding runs frozen Qwen3-Embedding-0.6B |
| CLAP | Pre-computed audio track vectors from dataset; query encoding runs frozen LAION CLAP |
| CF-BPR | Pre-computed from dataset |
| Fusion weights | Grid-searched manually (no gradient) |
