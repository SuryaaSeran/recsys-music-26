# ReccysMusicV2

Fresh-start approach to the TalkPlayData Challenge (ACM RecSys 2026 Music CRS).

Start here: **[CHALLENGE.md](CHALLENGE.md)** — task, goal, input/output/submission
schemas, metrics, dataset locations, and a verified loader snippet.

Predict the next track per `music` turn in a multi-turn music conversation, ranked
nDCG@20; on the blind set also generate the assistant response (LLM-judged).

## Layout

```
CHALLENGE.md   the spec (read first)
data/          local artifacts (gitignored)
src/           model + pipeline code
scripts/       train / inference / eval entrypoints
plan/          phase plans, score ladder
exp/           experiment outputs (gitignored)
```

Datasets are already in `~/.cache/huggingface/hub/` (see CHALLENGE.md).
