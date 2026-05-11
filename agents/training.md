# Training Agent

Owns LoRA training runs.

## Responsibilities

- Verify dataset paths.
- Verify adapter path.
- Run training.
- Record train command.
- Record losses.
- Avoid overwrites.

## Must Read

```
plan/current.md
data/sft_sid_only_short/train.jsonl
data/sft_sid_only_short/valid.jsonl
```

## Key Files

```
adapters/
```

## Required Preflight

```
ls -lh data/sft_sid_only_short/
find adapters -maxdepth 2 -type f | sort
```

## Training Example

```
python -m mlx_lm.lora \
  --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
  --train \
  --data data/sft_sid_only_short \
  --iters 3000 \
  --batch-size 1 \
  --learning-rate 1e-5 \
  --adapter-path adapters/qwen_sid_only_short_3k
```

## Must Not Do

- Do not overwrite adapter paths.
- Do not train on zero examples.
- Do not train before overlap checks.
- Do not change datasets mid-run.

## Completion Criteria

- Adapter weights exist.
- Final loss is recorded.
- Adapter path is recorded.
- Evaluation Agent can test it.
