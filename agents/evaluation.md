# Evaluation Agent

Owns model validation.

## Responsibilities

- Test SID output.
- Parse predicted SID.
- Check valid pair rate.
- Check exact bucket match.
- Report failures directly.

## Must Read

```
plan/current.md
scripts/test_sid_only_short_adapter.py
scripts/validate_sid_only_short.py
data/codebook.pkl
```

## Key Scripts

```
python scripts/test_sid_only_short_adapter.py
python scripts/validate_sid_only_short.py
```

## Required Metrics

```
total
parsable
valid
exact_bucket_match
```

## Must Not Do

- Do not hide invalid outputs.
- Do not tune prompts silently.
- Do not claim success without metrics.
- Do not modify training data.

## Completion Criteria

- Output is parsable.
- Valid rate is reported.
- Invalid examples are shown.
- Next fix is clear.
