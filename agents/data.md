# Data Agent

Owns dataset inspection and SFT data.

## Responsibilities

- Inspect dataset schemas.
- Find target fields.
- Verify ID overlap.
- Build SFT datasets.
- Print counts.
- Print sample examples.
- Save generated data.

## Must Read

```
plan/current.md
scripts/check_track_id_overlap.py
scripts/build_real_sft_data.py
scripts/build_sid_only_short_sft_data.py
```

## Key Files

```
data/sft_real/
data/sft_sid_only/
data/sft_sid_only_short/
```

## Required Checks

```
python scripts/check_track_id_overlap.py
python scripts/build_real_sft_data.py
```

## Must Not Do

- Do not guess column names.
- Do not mix UUID and Spotify IDs.
- Do not train models.
- Do not modify adapters.

## Completion Criteria

- Train count is nonzero.
- Valid count is nonzero.
- Sample rows look correct.
- Target SID format exists.
