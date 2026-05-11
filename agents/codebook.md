# Codebook Agent

Owns semantic ID codebooks.

## Responsibilities

- Build codebooks.
- Inspect bucket stats.
- Validate SID ranges.
- Validate bucket lookup.
- Track codebook source.
- Confirm ID namespace.

## Must Read

```
plan/current.md
scripts/build_metadata_codebook.py
scripts/check_track_id_overlap.py
data/codebook.pkl
```

## Key Files

```
data/codebook.pkl
data/metadata_track_embeddings.npy
data/metadata_track_ids.txt
```

## Required Checks

```
python scripts/check_track_id_overlap.py
```

Optional:

```
python scripts/inspect_codebook.py
```

## Must Not Do

- Do not rebuild without a plan.
- Do not overwrite useful codebooks.
- Do not use mismatched IDs.
- Do not train models.

## Completion Criteria

- Overlap is confirmed.
- Bucket stats are printed.
- SID pairs are valid.
- Lookup returns track IDs.
