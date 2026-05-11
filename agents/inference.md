# Inference Agent

Owns local recommender demo.

## Responsibilities

- Build demo pipeline.
- Generate SID from prompt.
- Parse SID pair.
- Resolve bucket.
- Select concrete track.
- Print metadata.

## Must Read

```
plan/current.md
data/codebook.pkl
scripts/validate_sid_only_short.py
```

## Pipeline

```
user request
-> SID-only model
-> semantic ID pair
-> codebook bucket
-> track UUID
-> track metadata
```

## Resolver Rules

Use this order:

```
1. Exact valid pair lookup
2. Highest popularity candidate
3. Metadata similarity fallback
4. Report invalid SID
```

## Must Not Do

- Do not fabricate tracks.
- Do not ignore invalid SIDs.
- Do not return bucket IDs only.
- Do not include all bucket tracks.

## Completion Criteria

- Demo accepts user text.
- Demo predicts SID.
- Demo resolves track UUID.
- Demo prints track metadata.
- Invalid outputs are handled.
