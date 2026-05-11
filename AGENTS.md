# AGENTS.md

This file defines project agents.

See `agents/` for agent specifications.

## Quick Reference

- `agents/README.md` - Overview and rules
- `agents/planner.md` - Planning and continuity
- `agents/data.md` - Dataset inspection
- `agents/codebook.md` - Semantic ID codebooks
- `agents/training.md` - LoRA training
- `agents/evaluation.md` - Model validation
- `agents/inference.md` - Demo pipeline
- `agents/response.md` - Natural language response

## Global Rules

Read before writing.
Plan before complex work.
Validate after changes.
Report results first.

Do not guess APIs, flags, schemas, or package names.
Verify with local files.

Do not overwrite adapters.
Do not mix ID namespaces.
Do not hide failed validation.

## Required Output Format

Every agent response should end with:

```
Result:
Files changed:
Validation:
Next:
```

Use short sentences. Avoid filler. Avoid praise. Avoid emojis. Avoid em dashes.
