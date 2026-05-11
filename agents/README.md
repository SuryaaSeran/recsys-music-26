# Agents

Agent specifications for the semantic ID music recommender.

All agents must read `CLAUDE.md` and `plan/current.md` before starting work.

## Global Rules

Read before writing.
Plan before complex work.
Validate after changes.
Report results first.

Do not guess APIs, flags, schemas, or package names.
Verify with local files.
Use official docs only when needed.

Do not overwrite adapters.
Do not mix ID namespaces.
Do not hide failed validation.

## Required Output Format

Every agent response ends with:

```
Result:
Files changed:
Validation:
Next:
```

Use short sentences. Avoid filler. Avoid praise. Avoid emojis. Avoid em dashes.

## Agents

1. **Planner** - Planning and continuity
2. **Data** - Dataset inspection and SFT data
3. **Codebook** - Semantic ID codebooks
4. **Training** - LoRA training runs
5. **Evaluation** - Model validation
6. **Inference** - Demo pipeline
7. **Response** - Natural language response

See individual `.md` files for details.
