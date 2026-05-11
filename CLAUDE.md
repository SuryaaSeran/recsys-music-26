# CLAUDE.md

## Approach
- Read existing files before writing. Don't re-read unless changed.
- Thorough in reasoning, concise in output.
- Skip files over 100KB unless required.
- No sycophantic openers or closing fluff.
- No emojis or em-dashes.
- Do not guess APIs, versions, flags, commit SHAs, or package names. Verify by reading code or docs before asserting.

## Working Style

Read this file before starting any task.

Be direct.
Be careful.
Be brief.
Do not sound like an assistant.

Use short sentences.
Avoid filler.
Avoid praise.
Avoid vague progress updates.

Do not use emojis.
Do not use em dashes.
Use normal punctuation only.

Prefer action over explanation.
Explain only when useful.
Ask only when blocked.

## Plan lifecycle (must follow)

Authoritative process is in `WORKFLOW.md`. Summary:

- One active phase plan at a time: `plan/0N_<name>.md`.
- Update `plan/CURRENT_BEST_ITERATION.md` the moment dev nDCG@20 (full 1000
  sessions) strictly beats the previous entry. Demote the old config to a
  one-liner in the "Previous bests" section.
- Update `plan/PLAN.md` score ladder only for full 1000-session results.
- Close a phase: move its plan into `plan/archive/` and replace the active entry
  in `plan/PLAN.md` with a one-line summary + archive link.
- Record only iterations that affected the conclusion, not every parameter tried.

When starting any session, read in this order:
1. `plan/PLAN.md`
2. `plan/CURRENT_BEST_ITERATION.md`
3. The single active `plan/0N_*.md`

## Planning Rules

Always check the `plan/` folder first.

Before complex work:
1. Read the latest plan.
2. Update or create a task plan.
3. List assumptions.
4. List exact files to inspect.
5. List exact files to modify.
6. Define success checks.

Do not execute complex workflows first.
Plan first, then act.

Use this structure:

```md
# Plan: <task name>

## Goal

## Current State

## Assumptions

## Files To Read

## Files To Modify

## Steps

## Validation

## Risks

## Notes
