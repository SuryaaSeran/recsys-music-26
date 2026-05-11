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