# Plan: Response Prompt Optimization

## Current Prompt Analysis

The existing `generate_responses_lmstudio.py` prompt is well-designed. Key strengths:

1. **Style hints (10 variants)** force lexical diversity via deterministic rotation.
   Each (session_id, turn_number) pair gets a unique opening style via SHA256 hash.
   This directly targets Distinct-2.

2. **Reasoning close requirement** (1-2 sentences explaining why this track is #1)
   targets the LLM-as-Judge explanation quality dimension.

3. **Hard rules** (no markdown, no greetings, plain prose, 3-5 sentences) prevent
   common LLM failure modes that would hurt both metrics.

4. **Concrete metadata grounding** ("the 1973 Stax production", "the acoustic
   finger-picking on the Folklore tag") targets explanation quality.

## Problems to Fix

### Problem 1: Conversation history is too thin

The prompt shows only last 3 user turns (`[TURN-3]`, `[TURN-2]`, `[TURN-1]`).
It does NOT show:
- **Assistant responses** from prior turns (what the system previously recommended)
- **Music turns** (what tracks were actually played)
- **The user's reaction** to prior recommendations

The Gemini judge evaluates the `(dialogue turn, recommended tracks, response)` triple.
If the user said "I loved that last track, give me more like it" and our response
doesn't acknowledge what "that last track" was, the judge will score low on
personalisation.

**Fix:** Include the last 3-4 full conversation turns (user + assistant + music)
as they actually happened, not just user turns. Show played tracks with their
metadata so the model can reference them.

### Problem 2: User profile is underutilized

The prompt passes `Goal: ...` and `Culture: ...` but does NOT include:
- **age_group** (e.g., "25-34")
- **country_name** (e.g., "United Kingdom")
- **gender**
- **listening_history** summary (what kind of music this user likes generally)

These are available in the dataset and directly target the personalisation dimension.
A response that says "as a listener drawn to Anglo-American rock" is more personalized
than one that doesn't reference the user at all.

**Fix:** Add demographics and culture to the context block. Optionally summarize
the user's listening history genre distribution (top 3 genres from their history).

### Problem 3: Only top 3 candidates shown

The model sees candidates #1-#3 but the response should primarily be about #1.
Showing #2-#3 sometimes confuses the model into mentioning all three (the hard rule
says "Do NOT mention any number of tracks. The reply is about ONE track" but models
sometimes ignore this).

More importantly, the model does NOT see what tracks were already recommended
earlier in the session. If tracks #2-#3 are from the same artist as a previously
played track, that's a strong personalization signal the model can't use.

**Fix:** Show top 3 candidates but also show "Previously played this session:"
with the actual tracks. This lets the model draw connections ("Since you enjoyed
Arctic Monkeys earlier, you'll love this track by...").

### Problem 4: Temperature 0.95 is too high

High temperature increases lexical diversity (Distinct-2) but can produce:
- Factual errors (hallucinating metadata details)
- Incoherent reasoning
- Drifting off-topic

The LLM-as-Judge penalizes incoherent or fabricated explanations. A temperature
of 0.7-0.8 would better balance diversity vs quality.

**Fix:** Lower temperature to 0.75. The style hints already force structural
diversity, so we don't need high temperature for Distinct-2.

### Problem 5: max_tokens 5000 is wasteful

The hard rule says 3-5 sentences (~60-100 words). max_tokens=5000 lets the model
ramble. Some models will fill the available space.

**Fix:** max_tokens=200. This forces conciseness and prevents rambling.

### Problem 6: No few-shot examples in the user message

The system prompt has 2 examples of acceptable responses. But they are in the
system prompt (seen once), not in the user message (seen last, most influential
for instruction-following models). Moving 1-2 examples to the user message as
"Here is an example of a good response for a different query:" would improve
instruction following.

**Fix:** Add one worked example in the user message, right before the final
instruction. Make it for a DIFFERENT genre/mood than the current query to avoid
the model copying the example's content.

### Problem 7: The banned openers list is too short

The prompt bans: "Built for", "Pulled together", "Loaded", "Queued", "For when",
"These", "This is", "Spinning up", "Threading". But models will find other
repetitive openers. Common ones to add:

- "Here's", "Here is"
- "Great choice", "Good pick"
- "Absolutely"
- "Sure thing"
- "You're going to love"
- "Check out"
- "I've got just the thing"
- "Perfect for"

**Fix:** Extend the banned list. Alternatively, instead of banning, use the style
hint more aggressively -- make the opening word/phrase mandatory, not just suggested.

### Problem 8: No explicit instruction about the listening history connection

The system prompt says "If the user previously enjoyed certain tracks, draw a
connection" but the model often ignores this because the history is formatted
as `[TURN-3]`, `[TURN-2]`, `[TURN-1]` (user text only, no track info).

**Fix:** In the user message, after the candidate tracks, add:
```
Tracks the user has enjoyed in this session (reference these if relevant):
- "Fluorescent Adolescent" by Arctic Monkeys (Tags: indie rock, alternative)
- "Heart-Shaped Box" by Nirvana (Tags: grunge, alternative rock, 90s)
```

This gives the model concrete material to draw connections with.

## Proposed Improved User Message Structure

```
CONVERSATION HISTORY (last 4 exchanges):
[User] "Play Heart-Shaped Box by Nirvana"
[Played] "Heart-Shaped Box" by Nirvana | Tags: grunge, alternative rock, 90s | Album: In Utero
[User] "Great! What other popular alternative rock tracks do you recommend?"
[Played] "Fluorescent Adolescent" by Arctic Monkeys | Tags: indie rock, alternative, 2000s
[User] "Another solid track. Can you recommend another highly popular alternative rock track?"

USER PROFILE:
- Goal: play highly popular alternative rock tracks
- Culture: Anglo-American Rock
- Demographics: 25-34, United Kingdom

TRACKS PLAYED THIS SESSION (for context/connections):
1. "Heart-Shaped Box" by Nirvana | grunge, alternative rock, 90s
2. "Fluorescent Adolescent" by Arctic Monkeys | indie rock, alternative, 2000s

YOUR TOP RECOMMENDATION (respond about this track):
1. "D Is For Dangerous" by Arctic Monkeys | Album: Favourite Worst Nightmare | Tags: indie rock, alternative, 2000s | Year: 2007
Also considered:
2. "Brianstorm" by Arctic Monkeys | Tags: indie rock, garage rock
3. "The View from the Afternoon" by Arctic Monkeys | Tags: indie rock

OPENING STYLE: Open by paraphrasing the user's request and lead into the track name.

Write 3-5 sentences recommending track #1. Personalize to the user's request,
name the track in double quotes with artist, explain why it fits using metadata,
reference their session history if relevant, and close with 1-2 sentences of
prediction reasoning.
```

## Proposed Improved System Prompt

Key changes from current:
1. Shorter, more focused (current is ~1200 words, target ~600 words)
2. Drop the 2 examples from system prompt (move 1 to user message)
3. Add explicit "reference session history" instruction
4. Extend banned openers list
5. Add: "The response will be evaluated by an AI judge for personalisation and
   explanation quality. Generic responses score poorly."

```
You are a music recommendation assistant in a multi-turn conversation. You write
the response the user sees when a track starts playing.

Your reply MUST do four things:

1. PERSONALIZE: Show you understood the specific thing the user asked for.
   Paraphrase their mood, artist, era, or feeling in your own words. If they
   said "melancholic like Nils Frahm", your reply must reference Frahm and
   melancholy. Generic replies fail.

2. NAME THE TRACK: Write the track name in double quotes followed by "by {artist}".
   Use the exact name and artist from the metadata provided.

3. EXPLAIN WITH EVIDENCE: Give one concrete reason from the track's metadata
   (year, album name, a specific tag, sonic character) that connects to the
   user's request. Not "it has great vibes" -- say "the 2007 Favourite Worst
   Nightmare production" or "the grunge and alternative rock tags".

4. CONNECT TO HISTORY: If the user has played tracks in this session, draw a
   connection. "Since you enjoyed Nirvana's grunge energy, this track channels
   a similar..." This is critical for personalisation scoring.

Close with 1-2 sentences explaining what combination of the user's preferences,
their listening history, and the track's characteristics made this the top pick.

HARD RULES:
- 3 to 5 sentences total. No 2-liners. No 6-liners.
- Plain prose. No headers, bullets, markdown, emojis, numbered lists.
- No greetings ("Hi!"), sign-offs ("Enjoy!"), or filler ("Sure thing!").
- Do NOT start with: "Here's", "Here is", "I recommend", "Based on",
  "Absolutely", "Great choice", "Check out", "I've got", "Perfect for",
  "Built for", "Loaded", "Queued", "Spinning up", "This is".
- NEVER invent track names, artists, lyrics, or albums not in the metadata.
- Begin with the first word of the actual reply.
- The response is about ONE track (#1). Do not list multiple tracks.

An AI judge will evaluate your response for personalisation and explanation
quality. Generic, templated responses score zero.
```

## Distinct-2 Optimization

The style hints are the primary Distinct-2 lever. Current 10 styles are good.
Additions to consider:

11. "Open with a contrast to the previous track ('Shifting gears from the grunge
    energy of Nirvana...')"
12. "Open with a rhetorical question ('Looking for that late-night acoustic feel?')"
13. "Open with a time/place anchor ('For those quiet 2am moments...')"

More styles = more bigram diversity across 80 blind predictions. 13 styles across
80 predictions means ~6 responses per style, which is reasonable variety.

## Implementation

Changes are to the prompt constants in `generate_responses_lmstudio.py`:
1. Update SYSTEM constant with the shorter, focused version above
2. Update `build_user()` to include full conversation history (user + assistant + music)
3. Update `build_user()` to include user demographics and played-tracks-this-session
4. Lower temperature to 0.75
5. Lower max_tokens to 200
6. Add 3 more style hints
7. Add one worked example in the user message

No structural code changes needed. The caching, API call, and fallback logic are fine.

## Validation

- Run on a 10-prediction subset first, manually inspect quality
- Measure Distinct-2 on the full 80 predictions using `evaluate_local.py`
- Target: Distinct-2 > 0.35 (current template: 0.2073, current Qwen: ~0.25 estimated)
