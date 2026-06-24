# Blind B v1 Predictions Audit

80 sessions (10 per turn, turns 1-8). Reviewed every query against its top-5 recommendations.

---

## Summary of Issues

| Category | Count | Severity |
|----------|-------|----------|
| Artist lock-in (stuck on one artist despite pivot request) | ~12 | High |
| Keyword/title matching over semantic intent | ~8 | Medium |
| Ignoring explicit user correction | ~10 | High |
| Cannot fulfill metadata queries (tempo, key, specific lyrics) | ~5 | Expected limitation |
| Genre mismatch (wrong vibe entirely) | ~6 | High |
| Album-flooding from a single album | ~7 | Medium |

---

## Detailed Findings

### 1. CRITICAL: Artist Lock-In (Model Cannot Break Away)

The most consistent failure. When the conversation establishes an artist context, the ranker floods results from that artist even when the user explicitly asks to move on.

**Session 04135d8a (Turn 5):** User says "I gotta break away from Deltron and Del completely. I'm out here looking for *new* established underground hip-hop artists." Top 5 is entirely Dan The Automator, Del The Funky Homosapien, and Deltron 3030 -- the exact artists the user asked to leave behind.

**Session 96c044be (Turn 3):** User says "I asked for hardcore or experimental alternative, and I'm still getting classic hip-hop." Top 5 is all Gang Starr -- classic hip-hop.

**Session 517ab271 (Turn 6):** User says "I was actually hoping to discover a new symphonic black metal band." All 5 recs are Cradle Of Filth, the same band from previous turns.

**Session dacd3a58 (Turn 4):** User explicitly asks "Could you please play Sarah McLachlan's 'Fallen'." Gets 5 Leonard Cohen tracks. The system cannot break out of the Cohen context.

**Session 94f9327f (Turn 7):** User says "I need recommendations from a *different* phase" (of God Is An Astronaut). All 5 recs are God Is An Astronaut, same stylistic phase.

**Session 954de66b (Turn 2):** User says "I wanted 'Watercolors' by Pat Metheny specifically." Does not get Watercolors. Gets the same Metheny tracks as before, reordered.

**Session 6c90a029 (Turn 5):** User says "I'm looking for a specific classic rock track, like from the late 70s or early 80s." Gets 3 Morcheeba tracks (trip-hop, wrong decade, wrong genre entirely). Only track 5 (ELO) is close.

**Session 59e1e7b3 (Turn 7):** User asks to branch out from Ryan Adams. Gets 5 Ryan Adams tracks.

### 2. CRITICAL: Ignoring Explicit User Corrections

**Session 6e2eb7e6 (Turn 8):** User states in all caps essentially: "I am *not* looking for punchy, high-energy alternative rock. I need wistful, reflective, introspective indie rock or classic soul for a *calm* evening." Gets 96 Tears, Kick Out the Jams, Little Talks -- punchy, high-energy tracks. Complete inversion of stated preference.

**Session 60f60edd (Turn 7):** User says "I keep getting nature sounds. I really need *electronic* ambient music, not actual forest or rain sounds." Top 5: Calm Rolling Thunder, Relaxing Constant Rain Storm, Rain Sounds, Rainforest Ambience, Serenity Stream. Identical failure mode repeated.

**Session 6f159f55 (Turn 4):** User says "It's still too loud and energetic. I really need something calm and gentle." Gets Los Campesinos! -- energetic indie rock.

**Session 46faad58 (Turn 3):** User asks for social/political message songs. Gets 5 Cradle Of Filth tracks -- gothic metal with zero social commentary.

**Session b5f75b36 (Turn 4):** User says "this isn't what I'm looking for at all. I'm trying to find a specific album, not just listen to more RVIVR." Top 5 starts with 2 RVIVR tracks.

### 3. MEDIUM: Keyword/Title Matching Over Semantic Understanding

**Session ff76b679 (Turn 1):** User asks about "striking abstract cover art." Gets "Painting (Masterpiece)" (title match) and "Painted Sun in Abstract" (keyword match). But ranks 2, 4, 5, 7, 8, 9, 10, 13 are all Kodak Black tracks with zero relevance to abstract art. The system matched some title keywords then filled with a dominant artist cluster.

**Session 7905bb71 (Turn 1):** User wants "happy, feel-good music." Gets "Happy Day" by Fee (Christian worship), "Glory to God Forever" by Fee. Title-keyword match ("happy") pulling in a worship artist with low general relevance.

**Session 76151251 (Turn 3):** User wants a song with the exact lyrical phrase "Your name, is a strong and mighty tower." Gets general worship songs (Matt Redman, Chris Tomlin) -- genre match but no lyric-level precision. This is a known limitation but worth noting.

### 4. MEDIUM: Album/Artist Flooding in Top-20

**Session ff76b679 (Turn 1):** 7 of the top 10 are Kodak Black from what appears to be the same album. No user signal pointed to Kodak Black.

**Session 5870e73f (Turn 4):** All 5 are Deltron 3030 / Dan The Automator despite user wanting "direct social commentary or vivid depictions of urban challenges."

**Session eec6b4e2 (Turn 2):** All 5 are Kalkbrenner brothers. User wanted "uplifting and positive" electronic -- valid genre but zero diversity.

**Session c75f8e41 (Turn 3):** All 5 are Beegie Adair Christmas songs despite user saying "I was actually hoping for something different."

### 5. EXPECTED LIMITATION: Metadata Queries the System Cannot Handle

**Session 46e8aa14 (Turn 6):** User asks for "tracks with a tempo of 155.01 bpm AND are in the key of C minor." System has no tempo/key metadata. Gets random The Word Alive tracks.

**Session bf27c872 (Turn 8):** User asks for tracks matching "126 BPM." Gets hardcore/metal. Same structural limitation.

**Session bdb28533 (Turn 5):** User wants songs in the "exact E major key signature." Not achievable with current features.

**Session fb33bd22 (Turn 5):** User wants the Husker Du track with exact lyrics "I'm never talking to you again." System cannot do lyric-level retrieval. Gets generic Husker Du tracks.

### 6. NOTABLE: What Works Well

**Session 68993adf (Turn 1):** "Strong narrative, life story, melancholic." Gets Story of My Life, Fix You, Stop This Train. Solid thematic match.

**Session ab87371b (Turn 1):** "Intense focused hip-hop, raw urban vibe." Gets m.A.A.d city, Hip Hop (Dead Prez), Bitch Don't Kill My Vibe. Strong.

**Session 25cc9533 (Turn 1):** Peter Doherty from Grace/Wastelands. All 5 are Peter Doherty tracks from the right album. Correct retrieval.

**Session fc6ba76a (Turn 1):** "Background music to relax." Gets Jose Gonzalez, C418 Minecraft, Birdy, Christopher Cross, George Benson. Good range.

**Session 4b239a62 (Turn 2):** Nu-jazz/downtempo follow-up. All 5 Parov Stelar -- correct artist, correct style.

**Session 40cc1c03 (Turn 6):** "Gritty raw blues rock." Gets Brother Dege, Lincoln Durham, Blues Saraceno. Well-targeted.

**Session 0802ac4a (Turn 6):** "Battle metal." Gets Turisas and Amon Amarth. Spot on.

**Session 909efb74 (Turn 8):** "Upbeat 90s country hits." Gets Tim McGraw, Tracy Byrd, George Strait, Tracy Lawrence, David Lee Murphy. All correct era and style.

---

## Root Cause Hypotheses

1. **Conversation context over-indexing on artist.** Once an artist appears in conversation history, the retrieval/ranking step heavily biases toward that artist's catalog. This is helpful for "more like this from the same artist" queries but catastrophic when the user explicitly asks to pivot.

2. **No negative-feedback signal.** When the user says "not this," "I don't want X," or "break away from Y," the system does not downweight the rejected entity. The conversation embedding likely still encodes the rejected artist/genre as a strong positive signal because the name appears frequently.

3. **Weak genre-pivot capability.** The system handles Turn 1 cold-start genre matching reasonably well (see "What Works Well") but degrades sharply when the user asks to shift genre mid-conversation.

4. **Title/keyword retrieval leaking into semantic slots.** Some top-1 picks are clearly title-keyword matches (e.g., "Painting (Masterpiece)" for a query about paintings). This works sometimes but pulls in irrelevant neighbors when the keyword-matched track's artist cluster dominates the re-ranking.

5. **No metadata retrieval.** Tempo, key, and exact-lyric queries are structurally unsupported. These failures are expected but should be documented as out-of-scope for this model version.

---

## Recommendations

1. **Implement negative-signal handling.** Parse explicit rejections ("not this," "break away from X," "I don't want Y") and suppress the rejected artist/genre in re-ranking.

2. **Cap per-artist slots in top-20.** Hard-cap at 3 tracks per artist unless the user explicitly asks for more from one artist. This alone would fix the Kodak Black flooding, Kalkbrenner flooding, etc.

3. **Detect pivot intent.** When the user asks for "something different," "a new artist," or "break away," apply artist diversification aggressively.

4. **Separate keyword-match score from semantic-match score.** Title keyword matches should boost but not dominate. Avoid letting a keyword hit pull in an unrelated artist cluster.

5. **Document metadata limitations.** Tempo, key, and lyric-exact queries should produce a graceful "I don't have that metadata" response rather than random results.
