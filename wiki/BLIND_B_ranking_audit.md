# Blind Set B Ranking Audit Report

## Overall Verdict

The ranking is **partially functional but with serious systematic problems**. Approximately **55-60% of sessions have a plausible #1 pick**, but this number masks large disparities between session types:

- Warm sessions where the system has a clear artist to anchor on are mostly acceptable.
- Cold sessions at turn 1 with vague or abstract requests frequently produce garbage at rank 1.
- Sessions where the user names a **specific track** and the system fails to surface it — even when the track exists in the corpus — are the most damaging failure mode. At least 8 sessions have this problem.

The most common pattern of failure: the model latches onto surface keywords (a user tag, a played artist, or a genre word) and floods the top 20 with that cluster, ignoring what the conversation actually evolved toward.

---

## GOOD Examples

**Session `4b239a62` (cold, turn 2) — Parov Stelar discovery**
User's active message: "Is it the rhythm or the overall instrumentation that makes it feel so good?"
Already played: "The Beach - Parov Stelar"
#1: "Beautiful Morning - Parov Stelar [2012]"
#2–#6: more Parov Stelar nu-jazz tracks, then adjacent artists (Koop, Nicola Conte).
Verdict: Correct. The user is deep in Parov Stelar territory; ranking his catalog densely at the top is exactly right.

**Session `f321d88d` (cold, turn 1) — Defiance, Ohio punk request**
User: "looking for songs with a raw, driving punk rhythm and passionate, almost shouting vocals, similar to Defiance, Ohio's sound."
#1: "Oh, Susquehanna! - Defiance, Ohio [2006]" — the band itself, correct match.
#2–#7: more Defiance, Ohio + Black Flag + Miss Murder + American Idiot.
Verdict: Placing the named artist at rank 1 is correct. The broader list is defensible punk/pop-punk, though Down With the Sickness (Disturbed, nu-metal) at #10 is a slight stretch.

**Session `49009ca7` (cold, turn 3) — Myrkur atmospheric black metal**
User explicitly asked for vocal versatility and Nordic folk-metal fusion after two rounds.
#1: "Ulvinde - Myrkur [2017]" — tagged "dark folk, atmospheric, post-black metal".
Top 11: entirely Myrkur. Rank 19: Marduk (Swedish black metal). Rank 20: Sólstafir (atmospheric).
Verdict: Strong. The artist focus is appropriate given the conversation, and the non-Myrkur picks at the bottom are genre-coherent.

**Session `ca9c5de4` (warm, turn 2) — Kyle Dixon / Stranger Things synthwave**
User: "find tracks that build tension with deep synth pads and arpeggiated melodies, like 'Danger Danger' or 'Abilities'."
Entire top 20: Kyle Dixon and Kyle Dixon & Michael Stein tracks from the Stranger Things OST.
Verdict: Correct given the played history and the hyper-specific request. The user wanted more of this exact artist/show; delivering 20 is reasonable.

**Session `0802ac4a` (warm, turn 6) — Viking/battle metal progression**
After five turns building from Amon Amarth to Turisas, user asked for more folk-metal.
#1–#5: Turisas. #6: Amon Amarth back in. Good genre coverage (Turisas, Amon Amarth).
Verdict: Coherent and musically appropriate.

---

## PROBLEM Examples

### P1 — Session `ff76b679` (cold, turn 1): Album cover art query flooded with Kodak Black

**User's last message:** "I'm looking for an album with a really striking, abstract cover art. It had some bold colors and shapes, almost like a modern painting."

**Top ranked tracks:**
1. "Painting (Masterpiece) - Lewis Del Mar [2016]" — triggered by "Painting" keyword
2. "Corrlinks and JPay - Kodak Black [2017]"
3. "Painted Sun in Abstract - Trent Reznor and Atticus Ross [2010]"
4–9: More Kodak Black (six tracks back to back)

**Problem:** Kodak Black has zero connection to abstract cover art, bold colors, or modern painting. The model apparently picked him up through some tag overlap and then duplicated him at ranks 2, 4, 5, 6, 7, 8, 13, 18 — nine appearances. The actual user need (artist with visually distinctive, bold-colored abstract album art) is unanswered. Track 3 is the only genuinely relevant pick (the word "Abstract" in title). Better answers would include Radiohead *Kid A*, Oneohtrix Point Never, Massive Attack *Mezzanine*, Flying Lotus, or Kanye West *Yeezus*.

---

### P2 — Session `dacd3a58` (cold, turn 4): User asked for Sarah McLachlan "Fallen" eight times and never got it

**User's last message:** "Could you please play Sarah McLachlan's 'Fallen', specifically the version that conveys that profound sense of introspection and longing?"

**Played history:** I Will Not Forget You, Fumbling Towards Ecstasy (both McLachlan), Hallelujah (Leonard Cohen — completely wrong artist)

**Ranked #1–#20:** Leonard Cohen catalog monopolizes ranks 1–5, 7–12, 14–16, 19–20. Sarah McLachlan appears only at rank 6 ("World on Fire"), rank 7 ("Hold On"), rank 13 ("Push"), and rank 18 ("Train Wreck"). "Fallen" by Sarah McLachlan is **absent entirely** from the top 20 despite the user asking for it by name four times.

**Problem:** The most severe failure mode in the dataset. The correct track — "Fallen" by Sarah McLachlan — either exists in the corpus or doesn't; if it does, it should be rank 1. If it doesn't exist, the system should surface her closest-matching tracks at the top, not Leonard Cohen. Drowning the list in Cohen after playing his "Hallelujah" is a confusion of conversation history with current intent.

---

### P3 — Session `93647ac5` (cold, turn 8): User asked for "Versace (Remix)" by Migos eight times — it's rank 2

**User's last message:** "This is getting ridiculous... Just play 'Versace (Remix)' by Migos."

**Played history:** Seven other Migos tracks (none of them "Versace").

**Ranked #1:** "China Town - Migos"
**Ranked #2:** "Versace (Remix) - Migos [2015]" — the exact requested track is in the corpus, ranked second.

**Problem:** The user has asked for this specific track eight consecutive times. The system has it in the catalog. It must be rank 1. Placing any other Migos track above it is a ranking failure that will directly hurt NDCG against any reasonable relevance judgment.

---

### P4 — Session `76151251` (cold, turn 3): Exact lyric search for "Your name, is a strong and mighty tower"

**User's last message:** "I really need to find a song with the exact lyrical phrase 'Your name, is a strong and mighty tower'."

**Top 20:** "10,000 Reasons (Bless the Lord) - Matt Redman", "How Great Is Our God - Chris Tomlin", and 18 other generic Christian Contemporary worship songs. None contain the target phrase.

**Problem:** The song containing that exact lyric is "Your Name" by Paul Baloche (2006), which is a well-known contemporary Christian worship song. It is not in the list at all. The system defaulted to generic CCM popularity instead of the specific track. This is a lookup failure, not a ranking failure per se, but surfacing a plausible candidate ("Your Name" or close variants) near the top was possible with better title/lyric matching.

---

### P5 — Session `37257e95` (warm, turn 8): "Czar Refaeli" by CZARFACE asked eight times, never appears

**User's last message:** "I need 'Czar Refaeli' by CZARFACE... Can you play 'Czar Refaeli' by CZARFACE, please?"

**Ranked top 20:** Nightcrawler, Czartacus, Sinister, Red Alert, Ka-Bang!, Escape from Czarkham Asylum (all CZARFACE) — but "Czar Refaeli" is not in the list at all. Apollo Brown and Black Milk fill the remaining slots.

**Problem:** "Czar Refaeli" appears to be absent from the corpus (it's a real CZARFACE track from the 2013 debut). The system correctly escalated CZARFACE to the top but did not prioritize by title similarity. If the track isn't in the corpus, the closest CZARFACE tracks should dominate top positions, which they mostly do — but rank 1 "Nightcrawler" has no particular link to the request over other CZARFACE tracks.

---

### P6 — Session `96c044be` (cold, turn 3): User explicitly asked for hardcore/experimental, got 20 Gang Starr tracks

**User's last message:** "I asked for hardcore or experimental alternative, and I'm still getting classic hip-hop. Can you please switch it up completely? Give me something raw and intense, far from my comfort zone."

**Played history:** Gang Starr (two tracks, explicitly the ones the user was tired of)

**Top 20:** 19 Gang Starr tracks + 1 Wu-Tang Clan.

**Problem:** The user's signal could not be clearer: "stop giving me Gang Starr." The system doubles down with a pure Gang Starr dump. A correct ranking would place no Gang Starr in the top 10 and lead with hardcore/noise/experimental tracks (Pig Destroyer, Hatebreed, Have A Nice Life, anything off-genre). This is a persona/conversation-state failure where the system treats played history as positive signal instead of reading the user's explicit rejection.

---

### P7 — Session `eec6b4e2` (cold, turn 2): User asked for "uplifting and positive" electronic, got Kalkbrenner minimal techno

**User's last message:** "I'm looking for something more uplifting and positive, really something to lift my spirits."

**Played:** A Fritz Kalkbrenner minimal techno track described as "too serious."

**Top 20:** Seventeen Paul/Fritz Kalkbrenner tracks, all minimal techno — the exact sound the user just rejected as "too serious." Ranks 19–20 finally escape the artist (Nicolas Jaar, also minimal).

**Problem:** The user gave direct negative feedback on the Kalkbrenner style. The system responded by flooding with 17 more Kalkbrenner tracks. No "happy," "uplifting," or mood-positive electronic music appears. Better answers: Daft Punk, Röyksopp, Disclosure, Kygo, Passion Pit, Hot Chip.

---

### P8 — Session `60f60edd` (warm, turn 7): User explicitly rejected nature sounds five times, ranked top 20 is all rain/forest soundscapes

**User's last message:** "I keep getting nature sounds. I really need to stress that I'm after electronic ambient music, not actual forest or rain sounds."

**Played history:** Includes nature sounds that were rejected repeatedly.

**Top 20 (ranked):** "Calm Rolling Thunder and Soothing Rain," "Relaxing Constant Rain Storm," "Rain Sounds," "Rainforest Ambience," "Serenity Stream," "Rain Forest and Tropical Beach Sound" — 14 of 20 tracks are literal rain/nature sound files. Brian Eno appears at rank 11 and rank 14. C418 appears at ranks 17–18.

**Problem:** This is the most extreme conversation-state failure. The user has explicitly rejected this category five turns in a row, yet the ranking fills 14 slots with it. Electronic ambient artists (Clubroot, Boards of Canada, Aphex Twin, Gas, Helios, Ólafur Arnalds) should lead the list.

---

### P9 — Session `5870e73f` (warm, turn 4): User asked for social commentary hip-hop; got Deltron 3030 deep cuts for the fourth straight turn

**User's last message:** "I need something more potent and thought-provoking, tracks that go explicitly at systemic issues or the grind of street life."

**Played history:** Deltron 3030 (three tracks), explicitly not what the user wanted.

**Top 20:** Deltron 3030 and Dan The Automator dominate ranks 1–16. C.R.E.A.M. (Wu-Tang) appears at rank 18.

**Problem:** The user is asking for classic underground social commentary ("something more direct, raw social commentary"). Artists like Public Enemy, Dead Prez, KRS-One, Immortal Technique, Talib Kweli, Mos Def, Common are entirely absent. The system stays trapped in the Deltron orbit despite multiple explicit rejections.

---

### P10 — Session `04135d8a` (warm, turn 5): User asked for non-Deltron underground hip-hop; top 13 are all Deltron/Del

**User's last message:** "I gotta break away from Deltron and Del completely. I'm out here looking for new established underground hip-hop artists."

**Top 20 ranks 1–13:** Dan The Automator, Deltron 3030, Del The Funky Homosapien — the exact artists the user explicitly banned. OutKast appears at 14, Gang Starr at 20.

**Problem:** Same failure as P9. The user gave a clear artist exclusion. The system ignored it entirely. MF Doom, Aesop Rock, Atmosphere, Sage Francis, People Under The Stairs, Company Flow, El-P, and many other relevant artists are absent.

---

### P11 — Session `6e2eb7e6` (warm, turn 8): User asked for wistful indie rock; got proto-punk and garage rock eight turns running

**User's last message:** "I am not looking for 'punchy, high-energy alternative rock.' I need wistful, reflective, introspective indie rock or classic soul."

**Top 20 ranks 1–2:** 96 Tears by ? & The Mysterians (proto-punk/energetic), Kick Out the Jams by MC5 (proto-punk). Rank 3: "Little Talks" (finally appropriate). Ranks 12, 15, 25 (Stooges, Soul Kitchen, more proto-punk still appear).

**Problem:** The system continues serving high-energy garage/proto-punk despite eight turns of the user's explicit rejection. The model cannot update away from the played history profile (The Cramps, MC5, ? & The Mysterians). Appropriate tracks (Elliott Smith, Big Star, Galaxie 500, Nick Drake, Yo La Tengo, Sparklehorse, Mazzy Star) are absent.

---

### P12 — Session `7905bb71` (cold, turn 1): "Happy music" returned Christian worship as #1

**User:** "I need some happy, feel-good music right now to lift my spirits."

**#1:** "Happy Day - Fee [2007] tags: christ, happy, gcc worship, upbeat, christian"
**#2:** "Feel - Robbie Williams" (bittersweet, not especially happy)
**#3:** "Glory to God Forever - Fee [2009]" (more CCW)

**Problem:** The #1 pick is a Christian Contemporary Worship track. While it's technically labeled "happy" in the tags, it is a highly genre-specific, non-mainstream choice that will fail for the overwhelming majority of users requesting "happy feel-good music." The system matched on the word "happy" in tags, ignoring the broad mainstream appeal expected. Better rank 1 candidates: Pharrell's "Happy," "Good as Hell" (Lizzo), "Walking on Sunshine," anything clearly secular and upbeat.

---

## Cold-Start Specific Findings

**Cold sessions are systematically worse than warm ones.** The primary reasons:

1. **Tag/keyword matching overrides semantic understanding.** When there is no user profile to weight on, the model relies heavily on tag overlap. Session `ff76b679` (Kodak Black flooding an art query) is the clearest example: "Painting" in a track title and "art" in a tag triggered an irrelevant cluster.

2. **Popularity bias is inconsistent and unpredictable.** Some cold sessions deliver reasonable mainstream picks (session `7905bb71` surfaces recognizable pop names, even if CCW at #1 is wrong). Others go deep into obscure niche artists with no explanation. There is no reliable fallback to "broad, popular" when the signal is vague.

3. **The system cannot handle requests that require world-knowledge lookup.** Cold sessions with specific factual queries (Peter Doherty's best track from 2009 — session `25cc9533`; album art identification sessions) are essentially information retrieval tasks. The ranking handles these by placing the artist's tracks in order, which is partially correct but misses the specific answer. Session `25cc9533` places "Arcady" at #1 as the top Doherty track from Grace/Wastelands, which is defensible (it is frequently cited), but then ranks 8–20 abandon the album entirely.

4. **Vague emotional requests produce random genre clusters.** "Happy music" (session `7905bb71`) led to Christian worship. "Relaxing background" (session `fc6ba76a`) gave a reasonable set. "Background for focus" requests were wildly inconsistent: session `db8ec85f` (Plaid focus) was handled well; session `60f60edd` (Clubroot/ambient) was a catastrophe because of nature-sound contamination in history.

5. **Cold sessions that name a specific artist are handled better than warm sessions that name a specific track.** Defiance, Ohio (cold, named artist) worked. Migos "Versace (Remix)" (cold, named specific track) failed at rank 1 despite the track existing in corpus.

---

## Recommendations

**1. Enforce hard constraint: if a specific track title is mentioned in the last user message AND that exact title exists in the corpus, it must rank #1.**
Sessions `dacd3a58` (Fallen), `93647ac5` (Versace Remix), `37257e95` (Czar Refaeli), `954de86b` (Watercolors), and `2bfd631e` (Everything Under the Sun tracks) all demonstrate this failure. A simple title-match override would fix multiple sessions.

**2. Track explicit negative feedback on artists/genres and demote those artists in the ranking.**
Sessions `eec6b4e2` (Kalkbrenner rejection), `96c044be` (Gang Starr rejection), `5870e73f` (Deltron rejection), `04135d8a` (Del rejection), and `6e2eb7e6` (proto-punk rejection) all show the system ignoring explicit "stop playing X" signals. Negative-signal propagation is completely absent. Every played track is being treated as positive signal, which contradicts the conversation content.

**3. Detect duplicate-artist flooding and cap at 3–4 tracks per artist in top 20 for cold sessions.**
Session `ff76b679`: 9 Kodak Black tracks. Session `eec6b4e2`: 17 Kalkbrenner tracks. Session `60f60edd`: 14 nature-sound files from different pseudo-artists (all effectively one category). Session `96c044be`: 19 Gang Starr tracks. This flooding destroys diversity and covers the correct answer. A hard per-artist cap (4 in top 20 for cold; 6 for warm artist-focused sessions) would materially improve all these cases.

**4. For requests with explicit decade/era constraints, demote tracks whose year contradicts the request.**
Session `b26791c3` (80s synth-pop from Breaking Bad): top result is R.E.M. "Stand (Remastered)" [1988] — borderline correct — but ranks 9–10 include "Howlin' For You" by The Black Keys [2010] and "My Own Worst Enemy" by Lit [1999]. The user asked for 80s synth-pop specifically. Non-80s tracks should be penalized. Similarly, session `ec727aa1` (mainstream 2017–2018 hit) places "Nebraska" by Lucy Rose (the played track, not mainstream) at rank 1.

**5. Treat nature sound / New Age / meditation audio as a distinct modality and suppress it unless explicitly requested.**
Session `60f60edd` is an extreme case. Nature sound recordings and ambient electronic music are completely different products. A user asking for "Clubroot-style electronic ambient" should never see rain sounds in the top 14 positions.

**6. For cold sessions with art/visual attribute queries, suppress keyword matching on track titles and rely on artist/tag similarity instead.**
The "Painting" keyword triggering Kodak Black (via the painting tag on a track that appears in a list with paintings) is a fundamental retrieval error. Visual attribute queries about album art cannot be answered by matching words in track titles.
