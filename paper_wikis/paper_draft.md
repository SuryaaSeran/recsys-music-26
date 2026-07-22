# Multi-Turn Preference Elicitation in Dialogue-Aware Music Recommendation: Fused Retrieval and Learned Ranking for the TalkPlayData Challenge

**Authors:** FirstName Surname; FirstName Surname; FirstName Surname
Department Name, Institution/University Name, City, State, Country

---

## ABSTRACT

We present a conversational music recommender for the TalkPlayData Challenge (ACM
RecSys 2026) that treats each dialogue turn's user reactions as a preference-
elicitation signal for both retrieval and ranking. The dataset's goal-progress
feedback is delayed by one turn and includes rejected tracks logged as if they
were positives; we align the signal in time, remove rejected-track contamination
from training, and reconstruct it from raw dialogue when it is withheld at test
time. This signal drives a role-tagged dialogue anchor -- encoding prior turns,
recommendations, and per-turn reactions -- behind a LoRA fine-tune of a
279M-parameter multilingual encoder, the system's only trained neural component.
Retrieval around this anchor is a union of nine inexpensive sources arbitrated by
a gradient-boosted learning-to-rank model over 67 features, with multi-source
agreement the strongest single feature. The full pipeline trains and serves on a
single consumer desktop with no discrete GPU, and reaches a composite score of
0.49 on the blind generalization set.

## CCS CONCEPTS

- Information systems -> Recommender systems; Learning to rank; Language models.

## KEYWORDS

multi-turn preference elicitation, dialogue-aware recommendation,
conversational recommendation, music recommendation, learning to rank, dense
retrieval, LoRA fine-tuning, CPU-only inference

---

## 1 Introduction

The TalkPlayData Challenge frames music recommendation as a multi-turn dialogue.
At each turn the system must (a) rank 20 tracks from a catalog of 47,071 so that
the single logged next track ranks high, and (b) produce the assistant's reply.
Submissions are scored by a composite metric:

> Score = 0.50 * nDCG@20 + 0.10 * CatalogDiversity + 0.10 * LexicalDiversity + 0.30 * JudgeNorm,
> JudgeNorm = (judge_score - 1) / 4.

The judge is a language model that reads only the response text (personalization
and explanation quality); it never sees the track list. The blind generalization
set ("Blind B") is harder than the labeled data: the conversation goal, the
per-turn goal-progress assessments, and the model's internal thoughts are removed,
and half of its 80 sessions are cold-start with no user profile.

Our system has three stages. *Recall* runs nine retrievers per turn and merges
their outputs into one deduplicated pool (mean ~3,300 candidates); no single
source exceeds ~59% gold recall at 500 candidates, but the fused pool reaches
~89% on the blind-simulated holdout. *Rescore* is a LightGBM LambdaMART model that
scores every candidate from 67 features and emits the top 20, replacing hand-tuned
score fusion. *Respond* is decoupled: because the judge reads only text, the
response may describe the ranked list but never reorders it. All three stages run
on one Apple M4 machine (16 GB unified memory, no GPU), and the only trained
neural component is the LoRA encoder of Section 3.

**Contributions.** (1) A treatment of the goal-progress feedback signal that is
off-by-one in time and contaminated with rejected tracks: we align it, decontaminate
it, and reconstruct it from raw dialogue when it is withheld at test time. (2) A
structured, role-tagged dialogue anchor that encodes prior turns, recommendations,
and per-turn reactions as explicit fields, paired with a priority-ordered
hard-negative recipe that recycles confirmed session rejections as training
negatives. (3) A dialogue-aware retriever: a LoRA fine-tune of `multilingual-e5-base`
over that anchor, combining (1) and (2) into a single trained artifact. (4) A recall
design that treats retrieval as *set union, not weighted fusion*: each source
contributes candidates and per-source features, and a learned ranker decides how far
to trust each; multi-source agreement is the single strongest feature.

## 2 Recall Architecture

Recall is a union of nine retrievers (Table 1). Each candidate carries the
per-source scores and origin flags that later become ranker features. The design
rule is to *add candidates, never re-weight sources by hand*: even low-precision
sources help, because source membership feeds the agreement feature of Section 4.

**Table 1: The nine recall sources.**

| # | Source | Budget | Signal |
|---|---|---|---|
| 1 | BM25 (Okapi) | top-500 | lexical match on name/artist/album/tags |
| 2 | Two-tower ANN | top-2000 | fine-tuned conversational-intent match |
| 3 | Last-track NN | 100 x last 2 | "more like what just played" |
| 4 | Artist expansion | discographies | tracks by any retrieved/mentioned artist |
| 5 | Qwen3-0.6B NN | top-500 | second dense opinion, different space |
| 6 | CF-BPR | top-200 | collaborative signal, warm users only |
| 7 | Session-mean NN | top-100 | neighbors of the session taste centroid |
| 8 | Co-occurrence | 300/150/50 | tracks that followed recent history |
| 9 | SASRec buckets | cap 300 | next semantic-bucket expansion (optional) |

*Lexical (source 1).* BM25 (via `bm25s`) indexes each track's name, artist, album,
and tag list; indexing the tags matters because conversational requests use
genre/mood/era vocabulary that only tags carry. The query concatenates the goal,
culture, the last four played tracks, and the last four dialogue turns (the
current request among them); older turns are excluded because they dilute the
current request's IDF mass.

*Dense (sources 2, 5).* Two independent dense retrievers give complementary
spaces. Source 2 is our fine-tuned two-tower encoder (Section 3): the dialogue
anchor is encoded once and matched against a precomputed 47,071 x 768
L2-normalized matrix by exact brute-force matmul (top-2000); at this catalog size
an approximate index is unnecessary and only loses recall. Source 5 is the
competition-provided `Qwen3-Embedding-0.6B` metadata embedding, whose different
pretraining recovers gold tracks the two-tower misses; both cosines become ranker
features.

*Neighborhood (sources 3, 7).* Query-side retrieval is complemented by expansion
around the history: the nearest neighbors of the last two played tracks (source 3)
and of the mean two-tower embedding of the last four history tracks, a recent
taste centroid (source 7). Both are cheap matmuls against the same matrix; rejected history tracks
are filtered from these seeds first (Section 3.3).

*Catalog structure (sources 4, 8, 9).* Artist expansion (source 4) is a
dictionary lookup that adds the full, popularity-capped discography of every
retrieved or mentioned artist; it is the cheapest high-recall source and matters
because sessions are often artist deep-dives. Co-occurrence (source 8) is a count
table of tracks that most often followed the last one/two/three history tracks in
training sessions. Semantic-bucket expansion (source 9) is an optional auxiliary:
an RQ-VAE assigns each track a semantic ID and a small SASRec predicts the next
bucket, but next-bucket prediction has a low ceiling (hit@3 = 42.1%), so an
eight-source configuration without it reaches parity.

*Collaborative (source 6).* The competition-provided BPR matrix-factorization
embeddings add each warm user's top-200 tracks; cold sessions simply receive nothing from this source and the
pipeline degrades gracefully.

## 3 Dialogue-Aware Retriever

Off-the-shelf dense retrieval failed here: conversational queries do not embed
near track-metadata text in general-purpose spaces, so dense recall requires
fine-tuning on (dialogue, gold-track) pairs.

### 3.1 Model and anchor

The base encoder is `intfloat/multilingual-e5-base` (XLM-RoBERTa, 768-dim,
512-token window, 279M parameters), chosen for its context length (an earlier
256-token model truncated the user's request on 81% of anchors). We fine-tune with
LoRA (rank 32, alpha 64) on the query/key/value projections: 1.8M trainable
parameters, 0.64% of the model; full fine-tuning does not fit in 16 GB, LoRA plus
gradient checkpointing does. The query anchor is *role-tagged* so the encoder sees
dialogue structure:

```
query: [PROFILE] {age}.{country}.{gender}.{culture} [GOAL] {goal}
[T1] USER:{q1} | REC:{track-artist} | ASST:{r1} | REACTION:liked/rejected
... [NOW] USER:{current request}
```

A tokenizer-aware greedy builder always keeps a fixed core (profile, goal, latest
request) and inserts history turns most-recent-first within a 510-token budget; the
current request can never be truncated. Documents use the symmetric form
`passage: {name} by {artist} | Album: {album} | Tags: {tags} | {year}`, with the
E5 `query:`/`passage:` prefixes identical across training, index build, and
inference.

### 3.2 Training and serving

We train with MultipleNegativesRankingLoss (InfoNCE with in-batch negatives) plus
two explicit hard negatives per row (Section 3.3): 3 epochs, effective
batch 32, learning rate 1e-4. Training uses ~46K anchor/positive pairs (plus a
7K-pair validation split), excluding the 6,000 sessions reserved for the ranker's
feature dump. At serving time we ship
a 7 MB LoRA adapter applied to the cached base encoder, and query a one-pass
47,071 x 768 track index by brute-force matmul.

### 3.3 Using the goal-progress feedback signal

Each training turn carries a `goal_progress_assessment` (GPA): a label recording
whether the *previous* recommendation moved the listener toward their goal, i.e. a
per-recommendation accept/reject signal. It is the richest preference feedback in
the dataset and the trickiest part of the system, because the raw label cannot be
used as-is and it is removed entirely from the blind set.

**First, align it in time.** The GPA at turn *T* is the reaction to the track
recommended at turn *T-1*: the system acts at one turn and the user responds at the
next. A naive reading attaches each reaction to the wrong track, so we shift every
GPA back one turn (`turn_number - 1`). This alignment is a prerequisite for
everything below.

**Second, stop rewarding rejected tracks.** After alignment, 43.2% of turns
carry a gold track the user *rejected* (`DOES_NOT_MOVE_TOWARD_GOAL`). Training
these as ordinary positives teaches the models to rank rejected tracks highly.
The ranker *drops* such turns entirely (a rejected track is not a valid
positive). The retriever additionally recycles every rejected track as a *hard
negative* for the rest of that session, exactly the near-miss the encoder
should learn to push away: negatives are drawn in priority order from confirmed
session rejections, then BM25 near-misses, then same-artist distractors, then
in-batch negatives, and an accepted track is never sampled as a negative.

**Third, act on the feedback at inference.** The aligned GPA drives three
behaviors: the history-based retrievers (sources 3, 7 and the BM25 query) are
seeded only from *accepted* tracks, so a streak of rejections does not fill the
pool with neighbors of refused tracks; the ranker receives features contrasting a
candidate against the accepted- and rejected-history centroids (Section 4); and
the anchor's goal slot is replaced by the most recently accepted track ("more like
the last thing that worked").

**Finally, reconstruct it when missing.** The blind set removes GPA, so we infer
it from the dialogue: a rule-based classifier labels each user follow-up accept or
reject from keyword banks (accept: "love it", "more like this"; reject: "not
what", "something different"). The inferred label fills the anchor's `REACTION`
slot and drives the same seed filtering and features. On the blind-simulated
holdout the system scores as well with this proxy as with the true label, which is
what makes it viable.

## 4 Learning-to-Rank Rescoring

The rescorer is a LightGBM LambdaMART model (nDCG@20 objective) that scores every
pooled candidate and returns the top 20. It replaces hand-tuned linear fusion and
is the pivotal choice: it *decouples recall from precision*, absorbs heterogeneous
features (cosines, ranks, count tables, metadata, session counters), and handles
extreme imbalance natively (one positive per ~3,300 candidates) via pairwise
gradients within each turn's group. Training runs the full retrieval in dump mode
on 6,000 held-out sessions, yielding 27,166 groups and 89.7M rows; the 8.2% of
groups whose gold track the pool missed carry no positive and are dropped,
leaving 24,951 training groups. The model has 67 features (5-fold session CV
nDCG@20 = 0.3144) and is a 0.6 MB file. The features fall into six groups (Table 2). Group B dominates:
multi-source agreement accounts for roughly 57% of total model gain and is the
feature that transfers best when goal and GPA are withheld on the blind set.

**Table 2: Rescorer feature groups (67 features).**

| Group | Features |
|---|---|
| A. Retrieval scores | two-tower cosine and rank; Qwen3 metadata and lyrics cosines; CLAP audio cosine; CF cosine; BM25 reciprocal rank; co-occurrence rank |
| B. Source agreement | `n_sources` and transforms: count of sources returning the candidate; a de-facto ensemble vote |
| C. Track metadata | popularity percentile; years since release; request-tag overlap |
| D. Session state (GPA) | cosine to accepted- and to rejected-history centroids; artist-in-rejected-set; accepted/rejected turn counts; consecutive rejections; CF distance to history |
| E. Intent extraction | era/genre/mood keywords parsed from the request; candidate era/genre match; album continuity; within-artist popularity and transition ranks; bucket-match counts |
| F. Sequential ID | SASRec bucket-origin and rank (optional configuration) |

## 5 Results

Table 3 lists only the changes that moved the development nDCG@20 (1,000
sessions). The two largest single jumps are non-model changes: excluding
already-played tracks from BM25 (+0.035), and replacing linear fusion with a
learned ranker, which unlocked every later feature. The goal-progress fixes of
Section 3.3 account for the final step to 0.1864. On the blind sets the same
system scores nDCG@20 = 0.3997 (Blind A) and composite 0.49 (Blind B).

**Table 3: Development nDCG@20 after each load-bearing change.**

| Change | nDCG@20 |
|---|---|
| BM25 over name + artist + album | 0.0861 |
| + index the tag list | 0.0960 |
| + exclude already-played tracks | 0.1313 |
| + fine-tuned two-tower, linear fusion | 0.1418 |
| + multi-source pool expansion | 0.1533 |
| + LightGBM LambdaMART replaces linear fusion | 0.1609 |
| + feature engineering, larger two-tower pool | 0.1684 |
| + E5 role-tagged dialogue anchor | 0.1748 |
| + GPA-aware inference and intent features | 0.1864 |

### 5.1 What We Tried and Why It Didn't Work

Several directions that looked promising on paper were dropped after measurement.
We report them because the reasons they failed are more informative than the
numbers in Table 3 alone.

**Reranking on top of LambdaMART.** Every attempt to re-rank the LambdaMART
output with a second, more "semantic" model underperformed the LambdaMART-only
baseline (Table 3a). A pre-trained cross-encoder with no TalkPlay fine-tuning
underperformed outright. A fine-tuned cross-encoder (bge-reranker-v2-m3,
pool-aware negatives) regressed to dev nDCG@20 = 0.1228, a large drop -- but the
cause was not the reranking idea itself, it was trained on a stale `[TURN-N]`
anchor format that no longer matched the shipped anchor once the encoder moved to
the v8d format, a data mismatch rather than a modeling failure. A local
Gemma-3-12b model doing conservative top-25-to-20 pruning regressed slightly
(0.0812 vs. 0.0818 baseline on a 50-session pilot). The clearest case was a
zero-shot Qwen3-8B reranker applied to the Blind A submission: nDCG@20 fell from
0.3990 to 0.3310, lexical diversity collapsed from 0.5909 to 0.0125, the judge
score fell from 1.90 to 1.15, and composite fell from 0.3291 to 0.1811. The
intuition that unifies all four cases: none of these rerankers had access to the
structural signal LambdaMART uses instead -- explicit multi-source agreement and
GPA-derived session-state features (Table 2, groups B and D) -- so an unfine-tuned
or mismatched model instead gravitates toward generically "safe" or near-identical
tracks across sessions, a failure mode invisible to nDCG@20 alone but catastrophic
for lexical diversity and judge quality. The retained system (LambdaMART only, no
added reranker) is the positive case here: its composite score beats every
reranked variant we tried.

**Table 3a: Reranking attempts on top of the LambdaMART baseline.**

| Attempt | Result | Verdict |
|---|---|---|
| Pre-trained cross-encoder (no fine-tune) | underperformed the LambdaMART baseline | no domain adaptation to TalkPlay dialogue/track text |
| Fine-tuned cross-encoder (bge-reranker-v2-m3) | dev nDCG@20 = 0.1228 (large regression) | trained on a stale anchor format, mismatched with the shipped encoder |
| Gemma-3-12b conservative pruning (25→20) | 0.0812 vs. 0.0818 baseline | regression |
| Zero-shot Qwen3-8B reranker | Blind A nDCG@20 0.3990→0.3310, lex. div. 0.5909→0.0125, judge 1.90→1.15, composite 0.3291→0.1811 | diversity collapse |

**Hard negatives that taught the wrong lesson.** An earlier two-tower iteration
(v4) trained with `MultipleNegativesRankingLoss` and BM25-top-5 hard negatives
regressed dev nDCG@20 to 0.1364, worse than the 0.1418 reached without hard
negatives at all. Our reading: BM25-top-5 negatives are lexically close to the
positive but not necessarily wrong in intent, so the loss pushed the encoder to
separate on superficial lexical features rather than dialogue intent. This result
is why the shipped model (Section 3.2) mines hard negatives in a strict priority
order -- confirmed session rejections first, then BM25 near-misses, then
same-artist distractors, then in-batch negatives -- and never treats a raw BM25
top-K result as a hard negative on its own.

**Development and blind generalization can disagree.** Adding progress-aware
retrieval and ranking (H1/H3, Section 3.3) to an intermediate two-tower model
(v8b) dropped dev nDCG@20 from 0.1684 to 0.1615, a regression on the development
metric, while Blind A nDCG rose to 0.3701, a gain over the prior no-GPA baseline.
Judged on dev alone, this change would have been reverted. We instead adopted two
working rules from this case: a feature whose importance swings by more than 50%
across the 5 cross-validation folds is flagged as unstable and treated with
suspicion, and a regression on the blind set is weighed as costing more than the
sum of all dev-set gains, since blind sessions -- with goal, GPA, and thoughts
removed -- are the harder, more representative target.

### 5.2 Behavior Across Goal-Category Types

Not everything trains the same behavior. Each session's goal falls into one of
11 categories (Table 3b); dev nDCG@20 for the shipped configuration ranges from
0.1326 (Category I) to 0.1953 (Category F). The weakest four categories (I, C,
K, J) are all narrow, exact-match requests; the strongest three (E, F, H) are
broad or multi-faceted requests.

**Table 3b: Dev nDCG@20 by goal category.**

| Cat | Type | nDCG@20 |
|---|---|---|
| I | specific globally popular song | 0.1326 |
| C | specific album by description | 0.1332 |
| K | discover, multi-instrumental, broad era | 0.1367 |
| J | specific hit | 0.1585 |
| G | multi mood | 0.1605 |
| A | specific instrumental (game/movie) | 0.1639 |
| D | specific soundtrack | 0.1653 |
| B | specific by lyric phrase | 0.1807 |
| H | multi artists | 0.1884 |
| E | musical journey, multi | 0.1945 |
| F | multi era/genre | 0.1953 |

A concrete pair from the dev set shows why. In one Category I session the
listener asks for "a catchy, dance-y track that was big in the 2000s, possibly
from a non-English speaking country" with no title or artist to anchor
retrieval; the system recommends "How Soon Is Now?" by The Smiths, then "Abra
Cadaver" by The Hives -- both alternative/punk rock, both rejected ("it's not
quite what I'm looking for... definitely had a Latin feel to it"). With no
lexical anchor, retrieval falls back to the listener's broader taste profile
(alternative rock) instead of the request's actual content (Latin pop,
danceable). In a Category F session, by contrast, the listener names the
artist directly ("Black Keys songs... from 'Thickfreakness' or 'Rubber
Factory'"); the system returns "Thickfreakness" on the first turn, the
listener confirms it is "absolutely what I had in mind," and the goal-progress
label reads MOVES_TOWARD_GOAL for the rest of the session. The difference is
not specificity in the abstract -- both requests are specific -- it is whether
the request supplies a lexical anchor, an artist or title, that BM25 and the
dialogue anchor can lock onto.

This also complicates the multi-source pool-expansion story of Section 2:
expansion helps ten of the eleven categories but actively hurts Category I
(-0.0039 dev nDCG vs. the simpler two-source baseline), because sources
calibrated to mainstream, taste-profile-driven sessions add noise exactly
where there is no anchor to filter against. More sources is not uniformly
better.

## 6 Efficiency and Scalability

Every stage is CPU-native; Table 4 (top) reports per-component latencies with all
neural blocks forced to CPU. Exact brute-force similarity over the 47,071 x 768
matrix is a single matmul, so no approximate index is needed at this catalog
scale, and every stage except the encoder passes is single-digit milliseconds.
Measured end-to-end over the 200-session evaluation set, the full recall-plus-
rescore pipeline runs at **2.17 s per session, about 0.27 s per turn** (Table 4,
bottom). Of this, the rescore stage is negligible: scoring the ~3,300-candidate
pool with LambdaMART takes ~3.4 ms, roughly 1% of turn latency, so recall (the nine
sources and their two query-encoder passes) accounts for essentially all of it.
Even with every neural block pinned to the CPU (Table 4, top), a full turn stays
under one second; dropping the frozen Qwen3-0.6B encoder in the eight-source
configuration removes the single largest cost. The trained artifact is a 24 MB
LoRA adapter and the serving process fits in under ~2 GB of RAM, so the entire
train/evaluate/iterate loop, including the one LoRA fine-tune, runs on a single
16 GB desktop, which is what made rapid iteration practical.

**Table 4: Measured latency and footprint. Top: per component, all neural blocks
forced to CPU (worst case), median over 20 runs. Bottom: end-to-end recall +
rescore over the 200-session set, encoders on the Metal backend.**

| Operation | Cost |
|---|---|
| Two-tower query encode (e5-base) | 47 ms |
| Exact ANN, 47,071 x 768, top-2000 | 2.2 ms |
| Qwen3-0.6B query encode | 601 ms |
| BM25 retrieve, top-500 | 0.5 ms |
| Rescore: LambdaMART, 3,100 x 67 features | 3.4 ms |
| Serving memory, all components loaded | 1.85 GB |
| Trained artifact (LoRA adapter) | 24 MB |
| Recall stage, per turn (nine sources) | ~0.26 s |
| Rescore stage, per turn | 3.4 ms |
| End-to-end, per session (per turn) | 2.17 s (0.27 s) |

## 7 Conclusion

Treating goal-progress feedback as a delayed, sometimes-withheld
preference-elicitation signal -- aligning it in time, removing rejected-track
contamination, and reconstructing it from dialogue when absent -- feeds directly
into recall, ranking, and the dialogue-aware retriever's anchor. Fusing many
inexpensive retrievers, letting a gradient-boosted ranker arbitrate them, and
adding a single LoRA-adapted dialogue encoder built around this signal together
reach composite 0.49 on the blind set, entirely on one desktop with no discrete
GPU.

## ACKNOWLEDGMENTS

Insert acknowledgments here.

## REFERENCES

[1] Wang, L. et al. 2024. Multilingual E5 Text Embeddings: A Technical Report. arXiv:2402.05672.

[2] Ke, G. et al. 2017. LightGBM: A Highly Efficient Gradient Boosting Decision Tree. In NeurIPS.

[3] Burges, C. J. C. 2010. From RankNet to LambdaRank to LambdaMART: An Overview. Microsoft Research Technical Report MSR-TR-2010-82.

[4] Hu, E. J. et al. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In ICLR.

[5] Rendle, S. et al. 2009. BPR: Bayesian Personalized Ranking from Implicit Feedback. In UAI.

[6] Kang, W.-C. and McAuley, J. 2018. Self-Attentive Sequential Recommendation. In ICDM.

[7] Robertson, S. and Zaragoza, H. 2009. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval 3, 4.
