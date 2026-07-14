# A CPU-Scale Conversational Music Recommender: Fused Retrieval and Learned Ranking for the TalkPlayData Challenge

*(Word-ready draft. Headings map to the ACM `acmart` template styles: the block
below the title is `Abstract`, `CCS CONCEPTS`, `KEYWORDS`, then numbered
`Head1`/`Head2` sections. Paste each block into the corresponding styled
paragraph of the interim ACM template.)*

**Authors:** FirstName Surname; FirstName Surname; FirstName Surname
Department Name, Institution/University Name, City, State, Country

---

## ABSTRACT

We present a conversational music recommender for the TalkPlayData Challenge
(ACM RecSys 2026) that ranks tracks and generates the assistant response at each
turn of a multi-turn dialogue, and that trains and serves in its entirety on a
single consumer desktop with no discrete GPU. The system has three stages:
(i) a *recall* stage that fuses nine inexpensive retrievers into a candidate pool
of roughly three thousand tracks; (ii) a *rescore* stage that arbitrates the pool
with a gradient-boosted learning-to-rank model over 67 per-candidate features;
and (iii) a decoupled *response* stage. The only trained neural component is a
LoRA adaptation of a 279M-parameter multilingual text encoder, fine-tuned on
(dialogue, next-track) pairs with a role-tagged dialogue anchor. We show that
correctly handling the dataset's goal-progress feedback signal, which is both
off-by-one in time and contaminated with user-rejected tracks, is the single
highest-value engineering step, and that the same feedback channel can be
reconstructed from raw dialogue text when it is withheld at test time. On the
blind generalization set our system reaches a composite score of 0.49.

## CCS CONCEPTS

- Information systems -> Recommender systems; Learning to rank; Language models.
- Computing methodologies -> Learning latent representations.

## KEYWORDS

conversational recommendation, music recommendation, learning to rank, dense
retrieval, LoRA fine-tuning, preference elicitation, CPU-scale systems

---

## 1 Introduction

The TalkPlayData Challenge frames music recommendation as a multi-turn dialogue.
At each turn the system must (a) rank 20 tracks from a catalog of 47,071 such
that the single logged next track ranks high, and (b) produce the assistant's
natural-language reply. Submissions are scored by a composite metric:

> Score = 0.50 * nDCG@20 + 0.10 * CatalogDiversity + 0.10 * LexicalDiversity + 0.30 * JudgeNorm,
> with JudgeNorm = (judge_score - 1) / 4.

The judge is a large language model that reads only the response text along two
disclosed axes (personalization, explanation quality); it never sees the track
list. With one relevant item per turn, nDCG@20 reduces to 1/log2(rank+1) when the
gold track appears in the top 20 and 0 otherwise.

The blind generalization set ("Blind B") is deliberately harder than the
labeled data: the conversation goal, the per-turn goal-progress assessments, and
the model's internal "thoughts" are all removed, and half of its 80 sessions are
cold-start with no user identifier or profile.

This paper describes the retrieval-and-ranking system behind our best blind
result (composite 0.49). We focus on three components: the fused recall
architecture and the exact construction of each retriever (Section 3); the
LoRA-adapted dialogue encoder and the goal-progress feedback fix that made it
work (Section 4); and the learning-to-rank rescorer and its full feature set
(Section 5). Section 6 gives a compact account of the iterative gains, Section 7
the efficiency profile, and Section 8 concludes.

**Contributions.**

1. A recall design that treats retrieval as *set union, not weighted fusion*:
   nine independent retrievers each contribute candidates and per-source
   features, and a learned ranker decides how much to trust each. The strongest
   single feature is multi-source agreement.
2. A dialogue-aware retriever: a LoRA fine-tune of `multilingual-e5-base` on a
   role-tagged anchor that encodes prior turns, recommendations, and per-turn
   user reactions.
3. A treatment of the goal-progress feedback signal, which we show is off-by-one
   in time and contaminated with rejected tracks, together with a test-time
   reconstruction of that signal from raw dialogue when it is withheld.

## 2 System Overview

The pipeline is three sequential stages.

**Recall.** Nine retrievers run per turn and their outputs are merged into one
deduplicated candidate pool (mean ~3,100 tracks). No single retriever exceeds
~59% gold recall at 500 candidates; the fused pool reaches ~89% on the
blind-simulated holdout.

**Rescore.** A LightGBM LambdaMART model scores every pooled candidate from 67
features and emits the top 20. This learned ranker replaces hand-tuned score
fusion and lets any retriever contribute candidates without a manually assigned
weight.

**Respond.** Because the judge reads only text, response generation is decoupled
from ranking: the response may describe the ranked list but never reorders it.
This stage is out of scope here.

All three stages run on one consumer machine (Apple M4, 16 GB unified memory, no
discrete GPU). The only trained neural component is the LoRA encoder of Section 4.

## 3 Recall Architecture

Recall is a union of nine retrievers (Table 1). Each candidate carries the
per-source scores and origin flags that later become ranker features. The design
rule is to *add candidates, never re-weight sources by hand*: even
low-precision sources help because source membership feeds the agreement feature
of Section 5.

**Table 1: The nine recall sources.**

| # | Source | Budget | Signal |
|---|---|---|---|
| 1 | BM25 (Okapi) | top-500 | lexical match on name/artist/album/tags |
| 2 | Two-tower ANN | top-2000 | fine-tuned conversational-intent match |
| 3 | Last-track NN | 100 x last 2 | "more like what just played" |
| 4 | Artist expansion | full discographies | tracks by any retrieved/mentioned artist |
| 5 | Qwen3-0.6B NN | top-500 | second dense opinion, different embedding space |
| 6 | CF-BPR | top-200 | collaborative signal, warm users only |
| 7 | Session-mean NN | top-100 | neighbors of the session taste centroid |
| 8 | Co-occurrence | 300/150/50 | tracks that followed recent history tracks |
| 9 | SASRec buckets | cap 300 | next semantic-bucket expansion (optional) |

### 3.1 Lexical retrieval (source 1)

BM25 (Okapi, via `bm25s`) indexes each track's name, artist, album, and tag
list. Indexing the *tag list* is important: conversational requests use
genre/mood/era vocabulary ("something mellow and acoustic") that only the tags
carry. The query is assembled from the latest user message, the stated goal, the
listener culture, the last four played tracks (name/artist/tags), and the last
four dialogue turns. Turns older than four are deliberately excluded: adding them
dilutes the inverse-document-frequency mass of the current request. Turns whose
vocabulary misses the index are smoothed by a score floor (0.05) so they are not
dropped from the pool.

### 3.2 Dense retrieval (sources 2, 5)

Two independent dense retrievers provide complementary embedding spaces.

*Source 2* is our fine-tuned two-tower encoder (Section 4). At query time the
dialogue anchor is encoded once and compared to a precomputed matrix of 47,071
track embeddings (768-dim, L2-normalized). Retrieval is an exact brute-force
matmul followed by a top-2000 partial sort; at this catalog size an approximate
index (FAISS/HNSW) is unnecessary and would only lose recall.

*Source 5* is the competition-provided `Qwen3-Embedding-0.6B` metadata
embedding. The track side is precomputed; the query side is encoded by the frozen
0.6B model with the model's instruction prefix, then matched by exact matmul
(top-500). Because it is a different pretraining distribution, it recovers gold
tracks that the two-tower misses, and both cosines become ranker features.

### 3.3 Neighborhood expansion (sources 3, 7)

Dense retrieval on the *query* is complemented by dense expansion around the
*history*. Source 3 takes the last two played tracks and adds each track's 100
nearest neighbors in two-tower space. Source 7 computes the mean two-tower
embedding of the session's history tracks (the session taste centroid) and adds
its 100 nearest neighbors. Both are cheap matmuls against the same track matrix
as source 2. When history contains user-rejected tracks these seeds are filtered
first (Section 4.3).

### 3.4 Catalog-structure expansion (sources 4, 8, 9)

These sources exploit structure in the catalog and in training sessions rather
than the query text.

*Artist expansion (source 4)* is a dictionary lookup: for every artist retrieved
by another source or named in the dialogue, add that artist's full discography
(capped by popularity). It is the cheapest high-recall source in the system (no
model), and it matters because TalkPlay sessions are frequently artist
deep-dives.

*Co-occurrence (source 8)* is a count table built from training sessions: for the
last one, two, and three history tracks it adds the tracks that most often
followed them (top 300/150/50). The table is built with all feature-dump
sessions excluded to avoid leakage (Section 4.3).

*Semantic-bucket expansion (source 9)* is optional. A two-level RQ-VAE (64
codes/level) over track attribute embeddings assigns each track a semantic ID; a
small SASRec model predicts the next bucket from the session's bucket sequence,
and members of the top-3 predicted buckets join the pool. Because next-bucket
prediction has a low ceiling (hit@3 = 42.1%), this source is a capped auxiliary
only, and an eight-source configuration without it reaches parity.

### 3.5 Collaborative signal (source 6)

Source 6 is a Bayesian Personalized Ranking (BPR-MF) model over user-track
interactions, restricted to warm users; it adds each user's top-200 tracks. Cold
sessions simply receive no candidates from this source and the pipeline degrades
gracefully.

## 4 Dialogue-Aware Retriever: A LoRA-Adapted E5 Encoder

Off-the-shelf dense retrieval failed on this task: conversational queries do not
embed near track-metadata text in general-purpose spaces. Dense retrieval here
requires fine-tuning on (dialogue, gold-track) pairs. The remaining problem is
context length and label quality.

### 4.1 Model and anchor construction

The base encoder is `intfloat/multilingual-e5-base` (XLM-RoBERTa, 12 layers,
768-dim, 512-token window, 279M parameters), chosen for its 512-token context
(an earlier 256-token model truncated the user's request on 81% of anchors). We
fine-tune with LoRA (rank 32, alpha 64, dropout 0.05) on the query/key/value
projections: 1.8M trainable parameters, 0.64% of the model. Full fine-tuning does
not fit in 16 GB of unified memory; LoRA plus gradient checkpointing does.

The query anchor is *role-tagged* so the encoder sees explicit dialogue
structure:

```
query: [PROFILE] {age} . {country} . {gender} . {culture} . {language}
[GOAL] {listener_goal} ({specificity})
[T1] USER: {q1} | REC: {track - artist} | ASST: {r1} | REACTION: liked/rejected
...
[NOW] USER: {current request}
```

A tokenizer-aware greedy builder always includes a fixed core (profile, goal,
latest request) and then inserts history turns most-recent-first, each only if it
fits a 510-token budget, with per-turn fallbacks (full -> short -> minimal). The
user's current request can never be truncated; only ~5% of anchors reach the
budget. Documents use the symmetric form
`passage: {name} by {artist} | Album: {album} | Tags: {tags} | {year}`. The
`query:`/`passage:` prefixes match E5 pretraining and must be identical across
training, index build, and inference.

### 4.2 Training and use

We train with MultipleNegativesRankingLoss (InfoNCE with in-batch negatives) plus
two to three explicit hard negatives per row (Section 4.3), for 3 epochs at
effective batch 32 (batch 8, gradient accumulation 4), learning rate 1e-4, 200
warmup steps, with gradient checkpointing. Training data is ~74K anchor/positive
pairs drawn from the 15,199 training sessions, excluding the 6,000 sessions
reserved for the ranker's feature dump.

At serving time we ship the LoRA adapter (a 7 MB `safetensors` file, 24 MB with
its tokenizer) and apply it to the base encoder loaded from cache; the adapter
can optionally be merged into a standalone encoder. The track index is one
forward pass over the catalog, stored as a 47,071 x 768 float32 matrix
(146.5 MB) and queried by brute-force matmul.

### 4.3 The goal-progress assessment fix

The dataset labels each turn with a `goal_progress_assessment` (GPA). Using it
correctly was the highest-value work in the project, and it required three
separate corrections.

**(a) The feedback is off by one turn.** The GPA at turn *T* is the listener's
verdict on the recommendation made at turn *T-1* (the action is taken at *t*, the
reaction is recorded at *t+1*). Any GPA-conditioned component that indexes turn
*T* directly is training on a label that judges the *previous* turn. All builders
now index `turn_number - 1`.

**(b) 43% of "positive" tracks were user-rejected.** In the ranker's training
dump, 43.2% of turns carry a gold track labeled `DOES_NOT_MOVE_TOWARD_GOAL`: a
track the logged system played and the user rejected. Training these as positives
teaches the model to rank rejected tracks highly. For the ranker we drop such
turns entirely (a rejected gold is not a usable positive). For the retriever we
remove them from positive pairs and *recycle the rejected tracks as
session-level hard negatives*: same intent context, explicitly refused. The
hard-negative priority per anchor is confirmed session rejections (up to 2), then
BM25-hard negatives for specific-goal sessions (up to 2), then artist-repeat
distractors (1), then in-batch negatives, with a protection rule that never
samples a user-accepted track as a negative.

**(c) Co-occurrence leakage.** The co-occurrence table (source 8) was first built
on sessions that overlapped the ranker feature dump. Gold tracks had
co-occurrence hits in 81.9% of leaky turns versus 28.1% of clean turns: a 3x
inflated feature that would collapse on blind data. Rebuilding the table with all
6,000 dump sessions excluded removes the leak.

**GPA at inference.** Beyond clean labels, GPA drives three inference behaviors.
*Seed filtering (H1)* removes rejected tracks from the history seeds that feed
sources 3, 7, and the BM25 query, so a run of rejections does not seed the pool
with tracks the user just refused. *Session-state features (H2)* summarize
accepted-versus-rejected history for the ranker (Section 5). *Goal substitution
(H3)* replaces the goal text in the anchor with the most recently accepted track
("more like the last thing that worked").

**Reconstruction on the blind set.** Blind B removes GPA entirely. We reconstruct
it from dialogue text: a rule-based classifier maps each user follow-up to accept
or reject via two small regex banks (acceptance: "love it", "perfect", "more like
this"; rejection: "not what", "something different", "too slow"). The inferred
label fills the anchor's `REACTION` slot and drives H1 and H2; goal substitution
fills the emptied `[GOAL]` slot. This reconstruction is noisier than true GPA but,
measured on the blind-simulated holdout, loses nothing relative to using the real
labels.

## 5 Learning-to-Rank Rescoring

The rescorer is a LightGBM LambdaMART model (`lambdarank` objective, nDCG@20)
that scores every pooled candidate and returns the top 20. It replaces hand-tuned
linear fusion and is the pivotal architectural choice: it *decouples recall from
precision*, absorbs heterogeneous feature types (cosines, reciprocal ranks, count
tables, metadata, session counters), and handles extreme class imbalance natively
(one positive per ~3,100 candidates) because LambdaMART forms pairwise gradients
within each turn's group.

**Training data.** The full nine-source retrieval is run in dump mode on 6,000
held-out training sessions, writing one row per (turn, candidate): 24,718 groups,
77.7M rows, positive rate 0.00035. Splits are at the session level, and the same
6,000 sessions are excluded from two-tower training and the co-occurrence table.

**Hyperparameters.** 31 leaves, learning rate 0.08, L2 0.1, minimum sum Hessian
0.1, path smoothing 1.0, feature and bagging fraction 0.8, truncation level 30,
5-fold session-level cross-validation with early stopping (75 rounds). The
production model has 67 features (cross-validated nDCG@20 = 0.3144) and is a
0.6 MB text file.

The 67 features fall into six groups (Table 2).

**Table 2: Rescorer feature groups.**

| Group | Features | Construction |
|---|---|---|
| A. Retrieval scores | two-tower cosine and a rank sigmoid; Qwen3 metadata and lyrics cosines; CLAP audio cosine; CF cosine; BM25 reciprocal rank; co-occurrence rank signal | the raw per-source votes, read directly from each retriever's output for the candidate |
| B. Multi-source agreement | `n_sources`, `log1p(n_sources)`, `n_sources/7` | count of independent sources that returned the candidate; a de-facto ensemble vote |
| C. Track metadata | popularity percentile; years since release; tag overlap with the request | from static catalog metadata and a set intersection of request tokens with track tags |
| D. Session state (from GPA/H2) | cosine to the mean embedding of accepted history; cosine to the mean embedding of rejected history; artist-in-rejected-set flag; counts of accepted and rejected turns; consecutive-rejection counter; CF distance to last and mean history | computed per candidate against the accepted/rejected partitions of the session history |
| E. Intent extraction (Tier 1) | era/genre/mood/instrument keywords parsed from the request; candidate era and genre match; album continuity (same album as last track, album seen in recent window); within-artist popularity and transition ranks; semantic-bucket match counts | keyword extraction over the request text matched against candidate tags and release year, plus per-artist ranking statistics |
| F. Sequential-ID (optional) | SASRec bucket-origin and bucket-rank features | present only in the nine-source configuration |

Group B is by far the most important: multi-source agreement accounts for roughly
57% of total model gain. It is also the feature that transfers best across
distributions, which is why the system is robust when goal and GPA are withheld
on the blind set.

## 6 Iterative Improvements

Rather than a full narrative, Table 3 lists only the changes that moved the
metric, with the resulting development nDCG@20 (1,000 sessions).

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

The two largest single jumps are non-model changes: excluding already-played
tracks (+0.035) and replacing linear fusion with a learned ranker (that unlocked
every later feature). The GPA fixes of Section 4.3 account for the final step to
0.1864. On the blind sets the same system scores nDCG@20 = 0.3997 (Blind A) and
composite 0.49 (Blind B).

## 7 Efficiency

Every stage is CPU-native. Exact brute-force similarity over the 47,071 x 768
matrix is one matmul (median 2.2 ms), so no approximate index is needed at this
catalog scale. Per-turn neural cost is dominated by the two encoder forward
passes; the classical stages are single-digit milliseconds (BM25 0.5 ms, ranking
3,100 candidates 3.4 ms). The trained artifact is a 24 MB LoRA adapter; the full
serving process fits under ~2 GB of RAM. The complete train/evaluate/iterate loop,
including the one LoRA fine-tune, runs on a single 16 GB consumer machine, which
is what made rapid iteration feasible.

## 8 Conclusion

A conversational music recommender does not require a GPU cluster or a
per-candidate neural reranker to be competitive. Fusing many inexpensive
retrievers and letting a gradient-boosted ranker arbitrate them, with a single
LoRA-adapted dialogue encoder for dense recall, reaches a composite of 0.49 on the
blind generalization set while training and serving on one desktop. The largest
gains came not from model capacity but from data hygiene: excluding seen tracks,
learning the fusion instead of tuning it, and correctly handling a goal-progress
feedback signal that is off-by-one in time, contaminated with rejected tracks, and
absent at test time.

## ACKNOWLEDGMENTS

Insert acknowledgments here.

## REFERENCES

[1] Wang, L. et al. Multilingual E5 Text Embeddings. (`intfloat/multilingual-e5-base`).

[2] Ke, G. et al. 2017. LightGBM: A Highly Efficient Gradient Boosting Decision
Tree. In NeurIPS.

[3] Burges, C. J. C. 2010. From RankNet to LambdaRank to LambdaMART: An Overview.
Microsoft Research Technical Report.

[4] Hu, E. J. et al. 2022. LoRA: Low-Rank Adaptation of Large Language Models. In
ICLR.

[5] Rendle, S. et al. 2009. BPR: Bayesian Personalized Ranking from Implicit
Feedback. In UAI.

[6] Kang, W.-C. and McAuley, J. 2018. Self-Attentive Sequential Recommendation
(SASRec). In ICDM.

[7] Robertson, S. and Zaragoza, H. 2009. The Probabilistic Relevance Framework:
BM25 and Beyond. Foundations and Trends in Information Retrieval.
