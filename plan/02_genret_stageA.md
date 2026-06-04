# Plan: Stage A - generative candidate generator (recall)

## Context
Stage 1 gave every track a semantic ID. Now build the retrieval pipeline as
generate-wide-then-rerank-sharp. Stage A (this plan) owns RECALL: a text-conditioned
generative model decodes cf-bpr semantic-ID tuples under trie constraint, over-
generating a ~150-200 track pool. Stage B (next phase) reranks with Qwen3-Reranker-4B
+ the old LTR, fusing the generative log-prob as a prior. Stage A is the dependency:
its beam misses become Stage B's hard negatives, and its recall@pool caps the system.

Decisions: generator base = Llama-3.2-1B-Instruct (ungated mirror
`unsloth/Llama-3.2-1B-Instruct`, Meta repo is 403 for this account). Stage A only now.

## Load-bearing facts (verified)
- cf-bpr 4-tuple is 1:1 with a track (46455 distinct tuples, all singletons). So
  beam width == pool size; no bucket expansion. Beam ~200 -> ~200 candidates.
- 616 tracks have no cf-bpr (all -1) -> ungeneratable cold tail. On dev that is
  130/8000 gold turns -> hard recall ceiling = 0.9838. LOGGED KNOWN GAP; planned
  patch = content-path union (a second recall path over a content branch), NOT in v1.
- Train: 15199 sessions x 8 = 121592 music turns, 100% gold have cf-bpr (fully
  trainable). Dev: 1000 sessions, 8000 turns, gold present.
- Compute MPS only; train float32 (bf16/fp16 flaky on MPS 2.11).

## Token scheme
Add 1024 tokens `<cf_{L}_{c}>` (L 0..3, c 0..255; per-level offset so a code is a
distinct token per level) + structural specials `<hist>`, `</hist>`, `<gen>`.
Resize embeddings. Llama ties input/output embeddings; train only the new rows
(+ LoRA on attention), base frozen via a gradient mask on the embedding weight.

## Prompt / target
Per music turn, condition on strictly-prior turns:
```
profile: <age_group>, <country>, <gender>, <lang>, <culture>
goal: <category> / <specificity> / <listener_goal>
<hist> <cf tokens of each prior in-session music turn> </hist>
user: ...   (last up-to-3 user/assistant text turns, query last)
assistant: ...
<gen>
```
Target (loss only): the gold track's 4 cf tokens + eos. History on by default.

## Trie + decode
CfTrie over the 46455 valid 4-tuples in TOKEN space. `prefix_allowed_tokens_fn`
walks the suffix after `<gen>`; depth 4 -> eos. Decoded tuple -> track via a
cf-only `tuple->track` map built from per_modality_codes.npy cols 0:4 (NOT
codes_to_tracks.json, which is keyed on the full 16-tuple).

## Generation
transformers generate(): beam=200, num_return_sequences=200, max_new_tokens=5,
prefix_allowed_tokens_fn=trie, return_dict_in_generate + output_scores. Per-candidate
score = length-normalized sequence log-prob (for Stage B fusion). Diverse/group beam
and prefix-bucket expansion are flags, off by default. EARLY: probe beam=200 decode
latency on MPS before full training (heavy KV cache x200).

## Training
Custom MPS loop. peft LoRA (q,k,v,o proj, r=16, alpha=32). New-token embed rows
trainable via grad-mask hook (zero grad on old rows; verify old rows bit-identical
after a step). CE on target tokens only. AdamW lr 2e-4, batch 8 x accum 4, 3 epochs.
Overfit-200 sanity first (top-1 tuple acc -> high). Checkpoint per epoch, select on
dev-subsample recall@200.

## Eval (the only gate)
recall@pool {20,50,100,200} over 8000 dev turns = gold track in expanded pool.
Report global + slices: cf-absent set (the 130, = ceiling 0.9838), low-popularity
cold slice, by turn position 1..8. Also exact-tuple top-1/top-10. Write
exp/genret/eval/report.json. Headline: recall@200 vs 0.9838 ceiling.

## Files
```
src/genret/config.py  tokens.py  data.py  trie.py  train.py  generate.py  eval.py
scripts/genret_build_data.py  genret_train.py  genret_eval.py
```
Outputs gitignored: exp/genret/{data,ckpt,eval}/.

## Risks
- MPS: float32 only; beam=200 KV-cache memory -> batch dev generation small. Custom
  loop avoids HF Trainer MPS edge cases. Partial-row embed training via grad hook is
  the load-bearing trick; if wrong, base vocab drifts.
- Near-unique tuples: one wrong token = miss; over-generation (beam 200) is the recall
  lever. Safety net (off by default): prefix-bucket expansion at 2-tuple level.
- Cold tail (1.6% dev) accepted for the milestone, logged, patched later via content
  union. Do not let it silently become the permanent ceiling.
```
