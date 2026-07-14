# Plan: Paper writeup — v2 Blind B system technical account + CPU benchmarks

## Goal

Write `wiki/PAPER_SYSTEM_ACCOUNT.md`: full technical narrative of the system that
produced the best Blind B submission (v2, composite 0.49), from BM25 baseline to
the final recall+rescore architecture, with CPU-oriented scalability analysis and
measured benchmarks. Source material for a paper.

## Current State

- v2 = v8d recall (9 sources) + 67-feat s3cap LTR + Claude pivot-aware responses.
- History is spread across plan/PLAN.md, plan/archive/*, wiki/BLIND_B.md,
  CURRENT_BEST_ITERATION.md, memory feedback.
- No consolidated benchmark table exists.

## Assumptions

- Hardware: Apple M4 Mac mini, 16GB unified memory (per plan/archive/07). Verify.
- Inference is CPU-friendly: encoders are small (279M e5-base LoRA-merged,
  0.6B Qwen3), ANN is brute-force numpy matmul, LTR is LightGBM. SASRec loader
  defaults to mps but is a tiny model. Benchmarks forced to device=cpu.
- Wall-times from logs: golden-200 = 7:14 (2.17 s/session); dev-1000 ~35-40 min.

## Files To Read

- plan/PLAN.md, plan/CURRENT_BEST_ITERATION.md [done]
- plan/archive/01,02,05,07,08,10 [done]
- wiki/BLIND_B.md [done]
- TRAIN_EVAL_DATA.md [done]
- memory feedback.md [done]

## Files To Modify

- NEW: scripts/analysis/bench_cpu_components.py (micro-benchmarks)
- NEW: wiki/PAPER_SYSTEM_ACCOUNT.md (the deliverable)
- plan/PLAN.md (add phase pointer) — optional

## Steps

1. Write bench_cpu_components.py: CPU-forced latency for TT encode, ANN matmul,
   Qwen3 encode, BM25 retrieve, LightGBM score, artifact sizes, RSS.
2. Run it; capture numbers.
3. Collect artifact sizes (du) + hardware spec (sysctl).
4. Write PAPER_SYSTEM_ACCOUNT.md: story, recall, rescore, training, responses,
   scalability section with benchmark tables.
5. Commit + push.

## Validation

- Every number in the doc traces to a measured run, a log, or a ledger entry.
- Benchmarks reproducible via the committed script.

## Risks

- Bash classifier outage delays benchmark runs (work around by preparing scripts).
- CLAP text-encode benchmark heavy to load; skip if it stalls.

## Notes

- Focus: everything trains and serves on one consumer machine; no CUDA GPU.
  Training used MPS (Apple unified memory); inference components run on CPU.
  Be precise about this distinction in the paper doc.
