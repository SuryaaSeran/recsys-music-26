"""
Bucketed failure-table analysis over a provenance JSONL.

Reads the JSONL emitted by run_inference_fusion_recall_expansion.py
--write_provenance and prints:

  - n_turns and top20_hit_rate split by source bucket:
    bm25_only / artist_only / tt_only / nn_only / multi / unreachable
  - For each rescued bucket: median + 90th-pct final_rank of the gold
    (so we can see how far below top-20 the gold is sitting).
  - Aggregate dev top-20 rate (should match nDCG-pipeline Hit@20).

Usage:
    python scripts/inference/analyze_provenance.py \
        --prov exp/analysis/prov_ltr_v2.jsonl
"""
import argparse
import json
from collections import defaultdict
from statistics import median

parser = argparse.ArgumentParser()
parser.add_argument("--prov", required=True)
args = parser.parse_args()


def bucket(found_by: list[str], found_in_pool: bool) -> str:
    if not found_in_pool:
        return "unreachable"
    s = set(found_by)
    if len(s) >= 2:
        return "multi"
    if "bm25"   in s: return "bm25_only"
    if "artist" in s: return "artist_only"
    if "tt"     in s: return "tt_only"
    if "nn"     in s: return "nn_only"
    return "other"


buckets: dict[str, list[dict]] = defaultdict(list)
with open(args.prov) as f:
    for line in f:
        row = json.loads(line)
        b = bucket(row.get("found_by") or [], row.get("found_in_pool", False))
        buckets[b].append(row)

total = sum(len(v) for v in buckets.values())

print(f"\nProvenance file: {args.prov}")
print(f"Total turns: {total}\n")

order = ["bm25_only", "artist_only", "tt_only", "nn_only", "multi", "unreachable"]
print(f"{'bucket':<14} {'n':>6} {'%':>6} {'top20':>8} {'med_rank':>10} {'p90_rank':>10}")
print("-" * 60)
agg_top20 = 0
for b in order:
    rows = buckets.get(b, [])
    n = len(rows)
    if n == 0:
        print(f"{b:<14} {0:>6}")
        continue
    if b == "unreachable":
        top20 = 0
        med = None
        p90 = None
    else:
        ranks = [r.get("final_rank") for r in rows if r.get("final_rank") is not None]
        hits = sum(1 for r in rows if (r.get("final_rank") or 999) <= 20)
        top20 = hits / n
        agg_top20 += hits
        med = median(ranks) if ranks else None
        ranks_sorted = sorted(ranks)
        p90 = ranks_sorted[int(0.9 * (len(ranks_sorted) - 1))] if ranks_sorted else None
    pct = 100 * n / total
    med_s = f"{med:.0f}" if med is not None else "-"
    p90_s = f"{p90}" if p90 is not None else "-"
    print(f"{b:<14} {n:>6} {pct:>5.1f}% {top20:>7.1%} {med_s:>10} {p90_s:>10}")

print("-" * 60)
print(f"{'TOTAL':<14} {total:>6} {100.0:>5.1f}% {agg_top20/total:>7.1%}")

# Rescued-but-not-ranked subtable
print("\nRescued-but-NOT-ranked (final_rank > 20 within bucket):")
for b in ("bm25_only", "artist_only", "tt_only", "nn_only", "multi"):
    rows = buckets.get(b, [])
    if not rows:
        continue
    over = [r.get("final_rank") for r in rows
            if r.get("final_rank") is not None and r.get("final_rank") > 20]
    if not over:
        print(f"  {b}: 0 misses")
        continue
    over_sorted = sorted(over)
    p50 = median(over)
    p90 = over_sorted[int(0.9 * (len(over_sorted) - 1))]
    print(f"  {b}: {len(over)} misses  median_rank={p50:.0f}  p90_rank={p90}")
