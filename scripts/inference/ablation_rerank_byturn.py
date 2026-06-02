"""Per-turn ablation: does the reranker help or hurt at each turn position?

Computes flat nDCG@20 by turn-number on the blind-sim dev-100 set for the LTR
baseline vs several rerank configs (built from saved score-logs + one 4B output).
"""
import argparse, json, math
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--spec", default="plan/DEV_BLINDSIM_100.json")
ap.add_argument("--baseline", default="exp/inference/devset/blindsim100v2_baseline_top100.json")
ap.add_argument("--scores_short", default="exp/inference/devset/blindsim100v2_qwen06b_short_scores.json")
ap.add_argument("--scores_full", default="exp/inference/devset/blindsim100v2_qwen06b_full_scores.json")
ap.add_argument("--out4b_full", default="exp/inference/devset/blindsim100v2_qwen4b_k50.json",
                help="4B full-history alpha=0 k50 output (pred file).")
ap.add_argument("--split", default="test")
ap.add_argument("--k", type=int, default=20)
args = ap.parse_args()

spec = json.load(open(args.spec))
want = {(s["session_id"], s["turn_number"]) for s in spec}
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")[args.split]
sids = {s["session_id"] for s in spec}
gold = {}
for it in ds:
    if it["session_id"] not in sids:
        continue
    for t in it["conversations"]:
        if t["role"] == "music" and (it["session_id"], t["turn_number"]) in want:
            gold[(it["session_id"], t["turn_number"])] = t["content"]


def ndcg(ids, g, k):
    top = ids[:k]
    return 1.0 / math.log2(top.index(g) + 2) if g in top else 0.0


def blend_order(entries, alpha, rk):
    head = entries[:rk]
    n = len(head)
    if n == 0:
        return []
    scs = [e[1] for e in head]
    lo, hi = min(scs), max(scs); span = (hi - lo) or 1.0
    rk_ = sorted(range(n), key=lambda i: alpha*((n-i)/n) + (1-alpha)*((scs[i]-lo)/span), reverse=True)
    return [head[i][0] for i in rk_]


# Build per-config ordered-id lookups keyed by (sid,tn)
baseline = {(p["session_id"], p["turn_number"]): p["predicted_track_ids"]
            for p in json.load(open(args.baseline))}
short = json.load(open(args.scores_short))
full = json.load(open(args.scores_full))
out4b = {(p["session_id"], p["turn_number"]): p["predicted_track_ids"]
         for p in json.load(open(args.out4b_full))}

configs = {
    "baseline (LTR)":        lambda key: baseline.get(key, []),
    "4B full a0 k50":        lambda key: out4b.get(key, []),
    "0.6B short a0 k50":     lambda key: blend_order(short.get(f"{key[0]}|{key[1]}", []), 0.0, 50),
    "0.6B short a0.5 k50":   lambda key: blend_order(short.get(f"{key[0]}|{key[1]}", []), 0.5, 50),
    "0.6B short a0.7 k20":   lambda key: blend_order(short.get(f"{key[0]}|{key[1]}", []), 0.7, 20),
}

# accumulate per-turn
turns = sorted({s["turn_number"] for s in spec})
per_turn = {name: {t: [] for t in turns} for name in configs}
overall = {name: [] for name in configs}
for s in spec:
    key = (s["session_id"], s["turn_number"])
    g = gold.get(key)
    if g is None:
        continue
    for name, fn in configs.items():
        nd = ndcg(fn(key), g, args.k)
        per_turn[name][s["turn_number"]].append(nd)
        overall[name].append(nd)

def mean(xs): return sum(xs)/len(xs) if xs else 0.0

# print table
ncfg = list(configs)
print(f"{'turn (n)':>10} | " + " | ".join(f"{c:>18}" for c in ncfg))
print("-" * (12 + 21*len(ncfg)))
for t in turns:
    n = len(per_turn[ncfg[0]][t])
    row = [f"{mean(per_turn[c][t]):.4f}" for c in ncfg]
    print(f"  t{t} ({n:>2}) | " + " | ".join(f"{v:>18}" for v in row))
print("-" * (12 + 21*len(ncfg)))
print(f"{'ALL (100)':>10} | " + " | ".join(f"{mean(overall[c]):>18.4f}" for c in ncfg))

# delta vs baseline per turn for the strongest rerank (4B)
print("\nDelta vs baseline (4B full a0 k50):")
for t in turns:
    b = mean(per_turn['baseline (LTR)'][t]); r = mean(per_turn['4B full a0 k50'][t])
    flag = "HELP" if r > b + 1e-9 else ("HURT" if r < b - 1e-9 else "flat")
    print(f"  t{t}: base={b:.4f}  4B={r:.4f}  delta={r-b:+.4f}  {flag}")
