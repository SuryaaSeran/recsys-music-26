"""Build a 100-session blind-A-simulating dev validation set.

Blind A evaluates ONE turn per session, heavily weighted to early turns
(turn 1 = 25%). All-turn dev eval misaligns with this. This set picks 100
dev (test-split) sessions and assigns each a single eval turn whose
distribution mirrors blind A, including 25 turn-1 (cold-start / single-turn)
cases.

Sampled from the first `--pool` dev sessions (default 200) so an existing
top-100 candidate dump covers them for reranking experiments.

Output: plan/DEV_BLINDSIM_100.json = [{"session_id":..., "turn_number":...}, ...]
"""
import argparse, json, random
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="plan/DEV_BLINDSIM_100.json")
ap.add_argument("--pool", type=int, default=200,
                help="Sample from the first N dev sessions (candidate dump coverage).")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

# Blind A predicted-turn distribution (80 sessions) scaled to 100:
#   turn: 1  2  3  4  5  6  7  8
#   blind:20 15 10  5  8  9  8  5
#   x1.25:25 19 13  6 10 11 10  6   (sums to 100)
TURN_COUNTS = {1: 25, 2: 19, 3: 13, 4: 6, 5: 10, 6: 11, 7: 10, 8: 6}
assert sum(TURN_COUNTS.values()) == 100

random.seed(args.seed)
ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")["test"]
pool = [ds[i] for i in range(min(args.pool, len(ds)))]

# Per-session MOVES_TOWARD_GOAL turn set. Turn 1 is always null (session start),
# so it is treated as the cold-start single-turn case and is always eligible.
# Turns 2-8 are only eligible as eval targets if MOVES_TOWARD_GOAL (clean positive).
moves_turns = {}
for s in pool:
    mt = {a["turn_number"] for a in (s.get("goal_progress_assessments") or [])
          if a.get("goal_progress_assessment") == "MOVES_TOWARD_GOAL"}
    moves_turns[s["session_id"]] = mt

all_ids = [s["session_id"] for s in pool]
random.shuffle(all_ids)

spec = []
used = set()
for turn, count in TURN_COUNTS.items():
    picked = 0
    for sid in all_ids:
        if picked >= count:
            break
        if sid in used:
            continue
        eligible = (turn == 1) or (turn in moves_turns.get(sid, set()))
        if not eligible:
            continue
        spec.append({"session_id": sid, "turn_number": turn})
        used.add(sid)
        picked += 1
    if picked < count:
        raise RuntimeError(f"turn {turn}: only {picked}/{count} eligible sessions "
                           f"in first {args.pool} (need a larger --pool)")

random.shuffle(spec)
with open(args.out, "w") as f:
    json.dump(spec, f, indent=2)

import collections
dist = collections.Counter(s["turn_number"] for s in spec)
print(f"Wrote {len(spec)} session/turn pairs to {args.out}")
print("turn distribution:", dict(sorted(dist.items())))
print(f"single-turn (turn 1, cold start): {dist[1]}")
print(f"distinct sessions: {len(set(s['session_id'] for s in spec))}")
