"""
Build a mixed-turn dev blind-sim spec from the 1000 test (dev) sessions.

One (session_id, turn_number) pair per session. For each session the
"blind turn" is chosen as follows:
  - Default (--mode last): the LAST eligible turn in the session.
  - --mode random: a uniformly random eligible turn.

Eligible: turns where progress_by_turn[T] == "MOVES_TOWARD_GOAL", i.e.
the recommendation at turn T was explicitly confirmed as moving toward goal.

gpa semantics (from TalkPlayData spec):
  - Turn 1 gpa is always null: no prior rec existed to assess.
  - gpa at dataset turn T assesses the rec made at turn T-1.
  - After re-keying (T -> T-1): progress_by_turn[T] = assessment of rec at T.
  - Eligible turns are 1-7 (turn 8 has no gpa_9, so never MOVES_TOWARD_GOAL).

Usage:
    python scripts/analysis/build_dev_blindsim_mixed.py \
        --out plan/DEV_BLINDSIM_MIXED.json
"""
import argparse, json, random
from collections import Counter
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="plan/DEV_BLINDSIM_MIXED.json")
ap.add_argument("--mode", choices=["last", "random"], default="last",
                help="'last' = last eligible turn per session (blind-A-like); "
                     "'random' = random eligible turn")
ap.add_argument("--seed", type=int, default=42)
args = ap.parse_args()

random.seed(args.seed)

ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="test")

spec = []
turn_dist = Counter()
skipped = 0

for item in ds:
    sid = item["session_id"]
    # gpa re-keyed: progress_by_turn[T] = assessment for rec at T
    progress = {a["turn_number"] - 1: a["goal_progress_assessment"]
                for a in (item.get("goal_progress_assessments") or [])}

    music_turns = sorted(
        t["turn_number"] for t in item["conversations"] if t["role"] == "music"
    )

    # A turn is eligible only when its recommendation was explicitly
    # confirmed as MOVES_TOWARD_GOAL. Turn 8 never qualifies (no gpa_9).
    eligible = [
        tn for tn in music_turns
        if progress.get(tn) == "MOVES_TOWARD_GOAL"
    ]

    if not eligible:
        skipped += 1
        continue

    if args.mode == "last":
        chosen = eligible[-1]
    else:
        chosen = random.choice(eligible)

    spec.append({"session_id": sid, "turn_number": chosen})
    turn_dist[chosen] += 1

random.shuffle(spec)

with open(args.out, "w") as f:
    json.dump(spec, f, indent=2)

print(f"Wrote {len(spec)} session/turn pairs to {args.out}")
print(f"Skipped {skipped} sessions with no eligible turn")
print("Turn distribution:", dict(sorted(turn_dist.items())))
