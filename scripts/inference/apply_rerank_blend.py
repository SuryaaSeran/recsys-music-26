"""Apply a chosen (alpha, rerank_k) blend to saved rerank scores -> prediction file.

Companion to sweep_rerank_blend.py (which only evaluates). This writes an actual
predicted_track_ids ordering so a tuned config can be turned into a submission
without rerunning the model.

Usage:
    python scripts/inference/apply_rerank_blend.py \
        --pred  exp/inference/blind_a/blind_a_phd_top100.json \
        --scores exp/inference/blind_a/blind_a_phd_qwen4b_short_scores.json \
        --alpha 0.3 --rerank_k 30 --final_k 20 \
        --out exp/inference/blind_a/blind_a_phd_qwen4b_tuned.json
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--pred", required=True, help="Original top-N pred (for tail + fields).")
ap.add_argument("--scores", required=True, help="score-log from rerank_qwen3 --save_scores")
ap.add_argument("--alpha", type=float, required=True)
ap.add_argument("--rerank_k", type=int, default=50)
ap.add_argument("--final_k", type=int, default=20)
ap.add_argument("--out", required=True)
ap.add_argument("--response", default=None,
                help="If set, overwrite predicted_response with this string.")
args = ap.parse_args()

score_log = json.load(open(args.scores))   # "sid|tn" -> [[tid, score], ...] LTR order
preds = json.load(open(args.pred))


def blended(entries, alpha, rk):
    head = entries[:rk]
    n = len(head)
    if n == 0:
        return []
    scs = [e[1] for e in head]
    lo, hi = min(scs), max(scs)
    span = (hi - lo) or 1.0
    scored = []
    for i, (tid, sc) in enumerate(head):
        ltr_s = (n - i) / n
        rr_s = (sc - lo) / span
        scored.append((alpha * ltr_s + (1 - alpha) * rr_s, tid))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [tid for _, tid in scored]


out = []
changed = 0
for p in preds:
    key = f"{p['session_id']}|{p['turn_number']}"
    ent = score_log.get(key)
    if ent is None:
        new_ids = p["predicted_track_ids"][: args.final_k]
    else:
        scored_ids = {e[0] for e in ent[: args.rerank_k]}
        head = blended(ent, args.alpha, args.rerank_k)
        # tail = original-order candidates beyond the scored head
        tail = [t for t in p["predicted_track_ids"] if t not in scored_ids]
        new_ids = (head + tail)[: args.final_k]
        if new_ids != p["predicted_track_ids"][: args.final_k]:
            changed += 1
    rec = {
        "session_id": p["session_id"],
        "user_id": p.get("user_id", ""),
        "turn_number": p["turn_number"],
        "predicted_track_ids": new_ids,
        "predicted_response": args.response if args.response is not None
                              else p.get("predicted_response", ""),
    }
    out.append(rec)

json.dump(out, open(args.out, "w"), ensure_ascii=False, indent=2)
print(f"Wrote {len(out)} preds to {args.out} "
      f"(alpha={args.alpha}, rerank_k={args.rerank_k}); order changed on {changed}.")
