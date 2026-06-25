"""Merge 4 Opus drop-decision batches and render a human-readable review report.

Inputs:  /tmp/blindb_drops_batch{0..3}.json  (Opus decisions)
         /tmp/blindb_full_review.json        (sessions + 25 candidates w/ metadata)
Outputs: wiki/BLIND_B_v3_drop_review.md       (human-readable, per session)
         /tmp/blindb_drops_merged.json        (machine-readable, for applying)
"""
import json
from pathlib import Path

review = {s["session_id"]: s for s in json.load(open("/tmp/blindb_full_review.json"))}
order = [s["session_id"] for s in json.load(open("/tmp/blindb_full_review.json"))]

dec = {}
for b in range(4):
    p = Path(f"/tmp/blindb_drops_batch{b}.json")
    if not p.exists():
        print(f"WARNING: missing {p}")
        continue
    for d in json.load(open(p)):
        dec[d["session_id"]] = d

json.dump([dec.get(sid, {"session_id": sid, "drops": []}) for sid in order],
          open("/tmp/blindb_drops_merged.json", "w"), indent=2)

conf_rank = {"high": 0, "medium": 1, "low": 2}
lines = ["# Blind B v3 — Drop Review (Opus, all 80 sessions, 25 candidates each)", ""]
tot_drops = 0
by_conf = {"high": 0, "medium": 0, "low": 0}
sess_with_drops = 0

for i, sid in enumerate(order, 1):
    s = review[sid]
    d = dec.get(sid, {})
    drops = {dr["rank"]: dr for dr in d.get("drops", [])}
    if drops:
        sess_with_drops += 1
    for dr in d.get("drops", []):
        tot_drops += 1
        by_conf[dr.get("confidence", "low")] = by_conf.get(dr.get("confidence", "low"), 0) + 1

    cold = "COLD" if s["cold"] else "warm"
    lines.append(f"## {i}. `{sid[:8]}` — turn {s['turn_number']} ({cold})")
    lines.append(f"**Intent (Opus):** {d.get('user_intent','-')}  ")
    lines.append(f"**List quality:** {d.get('list_quality','-')}  ")
    # user messages
    umsgs = [c["text"] for c in s["conversation"] if c.get("role") == "user"]
    if umsgs:
        lines.append(f"**User's last message:** {umsgs[-1][:240]}")
    if len(umsgs) > 1:
        lines.append(f"**Earlier asks:** " + " | ".join(m[:90] for m in umsgs[:-1][-3:]))
    lines.append("")
    if drops:
        lines.append(f"**Proposed drops: {len(drops)}** (ranks " +
                     ", ".join(str(r) for r in sorted(drops)) + ")")
    else:
        lines.append("**Proposed drops: none (list judged clean)**")
    lines.append("")
    lines.append("| # | Track | Year | Tags | Verdict |")
    lines.append("|---|---|---|---|---|")
    for c in s["candidates_25"]:
        r = c["rank"]
        tags = ", ".join(c.get("tags", [])[:6])
        td = f"{c['title']} — {c['artist']}"
        if r in drops:
            dr = drops[r]
            verdict = f"**DROP [{dr.get('confidence','?')}]** {dr.get('reason','')}"
        else:
            verdict = "keep"
        lines.append(f"| {r} | {td} | {c.get('year','')} | {tags} | {verdict} |")
    lines.append("")

header = [
    f"**Summary:** {sess_with_drops}/80 sessions have proposed drops, {tot_drops} drops total "
    f"(high={by_conf.get('high',0)}, medium={by_conf.get('medium',0)}, low={by_conf.get('low',0)}).",
    "",
    "How to use: scan each session. Drops are proposals only — nothing is applied yet.",
    "Tell me which to accept (e.g. 'high-confidence only', or per-session rank lists) and",
    "I will remove them, backfill from ranks 21-25, and rebuild the submission.",
    "",
]
out = Path("wiki/BLIND_B_v3_drop_review.md")
out.write_text("\n".join(header + lines))
print(f"Wrote {out}  ({sess_with_drops} sessions w/ drops, {tot_drops} drops: {by_conf})")
