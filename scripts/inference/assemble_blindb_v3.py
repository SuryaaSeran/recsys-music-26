"""Assemble Blind B v3 from the 2-round Opus prune/deepen loop.

Round 1: top-25 reviewed, drop high+medium-confidence mismatches.
Round 2: short sessions (kept<20) vet ranks 26-100, Opus returns keeps.
Assembly: final 20 = round1_kept (rank order) + round2_keeps (priority order),
          dedup, then PAD to exactly 20 with highest-LTR remaining if still short.

Outputs:
  exp/inference/blind_b/blind_b_v8d_s3cap_v3.json   (prediction, 20 ids each)
  wiki/BLIND_B_v3_final_submission.md               (human-readable, per session)
"""
import json
from pathlib import Path
from datasets import load_dataset, concatenate_datasets

ROOT = Path(__file__).resolve().parents[2]

pred100 = {x["session_id"]: x for x in json.load(open(ROOT / "exp/inference/blind_b/blind_b_v8d_s3cap_100.json"))}
order = [x["session_id"] for x in json.load(open(ROOT / "exp/inference/blind_b/blind_b_v8d_s3cap_100.json"))]

# Round 1 decisions (high+medium drops)
r1 = {}
for b in range(4):
    for d in json.load(open(f"/tmp/blindb_drops_batch{b}.json")):
        r1[d["session_id"]] = d

# Round 2 decisions (deepening keeps for short sessions)
r2 = {}
for b in range(4):
    p = Path(f"/tmp/blindb_round2_decisions_batch{b}.json")
    if p.exists():
        for d in json.load(open(p)):
            r2[d["session_id"]] = d

# Metadata + responses
meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
allt = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
meta = {r["track_id"]: r for r in allt}
def line(t):
    r = meta.get(t, {})
    return (f'{(r.get("track_name") or ["?"])[0]} - {(r.get("artist_name") or ["?"])[0]} '
            f'[{str(r.get("release_date","") or "")[:4]}] {", ".join((r.get("tag_list") or [])[:5])}')

v2resp = {x["session_id"]: x["predicted_response"]
          for x in json.load(open(ROOT / "exp/inference/blind_b/blind_b_v8d_s3cap_v2.json"))}

out_pred = []
report = ["# Blind B v3 — Final Submission (2-round Opus prune+deepen)", ""]
n_padded = 0
n_deepened = 0
for sid in order:
    ids100 = pred100[sid]["predicted_track_ids"]
    drops_hm = {dr["rank"] for dr in r1.get(sid, {}).get("drops", []) if dr.get("confidence") in ("high", "medium")}
    r1_ranks = [r for r in range(1, 26) if r not in drops_hm]      # kept ranks from top-25
    prov = {}                                                       # rank -> provenance
    for r in r1_ranks:
        prov[r] = "kept(top25)"
    final_ranks = list(r1_ranks)

    if len(final_ranks) < 20 and sid in r2:
        n_deepened += 1
        for r in r2[sid].get("keeps", []):
            if r not in final_ranks and 26 <= r <= 100:
                final_ranks.append(r)
                prov[r] = "added(R2)"
            if len(final_ranks) >= 20:
                break

    # Pad to exactly 20 with highest-LTR remaining ranks (even if flagged)
    if len(final_ranks) < 20:
        for r in range(1, 101):
            if r not in final_ranks:
                final_ranks.append(r)
                prov[r] = "PAD(least-bad)"
            if len(final_ranks) >= 20:
                break
        n_padded += 1

    final_ranks = final_ranks[:20]
    final_ids = [ids100[r - 1] for r in final_ranks]
    assert len(final_ids) == 20 and len(set(final_ids)) == 20, sid

    # Response: keep v2 unless rank-1 track changed
    resp = v2resp.get(sid, "")
    rank1_changed = (final_ids[0] != pred100[sid]["predicted_track_ids"][0])

    out_pred.append({
        "session_id": sid, "user_id": pred100[sid]["user_id"],
        "turn_number": pred100[sid]["turn_number"],
        "predicted_track_ids": final_ids, "predicted_response": resp,
        "_rank1_changed": rank1_changed,
    })

    intent = r1.get(sid, {}).get("user_intent", "")
    note2 = r2.get(sid, {}).get("notes", "") if sid in r2 else ""
    report.append(f"## `{sid[:8]}` t{pred100[sid]['turn_number']}"
                  + (" — RANK1 CHANGED (response will be regenerated)" if rank1_changed else ""))
    report.append(f"**Intent:** {intent}  ")
    if note2:
        report.append(f"**Deepen note:** {note2}  ")
    report.append("")
    report.append("| slot | rank | track | source |")
    report.append("|---|---|---|---|")
    for i, r in enumerate(final_ranks, 1):
        report.append(f"| {i} | {r} | {line(ids100[r-1])} | {prov.get(r,'?')} |")
    report.append("")

# Strip helper key for the actual prediction file
clean = [{k: v for k, v in p.items() if not k.startswith("_")} for p in out_pred]
json.dump(clean, open(ROOT / "exp/inference/blind_b/blind_b_v8d_s3cap_v3.json", "w"), indent=2, ensure_ascii=False)

changed = [p["session_id"] for p in out_pred if p["_rank1_changed"]]
json.dump(changed, open("/tmp/blindb_v3_rank1_changed.json", "w"))

hdr = [f"**Sessions deepened (round 2): {n_deepened}** | **padded with least-bad: {n_padded}** | "
       f"**rank-1 changed (need response regen): {len(changed)}**", ""]
Path(ROOT / "wiki/BLIND_B_v3_final_submission.md").write_text("\n".join(hdr + report))
print(f"v3 assembled. deepened={n_deepened} padded={n_padded} rank1_changed={len(changed)}")
print(f"rank1-changed sessions: {[s[:8] for s in changed]}")
