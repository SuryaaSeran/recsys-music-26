"""Stage A evaluation: recall@pool over dev, with ceiling / cold / position slices."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _recall(records, k, key=lambda r: True):
    sub = [r for r in records if key(r)]
    if not sub:
        return None
    return round(float(np.mean([r["hit"][k] for r in sub])), 4), len(sub)


def evaluate(retriever, examples, pool_sizes=(20, 50, 100, 200), num_beams=None,
             diverse=False, max_examples=None, seed=0):
    from src.tracks import load_catalog
    cat = load_catalog()

    ex = list(examples)
    if max_examples and len(ex) > max_examples:
        rng = np.random.default_rng(seed)
        ex = [ex[i] for i in rng.choice(len(ex), max_examples, replace=False)]

    pmax = max(pool_sizes)
    records = []
    for j, e in enumerate(ex):
        gold = e["gold_track_id"]
        pool = retriever.generate_pool(e["context"], pool_size=pmax, num_beams=num_beams,
                                       diverse=diverse) if e["gold_has_cf"] or True else []
        ids = [c.track_id for c in pool]
        rank = ids.index(gold) + 1 if gold in ids else None
        hit = {k: bool(rank and rank <= k) for k in pool_sizes}
        gcf = e.get("gold_cf")
        firsts = {c.cf_tuple[0] for c in pool}
        prefix2 = {c.cf_tuple[:2] for c in pool}
        records.append({
            "hit": hit,
            "rank": rank,
            "gold_has_cf": e["gold_has_cf"],
            "turn": e["turn_number"],
            "pop": cat[gold].popularity if gold in cat else 0.0,
            "first_in_pool": bool(gcf and gcf[0] in firsts),
            "prefix2_in_pool": bool(gcf and tuple(gcf[:2]) in prefix2),
            "top1": bool(rank == 1),
            "top10": bool(rank and rank <= 10),
        })
        if (j + 1) % 50 == 0:
            print(f"  eval {j+1}/{len(ex)}")

    pop_thresh = float(np.percentile([r["pop"] for r in records], 25))
    rep = {
        "n": len(records),
        "pool_sizes": list(pool_sizes),
        "recall_global": {k: _recall(records, k) for k in pool_sizes},
        "recall_generatable": {k: _recall(records, k, lambda r: r["gold_has_cf"]) for k in pool_sizes},
        "recall_cold": {k: _recall(records, k, lambda r: r["pop"] <= pop_thresh) for k in pool_sizes},
        "ceiling": round(float(np.mean([r["gold_has_cf"] for r in records])), 4),
        "diagnostics": {
            "gold_first_token_in_pool": round(float(np.mean([r["first_in_pool"] for r in records])), 4),
            "gold_prefix2_in_pool": round(float(np.mean([r["prefix2_in_pool"] for r in records])), 4),
            "exact_top1": round(float(np.mean([r["top1"] for r in records])), 4),
            "exact_top10": round(float(np.mean([r["top10"] for r in records])), 4),
        },
        "recall200_by_turn": {
            int(t): _recall(records, max(pool_sizes), lambda r, t=t: r["turn"] == t)
            for t in sorted({r["turn"] for r in records})
        },
        "cold_pop_threshold": pop_thresh,
    }
    return rep, records


def write_report(rep, out_dir="exp/genret/eval"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(rep, indent=2))
    return out / "report.json"
