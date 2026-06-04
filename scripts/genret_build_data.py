"""Build Stage A train/dev examples.

  python scripts/genret_build_data.py
Train = one terminal example per session (Blind A turn-depth histogram).
Dev   = music turns with goal_progress_assessment == MOVES_TOWARD_GOAL.
"""
import argparse
import json
import pathlib
import sys
from collections import Counter

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from transformers import AutoTokenizer

from src.genret.config import BASE_MODEL, GenRetConfig
from src.genret.data import (build_dev_examples, build_train_examples, load_cf_map,
                             load_split)
from src.genret.tokens import SemTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="exp/genret/data")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = GenRetConfig()
    out = pathlib.Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    sem = SemTokenizer(tok)
    cf_map = load_cf_map()

    kw = dict(with_history=cfg.with_history, max_recent_turns=cfg.max_recent_turns)

    train_df = load_split("train")
    train = build_train_examples(train_df, sem, cf_map, tok, seed=a.seed,
                                 max_ctx_tokens=cfg.max_ctx_tokens, **kw)
    with open(out / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")

    dev_df = load_split("test")
    dev = build_dev_examples(dev_df, sem, cf_map, moves_only=True, **kw)
    with open(out / "dev.jsonl", "w") as f:
        for ex in dev:
            f.write(json.dumps(ex) + "\n")

    # quick stats
    def tn_hist(rows):
        return dict(sorted(Counter(r["turn_number"] for r in rows).items()))

    train_tn = Counter()
    for ex in train:
        # recover turn count is not stored; skip (terminal sampling already logged below)
        pass
    dev_cf = sum(r["gold_has_cf"] for r in dev)
    print(json.dumps({
        "train_examples": len(train),
        "dev_examples": len(dev),
        "dev_turn_hist": tn_hist(dev),
        "dev_gold_has_cf": dev_cf,
        "dev_cf_absent": len(dev) - dev_cf,
        "dev_recall_ceiling": round(dev_cf / len(dev), 4) if dev else None,
    }, indent=2))


if __name__ == "__main__":
    main()
