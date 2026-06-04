"""Evaluate the Stage A generator: dev recall@pool with slices.

  python scripts/genret_eval.py --ckpt exp/genret/ckpt --max-examples 1000
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.genret.config import BASE_MODEL
from src.genret.eval import evaluate, write_report
from src.genret.generate import GenRetriever


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="exp/genret/ckpt")
    ap.add_argument("--base", default=BASE_MODEL)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--data", default="exp/genret/data/dev.jsonl")
    ap.add_argument("--max-examples", type=int, default=1000)
    ap.add_argument("--num-beams", type=int)
    ap.add_argument("--diverse", action="store_true")
    ap.add_argument("--out-dir", default="exp/genret/eval")
    a = ap.parse_args()

    examples = [json.loads(l) for l in open(a.data)]
    retr = GenRetriever.load(a.ckpt, a.base, a.device)
    rep, _ = evaluate(retr, examples, num_beams=a.num_beams, diverse=a.diverse,
                      max_examples=a.max_examples)
    path = write_report(rep, a.out_dir)
    print(json.dumps(rep, indent=2))
    print("wrote", path)


if __name__ == "__main__":
    main()
