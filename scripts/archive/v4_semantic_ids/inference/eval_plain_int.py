"""
Evaluate plain-int adapter.

Metrics:
  cluster_hit@K  - gold track is in ANY of the K predicted clusters
  coarse_hit@K   - gold coarse code matches any predicted coarse
  exact_hit@K    - gold (c,f) exactly matches any of K predictions

Usage: python scripts/eval_plain_int.py [--ckpt 1600] [--n 200] [--k 10]
"""
import argparse
import json
import pickle
import re
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

parser = argparse.ArgumentParser()
parser.add_argument("--ckpt", type=int, default=0, help="checkpoint step (0=final)")
parser.add_argument("--n", type=int, default=200)
parser.add_argument("--k", type=int, default=10)
parser.add_argument("--temp", type=float, default=1.0)
args = parser.parse_args()

with open("data/codebook_v2.pkl", "rb") as f:
    cb = pickle.load(f)

track_to_codes = cb["track_to_codes"]
codes_to_tracks = cb["codes_to_tracks"]  # (c,f) -> [track_id, ...]

adapter_dir = Path("adapters/qwen_plain_int_v1")
if args.ckpt:
    tmp = Path(tempfile.mkdtemp())
    shutil.copy(adapter_dir / f"{args.ckpt:07d}_adapters.safetensors", tmp / "adapters.safetensors")
    shutil.copy(adapter_dir / "adapter_config.json", tmp / "adapter_config.json")
    adapter_path = str(tmp)
else:
    adapter_path = str(adapter_dir)

print(f"Loading adapter: {adapter_path}")
model, tokenizer = load("models/qwen_sid_patched", adapter_path=adapter_path)
sampler = make_sampler(temp=args.temp)

examples = []
with open("data/sft_plain_int_v1/valid.jsonl") as f:
    for line in f:
        ex = json.loads(line)
        gold_c, gold_f = map(int, ex["messages"][2]["content"].split())
        if (gold_c, gold_f) in codes_to_tracks:
            examples.append(ex)
        if len(examples) >= args.n:
            break

cluster_hits = 0
coarse_hits = 0
exact_hits = 0
format_fails = 0
total_candidates = 0

for ex in examples:
    gold_content = ex["messages"][2]["content"]
    gold_c, gold_f = map(int, gold_content.split())
    gold_tracks = set(codes_to_tracks.get((gold_c, gold_f), []))

    messages = [ex["messages"][0], ex["messages"][1]]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    candidates = []
    for _ in range(args.k):
        out = generate(model, tokenizer, prompt=prompt, max_tokens=10, sampler=sampler).strip()
        m = re.fullmatch(r"(\d+)\s+(\d+)", out)
        if m:
            c, f = int(m.group(1)), int(m.group(2))
            if 0 <= c <= 127 and 0 <= f <= 127:
                candidates.append((c, f))
        else:
            format_fails += 1

    total_candidates += len(candidates)

    # Cluster hit: any predicted cluster contains a track also in gold cluster
    predicted_tracks = set()
    for c, f in candidates:
        predicted_tracks.update(codes_to_tracks.get((c, f), []))

    if predicted_tracks & gold_tracks:
        cluster_hits += 1

    # Coarse hit: gold coarse code appears in any prediction
    if any(c == gold_c for c, f in candidates):
        coarse_hits += 1

    # Exact hit: gold (c,f) pair exactly predicted
    if (gold_c, gold_f) in candidates:
        exact_hits += 1

n = len(examples)
print(f"N={n}, K={args.k}, temp={args.temp}")
print(f"Cluster hit@{args.k}: {cluster_hits}/{n} = {100*cluster_hits/n:.1f}%")
print(f"Coarse hit@{args.k}:  {coarse_hits}/{n} = {100*coarse_hits/n:.1f}%")
print(f"Exact hit@{args.k}:   {exact_hits}/{n} = {100*exact_hits/n:.1f}%")
print(f"Format fails: {format_fails}/{n*args.k} = {100*format_fails/(n*args.k):.1f}%")
print(f"Avg valid candidates per query: {total_candidates/n:.1f}")
