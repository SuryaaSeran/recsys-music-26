"""
Test plain integer adapter. Checks format validity and diversity across 30 valid examples.
"""
import json
import pickle
import re
from pathlib import Path
from collections import Counter

from mlx_lm import load, generate

valid_path = Path("data/sft_plain_int_v1/valid.jsonl")
cb_path = Path("data/codebook_v2.pkl")

with open(cb_path, "rb") as f:
    cb = pickle.load(f)
codes_to_tracks = cb["codes_to_tracks"]

examples = []
with open(valid_path) as f:
    for line in f:
        examples.append(json.loads(line))
        if len(examples) >= 30:
            break

model, tokenizer = load(
    "models/qwen_sid_patched",
    adapter_path="adapters/qwen_plain_int_v1",
)

outputs = []
format_ok = 0
valid_range = 0

for ex in examples:
    messages = [ex["messages"][0], ex["messages"][1]]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    gold = ex["messages"][2]["content"]

    out = generate(model, tokenizer, prompt=prompt, max_tokens=10).strip()
    outputs.append(out)

    m = re.fullmatch(r"(\d+)\s+(\d+)", out)
    if m:
        format_ok += 1
        c, f = int(m.group(1)), int(m.group(2))
        if 0 <= c <= 127 and 0 <= f <= 127:
            valid_range += 1

    print(f"GOLD: {gold:>7}  |  OUT: {out}")

print(f"\nFormat OK: {format_ok}/30")
print(f"Valid range (0-127): {valid_range}/30")

coarse_counts = Counter()
for o in outputs:
    m = re.fullmatch(r"(\d+)\s+(\d+)", o)
    if m:
        coarse_counts[int(m.group(1))] += 1

if coarse_counts:
    top_c, top_n = coarse_counts.most_common(1)[0]
    print(f"Most common coarse: {top_c} ({top_n}/30, {100*top_n/30:.0f}%)")
    print(f"Unique coarse codes seen: {len(coarse_counts)}")
