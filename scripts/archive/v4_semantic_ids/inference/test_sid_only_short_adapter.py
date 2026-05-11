import json
from pathlib import Path

from mlx_lm import load, generate

valid_path = Path("data/sft_sid_only_short_v2/valid.jsonl")

with open(valid_path) as f:
    ex = json.loads(next(f))

model, tokenizer = load(
    "models/qwen_sid_patched",
    adapter_path="adapters/qwen_sid_ml_v2_3k",
)

messages = [
    ex["messages"][0],
    ex["messages"][1],
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)

gold = ex["messages"][2]["content"]

print("GOLD:")
print(gold)
print("\n" + "=" * 80)
print("MODEL OUTPUT:")

out = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=30,
)

print(out)
