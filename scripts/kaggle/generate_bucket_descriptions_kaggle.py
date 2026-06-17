"""Kaggle script: Generate L0 bucket descriptions and names using Gemma 4 (12B-it).

Kaggle setup:
  - GPU: T4 x2 (recommended) or P100
  - Internet: ON
  - Secrets: HF_TOKEN, GITHUB_TOKEN
  - Input dataset: suryaseran/reccysmusic-bucket-members
    (contains bucket_members.json — top-25 tracks per L0 bucket)

What it does:
  1. Loads bucket_members.json (64 L0 buckets x 25 tracks)
  2. For each bucket: generates a short name + 2-3 sentence description
  3. Writes bucket_descriptions.json to /kaggle/working/
  4. Commits and pushes to GitHub:
       cache/semantic_ids/runF_v8e_L2C64/bucket_descriptions.json

CLI (from repo root):
  kaggle kernels push -p scripts/kaggle/
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ── Kaggle secrets ────────────────────────────────────────────────────────────
try:
    from kaggle_secrets import UserSecretsClient
    secrets = UserSecretsClient()
    os.environ["HF_TOKEN"]      = secrets.get_secret("HF_TOKEN")
    os.environ["GITHUB_TOKEN"]  = secrets.get_secret("GITHUB_TOKEN")
    print("Secrets loaded from Kaggle")
except Exception:
    print("Not on Kaggle or secrets missing — continuing with env vars")

import torch
from transformers import AutoProcessor, AutoModelForMultimodalLM

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_ID       = "google/gemma-4-12B-it"
# Path relative to cloned repo — works when run via:
# !python /kaggle/working/repo/scripts/kaggle/generate_bucket_descriptions_kaggle.py
_SCRIPT_DIR    = Path(__file__).resolve().parent
_REPO_ROOT     = _SCRIPT_DIR.parent.parent
MEMBERS_PATH   = str(_REPO_ROOT / "kaggle_datasets/reccysmusic-bucket-members/bucket_members.json")
OUT_PATH       = Path("/kaggle/working/bucket_descriptions.json")
GIT_REPO       = "https://github.com/SuryaaSeran/recsys-music-26.git"
GIT_OUT_PATH   = "cache/semantic_ids/runF_v8e_L2C64/bucket_descriptions.json"

# ── Load bucket members ───────────────────────────────────────────────────────
print(f"Loading: {MEMBERS_PATH}")
with open(MEMBERS_PATH) as f:
    bucket_members: dict[str, list[str]] = json.load(f)
n_buckets = len(bucket_members)
print(f"  {n_buckets} buckets, {sum(len(v) for v in bucket_members.values())} total track entries")

# ── Load Gemma 4 ──────────────────────────────────────────────────────────────
print(f"\nLoading {MODEL_ID}...")
print(f"  CUDA available: {torch.cuda.is_available()}  GPUs: {torch.cuda.device_count()}")

processor = AutoProcessor.from_pretrained(MODEL_ID)
model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    dtype="auto",
    device_map="auto",
)
model.eval()
print("  Model loaded")

# ── Prompt ────────────────────────────────────────────────────────────────────
SYSTEM = (
    "You are a music expert. You will receive a list of tracks that have been "
    "automatically clustered by an embedding model. Your task is to characterise "
    "the cluster.\n\n"
    "Respond in EXACTLY this format (two lines, nothing else):\n"
    "NAME: <3-6 word cluster name e.g. '90s Alternative Rock' or 'Latin Dance Pop'>\n"
    "DESCRIPTION: <2-3 sentences about genre, mood, era, tempo, sonic texture. "
    "Do NOT name specific artists or tracks.>"
)


def build_messages(l0: int) -> list[dict]:
    tracks = bucket_members[str(l0)]
    numbered = "\n".join(f"{i+1}. {t}" for i, t in enumerate(tracks))
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": f"Tracks in cluster {l0}:\n\n{numbered}"},
    ]


def generate(l0: int) -> dict:
    messages = build_messages(l0)
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=True,
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=512)
    raw = processor.decode(outputs[0][input_len:], skip_special_tokens=False)
    parsed = processor.parse_response(raw)           # → {role, content, thinking}
    return {"raw": raw, "content": parsed.get("content", ""), "thinking": parsed.get("thinking", "")}


def parse_content(content: str) -> tuple[str, str]:
    """Extract NAME and DESCRIPTION from content string."""
    name, desc = "", ""
    for line in content.strip().split("\n"):
        if line.startswith("NAME:"):
            name = line[5:].strip()
        elif line.startswith("DESCRIPTION:"):
            desc = line[12:].strip()
        elif desc:
            desc += " " + line.strip()
    return name.strip(), desc.strip()


# ── Generate descriptions ─────────────────────────────────────────────────────
results: dict[str, dict] = {}
if OUT_PATH.exists():
    results = json.loads(OUT_PATH.read_text())
    print(f"\nResuming: {len(results)}/{n_buckets} already done")

buckets = sorted(bucket_members.keys(), key=int)

for bucket_id in buckets:
    if bucket_id in results:
        continue

    l0 = int(bucket_id)
    t0 = time.time()
    gen = generate(l0)
    name, desc = parse_content(gen["content"])
    elapsed = time.time() - t0

    results[bucket_id] = {
        "bucket_id":       l0,
        "name":            name,
        "description":     desc,
        "n_tracks_shown":  len(bucket_members[bucket_id]),
        "raw_response":    gen["content"],
        "thinking":        gen["thinking"][:500] if gen["thinking"] else "",
    }

    print(f"  [{l0:>2}/63] ({elapsed:.1f}s)  {name!r}")
    if not name:
        print(f"    WARNING: no NAME parsed from: {gen['content'][:120]!r}")

    # Incremental save
    OUT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False))

print(f"\nAll {len(results)} buckets done → {OUT_PATH}")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n=== Bucket Names ===")
for k in sorted(results.keys(), key=int):
    print(f"  {k:>2}: {results[k]['name']}")

# ── Push to GitHub ────────────────────────────────────────────────────────────
github_token = os.environ.get("GITHUB_TOKEN", "")
if not github_token:
    print("\nNo GITHUB_TOKEN — skipping git push")
    sys.exit(0)

print("\nPushing to GitHub...")
repo_dir = "/kaggle/working/repo"
repo_url = GIT_REPO.replace("https://", f"https://{github_token}@")

steps = [
    ["git", "clone", "--depth=1", repo_url, repo_dir],
    ["cp", str(OUT_PATH), f"{repo_dir}/{GIT_OUT_PATH}"],
    ["git", "-C", repo_dir, "config", "user.email", "kaggle@reccysmusic"],
    ["git", "-C", repo_dir, "config", "user.name", "Kaggle Runner"],
    ["git", "-C", repo_dir, "add", GIT_OUT_PATH],
    ["git", "-C", repo_dir, "commit", "-m",
     "kaggle: bucket descriptions + names via Gemma 4-12B (runF_v8e_L2C64)"],
    ["git", "-C", repo_dir, "push"],
]
for cmd in steps:
    print("  " + " ".join(cmd[:5]))
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"  FAILED: {r.stderr[:400]}")
        sys.exit(1)

print("Done — output committed to GitHub.")
