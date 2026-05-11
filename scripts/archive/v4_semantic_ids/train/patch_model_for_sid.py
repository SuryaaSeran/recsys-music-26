"""
Patch Qwen2.5-0.5B-Instruct-4bit to add <SID_0>..<SID_255> as single tokens.

Qwen2.5's embedding table has 151,936 rows but only 151,665 are assigned to
active tokens (151,643 base vocab + 22 special tokens). IDs 151,665..151,935
are pre-allocated unused rows — no weight modification needed.

We simply register the SID token strings in the tokenizer files.

Output: models/qwen_sid_patched/
"""

import json
import shutil
from pathlib import Path

SRC = Path("/Users/stealthmacmini/.cache/huggingface/hub/models--mlx-community--Qwen2.5-0.5B-Instruct-4bit/snapshots/a5339a4131f135d0fdc6a5c8b5bbed2753bbe0f3")
DST = Path("models/qwen_sid_patched")

N_SID = 256          # <SID_0> .. <SID_255>
FIRST_SID_ID = 151665


def main():
    DST.mkdir(parents=True, exist_ok=True)

    # Copy all model files
    for f in SRC.iterdir():
        shutil.copy2(f, DST / f.name)
    print(f"Copied model to {DST}")

    sid_tokens = [f"<SID_{i}>" for i in range(N_SID)]
    sid_id_map = {tok: FIRST_SID_ID + i for i, tok in enumerate(sid_tokens)}

    # 1. added_tokens.json
    added = json.loads((DST / "added_tokens.json").read_text())
    added.update(sid_id_map)
    (DST / "added_tokens.json").write_text(json.dumps(added, indent=2))
    print(f"added_tokens.json: added {N_SID} SID tokens (IDs {FIRST_SID_ID}..{FIRST_SID_ID+N_SID-1})")

    # 2. tokenizer_config.json — append to additional_special_tokens
    tc = json.loads((DST / "tokenizer_config.json").read_text())
    existing = tc.get("additional_special_tokens", [])
    tc["additional_special_tokens"] = existing + sid_tokens
    (DST / "tokenizer_config.json").write_text(json.dumps(tc, indent=2))
    print("tokenizer_config.json: updated additional_special_tokens")

    # 3. tokenizer.json — add to model.added_tokens and added_tokens list
    tj = json.loads((DST / "tokenizer.json").read_text())

    # model.added_tokens (list of {id, content, single_word, lstrip, rstrip, normalized, special})
    existing_added = tj.get("added_tokens", [])
    existing_ids = {e["id"] for e in existing_added}
    for i, tok in enumerate(sid_tokens):
        tid = FIRST_SID_ID + i
        if tid not in existing_ids:
            existing_added.append({
                "id": tid,
                "content": tok,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            })
    tj["added_tokens"] = sorted(existing_added, key=lambda x: x["id"])

    # Also add to the normalizer/post-processor if it has added_tokens there
    # (Qwen2.5 uses ByteLevel BPE; special tokens are handled via added_tokens list only)

    (DST / "tokenizer.json").write_text(json.dumps(tj, indent=2))
    print("tokenizer.json: updated added_tokens list")

    # 4. special_tokens_map.json — add additional_special_tokens if key exists
    stm = json.loads((DST / "special_tokens_map.json").read_text())
    if "additional_special_tokens" in stm:
        existing_stm = stm["additional_special_tokens"]
        stm["additional_special_tokens"] = existing_stm + sid_tokens
        (DST / "special_tokens_map.json").write_text(json.dumps(stm, indent=2))
        print("special_tokens_map.json: updated")

    # Verify
    print("\nVerification:")
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(DST))
    for test in ["<SID_0>", "<SID_10>", "<SID_127>", "<SID_255>"]:
        ids = tok.encode(test, add_special_tokens=False)
        status = "OK (1 token)" if len(ids) == 1 else f"FAIL ({len(ids)} tokens: {ids})"
        print(f"  {test} -> {ids} {status}")

    print(f"\nVocab size: {len(tok)}")
    print(f"Expected:   {FIRST_SID_ID + N_SID} = {FIRST_SID_ID + N_SID}")


if __name__ == "__main__":
    main()
