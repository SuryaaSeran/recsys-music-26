"""Qwen3-Reranker-8B reranker for Blind A — runs on Kaggle GPU.

Reads blind_a_v8d_tier1_stage3_top100.json (100-candidate prediction file
produced locally by run_inference_fusion_recall_expansion --emit_topk 100).
Loads track metadata + conversation from HF datasets (needs HF_TOKEN env var).
Scores all 100 candidates with Qwen3-Reranker-8B, blends with LTR rank prior,
writes final prediction.json with top-20.

Kaggle setup:
  - Secrets: HF_TOKEN
  - GPU: T4 x2 (16GB) or P100 (16GB). 8B fp16 needs ~16GB.
  - Enable internet access
  - pip install -U transformers>=4.45 datasets

Run:
  python rerank_qwen3_8b_kaggle.py \\
    --pred /kaggle/input/blind-a-top100/blind_a_v8d_tier1_stage3_top100.json \\
    --out  /kaggle/working/prediction.json \\
    --model Qwen/Qwen3-Reranker-8B \\
    --alpha 0.5 --rerank_k 100 --batch_size 8

Then zip /kaggle/working/prediction.json -> submission.zip and upload.
"""
import argparse, json, os, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset, concatenate_datasets

ap = argparse.ArgumentParser()
ap.add_argument("--pred",      required=True,
                help="blind_a top-100 prediction JSON from --emit_topk 100")
ap.add_argument("--out",       default="prediction.json")
ap.add_argument("--scores_out",default="rerank_scores.json")
ap.add_argument("--model",     default="Qwen/Qwen3-Reranker-8B")
ap.add_argument("--alpha",     type=float, default=0.5,
                help="blend: final = alpha*ltr_prior + (1-alpha)*norm_rerank. "
                     "0=pure rerank, 1=keep LTR order.")
ap.add_argument("--rerank_k",  type=int, default=100,
                help="rerank top-K candidates per turn (use 100 to see Stage 3 recall)")
ap.add_argument("--final_k",   type=int, default=20)
ap.add_argument("--batch_size",type=int, default=8)
ap.add_argument("--max_length",type=int, default=2048)
ap.add_argument("--response",  default="This is my suggestion.")
args = ap.parse_args()

# ── Load metadata (track_id → doc string) ────────────────────────────────────
print("Loading track metadata from HF...")
meta_ds = concatenate_datasets([
    load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")["all_tracks"],
    load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")["test_tracks"],
])
track_meta = {}
for row in meta_ds:
    tid = row["track_id"]
    name   = (row.get("track_name")  or [""])[0] or ""
    artist = (row.get("artist_name") or [""])[0] or ""
    album  = (row.get("album_name")  or [""])[0] or ""
    tags   = ", ".join((row.get("tag_list") or [])[:5])
    rel    = (row.get("release_date") or "")[:4]
    parts  = [p for p in [name, f"by {artist}" if artist else "",
                           f"({rel})" if rel else "",
                           f"[{album}]" if album else "",
                           f"tags: {tags}" if tags else ""] if p]
    track_meta[tid] = " ".join(parts)

print(f"  {len(track_meta):,} tracks loaded")

# ── Load conversations (session_id → turns) ───────────────────────────────────
print("Loading blind A conversations from HF...")
blind_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Blind-A")
blind_sessions = {row["session_id"]: row for row in blind_ds["test"]}

def build_query(session_id: str, turn_number: int) -> str:
    """Reconstruct conversation query up to turn_number."""
    row = blind_sessions.get(session_id, {})
    goal = (row.get("conversation_goal") or {})
    goal_text = goal.get("goal_text", "")
    convs = row.get("conversations") or []

    # Collect user/assistant turns up to this turn_number
    history = []
    music_n = 0
    for t in convs:
        role = t.get("role", "")
        if role == "music":
            music_n += 1
            if music_n >= turn_number:
                break
        elif role == "user":
            history.append(f"User: {t.get('content','')}")
        elif role == "assistant":
            history.append(f"Assistant: {t.get('content','')}")

    ctx = " | ".join(history[-6:])  # last 6 turns of context
    if goal_text:
        ctx = f"Goal: {goal_text} | {ctx}"
    return ctx.strip() or "music recommendation"

# ── Official Qwen3-Reranker scaffolding ──────────────────────────────────────
INSTRUCTION = ("Given a user's conversational music request and the dialogue "
               "history, determine whether the track satisfies the user's goal.")

PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements '
          'based on the Query and the Instruct provided. Note that the answer can '
          'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX  = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

print(f"Loading {args.model} ...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.float16,
    device_map="auto" if torch.cuda.is_available() else None,
).eval()
device = next(model.parameters()).device
print(f"  loaded in {time.time()-t0:.0f}s  device={device}")

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id  = tokenizer.convert_tokens_to_ids("yes")
prefix_tokens  = tokenizer.encode(PREFIX, add_special_tokens=False)
suffix_tokens  = tokenizer.encode(SUFFIX,  add_special_tokens=False)


def fmt(query, doc):
    return f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"


@torch.no_grad()
def score_pairs(pairs):
    enc = tokenizer(pairs, padding=False, truncation="longest_first",
                    return_attention_mask=False,
                    max_length=args.max_length - len(prefix_tokens) - len(suffix_tokens))
    for i, ids in enumerate(enc["input_ids"]):
        enc["input_ids"][i] = prefix_tokens + ids + suffix_tokens
    enc = tokenizer.pad(enc, padding=True, return_tensors="pt", max_length=args.max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits[:, -1, :]
    true_v  = logits[:, token_true_id]
    false_v = logits[:, token_false_id]
    probs = torch.nn.functional.log_softmax(
        torch.stack([false_v, true_v], dim=1), dim=1)
    return probs[:, 1].exp().tolist()


def blended_order(cands, scores, alpha, rk):
    head = list(zip(cands[:rk], scores[:rk]))
    n = len(head)
    if n == 0:
        return cands[:0]
    scs = [s for _, s in head]
    lo, hi = min(scs), max(scs)
    span = (hi - lo) or 1.0
    ranked = []
    for i, (tid, sc) in enumerate(head):
        ltr_s = (n - i) / n
        rr_s  = (sc - lo) / span
        ranked.append((alpha * ltr_s + (1 - alpha) * rr_s, tid))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [tid for _, tid in ranked]


# ── Rerank loop ───────────────────────────────────────────────────────────────
preds = json.load(open(args.pred))
print(f"Reranking {len(preds)} turns, top-{args.rerank_k}, batch {args.batch_size} ...")

results, score_log = [], {}
t_start = time.time()

for j, p in enumerate(preds):
    sid  = p["session_id"]
    tn   = p["turn_number"]
    cands = p.get("predicted_track_ids") or []

    query     = build_query(sid, tn)
    head_ids  = cands[:args.rerank_k]
    tail      = cands[args.rerank_k:]
    head_docs = [track_meta.get(t, t) for t in head_ids]

    scores = []
    for b in range(0, len(head_docs), args.batch_size):
        pairs = [fmt(query, d) for d in head_docs[b:b + args.batch_size]]
        scores.extend(score_pairs(pairs))

    order   = blended_order(head_ids, scores, args.alpha, args.rerank_k)
    new_ids = (order + tail)[:args.final_k]

    score_log[f"{sid}|{tn}"] = [[head_ids[i], scores[i]] for i in range(len(head_ids))]
    results.append({
        "session_id":          sid,
        "user_id":             p.get("user_id", ""),
        "turn_number":         tn,
        "predicted_track_ids": new_ids,
        "predicted_response":  args.response,
    })

    if (j + 1) % 10 == 0:
        elapsed = time.time() - t_start
        eta = elapsed / (j + 1) * (len(preds) - j - 1)
        print(f"  {j+1}/{len(preds)}  elapsed={elapsed:.0f}s  eta={eta:.0f}s", flush=True)

json.dump(results,   open(args.out,        "w"), ensure_ascii=False, indent=2)
json.dump(score_log, open(args.scores_out, "w"), ensure_ascii=False)
print(f"Saved {len(results)} predictions → {args.out}")
print(f"Saved raw scores     → {args.scores_out}")
print("Zip prediction.json → submission.zip and upload.")
