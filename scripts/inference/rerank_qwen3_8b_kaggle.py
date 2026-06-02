"""Qwen3-Reranker-8B last-stage reranker for Blind A — runs on Kaggle GPU.

Self-contained: reads blind_a_rerank_input.json (queries + candidate docs already
built, future-pruned, short-history) and the Qwen3-Reranker-8B weights. No HF
datasets needed. Uses the official Qwen3-Reranker scoring (yes/no logit on a
causal LM), blends with the LTR rank prior, and writes a submission prediction.json.

Kaggle setup:
  - Add input dataset: upload blind_a_rerank_input.json
  - Enable GPU (T4 x2 / P100 / A100). 8B in fp16 needs ~16GB; T4 16GB is enough
    with batch_size small. Use bf16/fp16.
  - pip install -U transformers torch  (Kaggle usually has these)

Run:
  python rerank_qwen3_8b_kaggle.py \
    --input /kaggle/input/blind-a-rerank/blind_a_rerank_input.json \
    --out /kaggle/working/prediction.json \
    --model Qwen/Qwen3-Reranker-8B --alpha 0.5 --rerank_k 50 --batch_size 8

Then zip /kaggle/working/prediction.json -> submission.zip.
"""
import argparse, json, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

ap = argparse.ArgumentParser()
ap.add_argument("--input", required=True, help="blind_a_rerank_input.json")
ap.add_argument("--out", default="prediction.json")
ap.add_argument("--scores_out", default="rerank_scores.json",
                help="raw per-candidate scores, for offline alpha/k re-blend.")
ap.add_argument("--model", default="Qwen/Qwen3-Reranker-8B")
ap.add_argument("--alpha", type=float, default=0.5,
                help="blend: final = alpha*ltr_prior + (1-alpha)*norm_rerank. "
                     "0=pure rerank, 1=keep LTR order.")
ap.add_argument("--rerank_k", type=int, default=50, help="rerank top-K LTR candidates.")
ap.add_argument("--final_k", type=int, default=20)
ap.add_argument("--batch_size", type=int, default=8)
ap.add_argument("--max_length", type=int, default=2048)
ap.add_argument("--response", default="This is my suggestion.")
args = ap.parse_args()

# ── Official Qwen3-Reranker scaffolding ──────────────────────────────────────
INSTRUCTION = ("Given a user's music listening request and the conversation so far, "
               "retrieve the track that best satisfies the request and continues "
               "the listening session.")

PREFIX = ('<|im_start|>system\nJudge whether the Document meets the requirements '
          'based on the Query and the Instruct provided. Note that the answer can '
          'only be "yes" or "no".<|im_end|>\n<|im_start|>user\n')
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'

print(f"loading {args.model} ...")
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
model = AutoModelForCausalLM.from_pretrained(
    args.model, torch_dtype=torch.float16,
    device_map="auto" if torch.cuda.is_available() else None,
).eval()
device = next(model.parameters()).device
print(f"  loaded in {time.time()-t0:.0f}s on {device}")

token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
prefix_tokens = tokenizer.encode(PREFIX, add_special_tokens=False)
suffix_tokens = tokenizer.encode(SUFFIX, add_special_tokens=False)


def fmt(query, doc):
    return f"<Instruct>: {INSTRUCTION}\n<Query>: {query}\n<Document>: {doc}"


@torch.no_grad()
def score_pairs(pairs):
    """Return P(yes) for each (query, doc) pair, following the official recipe."""
    enc = tokenizer(pairs, padding=False, truncation="longest_first",
                    return_attention_mask=False,
                    max_length=args.max_length - len(prefix_tokens) - len(suffix_tokens))
    for i, ids in enumerate(enc["input_ids"]):
        enc["input_ids"][i] = prefix_tokens + ids + suffix_tokens
    enc = tokenizer.pad(enc, padding=True, return_tensors="pt", max_length=args.max_length)
    enc = {k: v.to(device) for k, v in enc.items()}
    logits = model(**enc).logits[:, -1, :]
    true_v = logits[:, token_true_id]
    false_v = logits[:, token_false_id]
    stacked = torch.stack([false_v, true_v], dim=1)
    probs = torch.nn.functional.log_softmax(stacked, dim=1)
    return probs[:, 1].exp().tolist()


def blended_order(cands, scores, alpha, rk):
    head = list(zip(cands[:rk], scores[:rk]))   # cands in LTR order
    n = len(head)
    if n == 0:
        return [c for c, _ in zip(cands, [])][:0]
    scs = [s for _, s in head]
    lo, hi = min(scs), max(scs)
    span = (hi - lo) or 1.0
    ranked = []
    for i, (tid, sc) in enumerate(head):
        ltr_s = (n - i) / n
        rr_s = (sc - lo) / span
        ranked.append((alpha * ltr_s + (1 - alpha) * rr_s, tid))
    ranked.sort(key=lambda x: x[0], reverse=True)
    return [tid for _, tid in ranked]


# ── Rerank loop ──────────────────────────────────────────────────────────────
data = json.load(open(args.input))
print(f"reranking {len(data)} turns, top-{args.rerank_k}, batch {args.batch_size}...")
results, score_log = [], {}
for j, turn in enumerate(data):
    q = turn["query"]
    cands = [c[0] for c in turn["candidates"]]          # track_ids, LTR order
    docs = [c[1] for c in turn["candidates"]]
    head_ids = cands[: args.rerank_k]
    head_docs = docs[: args.rerank_k]
    tail = cands[args.rerank_k:]

    scores = []
    for b in range(0, len(head_docs), args.batch_size):
        pairs = [fmt(q, d) for d in head_docs[b:b + args.batch_size]]
        scores.extend(score_pairs(pairs))

    order = blended_order(head_ids, scores, args.alpha, args.rerank_k)
    new_ids = (order + tail)[: args.final_k]
    score_log[f"{turn['session_id']}|{turn['turn_number']}"] = \
        [[head_ids[i], scores[i]] for i in range(len(head_ids))]
    results.append({
        "session_id": turn["session_id"],
        "user_id": turn.get("user_id", ""),
        "turn_number": turn["turn_number"],
        "predicted_track_ids": new_ids,
        "predicted_response": args.response,
    })
    if (j + 1) % 10 == 0:
        print(f"  {j+1}/{len(data)} done", flush=True)

json.dump(results, open(args.out, "w"), ensure_ascii=False, indent=2)
json.dump(score_log, open(args.scores_out, "w"))
print(f"Saved {len(results)} preds to {args.out}")
print(f"Saved raw scores to {args.scores_out} (re-blend alpha/k offline).")
print("Zip prediction.json -> submission.zip and upload.")
