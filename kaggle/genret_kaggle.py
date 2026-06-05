"""Stage A generative candidate generator -- self-contained Kaggle runner (CUDA).

Trains Llama-3.2-1B to map conversational context -> the gold track's cf-bpr 4-token
semantic ID, then evaluates dev recall@pool with trie-constrained beam decode.

Inputs:
  --sem  path to semantic_ids.json   (upload as a Kaggle dataset; has cf-bpr per track)
  sessions + base model are pulled from HF (enable Internet).

Run on Kaggle (GPU T4/P100, Internet ON):
  !pip -q install -U "transformers>=4.45" peft accelerate datasets
  !python genret_kaggle.py --sem /kaggle/input/reccys-cf-ids/semantic_ids.json \
         --epochs 8 --batch-size 16 --out /kaggle/working/ckpt
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

# ----------------------------------------------------------------------------- config
BASE_MODEL = "unsloth/Llama-3.2-1B-Instruct"
CF_LEVELS, CF_K = 4, 256
CF_TOKEN = "<cf_{l}_{c}>"
HIST_OPEN, HIST_CLOSE, GEN = "<hist>", "</hist>", "<gen>"
BLIND_TURN_DIST = {1: 20, 2: 15, 3: 10, 4: 5, 5: 8, 6: 9, 7: 8, 8: 5}
MOVES = "MOVES_TOWARD_GOAL"

# --------------------------------------------------------------------------- tokens
class SemTokenizer:
    def __init__(self, tok):
        self.tok = tok
        cf = [CF_TOKEN.format(l=l, c=c) for l in range(CF_LEVELS) for c in range(CF_K)]
        self.n_added = tok.add_special_tokens(
            {"additional_special_tokens": cf + [HIST_OPEN, HIST_CLOSE, GEN]})
        self.grid = np.empty((CF_LEVELS, CF_K), dtype=np.int64)
        self.tok2lc = {}
        for l in range(CF_LEVELS):
            for c in range(CF_K):
                tid = tok.convert_tokens_to_ids(CF_TOKEN.format(l=l, c=c))
                self.grid[l, c] = tid
                self.tok2lc[tid] = (l, c)
        self.gen_id = tok.convert_tokens_to_ids(GEN)
        self.eos_id = tok.eos_token_id
        cf_ids = self.grid.reshape(-1)
        self.new_lo = int(min(cf_ids.min(), tok.convert_tokens_to_ids(HIST_OPEN),
                              tok.convert_tokens_to_ids(GEN)))

    def cf_to_tokens(self, q):
        return [int(self.grid[l, int(q[l])]) for l in range(CF_LEVELS)]

    def cf_to_str(self, q):
        return "".join(CF_TOKEN.format(l=l, c=int(q[l])) for l in range(CF_LEVELS))

    def tokens_to_cf(self, ids):
        return tuple(self.tok2lc[int(t)][1] for t in ids)


# ----------------------------------------------------------------------------- trie
class CfTrie:
    def __init__(self, root, tuple_to_tracks, gen_id, eos_id):
        self.root, self.tuple_to_tracks = root, tuple_to_tracks
        self.gen_id, self.eos_id = gen_id, eos_id

    @classmethod
    def build(cls, sem, cf_map):
        root, t2t = {}, {}
        for tid, cf in cf_map.items():
            if cf is None:
                continue
            quad = tuple(cf)
            t2t.setdefault(quad, []).append(tid)
            node = root
            for t in sem.cf_to_tokens(quad):
                node = node.setdefault(t, {})
            node[sem.eos_id] = None
        return cls(root, t2t, sem.gen_id, sem.eos_id)

    def fn(self):
        def f(batch_id, input_ids):
            ids = input_ids.tolist()
            pos = len(ids) - 1 - ids[::-1].index(self.gen_id) + 1 if self.gen_id in ids else len(ids)
            node = self.root
            for t in ids[pos:]:
                node = node.get(int(t))
                if node is None:
                    return [self.eos_id]
            return list(node.keys())
        return f

    def to_tracks(self, quad):
        return self.tuple_to_tracks.get(tuple(int(c) for c in quad), [])


# --------------------------------------------------------------- lean trainable vocab
class LeanVocab(nn.Module):
    def __init__(self, full_weight, new_lo):
        super().__init__()
        self.new_lo = new_lo
        self.register_buffer("frozen", full_weight.detach().clone(), persistent=False)
        self.frozen.requires_grad_(False)
        self.new_emb = nn.Parameter(full_weight[new_lo:].detach().clone())

    def embed(self, ids):
        fe = F.embedding(ids, self.frozen)
        ne = F.embedding((ids - self.new_lo).clamp_min(0), self.new_emb)
        return torch.where((ids >= self.new_lo).unsqueeze(-1), ne.to(fe.dtype), fe)

    def head(self, h):
        return torch.cat([h @ self.frozen[:self.new_lo].t(), h @ self.new_emb.t()], -1)


class _Emb(nn.Module):
    def __init__(self, lv): super().__init__(); self.lv = lv
    def forward(self, ids): return self.lv.embed(ids)


class _Head(nn.Module):
    def __init__(self, lv): super().__init__(); self._lv = [lv]
    def forward(self, h): return self._lv[0].head(h)


def attach_lean_vocab(model, new_lo):
    lv = LeanVocab(model.get_input_embeddings().weight.data, new_lo)
    model.set_input_embeddings(_Emb(lv))
    model.lm_head = _Head(lv)
    model.config.tie_word_embeddings = False
    return lv


# ----------------------------------------------------------------------------- data
def load_cf_map(sem_path):
    sem = json.loads(Path(sem_path).read_text())
    out = {}
    for tid, e in sem.items():
        cf = e["cf-bpr"]
        out[tid] = None if cf[0] < 0 else cf
    return out


def _profile(p):
    f = [p.get("age_group"), p.get("country_name"), p.get("gender"),
         p.get("preferred_language"), p.get("preferred_musical_culture")]
    return "profile: " + ", ".join(str(x) for x in f if x)


def _goal(g):
    return f"goal: {g.get('category')} / {g.get('specificity')} / {g.get('listener_goal')}"


def iter_music_turns(conv):
    for i, t in enumerate(conv):
        if t["role"] == "music":
            yield i, t["content"], conv[:i], t["turn_number"]


def render_context(row, prior, sem, cf_map, with_history=True, max_recent=3):
    lines = [_profile(row["user_profile"]), _goal(row["conversation_goal"])]
    if with_history:
        blocks = [sem.cf_to_str(cf_map[t["content"]]) for t in prior
                  if t["role"] == "music" and cf_map.get(t["content"]) is not None]
        if blocks:
            lines.append("<hist> " + " ".join(blocks) + " </hist>")
    for t in [t for t in prior if t["role"] in ("user", "assistant")][-max_recent:]:
        lines.append(f"{t['role']}: {t['content']}")
    lines.append("<gen>")
    return "\n".join(lines)


def sample_terminal(avail, dist, rng):
    a = sorted(set(avail) & set(dist)) or sorted(avail)
    w = np.array([dist.get(t, 1) for t in a], float)
    return int(rng.choice(a, p=w / w.sum()))


def build_train(rows, sem, cf_map, tok, seed, max_ctx=256):
    """One terminal example per session (resampled by `seed` each epoch)."""
    rng = np.random.default_rng(seed)
    out = []
    for row in rows:
        conv = list(row["conversations"])
        turns = list(iter_music_turns(conv))
        T = sample_terminal([tn for _, _, _, tn in turns], BLIND_TURN_DIST, rng)
        for _, gold, prior, tn in turns:
            if tn != T:
                continue
            cf = cf_map.get(gold)
            if cf is None:
                break
            ctx = render_context(row, prior, sem, cf_map)
            ids = tok(ctx, add_special_tokens=True).input_ids
            tgt = sem.cf_to_tokens(cf) + [sem.eos_id]
            ids = ids[-max_ctx:]
            out.append({"input_ids": ids + tgt, "labels": [-100] * len(ids) + tgt})
            break
    return out


def build_dev(rows, sem, cf_map, with_history=True, max_recent=3):
    out = []
    for row in rows:
        amap = {int(a["turn_number"]): a["goal_progress_assessment"]
                for a in row["goal_progress_assessments"]}
        conv = list(row["conversations"])
        for _, gold, prior, tn in iter_music_turns(conv):
            v = amap.get(int(tn) + 1)
            if v is not None and v != MOVES:           # keep MOVES + verdict-less final turn
                continue
            cf = cf_map.get(gold)
            out.append({"context": render_context(row, prior, sem, cf_map, with_history, max_recent),
                        "gold_track_id": gold, "gold_cf": cf, "gold_has_cf": cf is not None,
                        "turn_number": int(tn)})
    return out


def collate(batch, pad_id):
    m = max(len(b["input_ids"]) for b in batch)
    ii, ll, aa = [], [], []
    for b in batch:
        n = m - len(b["input_ids"])
        ii.append(b["input_ids"] + [pad_id] * n)
        ll.append(b["labels"] + [-100] * n)
        aa.append([1] * len(b["input_ids"]) + [0] * n)
    return {"input_ids": torch.tensor(ii), "labels": torch.tensor(ll),
            "attention_mask": torch.tensor(aa)}


# ----------------------------------------------------------------------------- model
def build_model(tok, sem, dtype, device):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    raw = AutoModelForCausalLM.from_pretrained(BASE_MODEL, dtype=dtype)
    raw.resize_token_embeddings(len(tok), mean_resizing=False)
    lv = attach_lean_vocab(raw, sem.new_lo)
    peft_model = get_peft_model(raw, LoraConfig(task_type="CAUSAL_LM", r=16, lora_alpha=32,
                   lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    lv.new_emb.requires_grad_(True)
    raw.to(device)
    return raw, peft_model, lv


def loss_and_acc(raw, lv, b):
    h = raw.model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).last_hidden_state
    mask = b["labels"] != -100
    bi, ti = mask.nonzero(as_tuple=True)
    logits = lv.head(h[bi, ti - 1])
    tgt = b["labels"][bi, ti]
    return F.cross_entropy(logits, tgt), (logits.argmax(-1) == tgt).float().mean()


# ----------------------------------------------------------------------------- eval
@torch.no_grad()
def generate_pool(raw, lv, sem, trie, tok, context, device, pool=200, num_beams=256):
    enc = tok(context, return_tensors="pt").to(device)
    out = raw.generate(**enc, num_beams=num_beams, num_return_sequences=pool, max_new_tokens=5,
                       prefix_allowed_tokens_fn=trie.fn(), do_sample=False,
                       return_dict_in_generate=True, output_scores=True, length_penalty=1.0,
                       pad_token_id=tok.eos_token_id)
    gen = out.sequences[:, enc.input_ids.shape[1]:]
    trans = raw.compute_transition_scores(out.sequences, out.scores, out.beam_indices,
                                          normalize_logits=True)
    best = {}
    for row, sc in zip(gen.tolist(), trans):
        cf = [t for t in row if t in sem.tok2lc]
        if len(cf) != 4:
            continue
        lp = float(sc[:4].sum())
        for tid in trie.to_tracks(sem.tokens_to_cf(cf)):
            if tid not in best or lp > best[tid][0]:
                best[tid] = (lp, sem.tokens_to_cf(cf))
    return sorted(([t, v[0], v[1]] for t, v in best.items()), key=lambda x: x[1], reverse=True)


def evaluate(raw, lv, sem, trie, tok, dev, device, pools=(20, 50, 100, 200), n=1000, seed=0):
    rng = np.random.default_rng(seed)
    ex = [dev[i] for i in rng.choice(len(dev), min(n, len(dev)), replace=False)]
    raw.eval()
    rec = []
    for e in tqdm(ex, desc="eval"):
        pool = generate_pool(raw, lv, sem, trie, tok, e["context"], device, pool=max(pools))
        ids = [p[0] for p in pool]
        rank = ids.index(e["gold_track_id"]) + 1 if e["gold_track_id"] in ids else None
        firsts = {p[2][0] for p in pool}
        rec.append({"hit": {k: bool(rank and rank <= k) for k in pools}, "rank": rank,
                    "has_cf": e["gold_has_cf"], "turn": e["turn_number"],
                    "first_ok": bool(e["gold_cf"] and e["gold_cf"][0] in firsts)})
    def R(k, key=lambda r: True):
        s = [r for r in rec if key(r)]
        return round(float(np.mean([r["hit"][k] for r in s])), 4) if s else None
    return {
        "n": len(rec),
        "recall": {k: R(k) for k in pools},
        "recall_generatable": {k: R(k, lambda r: r["has_cf"]) for k in pools},
        "ceiling": round(float(np.mean([r["has_cf"] for r in rec])), 4),
        "gold_first_token_in_pool": round(float(np.mean([r["first_ok"] for r in rec])), 4),
        "exact_top1": round(float(np.mean([bool(r["rank"] == 1) for r in rec])), 4),
        "recall200_by_turn": {int(t): R(max(pools), lambda r, t=t: r["turn"] == t)
                              for t in sorted({r["turn"] for r in rec})},
    }


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sem", required=True, help="path to semantic_ids.json")
    ap.add_argument("--out", default="/kaggle/working/ckpt")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max-ctx", type=int, default=256)
    ap.add_argument("--eval-n", type=int, default=1000)
    ap.add_argument("--eval-every", type=int, default=2)
    ap.add_argument("--patience", type=int, default=3, help="evals w/o recall@200 gain before stop")
    ap.add_argument("--min-delta", type=float, default=0.0)
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"device={device} dtype={dtype}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    sem = SemTokenizer(tok)
    cf_map = load_cf_map(args.sem)
    trie = CfTrie.build(sem, cf_map)
    print(f"trie leaves={len(trie.tuple_to_tracks)}  cf_tracks={sum(v is not None for v in cf_map.values())}", flush=True)

    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
    train_rows = list(ds["train"])
    dev = build_dev(list(ds["test"]), sem, cf_map)
    print(f"train_sessions={len(train_rows)}  dev_examples={len(dev)} "
          f"ceiling={np.mean([e['gold_has_cf'] for e in dev]):.4f}", flush=True)

    raw, peft_model, lv = build_model(tok, sem, dtype, device)
    params = [p for p in raw.parameters() if p.requires_grad]
    print(f"trainable={sum(p.numel() for p in params)/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    pad = tok.eos_token_id
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    def save_ckpt(d):
        d = Path(d); d.mkdir(parents=True, exist_ok=True)
        peft_model.save_pretrained(d); tok.save_pretrained(d)   # LoRA adapter only
        torch.save({"new_lo": lv.new_lo, "new_rows": lv.new_emb.detach().cpu()},
                   d / "new_token_embeddings.pt")

    best, best_epoch, no_improve = -1.0, -1, 0
    for epoch in range(args.epochs):
        data = build_train(train_rows, sem, cf_map, tok, seed=epoch, max_ctx=args.max_ctx)
        rng = np.random.default_rng(epoch)
        order = rng.permutation(len(data))
        raw.train(); t0 = time.time(); rl = ra = nb = 0; opt.zero_grad()
        pbar = tqdm(range(0, len(order), args.batch_size), desc=f"epoch {epoch}")
        for step, i in enumerate(pbar):
            b = {k: v.to(device) for k, v in
                 collate([data[j] for j in order[i:i + args.batch_size]], pad).items()}
            loss, acc = loss_and_acc(raw, lv, b)
            (loss / args.grad_accum).backward()
            rl += loss.item(); ra += acc.item(); nb += 1
            if (step + 1) % args.grad_accum == 0:
                opt.step(); opt.zero_grad()
            pbar.set_postfix(loss=f"{rl/nb:.3f}", acc=f"{ra/nb:.3f}")
        opt.step(); opt.zero_grad()
        print(f"epoch {epoch}: loss {rl/nb:.4f} tgt_acc {ra/nb:.4f} {(time.time()-t0)/60:.1f} min", flush=True)
        save_ckpt(out)                                   # latest

        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            rep = evaluate(raw, lv, sem, trie, tok, dev, device, n=args.eval_n)
            metric = rep["recall"][200]                  # headline: recall@200
            print("EVAL", json.dumps(rep), flush=True)
            (out / f"eval_epoch{epoch}.json").write_text(json.dumps(rep, indent=2))
            if metric > best + args.min_delta:
                best, best_epoch, no_improve = metric, epoch, 0
                save_ckpt(out / "best")
                (out / "best" / "eval.json").write_text(json.dumps({"epoch": epoch, **rep}, indent=2))
                print(f"** new best recall@200 {best:.4f} @ epoch {epoch} -> saved {out}/best", flush=True)
            else:
                no_improve += 1
                print(f"no improvement {no_improve}/{args.patience} "
                      f"(best {best:.4f} @ epoch {best_epoch})", flush=True)
                if no_improve >= args.patience:
                    print(f"early stop at epoch {epoch}; best recall@200 {best:.4f} @ epoch {best_epoch}", flush=True)
                    break
    print(f"DONE best recall@200 {best:.4f} @ epoch {best_epoch} (weights in {out}/best)", flush=True)


if __name__ == "__main__":
    main()
