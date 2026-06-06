"""Diagnostic: dense conversation->track recall per modality (no training).

For each dev music turn, build a text query (profile + goal + last-3 turns), encode it
with Qwen3-Embedding-0.6B, cosine-search each modality's provided track embeddings, and
measure recall@K. Tells us which signal supports conversational recall (vs the generative
cf-bpr ~0.20). cf-bpr has no text encoder -> not a dense-retrievable signal from text.
"""
import argparse
import glob
import json
import os
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

HUB = os.path.expanduser("~/.cache/huggingface/hub")
DS = "datasets--talkpl-ai--TalkPlayData-Challenge-Dataset"
EMB = "datasets--talkpl-ai--TalkPlayData-Challenge-Track-Embeddings"
QWEN_MODS = ["metadata-qwen3_embedding_0.6b", "attributes-qwen3_embedding_0.6b",
             "lyrics-qwen3_embedding_0.6b"]


def load_split(split):
    p = sorted(glob.glob(f"{HUB}/{DS}/snapshots/*/data/{split}-*.parquet"))
    return pd.concat([pd.read_parquet(x) for x in p], ignore_index=True)


def build_queries(df, n, seed=0):
    rng = np.random.default_rng(seed)
    rows = df.to_dict("records")
    turns = []
    for r in rows:
        conv = list(r["conversations"])
        prof = r["user_profile"]; g = r["conversation_goal"]
        head = (f"profile: {prof.get('age_group')}, {prof.get('country_name')}, "
                f"{prof.get('gender')}, {prof.get('preferred_musical_culture')}\n"
                f"goal: {g.get('category')} / {g.get('specificity')} / {g.get('listener_goal')}")
        for i, t in enumerate(conv):
            if t["role"] == "music":
                dlg = [x for x in conv[:i] if x["role"] in ("user", "assistant")][-3:]
                q = head + "\n" + "\n".join(f"{x['role']}: {x['content']}" for x in dlg)
                turns.append((q, t["content"]))
    idx = rng.choice(len(turns), min(n, len(turns)), replace=False)
    return [turns[i] for i in idx]


def load_track_emb(modality):
    p = sorted(glob.glob(f"{HUB}/{EMB}/snapshots/*/data/all_tracks-*.parquet"))
    ids, vecs = [], []
    for f in p:
        d = pd.read_parquet(f, columns=["track_id", modality])
        ids += d["track_id"].astype(str).tolist()
        vecs.append(np.vstack([v if (v is not None and len(v)) else np.zeros(1024, np.float32)
                               for v in d[modality].to_numpy()]).astype(np.float32))
    M = np.concatenate(vecs)
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    return ids, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    from sentence_transformers import SentenceTransformer
    enc = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=a.device)

    qs = build_queries(load_split("test"), a.n)
    golds = [g for _, g in qs]
    Q = enc.encode([q for q, _ in qs], prompt_name="query", batch_size=16,
                   normalize_embeddings=True, show_progress_bar=True)

    for mod in QWEN_MODS:
        ids, M = load_track_emb(mod)
        pos = {t: i for i, t in enumerate(ids)}
        sims = Q @ M.T                                  # [nq, ntrack]
        order = np.argpartition(-sims, 200, axis=1)[:, :200]
        rec = {k: 0 for k in (20, 50, 100, 200)}
        miss = 0
        for r in range(len(qs)):
            gi = pos.get(golds[r])
            if gi is None:
                miss += 1; continue
            row = order[r][np.argsort(-sims[r, order[r]])]
            rank = np.where(row == gi)[0]
            if len(rank):
                for k in rec:
                    if rank[0] < k:
                        rec[k] += 1
        n = len(qs) - miss
        print(f"{mod:35} recall@20={rec[20]/n:.3f} @50={rec[50]/n:.3f} "
              f"@100={rec[100]/n:.3f} @200={rec[200]/n:.3f}  (n={n}, gold_missing={miss})", flush=True)


if __name__ == "__main__":
    main()
