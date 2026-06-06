"""Diagnostic: fusion recall (BM25 + dense content), RRF-fused, on dev.

BM25 over track text (name/artist/album/tags/year) catches named entities the user
states; dense Qwen3 over metadata/attributes catches semantic intent. RRF fuses them.
Reports each source alone + the fusion, recall@{50,200,500,1000}.
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
KS = (50, 200, 500, 1000)


def load_split(split):
    p = sorted(glob.glob(f"{HUB}/{DS}/snapshots/*/data/{split}-*.parquet"))
    return pd.concat([pd.read_parquet(x) for x in p], ignore_index=True)


def build_queries(df, n, seed=0):
    rng = np.random.default_rng(seed)
    turns = []
    for r in df.to_dict("records"):
        conv = list(r["conversations"]); prof = r["user_profile"]; g = r["conversation_goal"]
        head = (f"{prof.get('preferred_musical_culture')}. "
                f"{g.get('listener_goal')}")
        for i, t in enumerate(conv):
            if t["role"] == "music":
                dlg = [x for x in conv[:i] if x["role"] in ("user", "assistant")][-3:]
                q = head + " " + " ".join(x["content"] for x in dlg)
                turns.append((q, t["content"]))
    idx = rng.choice(len(turns), min(n, len(turns)), replace=False)
    return [turns[i] for i in idx]


def load_emb(modality, track_ids_order):
    p = sorted(glob.glob(f"{HUB}/{EMB}/snapshots/*/data/all_tracks-*.parquet"))
    d = pd.concat([pd.read_parquet(f, columns=["track_id", modality]) for f in p], ignore_index=True)
    pos = {str(t): i for i, t in enumerate(d["track_id"].astype(str))}
    M = np.zeros((len(track_ids_order), 1024), np.float32)
    for i, tid in enumerate(track_ids_order):
        v = d[modality].iloc[pos[tid]]
        if v is not None and len(v):
            M[i] = v
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    return M


def topk_from_sims(sims, k):
    idx = np.argpartition(-sims, k, axis=1)[:, :k]
    return np.take_along_axis(idx, np.argsort(-np.take_along_axis(sims, idx, 1), 1), 1)


def rrf(rank_lists, k_out, c=60):
    """rank_lists: list of [nq, k] arrays of track indices (best first). -> [nq, k_out]."""
    nq = rank_lists[0].shape[0]
    out = []
    for q in range(nq):
        score = {}
        for rl in rank_lists:
            for rank, idx in enumerate(rl[q]):
                score[idx] = score.get(idx, 0.0) + 1.0 / (c + rank)
        out.append([i for i, _ in sorted(score.items(), key=lambda x: -x[1])[:k_out]])
    return out


def recall(ranked, gold_idx, ks):
    res = {k: 0 for k in ks}
    for r, g in zip(ranked, gold_idx):
        rr = list(r)
        pos = rr.index(g) if g in rr else None
        if pos is not None:
            for k in ks:
                if pos < k:
                    res[k] += 1
    n = len(gold_idx)
    return {k: round(res[k] / n, 3) for k in ks}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--device", default="mps")
    a = ap.parse_args()

    from src.tracks import load_catalog
    import bm25s
    from sentence_transformers import SentenceTransformer

    track_ids = json.loads(open("data/cache/track_ids.json").read())
    cat = load_catalog()
    pos = {t: i for i, t in enumerate(track_ids)}

    qs = build_queries(load_split("test"), a.n)
    golds = [pos[g] for _, g in qs if g in pos]
    qtext = [q for q, g in qs if g in pos]

    # BM25
    corpus = [cat[t].text() if t in cat else "" for t in track_ids]
    bm = bm25s.BM25()
    bm.index(bm25s.tokenize(corpus, show_progress=False))
    bm_idx, _ = bm.retrieve(bm25s.tokenize(qtext, show_progress=False), k=1000, show_progress=False)

    # dense
    enc = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", device=a.device)
    Q = enc.encode(qtext, prompt_name="query", batch_size=16, normalize_embeddings=True,
                   show_progress_bar=False)
    dense_ranks = {}
    for mod in ["metadata-qwen3_embedding_0.6b", "attributes-qwen3_embedding_0.6b"]:
        M = load_emb(mod, track_ids)
        dense_ranks[mod] = topk_from_sims(Q @ M.T, 1000)

    print(f"n={len(golds)}")
    print("BM25            ", recall(bm_idx, golds, KS))
    print("metadata        ", recall(dense_ranks["metadata-qwen3_embedding_0.6b"], golds, KS))
    print("attributes      ", recall(dense_ranks["attributes-qwen3_embedding_0.6b"], golds, KS))
    fused = rrf([bm_idx, dense_ranks["metadata-qwen3_embedding_0.6b"],
                 dense_ranks["attributes-qwen3_embedding_0.6b"]], 1000)
    print("RRF fusion      ", recall(fused, golds, KS))


if __name__ == "__main__":
    main()
