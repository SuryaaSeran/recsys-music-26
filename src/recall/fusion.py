"""Stage A recall: BM25 (named entities) + dense Qwen3 (content), RRF-fused.

Replaces the generative cf-bpr retriever (recall@200 ~0.20) with classic fusion
(validated ~0.43@200, ~0.60@1000, untrained). BM25 over track text catches the
artists/songs the user names; dense Qwen3 over metadata/attributes catches intent.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HUB = os.path.expanduser("~/.cache/huggingface/hub")
EMB = "datasets--talkpl-ai--TalkPlayData-Challenge-Track-Embeddings"
DENSE_MODS = ["metadata-qwen3_embedding_0.6b", "attributes-qwen3_embedding_0.6b"]
QWEN = "Qwen/Qwen3-Embedding-0.6B"


def render_query(session, prior_turns) -> str:
    """Conversation -> retrieval query: culture + goal + last-3 user/assistant turns."""
    p, g = session["user_profile"], session["conversation_goal"]
    head = f"{p.get('preferred_musical_culture')}. {g.get('listener_goal')}"
    dlg = [t for t in prior_turns if t["role"] in ("user", "assistant")][-3:]
    return head + " " + " ".join(t["content"] for t in dlg)


def _load_dense(modality: str, track_ids: list[str]) -> np.ndarray:
    paths = sorted(glob.glob(f"{HUB}/{EMB}/snapshots/*/data/all_tracks-*.parquet"))
    d = pd.concat([pd.read_parquet(f, columns=["track_id", modality]) for f in paths],
                  ignore_index=True)
    pos = {str(t): i for i, t in enumerate(d["track_id"].astype(str))}
    dim = len(next(v for v in d[modality] if v is not None and len(v)))
    M = np.zeros((len(track_ids), dim), np.float32)
    col = d[modality].to_numpy()
    for i, tid in enumerate(track_ids):
        v = col[pos[tid]]
        if v is not None and len(v):
            M[i] = v
    M /= (np.linalg.norm(M, axis=1, keepdims=True) + 1e-8)
    return M


class FusionRetriever:
    def __init__(self, track_ids, bm25, dense, encoder, rrf_c=60):
        self.track_ids = track_ids
        self.bm25 = bm25
        self.dense = dense              # {modality: [N, d] normalized}
        self.encoder = encoder
        self.rrf_c = rrf_c

    @classmethod
    def build(cls, cache_dir="data/cache", device="mps", dense_mods=DENSE_MODS):
        import bm25s
        from sentence_transformers import SentenceTransformer
        from src.tracks import load_catalog

        track_ids = json.loads((Path(cache_dir) / "track_ids.json").read_text())
        cat = load_catalog()
        corpus = [cat[t].text() if t in cat else "" for t in track_ids]
        bm = bm25s.BM25()
        bm.index(bm25s.tokenize(corpus, show_progress=False))
        dense = {m: _load_dense(m, track_ids) for m in dense_mods}
        enc = SentenceTransformer(QWEN, device=device)
        return cls(track_ids, bm, dense, enc)

    def _dense_topk(self, Q, M, k):
        sims = Q @ M.T
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        return np.take_along_axis(idx, np.argsort(-np.take_along_axis(sims, idx, 1), 1), 1)

    def retrieve(self, queries: list[str], k: int = 200, per_source_k: int = 1000) -> list[list[str]]:
        import bm25s
        rank_lists = [self.bm25.retrieve(bm25s.tokenize(queries, show_progress=False),
                                         k=per_source_k, show_progress=False)[0]]
        Q = self.encoder.encode(queries, prompt_name="query", normalize_embeddings=True,
                                batch_size=16, show_progress_bar=False)
        for M in self.dense.values():
            rank_lists.append(self._dense_topk(Q, M, per_source_k))

        out = []
        for qi in range(len(queries)):
            score: dict[int, float] = {}
            for rl in rank_lists:
                for rank, ti in enumerate(rl[qi]):
                    score[int(ti)] = score.get(int(ti), 0.0) + 1.0 / (self.rrf_c + rank)
            top = sorted(score, key=score.get, reverse=True)[:k]
            out.append([self.track_ids[i] for i in top])
        return out
