"""Prefix trie over valid cf-bpr 4-tuples for constrained decoding."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.genret.config import CF_LEVELS, CODES_NPY, TRACK_IDS_JSON


class CfTrie:
    def __init__(self, root: dict, tuple_to_tracks: dict, gen_id: int, eos_id: int):
        self.root = root                      # nested dict in TOKEN space, depth 4 -> {eos_id: None}
        self.tuple_to_tracks = tuple_to_tracks  # (c0,c1,c2,c3) -> [track_id]
        self.gen_id = gen_id
        self.eos_id = eos_id

    @classmethod
    def build(cls, sem_tok, codes_npy: str = CODES_NPY, track_ids_json: str = TRACK_IDS_JSON) -> "CfTrie":
        codes = np.load(codes_npy)[:, :CF_LEVELS]                 # [N,4] int32
        track_ids = json.loads(Path(track_ids_json).read_text())
        valid = (codes >= 0).all(axis=1)

        root: dict = {}
        tuple_to_tracks: dict = {}
        for i in np.where(valid)[0]:
            quad = tuple(int(x) for x in codes[i])
            tuple_to_tracks.setdefault(quad, []).append(track_ids[i])
            toks = sem_tok.cf_codes_to_tokens(quad)
            node = root
            for t in toks:
                node = node.setdefault(t, {})
            node[sem_tok.eos_id] = None
        return cls(root, tuple_to_tracks, sem_tok.gen_id, sem_tok.eos_id)

    def n_leaves(self) -> int:
        return len(self.tuple_to_tracks)

    def prefix_allowed_tokens_fn(self):
        """Returns fn(batch_id, input_ids) -> list[int] of allowed next tokens.
        Walks the suffix after the last <gen> through the trie."""
        gen_id, root = self.gen_id, self.root

        def fn(batch_id, input_ids):
            ids = input_ids.tolist()
            # position after the last <gen>
            pos = len(ids) - 1 - ids[::-1].index(gen_id) + 1 if gen_id in ids else len(ids)
            node = root
            for t in ids[pos:]:
                nxt = node.get(int(t))
                if nxt is None:
                    return [self.eos_id]
                node = nxt
            return list(node.keys())

        return fn

    def decode_tuple_to_tracks(self, cf_token_codes) -> list[str]:
        quad = tuple(int(c) for c in cf_token_codes)
        return self.tuple_to_tracks.get(quad, [])
