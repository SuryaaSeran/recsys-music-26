"""Build song -> song co-occurrence table from TRAIN sessions.

Decayed window: weight[d] = 1.0 if d==1, 0.5 if d==2, 0.25 if d==3.
For each (a, b) with b played d turns after a (d in {1,2,3}) within the same
session, add weight[d] to count[a][b].

Output: cache/cooccur/next_song.npz
    track_ids : (N,) array of track-id strings (catalog order)
    neigh_ids : (N, top_k) int32 array of neighbour indices into track_ids
    neigh_w   : (N, top_k) float32 array of decayed weights (descending per row)

A row with fewer than top_k neighbours is padded with neigh_ids=-1, neigh_w=0.
"""
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top_k", type=int, default=300)
    ap.add_argument("--window", type=int, default=3, help="Max gap d to count.")
    ap.add_argument("--min_weight", type=float, default=1.0,
                    help="Drop neighbours with total weight < this.")
    ap.add_argument("--out", default="cache/cooccur/next_song.npz")
    ap.add_argument("--exclude_seed", type=int, default=-1,
                    help="If >=0, shuffle TRAIN sessions with this seed and skip the first --exclude_n.")
    ap.add_argument("--exclude_n", type=int, default=0,
                    help="Number of sessions to skip from the front of the shuffled order.")
    args = ap.parse_args()

    weights = {1: 1.0, 2: 0.5, 3: 0.25}
    if args.window != 3:
        weights = {d: 1.0 / d for d in range(1, args.window + 1)}

    print("Loading TRAIN sessions...")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset", split="train")
    sessions = list(ds)
    if args.exclude_seed >= 0 and args.exclude_n > 0:
        import random as _r
        _r.Random(args.exclude_seed).shuffle(sessions)
        excluded = sessions[: args.exclude_n]
        sessions = sessions[args.exclude_n :]
        print(f"  Holding out {len(excluded)} sessions (seed {args.exclude_seed}), "
              f"counting on {len(sessions)}.")

    counts: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for sess in tqdm(sessions, desc="counting"):
        plays = [t["content"] for t in sess["conversations"] if t["role"] == "music"]
        n = len(plays)
        for i, a in enumerate(plays):
            for d in range(1, args.window + 1):
                j = i + d
                if j >= n:
                    break
                b = plays[j]
                if a == b:
                    continue
                counts[a][b] += weights[d]

    print(f"Unique source tracks with co-occurrences: {len(counts)}")

    # Build catalog index over the union of all tracks seen (we want one table
    # row per source track in our catalog; load catalog for ordering).
    print("Loading catalog...")
    meta = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata",
                        split="all_tracks")
    track_ids = [r["track_id"] for r in meta]
    tid2idx = {t: i for i, t in enumerate(track_ids)}
    n_tracks = len(track_ids)

    neigh_ids = np.full((n_tracks, args.top_k), -1, dtype=np.int32)
    neigh_w = np.zeros((n_tracks, args.top_k), dtype=np.float32)

    n_with_neighbours = 0
    for src, ngh in tqdm(counts.items(), desc="sorting"):
        si = tid2idx.get(src)
        if si is None:
            continue
        filt = [(t, w) for t, w in ngh.items() if w >= args.min_weight and t in tid2idx]
        if not filt:
            continue
        filt.sort(key=lambda x: -x[1])
        filt = filt[: args.top_k]
        for k, (tid, w) in enumerate(filt):
            neigh_ids[si, k] = tid2idx[tid]
            neigh_w[si, k] = w
        n_with_neighbours += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        track_ids=np.array(track_ids),
        neigh_ids=neigh_ids,
        neigh_w=neigh_w,
    )
    nz = (neigh_ids[:, 0] >= 0).sum()
    mean_nbrs = (neigh_ids >= 0).sum(axis=1)[neigh_ids[:, 0] >= 0].mean()
    print(f"Saved {args.out}")
    print(f"  catalog tracks: {n_tracks}")
    print(f"  rows with neighbours: {nz}  ({100*nz/n_tracks:.1f}%)")
    print(f"  mean neighbours per row (where any): {mean_nbrs:.1f}")


if __name__ == "__main__":
    main()
