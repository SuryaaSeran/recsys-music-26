"""
Compute Phase G within-artist features for the existing 6K LTR dump.
Saves only the 2 new feature columns (not the full 13GB X) to avoid
a 3-hour re-dump and excessive peak RAM.

Output:
  exp/analysis/ltr_phaseG_extra_cols.npz  -- shape (N, 2), float32
    col 0: within_artist_trans_rank
    col 1: within_artist_pop_rank

At train time, hstack original X with this file to get 44-feature X.
"""
import json, time
import numpy as np
from datasets import load_dataset, concatenate_datasets

SRC_NPZ  = "exp/analysis/ltr_phase_d_v8b_6k_features.npz"
SRC_META = "exp/analysis/ltr_phase_d_v8b_6k_features.meta.json"
OUT_NPZ  = "exp/analysis/ltr_phaseG_extra_cols.npz"
COLLAB_IDX = 25  # collab_score column

def main():
    t0 = time.time()

    print("Loading sidecar metadata...")
    meta = json.load(open(SRC_META))
    turn_meta = meta["turn_meta"]
    n_turns = meta["n_turns"]
    N = meta["n_rows"]
    print(f"  {n_turns} turns, {N} rows")

    print("Loading HuggingFace track metadata...")
    meta_ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
    all_tracks = concatenate_datasets([meta_ds["all_tracks"], meta_ds["test_tracks"]])
    metadata_dict = {row["track_id"]: row for row in all_tracks}
    print(f"  {len(metadata_dict)} tracks")

    print("Reading collab_score column via mmap...")
    d = np.load(SRC_NPZ, allow_pickle=True, mmap_mode="r")
    # group is monotone-sorted, so rows are already in turn order
    group = np.array(d["group"])          # 300MB -- needed for boundary check
    collab_col = np.array(d["X"][:, COLLAB_IDX], dtype=np.float32)  # 300MB
    del d
    print(f"  collab nonzero: {np.count_nonzero(collab_col)}")

    # find turn boundaries using the fact group is sorted
    boundaries = np.searchsorted(group, np.arange(n_turns + 1))
    del group

    print("Computing within-artist features...")
    wa_trans = np.zeros(N, dtype=np.float32)
    wa_pop   = np.zeros(N, dtype=np.float32)

    for tid_idx, tm in enumerate(turn_meta):
        cand_ids = tm["cand_ids"]
        ng = len(cand_ids)
        lo = int(boundaries[tid_idx])
        hi = int(boundaries[tid_idx + 1])
        if hi - lo != ng or ng <= 1:
            continue

        cs  = collab_col[lo:hi]   # collab scores, same order as cand_ids
        pop = np.zeros(ng, dtype=np.float32)
        artist = [""] * ng
        for i, t in enumerate(cand_ids):
            row_m = metadata_dict.get(t, {})
            a = (row_m.get("artist_name") or [""])[0] or ""
            artist[i] = a.lower()
            pr = row_m.get("popularity")
            try:
                pop[i] = float(pr) if pr is not None else 0.0
            except (TypeError, ValueError):
                pop[i] = 0.0

        artist_to_idxs: dict = {}
        for i, a in enumerate(artist):
            artist_to_idxs.setdefault(a, []).append(i)

        for a, idxs in artist_to_idxs.items():
            if len(idxs) <= 1:
                continue
            local_cs  = cs[idxs]
            local_pop = pop[idxs]
            any_t     = np.any(local_cs > 0)
            n_peers   = len(idxs)
            for rank_i, i in enumerate(idxs):
                r = lo + i
                if any_t:
                    wa_trans[r] = float(np.sum(local_cs < local_cs[rank_i])) / (n_peers - 1)
                wa_pop[r] = float(np.sum(local_pop < local_pop[rank_i])) / (n_peers - 1)

        if (tid_idx + 1) % 5000 == 0:
            print(f"  {tid_idx+1}/{n_turns} [{time.time()-t0:.0f}s]")

    print(f"wa_trans nonzero: {np.count_nonzero(wa_trans)} / {N}")
    print(f"wa_pop   nonzero: {np.count_nonzero(wa_pop)}   / {N}")
    extra = np.stack([wa_trans, wa_pop], axis=1)  # (N, 2)
    print(f"Saving {OUT_NPZ} shape {extra.shape}...")
    np.savez_compressed(OUT_NPZ, extra=extra,
                        feature_names=np.array(["within_artist_trans_rank", "within_artist_pop_rank"]))
    print(f"Saved. Total time: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
