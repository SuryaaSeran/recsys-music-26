#!/usr/bin/env python3
"""Evaluate RQ-VAE codebook quality on TalkPlay catalog (5-axis check).

Inputs: semantic_ids dump from extract_semantic_ids.py + track metadata
Outputs: prints purity metrics + saves per-bucket samples for inspection
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from datasets import load_dataset


def entropy(counts):
    counts = np.array(counts, dtype=np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    p = counts / total
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sids_dir", required=True, help="cache/semantic_ids/<run>")
    ap.add_argument("--print_buckets", type=int, default=5,
                    help="Show top-K populated L0 buckets in detail")
    args = ap.parse_args()

    sids_dir = Path(args.sids_dir)
    track_ids = np.load(sids_dir / "track_ids.npy", allow_pickle=True).tolist()
    codes = np.load(sids_dir / "semantic_ids.npy")  # (N, L)
    meta = json.load(open(sids_dir / "meta.json"))
    L = codes.shape[1]
    K = meta["codebook_size"]
    print(f"Loaded {len(track_ids):,} tracks × {L} levels × {K} codes")

    print("Loading track metadata from HF...")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")["all_tracks"]
    meta_dict = {row["track_id"]: row for row in ds}

    # Build per-track attributes
    tid_to_info = {}
    for tid in track_ids:
        m = meta_dict.get(tid, {})
        artists = sorted(a for a in (m.get("artist_id") or []) if a)
        primary_artist = artists[0] if artists else None
        tags = [(t or "").lower() for t in (m.get("tag_list") or []) if t]
        primary_tag = tags[0] if tags else None
        rel = m.get("release_date") or ""
        year = int(rel[:4]) if rel and rel[:4].isdigit() else None
        decade = (year // 10) * 10 if year else None
        tid_to_info[tid] = {
            "artist": primary_artist,
            "primary_tag": primary_tag,
            "tags": tags[:5],
            "decade": decade,
            "name": (m.get("track_name") or [""])[0],
            "artist_name": (m.get("artist_name") or [""])[0],
        }

    # Group tracks by each level / leaf
    l0_to_tids = defaultdict(list)
    leaf_to_tids = defaultdict(list)
    for i, tid in enumerate(track_ids):
        l0 = int(codes[i, 0])
        leaf = tuple(int(c) for c in codes[i])
        l0_to_tids[l0].append(tid)
        leaf_to_tids[leaf].append(tid)

    # ─── Axis 1: L0 bucket purity by artist/genre/decade ───
    print("\n=== L0 bucket purity ===")
    l0_purity_artist = []
    l0_purity_tag = []
    l0_purity_decade = []
    l0_entropy_tag = []
    for l0, tids in l0_to_tids.items():
        artists = [tid_to_info[t]["artist"] for t in tids if tid_to_info[t]["artist"]]
        tags = [tid_to_info[t]["primary_tag"] for t in tids if tid_to_info[t]["primary_tag"]]
        decades = [tid_to_info[t]["decade"] for t in tids if tid_to_info[t]["decade"]]
        if artists:
            top = Counter(artists).most_common(1)[0][1]
            l0_purity_artist.append(top / len(artists))
        if tags:
            counter = Counter(tags)
            top = counter.most_common(1)[0][1]
            l0_purity_tag.append(top / len(tags))
            l0_entropy_tag.append(entropy(list(counter.values())))
        if decades:
            top = Counter(decades).most_common(1)[0][1]
            l0_purity_decade.append(top / len(decades))

    print(f"  artist purity (top artist's share):  mean={np.mean(l0_purity_artist):.3f}  median={np.median(l0_purity_artist):.3f}  p90={np.percentile(l0_purity_artist, 90):.3f}")
    print(f"  primary-tag purity:                  mean={np.mean(l0_purity_tag):.3f}  median={np.median(l0_purity_tag):.3f}  p90={np.percentile(l0_purity_tag, 90):.3f}")
    print(f"  decade purity:                        mean={np.mean(l0_purity_decade):.3f}  median={np.median(l0_purity_decade):.3f}  p90={np.percentile(l0_purity_decade, 90):.3f}")
    print(f"  primary-tag entropy (bits):           mean={np.mean(l0_entropy_tag):.2f}  median={np.median(l0_entropy_tag):.2f}  max-uniform={np.log2(K):.2f}")

    # ─── Axis 2: leaf bucket size distribution (already in meta.json) ───
    print("\n=== Leaf bucket size distribution ===")
    for k, v in meta["leaf_stats"].items():
        print(f"  {k}: {v}")

    # ─── Axis 3: per-level code usage (already in meta.json) ───
    print("\n=== Per-level code usage ===")
    for s in meta["per_level"]:
        print(f"  L{s['level']}: {s['n_used']}/{K} used  min={s['min']}  median={s['median']}  max={s['max']}")

    # ─── Axis 4: inspect top-K populated L0 buckets for human readability ───
    print(f"\n=== Top {args.print_buckets} populated L0 buckets (inspect for purity) ===")
    sorted_l0 = sorted(l0_to_tids.items(), key=lambda kv: -len(kv[1]))
    for l0, tids in sorted_l0[:args.print_buckets]:
        artists = Counter(tid_to_info[t]["artist_name"] for t in tids if tid_to_info[t]["artist_name"])
        tags = Counter(tid_to_info[t]["primary_tag"] for t in tids if tid_to_info[t]["primary_tag"])
        decades = Counter(tid_to_info[t]["decade"] for t in tids if tid_to_info[t]["decade"])
        print(f"\n  L0={l0} ({len(tids)} tracks)")
        print(f"    top artists: {[a for a, _ in artists.most_common(5)]}")
        print(f"    top tags:    {[t for t, _ in tags.most_common(8)]}")
        print(f"    top decades: {[d for d, _ in decades.most_common(4)]}")
        sample_names = [
            f"{tid_to_info[t]['name']} – {tid_to_info[t]['artist_name']}"
            for t in tids[:3]
        ]
        print(f"    sample tracks: {sample_names}")

    # ─── Axis 5: also inspect leaf-bucket samples ───
    print(f"\n=== Top {args.print_buckets} populated leaf buckets ===")
    sorted_leaf = sorted(leaf_to_tids.items(), key=lambda kv: -len(kv[1]))
    for leaf, tids in sorted_leaf[:args.print_buckets]:
        tags = Counter(tid_to_info[t]["primary_tag"] for t in tids if tid_to_info[t]["primary_tag"])
        artists = Counter(tid_to_info[t]["artist_name"] for t in tids if tid_to_info[t]["artist_name"])
        print(f"\n  leaf={leaf} ({len(tids)} tracks)")
        print(f"    top tags:    {[t for t, _ in tags.most_common(8)]}")
        print(f"    top artists: {[a for a, _ in artists.most_common(5)]}")
        sample_names = [
            f"{tid_to_info[t]['name']} – {tid_to_info[t]['artist_name']}"
            for t in tids[:3]
        ]
        print(f"    sample tracks: {sample_names}")


if __name__ == "__main__":
    main()
