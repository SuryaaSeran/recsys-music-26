"""Encode all tracks, concatenate per-modality codes into semantic IDs, verify.

  python scripts/export_ids.py --verify
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.rqvae.encode import export


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--ckpt-dir", default="exp/codebooks")
    ap.add_argument("--out-dir", default="exp/ids")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()

    rep = export(a.cache_dir, a.ckpt_dir, a.out_dir, a.device, a.verify)
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
