"""Spotify enrichment cache build.

Resumable pipeline. Stages:
  1  resolve  : catalog artist name -> Spotify artist id (exact match only)
  2  artist   : per resolved id -> top-tracks (10) + related-artists (10-20)
  3  related  : top-tracks for the deduped union of related-artist ids
  4  match    : map Spotify tracks back to our 47K catalog by (name, artist)

State + intermediate outputs:
  cache/spotify_enrich/resolve.json
  cache/spotify_enrich/artist_info.json
  cache/spotify_enrich/related_top_tracks.json
  cache/spotify_enrich/expansion.json   <- final usable cache

Usage:
  python scripts/build/spotify_enrich.py --stage all
  python scripts/build/spotify_enrich.py --stage 1
  python scripts/build/spotify_enrich.py --stage match    (alias for stage 4)
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path
from urllib.parse import quote

import requests
from datasets import load_dataset

CLIENT_ID = os.environ.get("SPOTIFY_CLIENT_ID", "33f5a2096dfc46d6a420901dd09e6f01")
CLIENT_SECRET = os.environ.get("SPOTIFY_CLIENT_SECRET", "c63c41a2313943c693004a0ac412bed7")

CACHE = Path("cache/spotify_enrich")
CACHE.mkdir(parents=True, exist_ok=True)

RESOLVE_PATH  = CACHE / "resolve.json"
ARTIST_PATH   = CACHE / "artist_info.json"
RELATED_PATH  = CACHE / "related_top_tracks.json"
EXPANSION_PATH = CACHE / "expansion.json"


# ─── Token + HTTP ──────────────────────────────────────────────────────────────

_TOKEN = {"value": None, "exp": 0}

def get_token() -> str:
    if _TOKEN["value"] and time.time() < _TOKEN["exp"] - 60:
        return _TOKEN["value"]
    creds = base64.b64encode(f"{CLIENT_ID}:{CLIENT_SECRET}".encode()).decode()
    r = requests.post(
        "https://accounts.spotify.com/api/token",
        data={"grant_type": "client_credentials"},
        headers={"Authorization": f"Basic {creds}"},
        timeout=30,
    )
    r.raise_for_status()
    j = r.json()
    _TOKEN["value"] = j["access_token"]
    _TOKEN["exp"] = time.time() + j.get("expires_in", 3600)
    return _TOKEN["value"]


def spotify_get(path: str, max_retries: int = 6) -> dict | None:
    """GET with token refresh + 429 backoff. Returns body dict or None on 404."""
    url = f"https://api.spotify.com{path}"
    delay = 0.4
    for attempt in range(max_retries):
        t = get_token()
        r = requests.get(url, headers={"Authorization": f"Bearer {t}"}, timeout=30)
        if r.status_code == 200:
            return r.json()
        if r.status_code == 401:
            _TOKEN["value"] = None
            continue
        if r.status_code == 429:
            wait = int(r.headers.get("Retry-After", "5")) + 1
            time.sleep(wait)
            continue
        if r.status_code in (404, 400):
            return None
        if 500 <= r.status_code < 600:
            time.sleep(delay)
            delay = min(delay * 2, 30)
            continue
        # unknown
        print(f"  ! status={r.status_code} for {path}: {r.text[:200]}", file=sys.stderr)
        return None
    return None


# ─── Catalog ───────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"[\(\[].*?[\)\]]", "", s)
    s = re.sub(r"\s*(feat\.?|ft\.?)\s.*$", "", s, flags=re.I)
    s = re.sub(r"[^a-z0-9 ]", " ", s.lower())
    return re.sub(r"\s+", " ", s).strip()


def load_catalog():
    """Return: (catalog_artists_list, name_artist_to_tid)."""
    print("Loading catalog...")
    ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata", split="all_tracks")
    artists = set()
    name_artist_to_tid: dict[tuple[str, str], list[str]] = {}
    track_meta: dict[str, dict] = {}
    for r in ds:
        tid = r["track_id"]
        track_meta[tid] = {
            "track_name": (r["track_name"] or [""])[0],
            "artist_name": r["artist_name"] or [],
            "popularity": float(r.get("popularity") or 0.0),
        }
        n = _norm((r["track_name"] or [""])[0])
        for a in (r["artist_name"] or []):
            if a and a.strip():
                artists.add(a.strip())
                k = (n, _norm(a))
                name_artist_to_tid.setdefault(k, []).append(tid)
    return sorted(artists), name_artist_to_tid, track_meta


def load_json(p: Path) -> dict:
    return json.loads(p.read_text()) if p.exists() else {}


def save_json(p: Path, obj):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(p)


# ─── Stage 1: resolve catalog artists -> spotify ids ───────────────────────────

def stage_resolve(catalog_artists: list[str], limit: int = 0):
    out = load_json(RESOLVE_PATH)
    done = set(out.keys())
    todo = [a for a in catalog_artists if a not in done]
    if limit:
        todo = todo[:limit]
    print(f"Stage 1 (resolve): {len(done)} done, {len(todo)} todo (of {len(catalog_artists)})")
    last_save = time.time()
    for i, name in enumerate(todo):
        q = quote(name)
        body = spotify_get(f"/v1/search?q={q}&type=artist&limit=1")
        items = (body or {}).get("artists", {}).get("items", []) if body else []
        hit = items[0] if items else None
        if hit and hit.get("name", "").lower() == name.lower():
            out[name] = {"id": hit["id"], "name": hit["name"],
                         "followers": hit.get("followers", {}).get("total", 0),
                         "genres": hit.get("genres", []),
                         "exact": True}
        elif hit:
            out[name] = {"id": hit["id"], "name": hit["name"],
                         "followers": hit.get("followers", {}).get("total", 0),
                         "genres": hit.get("genres", []),
                         "exact": False}
        else:
            out[name] = None
        if time.time() - last_save > 30 or (i + 1) % 200 == 0:
            save_json(RESOLVE_PATH, out)
            last_save = time.time()
            ok = sum(1 for v in out.values() if v and v.get("exact"))
            print(f"  [{i+1}/{len(todo)}] saved. exact={ok}/{len(out)}")
    save_json(RESOLVE_PATH, out)
    ok = sum(1 for v in out.values() if v and v.get("exact"))
    fuzzy = sum(1 for v in out.values() if v and not v.get("exact"))
    print(f"Done. exact={ok}  fuzzy={fuzzy}  miss={sum(1 for v in out.values() if not v)}")


# ─── Stage 2: artist info (top-tracks + related) ──────────────────────────────

def stage_artist_info():
    resolve = load_json(RESOLVE_PATH)
    out = load_json(ARTIST_PATH)
    ids = sorted({v["id"] for v in resolve.values() if v and v.get("exact")})
    todo = [aid for aid in ids if aid not in out]
    print(f"Stage 2 (artist_info): {len(out)} done, {len(todo)} todo (of {len(ids)})")
    last_save = time.time()
    for i, aid in enumerate(todo):
        top  = spotify_get(f"/v1/artists/{aid}/top-tracks?market=US")
        rel  = spotify_get(f"/v1/artists/{aid}/related-artists")
        out[aid] = {
            "top_tracks": [
                {"id": t["id"], "name": t["name"],
                 "artists": [a["name"] for a in t.get("artists", [])],
                 "popularity": t.get("popularity", 0)}
                for t in (top or {}).get("tracks", [])
            ] if top else [],
            "related": [
                {"id": a["id"], "name": a["name"],
                 "popularity": a.get("popularity", 0)}
                for a in (rel or {}).get("artists", [])
            ] if rel else [],
        }
        if time.time() - last_save > 30 or (i + 1) % 100 == 0:
            save_json(ARTIST_PATH, out)
            last_save = time.time()
            print(f"  [{i+1}/{len(todo)}] saved.")
    save_json(ARTIST_PATH, out)
    print(f"Done. {len(out)} artists have info.")


# ─── Stage 3: top tracks for related artists ───────────────────────────────────

def stage_related_tracks():
    artist_info = load_json(ARTIST_PATH)
    # union of related ids that we don't already have top-tracks for
    have = {aid for aid in artist_info if artist_info[aid].get("top_tracks")}
    related_ids: set[str] = set()
    for a in artist_info.values():
        for r in a.get("related", []) or []:
            related_ids.add(r["id"])
    to_fetch = sorted(related_ids - have)
    out = load_json(RELATED_PATH)
    todo = [aid for aid in to_fetch if aid not in out]
    print(f"Stage 3 (related top-tracks): {len(out)} cached, {len(todo)} todo "
          f"({len(related_ids)} unique related, {len(have)} already known)")
    last_save = time.time()
    for i, aid in enumerate(todo):
        top = spotify_get(f"/v1/artists/{aid}/top-tracks?market=US")
        out[aid] = [
            {"id": t["id"], "name": t["name"],
             "artists": [a["name"] for a in t.get("artists", [])],
             "popularity": t.get("popularity", 0)}
            for t in (top or {}).get("tracks", [])
        ] if top else []
        if time.time() - last_save > 30 or (i + 1) % 200 == 0:
            save_json(RELATED_PATH, out)
            last_save = time.time()
            print(f"  [{i+1}/{len(todo)}] saved.")
    save_json(RELATED_PATH, out)
    print(f"Done. related top-tracks cached for {len(out)} artists.")


# ─── Stage 4: build expansion table ────────────────────────────────────────────

def stage_match(name_artist_to_tid):
    resolve = load_json(RESOLVE_PATH)
    artist_info = load_json(ARTIST_PATH)
    related_tracks = load_json(RELATED_PATH)

    # spotify_artist_id -> [our catalog tids] (direct top-tracks)
    sp_to_tids: dict[str, list[tuple[str, int]]] = {}

    def match_track_list(tracks):
        out = []
        for t in tracks:
            name = t.get("name", "")
            for a in t.get("artists", []):
                k = (_norm(name), _norm(a))
                tids = name_artist_to_tid.get(k)
                if tids:
                    for tid in tids:
                        out.append((tid, t.get("popularity", 0)))
                    break
        return out

    for aid, info in artist_info.items():
        sp_to_tids[aid] = match_track_list(info.get("top_tracks", []))
    for aid, tracks in related_tracks.items():
        sp_to_tids.setdefault(aid, [])
        if not sp_to_tids[aid]:
            sp_to_tids[aid] = match_track_list(tracks)

    # Build catalog_artist (lowercased) -> {direct: [...], related: [...]}
    expansion: dict[str, dict] = {}
    for catalog_name, v in resolve.items():
        if not v or not v.get("exact"):
            continue
        aid = v["id"]
        direct = sp_to_tids.get(aid, [])
        related_pool = []
        related_ids = [r["id"] for r in (artist_info.get(aid, {}).get("related") or [])]
        for rid in related_ids:
            for tid, pop in sp_to_tids.get(rid, []):
                related_pool.append((tid, pop, rid))
        expansion[catalog_name.lower()] = {
            "spotify_id": aid,
            "direct": [{"tid": tid, "pop": pop} for tid, pop in direct],
            "related": [{"tid": tid, "pop": pop, "via": rid}
                        for tid, pop, rid in related_pool],
        }

    save_json(EXPANSION_PATH, expansion)

    # Summary
    n_with_direct = sum(1 for v in expansion.values() if v["direct"])
    n_with_related = sum(1 for v in expansion.values() if v["related"])
    direct_counts = [len(v["direct"]) for v in expansion.values()]
    related_counts = [len(v["related"]) for v in expansion.values()]
    print(f"Saved {EXPANSION_PATH}")
    print(f"  catalog artists covered : {len(expansion)}")
    print(f"  artists w/ direct match : {n_with_direct}  (mean {sum(direct_counts)/max(1,len(direct_counts)):.1f})")
    print(f"  artists w/ related      : {n_with_related}  (mean {sum(related_counts)/max(1,len(related_counts)):.1f})")


# ─── CLI ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="all",
                    choices=["all", "1", "2", "3", "4", "resolve", "artist", "related", "match"])
    ap.add_argument("--limit", type=int, default=0, help="cap on artists per stage (for testing)")
    args = ap.parse_args()

    catalog_artists, name_artist_to_tid, _ = load_catalog()
    print(f"  catalog artists: {len(catalog_artists)}")

    stage = args.stage
    if stage in ("all", "1", "resolve"):
        stage_resolve(catalog_artists, limit=args.limit)
    if stage in ("all", "2", "artist"):
        stage_artist_info()
    if stage in ("all", "3", "related"):
        stage_related_tracks()
    if stage in ("all", "4", "match"):
        stage_match(name_artist_to_tid)


if __name__ == "__main__":
    main()
