"""Track catalog parsing for TalkPlayData.

Gotchas (verified against all_tracks, 47071 rows):
- Most metadata fields are numpy arrays, NOT scalars, and NOT reliably length-1.
    track_name   1..3      album_name 1..2      album_id 1..10
    artist_name  1..31     artist_id  1..33     tag_list  0..105   ISRC 0..1
- artist_name and artist_id counts can DIFFER on the same row. Do not zip them.
- Empty arrays exist: 1334 tracks have no ISRC, 87 have no tags.
- release_date is "YYYY-MM-DD" or "" (644 empty). popularity/duration never null.
- Track embeddings parquet has a row for all 47071 track_ids, but each MODALITY has
  gaps (empty array): clap/attributes/lyrics/metadata miss 492 each, siglip2 586,
  cf-bpr 616. Missing modality vectors are zero-filled; a presence mask is returned.
"""
from __future__ import annotations
import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

HUB = os.path.expanduser("~/.cache/huggingface/hub")


def _pq(ds: str, name: str) -> pd.DataFrame:
    paths = glob.glob(f"{HUB}/datasets--talkpl-ai--{ds}/snapshots/*/data/{name}*.parquet")
    if not paths:
        raise FileNotFoundError(f"no parquet for {ds}/{name}*")
    return pd.concat([pd.read_parquet(p) for p in sorted(paths)], ignore_index=True)


def _as_list(v) -> list:
    """Coerce any cell (numpy array / None / scalar) to a plain list."""
    if v is None:
        return []
    if isinstance(v, np.ndarray):
        return [x for x in v.tolist() if x is not None]
    if isinstance(v, (list, tuple)):
        return [x for x in v if x is not None]
    return [v]


def _first(v, default: str = "") -> str:
    lst = _as_list(v)
    return str(lst[0]) if lst else default


def _year(release_date) -> int | None:
    if not release_date or not isinstance(release_date, str) or len(release_date) < 4:
        return None
    head = release_date[:4]
    return int(head) if head.isdigit() else None


@dataclass
class Track:
    track_id: str
    name: str                       # primary title (first element)
    artists: list[str]              # all artist names, order preserved
    album: str
    tags: list[str]
    popularity: float
    year: int | None
    duration_ms: int
    isrc: str | None
    artist_ids: list[str]           # may differ in length from artists
    album_ids: list[str]
    names_all: list[str] = field(default_factory=list)  # alt titles if any

    @property
    def artist(self) -> str:
        return ", ".join(self.artists)

    def text(self) -> str:
        """Compact textual surface form for BM25 / encoders."""
        parts = [self.name, "by", self.artist]
        if self.album:
            parts += ["| Album:", self.album]
        if self.tags:
            parts += ["| Tags:", " ".join(self.tags)]
        if self.year:
            parts += ["|", str(self.year)]
        return " ".join(parts)


def row_to_track(r) -> Track:
    return Track(
        track_id=str(r["track_id"]),
        name=_first(r["track_name"]),
        artists=[str(x) for x in _as_list(r["artist_name"])],
        album=_first(r["album_name"]),
        tags=[str(x) for x in _as_list(r["tag_list"])],
        popularity=float(r["popularity"]),
        year=_year(r["release_date"]),
        duration_ms=int(r["duration"]),
        isrc=_first(r["ISRC"]) or None,
        artist_ids=[str(x) for x in _as_list(r["artist_id"])],
        album_ids=[str(x) for x in _as_list(r["album_id"])],
        names_all=[str(x) for x in _as_list(r["track_name"])],
    )


def load_catalog() -> dict[str, Track]:
    """track_id -> Track for all 47071 catalog tracks."""
    m = _pq("TalkPlayData-Challenge-Track-Metadata", "all_tracks")
    return {str(r["track_id"]): row_to_track(r) for _, r in m.iterrows()}


# --- embeddings (kept per-modality, NOT concatenated) -------------------------

# Each modality is a separate vector space. Keep them apart so the model decides
# how to combine (separate towers / per-modality similarity / late fusion).
EMB_DIMS = {
    "audio-laion_clap": 512,
    "image-siglip2": 768,
    "cf-bpr": 128,
    "attributes-qwen3_embedding_0.6b": 1024,
    "lyrics-qwen3_embedding_0.6b": 1024,
    "metadata-qwen3_embedding_0.6b": 1024,
}


@dataclass
class TrackEmbeddings:
    track_ids: list[str]                 # shared row order across all modalities
    index: dict[str, int]                # track_id -> row
    matrices: dict[str, np.ndarray]      # modality -> [N, dim] float32 (0-filled if missing)
    present: dict[str, np.ndarray]       # modality -> [N] bool, True where real vector

    def vec(self, track_id: str, modality: str) -> np.ndarray | None:
        i = self.index.get(track_id)
        if i is None or not self.present[modality][i]:
            return None
        return self.matrices[modality][i]


def load_track_embeddings(modalities: list[str] | None = None) -> TrackEmbeddings:
    """Load each modality as its own [N, dim] matrix, aligned to one track_id list.

    Modalities are NOT concatenated. Rows missing a given modality are zero-filled
    and flagged False in `present[modality]`.
    """
    cols = modalities or list(EMB_DIMS)
    e = _pq("TalkPlayData-Challenge-Track-Embeddings", "all_tracks")
    ids = e["track_id"].astype(str).tolist()
    n = len(ids)
    matrices, present = {}, {}
    for c in cols:
        dim = EMB_DIMS[c]
        mat = np.zeros((n, dim), dtype=np.float32)
        mask = np.zeros(n, dtype=bool)
        for i, v in enumerate(e[c].to_numpy()):
            if v is not None and len(v) == dim:
                mat[i] = v
                mask[i] = True
        matrices[c], present[c] = mat, mask
    return TrackEmbeddings(ids, {t: i for i, t in enumerate(ids)}, matrices, present)


if __name__ == "__main__":
    cat = load_catalog()
    print("catalog tracks:", len(cat))
    # show a collaboration where artist_name/artist_id lengths differ
    for t in cat.values():
        if len(t.artists) != len(t.artist_ids):
            print("mismatch example:", t.track_id)
            print("  artists   :", t.artists)
            print("  artist_ids:", t.artist_ids)
            print("  text():", t.text())
            break
    emb = load_track_embeddings()
    print("embedding rows:", len(emb.track_ids))
    for c in EMB_DIMS:
        print(f"  {c:35} dim={emb.matrices[c].shape[1]:4d} present={int(emb.present[c].sum())}")
