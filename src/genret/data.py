"""Build (conversational context -> gold cf-bpr tuple) examples from sessions."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

HUB = os.path.expanduser("~/.cache/huggingface/hub")
DATASET = "datasets--talkpl-ai--TalkPlayData-Challenge-Dataset"

# Blind A terminating-turn histogram (counts out of 80 sessions). Training mimics
# Blind A: one terminal prediction per session, with this turn-depth distribution.
BLIND_TURN_DIST = {1: 20, 2: 15, 3: 10, 4: 5, 5: 8, 6: 9, 7: 8, 8: 5}
MOVES = "MOVES_TOWARD_GOAL"


def load_split(split: str) -> pd.DataFrame:
    """split: 'train' or 'test' (test = dev)."""
    paths = sorted(glob.glob(f"{HUB}/{DATASET}/snapshots/*/data/{split}-*.parquet"))
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def load_cf_map(path: str = "exp/ids/semantic_ids.json") -> dict[str, list[int] | None]:
    """track_id -> cf-bpr 4-codes, or None if that track has no cf-bpr."""
    sem = json.loads(Path(path).read_text())
    out: dict[str, list[int] | None] = {}
    for tid, e in sem.items():
        cf = e["cf-bpr"]
        out[tid] = None if cf[0] < 0 else cf
    return out


def _profile_line(p: dict) -> str:
    f = [p.get("age_group"), p.get("country_name"), p.get("gender"),
         p.get("preferred_language"), p.get("preferred_musical_culture")]
    return "profile: " + ", ".join(str(x) for x in f if x)


def _goal_line(g: dict) -> str:
    return f"goal: {g.get('category')} / {g.get('specificity')} / {g.get('listener_goal')}"


def iter_music_turns(session):
    """Yield (music_index, gold_track_id, prior_turns, turn_number) for each music turn.
    prior_turns = ordered list of turn dicts strictly before the music turn."""
    conv = list(session["conversations"])
    for i, t in enumerate(conv):
        if t["role"] == "music":
            yield i, t["content"], conv[:i], t["turn_number"]


def assessment_map(session) -> dict[int, str]:
    """turn_number -> goal_progress_assessment."""
    out = {}
    for a in session["goal_progress_assessments"]:
        out[int(a["turn_number"])] = a["goal_progress_assessment"]
    return out


def sample_terminal_turn(available: list[int], rng) -> int:
    """Sample one terminating turn_number from the Blind A histogram, restricted to
    the turns actually present in this session."""
    avail = sorted(set(available) & set(BLIND_TURN_DIST))
    if not avail:
        avail = sorted(available)
        w = np.ones(len(avail))
    else:
        w = np.array([BLIND_TURN_DIST[t] for t in avail], dtype=float)
    return int(rng.choice(avail, p=w / w.sum()))


def render_context(session, prior_turns, sem_tok, cf_map, with_history=True,
                   max_recent_turns=3) -> str:
    lines = [_profile_line(session["user_profile"]), _goal_line(session["conversation_goal"])]

    if with_history:
        blocks = []
        for t in prior_turns:
            if t["role"] == "music":
                cf = cf_map.get(t["content"])
                if cf is not None:
                    blocks.append(sem_tok.cf_codes_to_str(cf))
        if blocks:
            lines.append("<hist> " + " ".join(blocks) + " </hist>")

    dialogue = [t for t in prior_turns if t["role"] in ("user", "assistant")]
    for t in dialogue[-max_recent_turns:]:
        lines.append(f"{t['role']}: {t['content']}")

    lines.append("<gen>")
    return "\n".join(lines)


def build_example(session, prior_turns, gold_track_id, sem_tok, cf_map, tokenizer,
                  with_history=True, max_recent_turns=3, max_ctx_tokens=1024):
    """Return {input_ids, labels} or None if gold has no cf-bpr (ungeneratable)."""
    gold_cf = cf_map.get(gold_track_id)
    if gold_cf is None:
        return None
    ctx = render_context(session, prior_turns, sem_tok, cf_map, with_history, max_recent_turns)
    ctx_ids = tokenizer(ctx, add_special_tokens=True).input_ids
    target_ids = sem_tok.cf_codes_to_tokens(gold_cf) + [sem_tok.eos_id]
    if len(ctx_ids) > max_ctx_tokens:
        ctx_ids = ctx_ids[-max_ctx_tokens:]          # keep the recent tail + <gen>
    input_ids = ctx_ids + target_ids
    labels = [-100] * len(ctx_ids) + target_ids
    return {"input_ids": input_ids, "labels": labels}


def build_train_examples(df, sem_tok, cf_map, tokenizer, seed=0, **kw) -> list[dict]:
    """One terminal example per session: sample a terminating turn from the Blind A
    histogram and predict that music turn given its prior context."""
    rng = np.random.default_rng(seed)
    out = []
    for _, s in df.iterrows():
        turns = list(iter_music_turns(s))
        avail = [tn for _, _, _, tn in turns]
        T = sample_terminal_turn(avail, rng)
        for _, gold, prior, tn in turns:
            if tn == T:
                ex = build_example(s, prior, gold, sem_tok, cf_map, tokenizer, **kw)
                if ex is not None:
                    out.append(ex)
                break
    return out


def build_dev_examples(df, sem_tok, cf_map, with_history=True, max_recent_turns=3,
                       moves_only=True) -> list[dict]:
    """For eval: store context string + gold metadata. Generation happens later.

    moves_only: keep recommendation R_T only when the listener's verdict on it is
    MOVES_TOWARD_GOAL. That verdict lives at gpa_{T+1} (gpa at turn T judges R_{T-1}).
    This auto-drops the last, unlabeled recommendation (R_8 has no gpa_9) and includes
    turn 1 (judged by gpa_2)."""
    out = []
    for _, s in df.iterrows():
        amap = assessment_map(s)
        for _, gold, prior, tn in iter_music_turns(s):
            # Drop only turns with an explicit non-MOVES verdict. Keep MOVES turns and
            # keep the verdict-less final turn (R_T at the last turn has no gpa_{T+1}
            # but still has a gold track, so it is valid for recall).
            if moves_only:
                verdict = amap.get(int(tn) + 1)
                if verdict is not None and verdict != MOVES:
                    continue
            gold_cf = cf_map.get(gold)
            out.append({
                "context": render_context(s, prior, sem_tok, cf_map, with_history, max_recent_turns),
                "gold_track_id": gold,
                "gold_cf": gold_cf,
                "gold_has_cf": gold_cf is not None,
                "session_id": s["session_id"],
                "turn_number": int(tn),
            })
    return out


def collate(batch, pad_id):
    import torch
    maxlen = max(len(b["input_ids"]) for b in batch)
    input_ids, labels, attn = [], [], []
    for b in batch:
        n = maxlen - len(b["input_ids"])
        input_ids.append(b["input_ids"] + [pad_id] * n)
        labels.append(b["labels"] + [-100] * n)
        attn.append([1] * len(b["input_ids"]) + [0] * n)
    return {
        "input_ids": torch.tensor(input_ids),
        "labels": torch.tensor(labels),
        "attention_mask": torch.tensor(attn),
    }
