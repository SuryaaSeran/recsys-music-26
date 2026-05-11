"""
Dataset builder for joint retrieval + response training.

Each training example is one dialogue turn. The model sees:
  [user profile] + [listening history as ID tokens] + [conversation so far]

And must produce:
  [20 semantic ID sequences] + [natural language response]

This collapses retrieval and generation into a single seq2seq objective,
which is the core contribution of the Speak Spotify architecture.
"""

import json
import pickle
import random
from pathlib import Path
from typing import Optional

from loguru import logger
from torch.utils.data import Dataset

from src.model.music_crs_model import N_COARSE, codes_to_token_str


SYSTEM_PROMPT = (
    "You are a conversational music recommender. "
    "Given the user profile, listening history, and conversation, "
    "recommend exactly 20 tracks as semantic ID pairs, one per line, "
    "then write a natural language response explaining your choices."
)


def format_user_profile(profile: dict) -> str:
    parts = []
    if profile.get("age"):
        parts.append(f"age={profile['age']}")
    if profile.get("gender"):
        parts.append(f"gender={profile['gender']}")
    if profile.get("country"):
        parts.append(f"country={profile['country']}")
    return "[USER PROFILE] " + ", ".join(parts) if parts else "[USER PROFILE] unknown"


def format_history(track_ids: list[str], track_to_codes: dict, max_tracks: int = 20) -> str:
    """Format listening history as a sequence of semantic ID tokens."""
    tokens = []
    for tid in track_ids[-max_tracks:]:
        if tid in track_to_codes:
            c1, c2 = track_to_codes[tid]
            tokens.append(codes_to_token_str(c1, c2))
    if not tokens:
        return "[HISTORY] (none)"
    return "[HISTORY] " + " | ".join(tokens)


def format_dialogue(turns: list[dict], current_turn_idx: int) -> str:
    """Format dialogue history up to (and including) the current user turn."""
    lines = ["[DIALOGUE]"]
    for i, turn in enumerate(turns[:current_turn_idx + 1]):
        role = turn.get("role", "user")
        text = turn.get("text", "")
        if role == "user":
            lines.append(f"  User: {text}")
        else:
            lines.append(f"  System: {text}")
    return "\n".join(lines)


def format_target(
    ground_truth_track_ids: list[str],
    track_to_codes: dict,
    response_text: str,
) -> str:
    """Format the target output: 20 ID pairs followed by the response."""
    id_lines = []
    for tid in ground_truth_track_ids[:20]:
        if tid in track_to_codes:
            c1, c2 = track_to_codes[tid]
            id_lines.append(codes_to_token_str(c1, c2))
        else:
            # Fallback: track not in codebook (cold-start) — skip
            continue
    # Pad to 20 if needed with most-common ID
    while len(id_lines) < 20:
        id_lines.append("<0> <256>")

    ids_block = "\n".join(id_lines)
    return f"{ids_block}\n[RESPONSE] {response_text}"


class MusicCRSDataset(Dataset):
    """
    Dataset for one split (train / dev).

    Expected data structure under data_path/split/:
        sessions.jsonl  — one session per line
        {
          "session_id": "...",
          "user_profile": {"age": 28, "gender": "F", "country": "US"},
          "listening_history": ["track_id_1", "track_id_2", ...],
          "turns": [
            {
              "turn_id": 1,
              "role": "user",
              "text": "...",
              "ground_truth_tracks": ["track_id_a", ...],  // only in train/dev
              "response": "..."                            // only in train/dev
            },
            ...
          ]
        }
    """

    def __init__(
        self,
        data_path: str,
        split: str,
        codebook_path: str,
        tokenizer,
        cfg: dict,
        english_mix_path: Optional[str] = None,
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.max_seq_length = cfg.get("max_seq_length", 2048)
        self.max_history = cfg.get("max_history_tracks", 20)
        self.english_mix_ratio = cfg.get("english_mix_ratio", 0.10)

        # Load codebook
        logger.info(f"Loading codebook from {codebook_path}...")
        with open(codebook_path, "rb") as f:
            codebook = pickle.load(f)
        self.track_to_codes: dict = codebook["track_to_codes"]

        # Load sessions
        session_file = self.data_path / split / "sessions.jsonl"
        logger.info(f"Loading sessions from {session_file}...")
        self.examples = []
        with open(session_file) as f:
            for line in f:
                session = json.loads(line)
                self._extract_examples(session)

        logger.info(f"[{split}] {len(self.examples):,} training examples (one per turn)")

        # Optional English mix-in (prevents catastrophic forgetting)
        self.english_examples = []
        if english_mix_path and Path(english_mix_path).exists():
            with open(english_mix_path) as f:
                self.english_examples = json.load(f)
            logger.info(f"Loaded {len(self.english_examples):,} English mix-in examples")

    def _extract_examples(self, session: dict):
        """Create one training example per user turn in the session."""
        profile = session.get("user_profile", {})
        history = session.get("listening_history", [])
        turns = session.get("turns", [])

        for i, turn in enumerate(turns):
            if turn.get("role") != "user":
                continue
            gt_tracks = turn.get("ground_truth_tracks", [])
            response = turn.get("response", "")
            if not gt_tracks or not response:
                continue

            self.examples.append({
                "session_id": session["session_id"],
                "turn_id": turn["turn_id"],
                "user_profile": profile,
                "listening_history": history,
                "turns": turns,
                "current_turn_idx": i,
                "ground_truth_tracks": gt_tracks,
                "response": response,
            })

    def _build_prompt(self, ex: dict) -> tuple[str, str]:
        """Returns (input_text, target_text)."""
        profile_str  = format_user_profile(ex["user_profile"])
        history_str  = format_history(ex["listening_history"], self.track_to_codes, self.max_history)
        dialogue_str = format_dialogue(ex["turns"], ex["current_turn_idx"])
        target_str   = format_target(ex["ground_truth_tracks"], self.track_to_codes, ex["response"])

        input_text = (
            f"<|system|>\n{SYSTEM_PROMPT}\n"
            f"{profile_str}\n"
            f"{history_str}\n"
            f"{dialogue_str}\n"
            f"<|assistant|>\n"
        )
        return input_text, target_str

    def __len__(self):
        n = len(self.examples)
        if self.english_examples:
            n += int(n * self.english_mix_ratio)
        return n

    def __getitem__(self, idx):
        # English mix-in: randomly sample from English examples
        if (
            self.english_examples
            and idx >= len(self.examples)
        ):
            ex = random.choice(self.english_examples)
            input_text  = ex["input"]
            target_text = ex["output"]
        else:
            ex = self.examples[idx % len(self.examples)]
            input_text, target_text = self._build_prompt(ex)

        full_text = input_text + target_text + self.tokenizer.eos_token

        tokenized = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_seq_length,
            padding=False,
            return_tensors=None,
        )

        # Build labels: -100 for input tokens (not trained), real IDs for target
        input_len = len(self.tokenizer(input_text, add_special_tokens=False)["input_ids"])
        labels = [-100] * input_len + tokenized["input_ids"][input_len:]

        tokenized["labels"] = labels
        return tokenized
