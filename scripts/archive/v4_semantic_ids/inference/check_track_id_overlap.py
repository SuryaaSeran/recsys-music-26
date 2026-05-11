import pickle
from datasets import load_dataset
from collections import Counter

with open("data/codebook_v2.pkl", "rb") as f:
    cb = pickle.load(f)

codebook_ids = set(map(str, cb["track_to_codes"].keys()))

ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
conv_ds = ds["train"]

music_ids = []

for row in conv_ds:
    for turn in row["conversations"]:
        if turn.get("role") == "music":
            tid = str(turn.get("content", "")).strip()
            if tid:
                music_ids.append(tid)

music_id_set = set(music_ids)
overlap = music_id_set & codebook_ids

print("Codebook track IDs:", len(codebook_ids))
print("Conversation music turns:", len(music_ids))
print("Unique conversation music IDs:", len(music_id_set))
print("Overlap:", len(overlap))
print("Overlap %:", round(100 * len(overlap) / max(1, len(music_id_set)), 2))

print("\nSample conversation music IDs:")
for tid in list(music_id_set)[:10]:
    print(" ", tid, "IN_CODEBOOK" if tid in codebook_ids else "MISSING")

print("\nMost common music IDs:")
for tid, count in Counter(music_ids).most_common(10):
    print(tid, count, "IN_CODEBOOK" if tid in codebook_ids else "MISSING")
