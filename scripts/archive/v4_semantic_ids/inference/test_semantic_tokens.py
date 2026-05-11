import pickle

with open("data/codebook_v2.pkl", "rb") as f:
    cb = pickle.load(f)

n_coarse = cb["n_coarse"]

def track_to_semantic_tokens(track_id: str):
    coarse, fine = cb["track_to_codes"][track_id]
    return f"<SID_{coarse}> <SID_{n_coarse + fine}>"

for i, tid in enumerate(cb["track_to_codes"].keys()):
    print(tid, "->", cb["track_to_codes"][tid], "->", track_to_semantic_tokens(tid))
    if i >= 10:
        break
