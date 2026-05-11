from datasets import load_dataset
import json

ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Dataset")
print(ds)

for split in ds.keys():
    print("\n" + "=" * 80)
    print("SPLIT:", split)
    d = ds[split]
    print("Rows:", len(d))
    print("Columns:", d.column_names)

    print("\nFirst 3 examples:")
    for i in range(min(3, len(d))):
        print("\n--- example", i, "---")
        ex = d[i]
        for k, v in ex.items():
            s = str(v)
            if len(s) > 1000:
                s = s[:1000] + "..."
            print(f"{k}: {s}")

    print("\nFields containing track/item/song/music/id/target/rec:")
    for k in d.column_names:
        lk = k.lower()
        if any(x in lk for x in ["track", "item", "song", "music", "target", "rec", "id"]):
            print(" ", k)
