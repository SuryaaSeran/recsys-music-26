from datasets import load_dataset

ds = load_dataset("talkpl-ai/TalkPlayData-Challenge-Track-Metadata")
print(ds)

for split in ds.keys():
    d = ds[split]
    print("\n" + "=" * 80)
    print("Split:", split)
    print("Rows:", len(d))
    print("Columns:", d.column_names)

    print("\nFirst example:")
    ex = d[0]
    for k, v in ex.items():
        s = str(v)
        if len(s) > 700:
            s = s[:700] + "..."
        print(f"{k}: {s}")
