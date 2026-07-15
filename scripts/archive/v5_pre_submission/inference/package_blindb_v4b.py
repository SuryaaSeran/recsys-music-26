"""Package Blind B v4b submission zip.

prediction.json = exp/inference/blind_b/blind_b_v8d_s3cap_v4b.json
zip at exp/inference/blind_b/submissions/v4b_v8d_untouched_richresp/submission.zip
"""
import json
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
src = ROOT / "exp/inference/blind_b/blind_b_v8d_s3cap_v4b.json"
outdir = ROOT / "exp/inference/blind_b/submissions/v4b_v8d_untouched_richresp"
outdir.mkdir(parents=True, exist_ok=True)

# Final validation before packaging
data = json.load(open(src))
assert len(data) == 80, len(data)
for r in data:
    assert set(r) == {"session_id", "user_id", "turn_number",
                      "predicted_track_ids", "predicted_response"}, r["session_id"]
    ids = r["predicted_track_ids"]
    assert len(ids) == 20 and len(set(ids)) == 20, r["session_id"]
    assert isinstance(r["predicted_response"], str) and r["predicted_response"].strip()

shutil.copy(src, outdir / "prediction.json")
zp = outdir / "submission.zip"
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    z.write(outdir / "prediction.json", "prediction.json")
print(f"packaged: {zp} ({zp.stat().st_size} bytes)")
