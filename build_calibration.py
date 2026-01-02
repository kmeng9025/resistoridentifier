import json
import numpy as np
from collections import defaultdict

SAMPLES_JSONL = "band_samples.jsonl"
OUT_JSON = "calibration.json"

data = []
with open(SAMPLES_JSONL, "r") as f:
    for line in f:
        rec = json.loads(line)
        if rec.get("label", "").strip() == "":
            continue
        data.append(rec)

by = defaultdict(list)
for rec in data:
    label = rec["label"].strip().lower()
    by[label].append(rec)

if not by:
    raise RuntimeError("No labeled samples found. Fill in the 'label' field in band_samples.jsonl.")

classes = {}
for label, items in by.items():
    chrom = np.array([it["chrom"] for it in items], dtype=np.float32)
    bright = np.array([it["bright"] for it in items], dtype=np.float32)

    classes[label] = {
        "n": int(len(items)),
        "chrom_mean": chrom.mean(axis=0).tolist(),
        "chrom_std": (chrom.std(axis=0) + 1e-4).tolist(),
        "bright_mean": float(bright.mean()),
        "bright_std": float(bright.std() + 1e-4),
    }

out = {"classes": classes, "notes": "Computed from your pipeline's band ROIs (median BGR -> chrom + brightness)."}
with open(OUT_JSON, "w") as f:
    json.dump(out, f, indent=2)

print("Wrote", OUT_JSON)
print("Labels:", ", ".join(sorted(classes.keys())))
for k in sorted(classes.keys()):
    print(k, "n=", classes[k]["n"], "chrom_mean=", np.round(classes[k]["chrom_mean"], 3), "bright_mean=", round(classes[k]["bright_mean"], 1))
