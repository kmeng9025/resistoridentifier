import csv
import numpy as np
import json

IN_CSV = "labeled_means.csv"
OUT_JSON = "centroids_norm.json"

def bgr_to_norm(bgr):
    b, g, r = bgr
    s = b + g + r
    if s <= 1e-6:
        return np.array([0.0, 0.0, 0.0], dtype=np.float32)
    return np.array([b/s, g/s, r/s], dtype=np.float32)

by = {}
with open(IN_CSV, "r", newline="") as f:
    r = csv.DictReader(f)
    for row in r:
        label = row["label"].strip().lower()
        b = float(row["b"]); g = float(row["g"]); rr = float(row["r"])
        by.setdefault(label, []).append(bgr_to_norm((b,g,rr)))

centroids = {k: np.mean(np.stack(v, axis=0), axis=0).tolist() for k, v in by.items()}

with open(OUT_JSON, "w") as f:
    json.dump({"centroids_norm": centroids}, f, indent=2)

print("Wrote", OUT_JSON)
print("Labels:", ", ".join(sorted(centroids.keys())))
