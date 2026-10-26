"""Score detection + localization baselines on scene files with the same metrics as GIST.

  uv run python -m gist.train.eval_baselines --train dev_train --test dev_test --out runs/baselines_dev.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_curve

from gist.models.baselines import FEAT, IForestBaseline, LSTMAEBaseline, gpsjam_scores
from gist.models.localize import centroid, horizon_score, nls
from gist.train.dataset import SceneDataset

KM_PER_DEG = 111.195


def detection_metrics(y: np.ndarray, p: np.ndarray, threshold_classes: bool) -> dict:
    """``threshold_classes``: baselines with one anomaly score get a class by thresholding at
    the score's 1 % FPR operating point on normals (they cannot tell jam from spoof)."""
    r = {"n": int(len(y))}
    for c, name in ((1, "jam"), (2, "spoof")):
        pos = y == c
        if pos.sum() and (~pos).sum():
            r[f"{name}_ap"] = float(average_precision_score(pos, p[:, c]))
            fpr, tpr, _ = roc_curve(pos, p[:, c])
            r[f"{name}_recall_at_1fpr"] = float(np.interp(0.01, fpr, tpr))
    if threshold_classes:
        normal = y == 0
        thr = np.quantile(p[normal, 1], 0.99) if normal.any() else 0.5
        pred = np.where(p[:, 1] > thr, 1, 0)     # anomaly → "jam"; no way to say "spoof"
    else:
        pred = p.argmax(1)
    r["macro_f1"] = float(f1_score(y, pred, average="macro", labels=[0, 1, 2], zero_division=0))
    return r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train")
    ap.add_argument("--test", default="test")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tr = [SceneDataset(a.train)[i] for i in range(len(SceneDataset(a.train)))]
    te_ds = SceneDataset(a.test)
    te = [te_ds[i] for i in range(len(te_ds))]
    models = {"iforest": IForestBaseline().fit(tr), "lstm_ae": LSTMAEBaseline().fit(tr)}
    results: dict = {}
    for name in ("gpsjam", "iforest", "lstm_ae"):
        per_kind = {"real": ([], []), "sim": ([], [])}
        for s in te:
            n = s["x"].shape[0]
            keep = ~np.isnan(s["y_soft"]).any(1)
            if name == "gpsjam":
                p = gpsjam_scores(s["x"], s["mask"], s["xy_node"])
            else:
                p = models[name].scores(s["x"], s["mask"], s["xy_node"])
            kind = "sim" if bool(s["is_sim"]) else "real"
            per_kind[kind][0].append(np.nan_to_num(s["y_soft"]).argmax(1)[keep])
            per_kind[kind][1].append(p[:n][keep])
        results[name] = {k: detection_metrics(np.concatenate(ys), np.concatenate(ps), threshold_classes=True)
                         for k, (ys, ps) in per_kind.items() if ys}

    loc = {"centroid": [], "horizon": [], "nls": []}
    for s in te:
        em = s["emitter"]
        if not (em[3] > 0 and em[2] == 1):
            continue
        m = s["mask"][:, ::2]
        xy = s["pos_km"][:, ::2][m]
        h = s["h_m"][:, ::2][m]
        aff = (s["x"][:, ::2, FEAT["nic_min"]] * 11 <= 6.5)[m].astype(float)
        if aff.sum() < 3:
            continue
        # Scene tensors are in local km; the baselines expect degrees → convert around (0, 0).
        lat, lon = xy[:, 1] / KM_PER_DEG, xy[:, 0] / KM_PER_DEG
        for k, f in (("centroid", lambda: centroid(lat, lon, aff)), ("horizon", lambda: horizon_score(lat, lon, h, aff)),
                     ("nls", lambda: nls(lat, lon, h, aff)[:2])):
            la, lo = f()
            loc[k].append(float(np.hypot(la * KM_PER_DEG - em[1], lo * KM_PER_DEG - em[0])))
    results["localization"] = {k: {"n": len(v), "median_km": float(np.median(v)), "p95_km": float(np.percentile(v, 95))}
                               for k, v in loc.items() if v}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(results, indent=2))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
