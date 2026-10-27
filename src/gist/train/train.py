"""Train / evaluate GIST (PLAN §8.2). Ablation switches map 1:1 to the paper's ablation table.

  uv run python -m gist.train.train --train dev_train --val dev_test --name dev
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as Fn
from sklearn.metrics import average_precision_score, f1_score, roc_curve
from torch.utils.data import DataLoader

from gist.models.gist import GIST
from gist.train.dataset import SceneDataset, collate
from gist.train.tensors import F

# Cap the MPS caching allocator (unified memory on an 8 GB Mac): without it the process grew to
# 7.6 GB over 12 epochs and pushed the machine into swap. Read when MPS first allocates.
os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

CLASS_W = torch.tensor([1.0, 2.0, 4.0])
LOC_SIGMA_KM = 40.0


def to_dev(batch, dev):
    return {k: v.to(dev) for k, v in batch.items()}


def losses(model, out, b):
    node_mask = b["node_mask"]
    y = b["y_soft"]
    has = node_mask & ~torch.isnan(y).any(-1)
    logp = out["node_logits"].log_softmax(-1)
    w = CLASS_W.to(y.device)
    yt = torch.nan_to_num(y)
    l_node = -(yt * logp * w).sum(-1)[has].sum() / has.sum().clamp(min=1)

    ys = b["y_step"].long()
    sm = (ys >= 0) & b["mask"]
    l_step = Fn.cross_entropy(out["step_logits"][sm], ys[sm], weight=w) if sm.any() else out["step_logits"].sum() * 0

    l_loc = out["node_logits"].sum() * 0
    em = b["emitter"]
    jam = (em[:, 3] > 0) & (em[:, 2] == 1)
    if "loc_logits" in out and jam.any():
        grid = model.loc.grid                                                # [G, 2]
        d2 = ((grid[None] - em[:, None, :2]) ** 2).sum(-1)                   # [B, G]
        target = (-d2 / (2 * LOC_SIGMA_KM**2)).softmax(-1)
        ll = out["loc_logits"].log_softmax(-1)
        l_ce = -(target * ll).sum(-1)[jam].mean()
        est = model.loc.soft_argmax(out["loc_logits"])
        l_l1 = (est - em[:, :2]).norm(dim=-1)[jam].mean() / 100
        l_loc = l_ce + l_l1
    return l_node, l_step, l_loc


@torch.no_grad()
def evaluate(model, loader, dev) -> dict:
    model.eval()
    rows = {"real": ([], []), "sim": ([], [])}
    loc_err, loc_err_ref = [], []
    for b in loader:
        b = to_dev(b, dev)
        out = model(b)
        p = out["node_logits"].softmax(-1)
        for i in range(p.shape[0]):
            kind = "sim" if bool(b["is_sim"][i]) else "real"
            m = b["node_mask"][i] & ~torch.isnan(b["y_soft"][i]).any(-1)
            rows[kind][0].append(b["y_soft"][i][m].argmax(-1).cpu().numpy())
            rows[kind][1].append(p[i][m].cpu().numpy())
            em = b["emitter"][i]
            if em[3] > 0 and em[2] == 1 and "loc_logits" in out:
                sl = slice(i, i + 1)
                est = model.loc.soft_argmax(out["loc_logits"][sl])[0]
                loc_err.append(float((est - em[:2]).norm()))
                ref = model.loc.refine(out["loc_logits"][sl], out["step_p"][sl], b["pos_km"][sl], b["h_m"][sl], b["mask"][sl])[0]
                loc_err_ref.append(float((ref - em[:2]).norm()))
    res = {}
    for kind, (ys, ps) in rows.items():
        if not ys:
            continue
        y, p = np.concatenate(ys), np.concatenate(ps)
        r = {"n": len(y), "macro_f1": float(f1_score(y, p.argmax(1), average="macro", labels=[0, 1, 2], zero_division=0))}
        for c, name in ((1, "jam"), (2, "spoof")):
            pos = y == c
            if pos.sum() and (~pos).sum():
                r[f"{name}_ap"] = float(average_precision_score(pos, p[:, c]))
                fpr, tpr, _ = roc_curve(pos, p[:, c])
                r[f"{name}_recall_at_1fpr"] = float(np.interp(0.01, fpr, tpr))
                r[f"n_{name}"] = int(pos.sum())
        res[kind] = r
    if loc_err:
        res["loc"] = {"n": len(loc_err), "median_km": float(np.median(loc_err)), "p95_km": float(np.percentile(loc_err, 95))}
        res["loc_refined"] = {"n": len(loc_err_ref), "median_km": float(np.median(loc_err_ref)),
                              "p95_km": float(np.percentile(loc_err_ref, 95))}
        res["loc_errors"] = {"GIST": loc_err_ref, "GIST (learned head only)": loc_err}
    model.train()
    release_memory()
    return res


def release_memory() -> None:
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="train")
    ap.add_argument("--val", default="val")
    ap.add_argument("--name", required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-graph", action="store_true")
    ap.add_argument("--no-physics-loc", action="store_true")
    ap.add_argument("--no-learned-loc", action="store_true")
    ap.add_argument("--no-sim", action="store_true", help="train on real scenes only")
    ap.add_argument("--integrity-only", action="store_true", help="zero all non-integrity features")
    ap.add_argument("--init", default=None, help="pretrained encoder checkpoint")
    ap.add_argument("--eval-only", default=None, help="evaluate a full checkpoint on --val and exit")
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    kinds = ("real",) if a.no_sim else ("real", "sim")
    tr = DataLoader(SceneDataset(a.train, kinds), batch_size=a.batch, shuffle=True, collate_fn=collate, num_workers=0)
    va = DataLoader(SceneDataset(a.val), batch_size=a.batch, shuffle=False, collate_fn=collate)
    model = GIST(F, use_graph=not a.no_graph, use_physics=not a.no_physics_loc, use_learned_loc=not a.no_learned_loc).to(dev)
    if a.init:
        model.enc.load_state_dict(torch.load(a.init, map_location=dev))
    if a.eval_only:
        model.load_state_dict(torch.load(a.eval_only, map_location=dev))
        res = evaluate(model, va, dev)
        errs = res.pop("loc_errors", {})
        print(json.dumps(res, indent=2))
        Path(a.eval_only).with_name(f"eval_{a.val}.json").write_text(json.dumps(res, indent=2))
        Path(a.eval_only).with_name(f"locerr_{a.val}.json").write_text(json.dumps(errs))
        return
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-2)
    steps_total = a.epochs * len(tr)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=max(steps_total, 1), pct_start=0.1)
    run = Path("runs") / a.name
    run.mkdir(parents=True, exist_ok=True)
    (run / "args.json").write_text(json.dumps(vars(a), indent=2))
    best, history = -1.0, []
    integrity_cols = torch.tensor([0, 1, 2, 3, 4, 5], device=dev)
    for ep in range(a.epochs):
        t0, agg = time.time(), np.zeros(3)
        for b in tr:
            b = to_dev(b, dev)
            if a.integrity_only:
                keep = torch.zeros(F, device=dev, dtype=torch.bool)
                keep[integrity_cols] = True
                b["x"] = b["x"] * keep
            out = model(b)
            l_node, l_step, l_loc = losses(model, out, b)
            loss = l_node + l_step + 0.5 * l_loc
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            agg += [l_node.item(), l_step.item(), l_loc.item()]
            del out, loss, l_node, l_step, l_loc, b
        release_memory()
        res = evaluate(model, va, dev)
        score = np.mean([res[k]["macro_f1"] for k in ("real", "sim") if k in res])
        slim = {k: v for k, v in res.items() if k != "loc_errors"}
        history.append({"epoch": ep, "loss": (agg / len(tr)).tolist(), "val": slim, "time_s": time.time() - t0})
        print(json.dumps(history[-1]), flush=True)
        if score > best:
            best = score
            torch.save(model.state_dict(), run / "best.pt")
            (run / "best_val.json").write_text(json.dumps(slim, indent=2))
    (run / "history.json").write_text(json.dumps(history, indent=2))


if __name__ == "__main__":
    main()
