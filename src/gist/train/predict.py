"""Run a trained GIST over every window of a day (or of selected regions) and aggregate.

Tiles each 10-minute window into ±BOX_DEG scenes on a regular lattice, so every aircraft is
scored with its neighbours. Outputs:
  runs/<name>/pred_<day>.parquet      hex, win, region, lat, lon, cell4, p_normal, p_jam, p_spoof
  runs/<name>/emitters_<day>.parquet  per (region tile, win) jammer estimate when jam evidence is strong
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.6")
os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.5")

import numpy as np
import polars as pl
import torch

from gist.data.regions import ALL
from gist.features.windows import WIN_STEPS
from gist.models.gist import GIST
from gist.train.dataset import collate
from gist.train.tensors import BOX_DEG, F, build_scene

MIN_JAM_STEPS = 30     # need this much predicted jamming evidence before localizing
KM_PER_DEG = 111.195


def tiles(region: str, step_deg: float = 2 * BOX_DEG - 1.0):
    la0, la1, lo0, lo1 = ALL[region]
    for la in np.arange(la0 + BOX_DEG / 2, la1 + 1e-6, step_deg):
        dlon = step_deg / max(np.cos(np.radians(la)), 0.2)
        for lo in np.arange(lo0 + BOX_DEG / 2, lo1 + 1e-6, dlon):
            yield float(la), float(lo)


@torch.no_grad()
def predict_day(model, day: str, regions: list[str], dev, wins: list[int] | None = None):
    steps = pl.read_parquet(f"data_store/steps/{day}.parquet").filter(pl.col("region").is_in(regions))
    steps = steps.with_columns(win=pl.col("step") // WIN_STEPS)
    out_rows, emitters = [], []
    all_wins = sorted(steps["win"].unique().to_list()) if wins is None else wins
    for region in regions:
        reg = steps.filter(pl.col("region") == region)
        for win in all_wins:
            wsteps = reg.filter(pl.col("win") == win)
            if wsteps.is_empty():
                continue
            for lat0, lon0 in tiles(region):
                dlon = BOX_DEG / max(np.cos(np.radians(lat0)), 0.2)
                sc = wsteps.filter(pl.col("lat_any").is_between(lat0 - BOX_DEG, lat0 + BOX_DEG)
                                   & pl.col("lon_any").is_between(lon0 - dlon, lon0 + dlon))
                if sc["hex"].n_unique() < 5:
                    continue
                scene = build_scene(sc.drop("win"), None, lat0, lon0)
                hexes = scene.meta["hexes"]
                item = {k: getattr(scene, k) for k in ("x", "mask", "pos_km", "h_m", "xy_node", "y_soft", "y_step", "emitter")}
                item["is_sim"] = np.array(False)
                b = {k: v.to(dev) for k, v in collate([item]).items()}
                o = model(b)
                p = o["node_logits"].softmax(-1)[0, : len(hexes)].cpu().numpy()
                xy = scene.xy_node
                lat = lat0 + xy[:, 1] / KM_PER_DEG
                lon = lon0 + xy[:, 0] / (KM_PER_DEG * np.cos(np.radians(lat0)))
                # Only keep nodes in the central part of the tile (tiles overlap by 1°).
                inner = (np.abs(lat - lat0) <= BOX_DEG - 0.5)
                for i in np.flatnonzero(inner):
                    out_rows.append((hexes[i], win, region, float(lat[i]), float(lon[i]), *map(float, p[i])))
                jam_steps = float((o["step_p"][0] > 0.5).sum())
                if jam_steps >= MIN_JAM_STEPS:
                    est = model.loc.refine(o["loc_logits"], o["step_p"], b["pos_km"], b["h_m"], b["mask"])[0].cpu().numpy()
                    emitters.append({"region": region, "win": win, "tile_lat": lat0, "tile_lon": lon0,
                                     "lat": lat0 + est[1] / KM_PER_DEG,
                                     "lon": lon0 + est[0] / (KM_PER_DEG * np.cos(np.radians(lat0))),
                                     "jam_steps": jam_steps})
    pred = pl.DataFrame(out_rows, schema=["hex", "win", "region", "lat", "lon", "p_normal", "p_jam", "p_spoof"], orient="row")
    pred = pred.group_by("hex", "win").agg(pl.all().first())   # de-duplicate overlapping tiles
    return pred, pl.DataFrame(emitters)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--regions", nargs="+", default=["BAL", "EMED", "BLK", "MEA", "INPK", "MYN"])
    ap.add_argument("--every", type=int, default=1, help="score every k-th window (speed)")
    a = ap.parse_args()
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = GIST(F).to(dev)
    model.load_state_dict(torch.load(a.ckpt, map_location=dev))
    model.eval()
    run = Path(a.ckpt).parent
    for day in a.days:
        wins = list(range(0, 144, a.every))
        pred, em = predict_day(model, day, a.regions, dev, wins)
        pred.write_parquet(run / f"pred_{day}.parquet")
        if em.height:
            em.write_parquet(run / f"emitters_{day}.parquet")
        print(day, pred.height, "windows,", em.height, "emitter estimates", flush=True)


if __name__ == "__main__":
    main()
