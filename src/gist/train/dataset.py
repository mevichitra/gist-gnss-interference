"""Build scene datasets on disk (one .npz per scene) and load them as padded batches.

  data_store/scenes/<split>/real_<day>_<i>.npz   real traffic, label-model soft labels
  data_store/scenes/<split>/sim_<day>_<i>.npz    simulated emitter, exact step truth + emitter

Splits are by *day* (PLAN §8.3): no day appears in two splits.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import polars as pl
import torch
from torch.utils.data import Dataset

from gist.features.windows import WIN_STEPS
from gist.sim.scenes import make_scene
from gist.train.tensors import BOX_DEG, Scene, build_scene

ROOT = Path("data_store/scenes")


def split_of(day: str) -> str:
    if day < "2025-07-01":
        return "train"
    if day < "2026-01-01":
        return "val"
    return "test"


def _save(scene: Scene, path: Path) -> None:
    d = asdict(scene)
    meta = d.pop("meta")
    np.savez_compressed(path, **{k: (v.astype(np.float16) if v.dtype == np.float32 and k == "x" else v) for k, v in d.items()},
                        meta=np.array(str(meta)))


def real_scenes(day: str, n: int, out: Path, rng: np.random.Generator, interesting_frac: float = 0.6) -> int:
    steps = pl.read_parquet(f"data_store/steps/{day}.parquet")
    wins = pl.read_parquet(f"data_store/windows/{day}.parquet").filter(pl.col("alt_med") > 5_000)
    hot = wins.filter(pl.col("label") > 0)
    made = 0
    for _ in range(n * 3):
        if made >= n:
            break
        pool = hot if (rng.random() < interesting_frac and hot.height) else wins
        row = pool.row(int(rng.integers(pool.height)), named=True)
        lat0, lon0, win = row["lat"], row["lon"], row["win"]
        dlon = BOX_DEG / max(np.cos(np.radians(lat0)), 0.2)
        sc = steps.filter(
            (pl.col("step") // WIN_STEPS == win)
            & pl.col("lat_any").is_between(lat0 - BOX_DEG, lat0 + BOX_DEG)
            & pl.col("lon_any").is_between(lon0 - dlon, lon0 + dlon)
        )
        if sc["hex"].n_unique() < 10:
            continue
        lab = wins.filter(pl.col("win") == win).select("hex", "p_normal", "p_jam", "p_spoof")
        scene = build_scene(sc, lab, lat0, lon0, rng=rng, meta={"day": day, "win": win, "region": row["region"],
                                                               "lat0": lat0, "lon0": lon0, "kind": "real"})
        _save(scene, out / f"real_{day}_{made:05d}.npz")
        made += 1
    return made


def sim_scenes(day: str, n: int, out: Path, rng: np.random.Generator) -> int:
    steps = pl.read_parquet(f"data_store/steps/{day}.parquet")
    wins = pl.read_parquet(f"data_store/windows/{day}.parquet")
    made = 0
    for _ in range(n * 4):
        if made >= n:
            break
        got = make_scene(steps, wins, rng)
        if got is None:
            continue
        sim, em = got
        lat0 = float(sim["lat_any"].median())
        lon0 = float(sim["lon_any"].median())
        scene = build_scene(sim, None, lat0, lon0, emitter=em.to_dict(), rng=rng,
                            meta={"day": day, "kind": "sim", **em.to_dict(), "lat0": lat0, "lon0": lon0})
        _save(scene, out / f"sim_{day}_{made:05d}.npz")
        made += 1
    return made


class SceneDataset(Dataset):
    def __init__(self, split: str, kinds=("real", "sim"), root: Path = ROOT):
        self.files = sorted(f for k in kinds for f in (root / split).glob(f"{k}_*.npz"))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        z = np.load(self.files[i], allow_pickle=False)
        d = {k: z[k] for k in ("x", "mask", "pos_km", "h_m", "xy_node", "y_soft", "y_step", "emitter")}
        d["x"] = d["x"].astype(np.float32)
        d["is_sim"] = np.array(self.files[i].name.startswith("sim_"))
        return d


def collate(items: list[dict]) -> dict[str, torch.Tensor]:
    # Pad to a few fixed sizes: MPS compiles a new graph for every distinct shape (~30 s each).
    n_max = max(it["x"].shape[0] for it in items)
    n = next(bk for bk in (64, 128, 192, 256, 1 << 30) if bk >= n_max)
    n = n_max if n == 1 << 30 else n
    out: dict[str, list] = {k: [] for k in items[0]}
    out["node_mask"] = []
    for it in items:
        m = it["x"].shape[0]
        pad = n - m
        for k, v in it.items():
            if v.ndim >= 1 and k not in ("emitter",) and v.shape[0] == m:
                fill = -1 if k == "y_step" else (np.nan if k == "y_soft" else 0)
                v = np.concatenate([v, np.full((pad, *v.shape[1:]), fill, dtype=v.dtype)]) if pad else v
            out[k].append(v)
        out["node_mask"].append(np.arange(n) < m)
    return {k: torch.as_tensor(np.stack(v)) for k, v in out.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("days", nargs="+")
    ap.add_argument("--real", type=int, default=200)
    ap.add_argument("--sim", type=int, default=200)
    ap.add_argument("--split", default=None, help="override day-based split (dev runs only)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    for day in a.days:
        split = a.split or split_of(day)
        out = ROOT / split
        out.mkdir(parents=True, exist_ok=True)
        rng = np.random.default_rng(int(day.replace("-", "")) * 100 + a.seed)
        r = real_scenes(day, a.real, out, rng)
        s = sim_scenes(day, a.sim, out, rng)
        print(day, split, "real", r, "sim", s, flush=True)


if __name__ == "__main__":
    main()
