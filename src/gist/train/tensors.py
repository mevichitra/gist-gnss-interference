"""Turn step tables into fixed-size scene tensors for GIST.

A *scene* is every aircraft-window (node) in one 10-minute window inside a box of
±BOX_DEG latitude around a centre. Each node carries a [WIN_STEPS, F] step-feature sequence,
a step-presence mask, per-step true-ish positions (for the physics localization head) and,
when available, labels:
  y_soft [3]      label-model posterior (real data) or one-hot truth (simulation)
  y_step [T]      per-step truth 0/1/2 (simulation only; -1 otherwise)
Feature columns deliberately exclude ``mlat_jump`` (held-out evaluation signal).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from gist.features.windows import WIN_STEPS
from gist.sim import propagation as P

BOX_DEG = 3.0
MAX_NODES = 256

STEP_FEATURES = [
    # name, scale, clip
    ("nic_min", 11.0, None), ("nic_max", 11.0, None), ("nacp_min", 11.0, None), ("sil_min", 3.0, None),
    ("gva_min", 2.0, None), ("version", 2.0, None), ("alt_baro", 40_000.0, (-0.1, 1.5)),
    ("gs", 500.0, (0, 2)), ("gdiff_dev", 1_000.0, (-10, 10)), ("wind_dev", 50.0, (0, 10)),
    ("res_max_log", 1.0, (0, 15)), ("adsb_gap_max", 60.0, (0, 10)), ("stale_frac", 1.0, None),
    ("has_mlat", 1.0, None), ("in_cluster", 1.0, None), ("nic_low_share", 1.0, None),
    ("n_neigh_log", 1.0, (0, 8)), ("track_range", 30.0, (0, 6)),
]
MISSING_FLAGS = ["nic_min", "alt_geom", "gdiff_dev", "wind_dev", "lat"]
F = len(STEP_FEATURES) + len(MISSING_FLAGS)


def _prep(steps: pl.DataFrame) -> pl.DataFrame:
    return steps.with_columns(
        res_max_log=pl.col("res_max").log1p(),
        has_mlat=(pl.col("n_mlat") > 0).cast(pl.Float32),
        in_cluster=pl.col("in_cluster").cast(pl.Float32),
        n_neigh_log=pl.col("n_neigh").cast(pl.Float32).log1p(),
        pos=(pl.col("step") % WIN_STEPS).cast(pl.Int32),
        win=pl.col("step") // WIN_STEPS,
    )


def step_matrix(df: pl.DataFrame) -> np.ndarray:
    cols = []
    for name, scale, clip in STEP_FEATURES:
        v = df[name].cast(pl.Float32).to_numpy() / scale
        if clip:
            v = np.clip(v, *clip)
        cols.append(np.nan_to_num(v, nan=0.0))
    for name in MISSING_FLAGS:
        missing = df[name].is_null().to_numpy() | np.isnan(df[name].cast(pl.Float32).fill_null(np.nan).to_numpy())
        cols.append(missing.astype(np.float32))
    return np.stack(cols, 1).astype(np.float32)


@dataclass
class Scene:
    x: np.ndarray        # [N, T, F]
    mask: np.ndarray     # [N, T] bool
    pos_km: np.ndarray   # [N, T, 2] east/north km from scene centre (reported positions)
    h_m: np.ndarray      # [N, T]
    xy_node: np.ndarray  # [N, 2] node position (median reported), km
    y_soft: np.ndarray   # [N, 3]
    y_step: np.ndarray   # [N, T] int8, -1 unknown
    emitter: np.ndarray  # [4] east km, north km, kind (1 jam, 2 spoof, 0 none), valid flag
    meta: dict


def to_km(lat, lon, lat0, lon0):
    y = (lat - lat0) * 111.195
    x = (lon - lon0) * 111.195 * np.cos(np.radians(lat0))
    return x, y


def build_scene(steps: pl.DataFrame, labels: pl.DataFrame | None, lat0: float, lon0: float,
                emitter: dict | None = None, rng: np.random.Generator | None = None, meta: dict | None = None) -> Scene | None:
    """``steps``: one window's steps inside the box. ``labels``: hex → p_normal/p_jam/p_spoof (real data)."""
    df = _prep(steps).sort("hex", "pos")
    hexes = df["hex"].unique(maintain_order=True).to_list()
    if len(hexes) > MAX_NODES:
        rng = rng or np.random.default_rng(0)
        keep = set(rng.choice(hexes, MAX_NODES, replace=False).tolist())
        if "truth" in df.columns:  # always keep affected aircraft
            keep |= set(df.filter(pl.col("truth") > 0)["hex"].unique().to_list()[:MAX_NODES // 2])
            keep = set(list(keep)[:MAX_NODES])
        df = df.filter(pl.col("hex").is_in(list(keep)))
        hexes = df["hex"].unique(maintain_order=True).to_list()
    n, t = len(hexes), WIN_STEPS
    idx = {h: i for i, h in enumerate(hexes)}
    ni = np.array([idx[h] for h in df["hex"].to_list()])
    ti = df["pos"].to_numpy()
    x = np.zeros((n, t, F), np.float32)
    x[ni, ti] = step_matrix(df)
    mask = np.zeros((n, t), bool)
    mask[ni, ti] = True
    la = df["lat_any"].cast(pl.Float64).to_numpy()
    lo = df["lon_any"].cast(pl.Float64).to_numpy()
    ex, ny = to_km(la, lo, lat0, lon0)
    pos = np.zeros((n, t, 2), np.float32)
    pos[ni, ti, 0], pos[ni, ti, 1] = np.nan_to_num(ex), np.nan_to_num(ny)
    h = np.zeros((n, t), np.float32)
    h[ni, ti] = np.nan_to_num(df["alt_baro"].cast(pl.Float64).to_numpy()) * P.FT
    cnt = np.maximum(mask.sum(1, keepdims=True), 1)
    xy_node = (pos * mask[..., None]).sum(1) / cnt
    y_step = np.full((n, t), -1, np.int8)
    y_soft = np.zeros((n, 3), np.float32)
    if "truth" in df.columns:
        y_step[ni, ti] = df["truth"].to_numpy()
        for c in (1, 2):
            y_soft[:, c] = ((y_step == c).sum(1) >= 2)
        y_soft[:, 0] = 1 - y_soft[:, 1:].max(1)
        y_soft[(y_soft[:, 1] > 0) & (y_soft[:, 2] > 0), 1] = 0
    elif labels is not None:
        lab = labels.filter(pl.col("hex").is_in(hexes))
        li = np.array([idx[h] for h in lab["hex"].to_list()], dtype=int)
        y_soft[li] = lab.select("p_normal", "p_jam", "p_spoof").to_numpy()
        unl = y_soft.sum(1) == 0
        y_soft[unl] = np.nan
    em = np.zeros(4, np.float32)
    if emitter is not None:
        ex0, ny0 = to_km(np.array(emitter["lat"]), np.array(emitter["lon"]), lat0, lon0)
        em[:] = [float(ex0), float(ny0), 1 if emitter["kind"] == "jam" else 2, 1]
    return Scene(x, mask, pos, h, xy_node.astype(np.float32), y_soft, y_step, em, meta or {})
