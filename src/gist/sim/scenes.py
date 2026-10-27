"""Sample real clean traffic scenes and turn them into simulated interference scenes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl

from gist.features.windows import WIN_STEPS
from gist.sim.inject import Emitter, inject, sample_emitter

BOX_DEG_LAT = 3.0            # ±3° lat ≈ ±330 km
MIN_AIRCRAFT = 25
CLEAN_REGIONS = ("CUS", "AUS", "CEU")


def clean_scene(steps: pl.DataFrame, windows: pl.DataFrame, rng: np.random.Generator) -> tuple[pl.DataFrame, float, float] | None:
    """All steps in one 10-min window within a box around a random clean aircraft."""
    cands = windows.filter(pl.col("region").is_in(CLEAN_REGIONS) & (pl.col("label") == 0) & (pl.col("alt_med") > 10_000))
    if cands.is_empty():
        return None
    row = cands.row(int(rng.integers(cands.height)), named=True)
    lat0, lon0, win = row["lat"], row["lon"], row["win"]
    dlon = BOX_DEG_LAT / max(np.cos(np.radians(lat0)), 0.2)
    scene = steps.filter(
        (pl.col("step") // WIN_STEPS == win)
        & pl.col("lat_any").is_between(lat0 - BOX_DEG_LAT, lat0 + BOX_DEG_LAT)
        & pl.col("lon_any").is_between(lon0 - dlon, lon0 + dlon)
    )
    # Remove aircraft that already look degraded, so injected truth is the only interference.
    bad = windows.filter((pl.col("win") == win) & (pl.col("label") > 0)).select("hex")
    scene = scene.join(bad, on="hex", how="anti")
    if scene["hex"].n_unique() < MIN_AIRCRAFT:
        return None
    return scene, lat0, lon0


def make_scene(steps, windows, rng, distractor_frac: float = 0.0, kind: str | None = None) -> tuple[pl.DataFrame, Emitter] | None:
    got = clean_scene(steps, windows, rng)
    if got is None:
        return None
    scene, lat0, lon0 = got
    em = sample_emitter(rng, lat0, lon0)
    if kind is not None and em.kind != kind:
        return None
    sim = inject(scene, em, rng)
    if (sim["truth"] > 0).sum() == 0:
        return None   # emitter too weak / out of view: no supervision signal
    if distractor_frac > 0:
        sim = add_distractors(sim, distractor_frac, rng)
    return sim, em


def add_distractors(sim: pl.DataFrame, frac: float, rng: np.random.Generator) -> pl.DataFrame:
    """Benign low-integrity aircraft (old / poor installations): NIC 3–6 for the whole window,
    otherwise normal. Truth stays 0. Real airspace has ~2–5 % such aircraft (CUS/CEU, LOG Phase 2)."""
    from gist.features.physics import neighbor_features, spoof_clusters
    from gist.sim.inject import DERIVED

    clean = sim.group_by("hex").agg(t=pl.col("truth").max()).filter(pl.col("t") == 0)["hex"].to_list()
    k = max(1, int(frac * sim["hex"].n_unique()))
    pick = set(rng.choice(clean, min(k, len(clean)), replace=False).tolist()) if clean else set()
    nic = {h: int(rng.integers(3, 7)) for h in pick}
    is_d = pl.col("hex").is_in(list(pick))
    nic_col = pl.col("hex").replace_strict(nic, default=None, return_dtype=pl.Int8)
    out = sim.with_columns(
        nic_min=pl.when(is_d).then(nic_col).otherwise(pl.col("nic_min")),
        nic_max=pl.when(is_d).then(nic_col).otherwise(pl.col("nic_max")),
        distractor=is_d,
    )
    out = neighbor_features(out.drop([c for c in DERIVED if c in out.columns]))
    return spoof_clusters(out)


def generate(day: str, n: int, out_dir: Path, seed: int = 0) -> int:
    steps = pl.read_parquet(f"data_store/steps/{day}.parquet")
    windows = pl.read_parquet(f"data_store/windows/{day}.parquet")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    frames, meta, made, tries = [], [], 0, 0
    while made < n and tries < n * 5:
        tries += 1
        got = make_scene(steps, windows, rng)
        if got is None:
            continue
        sim, em = got
        frames.append(sim.with_columns(scene=pl.lit(made, pl.Int32)))
        meta.append({"scene": made, "day": day, **em.to_dict()})
        made += 1
    if frames:
        pl.concat(frames, how="diagonal_relaxed").write_parquet(out_dir / f"{day}_s{seed}.parquet", compression="zstd")
        pl.DataFrame(meta).write_parquet(out_dir / f"{day}_s{seed}_meta.parquet")
    return made
