"""Evaluate localization baselines on simulated jammer scenes (Gate 4 / Gate 6)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import polars as pl

from gist.models.localize import centroid, horizon_score, nls
from gist.sim import propagation as P


def scene_observations(scene: pl.DataFrame, every: int = 2) -> pl.DataFrame:
    """One row per aircraft-step (aircraft move ~140 km per window, so no per-window averaging):
    true position, altitude, NIC-based hard 'affected' flag."""
    return (
        scene.filter(pl.col("step") % every == 0)
        .select(
            "hex", "step", lat=pl.col("lat_any"), lon=pl.col("lon_any"), h_m=pl.col("alt_baro") * P.FT,
            affected=(pl.col("nic_min") <= 6).cast(pl.Float64),
            truth=(pl.col("truth") > 0).cast(pl.Float64),
        )
        .drop_nulls(["lat", "lon", "h_m", "affected"])
    )


def evaluate(sim_file: Path, methods=("centroid", "horizon", "nls"), score_col="affected") -> pl.DataFrame:
    sims = pl.read_parquet(sim_file)
    meta = pl.read_parquet(str(sim_file).replace(".parquet", "_meta.parquet")).filter(pl.col("kind") == "jam")
    rows = []
    for em in meta.iter_rows(named=True):
        obs = scene_observations(sims.filter(pl.col("scene") == em["scene"]))
        if obs[score_col].sum() < 3:
            continue
        a = (obs["lat"].to_numpy(), obs["lon"].to_numpy())
        h, s = obs["h_m"].to_numpy(), obs[score_col].to_numpy()
        est = {}
        if "centroid" in methods:
            est["centroid"] = centroid(*a, s)
        if "horizon" in methods:
            est["horizon"] = horizon_score(*a, h, s)
        if "nls" in methods:
            est["nls"] = nls(*a, h, s)[:2]
        for k, (la, lo) in est.items():
            rows.append({"scene": em["scene"], "method": k, "err_km": P.haversine_m(la, lo, em["lat"], em["lon"]) / 1000,
                         "eirp": em["eirp_dbw"], "n_aff": int(s.sum()), "n": len(s)})
    return pl.DataFrame(rows)


def main(files: list[str]) -> None:
    res = pl.concat([evaluate(Path(f)) for f in files])
    print(res.group_by("method").agg(n=pl.len(), median_km=pl.col("err_km").median(), p95_km=pl.col("err_km").quantile(0.95)).sort("median_km"))


if __name__ == "__main__":
    main(sys.argv[1:])
