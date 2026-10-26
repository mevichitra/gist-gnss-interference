"""Physics-consistency features (PLAN §6.1).

Pipeline: raw trace points → per-point kinematics → 30 s steps per aircraft → neighbor
consensus (same H3 res-3 cell, same 5 min) → deviations a spoofer cannot fake cheaply.
"""

from __future__ import annotations

from pathlib import Path

import h3
import numpy as np
import polars as pl

R = 6_371_000.0
KT = 0.514444
STEP_S = 30
NEIGH_STEPS = 10       # neighbor consensus over 10 steps = 5 min
H3_NEIGH = 3           # ~60 km cells for consensus
H3_MAP = 4             # ~22 km cells for maps / gpsjam baseline


def _hav(lat1, lon1, lat2, lon2) -> pl.Expr:
    la1, la2 = lat1.radians(), lat2.radians()
    a = ((la2 - la1) / 2).sin() ** 2 + la1.cos() * la2.cos() * ((lon2.radians() - lon1.radians()) / 2).sin() ** 2
    return 2 * R * a.sqrt().arcsin()


def point_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Per-point kinematic residuals against the previous point of the same aircraft."""
    prev = [pl.col(c).shift(1).over("hex").alias(f"{c}_p") for c in ("t", "lat", "lon", "gs", "src")]
    return (
        lf.sort("hex", "t")
        .with_columns(prev)
        .with_columns(dt=pl.col("t") - pl.col("t_p"))
        .with_columns(dist=_hav(pl.col("lat_p"), pl.col("lon_p"), pl.col("lat"), pl.col("lon")))
        .with_columns(
            res=pl.when((pl.col("dt") > 0) & (pl.col("dt") < 120))
            .then((pl.col("dist") - pl.col("gs_p") * KT * pl.col("dt")).abs())
            .otherwise(None),
            is_adsb=pl.col("src").str.starts_with("adsb") | pl.col("src").str.starts_with("adsr"),
            is_mlat=pl.col("src") == "mlat",
            stale=(pl.col("flags") & 1) > 0,
            # readsb's wind estimate is only consistent on the point that carries a fresh
            # details object (same instant as gs/track/tas/heading).
            fresh=pl.col("details_age") == 0,
        )
        .with_columns(
            # Position disagreement when the source switches between MLAT and ADS-B.
            mlat_jump=pl.when(
                (pl.col("src") != pl.col("src_p"))
                & ((pl.col("src") == "mlat") | (pl.col("src_p") == "mlat"))
                & (pl.col("dt") < 30)
            ).then(pl.col("res")).otherwise(None),
            adsb_gap=pl.when(pl.col("is_adsb")).then(pl.col("dt")).otherwise(None),
        )
    )


def step_table(lf: pl.LazyFrame) -> pl.LazyFrame:
    """Aggregate points to 30 s steps per aircraft."""
    hdg = pl.coalesce("true_heading", "mag_heading")
    adsb = pl.col("is_adsb")
    fresh_wind = adsb & pl.col("fresh") & pl.col("ws").is_not_null()
    return (
        point_features(lf)
        .with_columns(step=(pl.col("t") // STEP_S).cast(pl.Int64))
        .group_by("hex", "step")
        .agg(
            region=pl.col("region").last(),
            type=pl.col("type").last(),
            category=pl.col("category").drop_nulls().last(),
            lat=pl.col("lat").filter(adsb).last(),
            lon=pl.col("lon").filter(adsb).last(),
            lat_any=pl.col("lat").last(),
            lon_any=pl.col("lon").last(),
            alt_baro=pl.col("alt_baro").median(),
            alt_geom=pl.col("alt_geom").median(),
            gs=pl.col("gs").median(),
            track=pl.col("track").median(),
            tas=pl.col("tas").median(),
            hdg=hdg.median(),
            ias=pl.col("ias").median(),
            nic_min=pl.col("nic").min(),
            nic_max=pl.col("nic").max(),
            nacp_min=pl.col("nac_p").min(),
            sil_min=pl.col("sil").min(),
            gva_min=pl.col("gva").min(),
            version=pl.col("version").max(),
            n_pts=pl.len(),
            n_adsb=adsb.sum(),
            n_mlat=pl.col("is_mlat").sum(),
            stale_frac=pl.col("stale").mean(),
            res_max=pl.col("res").filter(adsb).max(),
            adsb_gap_max=pl.col("adsb_gap").max(),
            mlat_jump=pl.col("mlat_jump").max(),
            track_range=pl.col("track").max() - pl.col("track").min(),
            # wind vector (kt) the wind blows *towards*; readsb wd is the direction it comes from
            wu=(-pl.col("ws") * pl.col("wd").radians().sin()).filter(fresh_wind).median(),
            wv=(-pl.col("ws") * pl.col("wd").radians().cos()).filter(fresh_wind).median(),
        )
        .with_columns(
            gdiff=pl.col("alt_geom") - pl.col("alt_baro"),
            # turning aircraft give unreliable wind estimates
            wu=pl.when(pl.col("track_range") < 5).then(pl.col("wu")),
            wv=pl.when(pl.col("track_range") < 5).then(pl.col("wv")),
        )
        .sort("hex", "step")
    )


def add_cells(df: pl.DataFrame) -> pl.DataFrame:
    lat = df["lat_any"].to_numpy()
    lon = df["lon_any"].to_numpy()
    ok = np.isfinite(lat) & np.isfinite(lon)
    c3 = np.zeros(len(df), dtype=np.uint64)
    c4 = np.zeros(len(df), dtype=np.uint64)
    for i in np.flatnonzero(ok):
        c4[i] = h3.str_to_int(h3.latlng_to_cell(float(lat[i]), float(lon[i]), H3_MAP))
    c3[ok] = [h3.str_to_int(h3.cell_to_parent(h3.int_to_str(int(c)), H3_NEIGH)) for c in c4[ok]]
    return df.with_columns(cell3=pl.Series(c3), cell4=pl.Series(c4))


def neighbor_features(df: pl.DataFrame) -> pl.DataFrame:
    """Deviation of each step from the consensus of other aircraft in the same cell & 5 min."""
    # Wind and ISA deviation change strongly with altitude: consensus per 6000 ft band.
    df = df.with_columns(slot=pl.col("step") // NEIGH_STEPS, band=(pl.col("alt_baro") // 6000).cast(pl.Int16))
    trusted = (pl.col("nic_min") >= 7) & pl.col("alt_baro").gt(5000)
    cell_stats = df.group_by("cell3", "slot").agg(
        n_neigh=pl.col("hex").n_unique(),
        nic_low_share=(pl.col("nic_min") <= 6).mean(),
    )
    cons = df.group_by("cell3", "slot", "band").agg(
        n_band=pl.col("hex").filter(trusted).n_unique(),
        n_neigh=pl.col("hex").n_unique(),
        gdiff_med=pl.col("gdiff").filter(trusted & pl.col("version").ge(1)).median(),
        wu_med=pl.col("wu").filter(trusted).median(),
        wv_med=pl.col("wv").filter(trusted).median(),
    )
    df = df.join(cell_stats, on=["cell3", "slot"], how="left").join(cons, on=["cell3", "slot", "band"], how="left")
    enough = pl.col("n_band") >= 3
    return df.with_columns(
        gdiff_dev=pl.when(enough).then(pl.col("gdiff") - pl.col("gdiff_med")),
        wind_dev=pl.when(enough).then(
            ((pl.col("wu") - pl.col("wu_med")) ** 2 + (pl.col("wv") - pl.col("wv_med")) ** 2).sqrt()
        ),
    )


def spoof_clusters(df: pl.DataFrame, grid_deg: float = 0.005, min_ac: int = 3, alt_spread_ft: float = 5000) -> pl.DataFrame:
    """Flag steps where several aircraft report (nearly) the same position but very different altitudes."""
    keyed = df.filter(pl.col("lat").is_not_null()).with_columns(
        gy=(pl.col("lat") / grid_deg).floor().cast(pl.Int32),
        gx=(pl.col("lon") / grid_deg).floor().cast(pl.Int32),
        slot=pl.col("step") // 2,
    )
    clusters = (
        keyed.group_by("slot", "gy", "gx")
        .agg(n_ac=pl.col("hex").n_unique(), alt_spread=pl.col("alt_baro").max() - pl.col("alt_baro").min())
        .filter((pl.col("n_ac") >= min_ac) & (pl.col("alt_spread") >= alt_spread_ft))
        .with_columns(in_cluster=pl.lit(True))
    )
    flagged = keyed.join(clusters, on=["slot", "gy", "gx"], how="inner").select("hex", "step", "in_cluster")
    return df.join(flagged.unique(["hex", "step"]), on=["hex", "step"], how="left").with_columns(
        pl.col("in_cluster").fill_null(False)
    )


def build_steps(points_dir: Path) -> pl.DataFrame:
    steps = step_table(pl.scan_parquet(str(points_dir / "*.parquet"))).collect()
    steps = add_cells(steps)
    steps = neighbor_features(steps)
    return spoof_clusters(steps)
