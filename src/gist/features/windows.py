"""10-minute windows (20 × 30 s steps) per aircraft and their aggregate features."""

from __future__ import annotations

import polars as pl

from gist.data.regions import CONTROLS

WIN_STEPS = 20
TELEPORT_M = 10_000
GDEV_FT = 1_000
WIND_KT = 50


def window_table(steps: pl.DataFrame) -> pl.DataFrame:
    good = pl.col("nic_min") >= 7
    modern = pl.col("version").max() >= 1
    return (
        steps.with_columns(win=pl.col("step") // WIN_STEPS)
        .group_by("hex", "win")
        .agg(
            region=pl.col("region").mode().first(),
            cell3=pl.col("cell3").mode().first(),
            cell4=pl.col("cell4").mode().first(),
            lat=pl.col("lat_any").median(),
            lon=pl.col("lon_any").median(),
            n_steps=pl.len(),
            alt_med=pl.col("alt_baro").median(),
            modern=modern,
            nic_low_frac=(pl.col("nic_min") <= 6).mean(),
            nic_zero_frac=(pl.col("nic_min") <= 1).mean(),
            nic_hi_any=(pl.col("nic_max") >= 8).any(),
            nic_all_hi=(pl.col("nic_min") >= 8).all(),
            adsb_gap_max=pl.col("adsb_gap_max").max(),
            n_mlat=pl.col("n_mlat").sum(),
            cell_nic_low=pl.col("nic_low_share").mean(),
            teleport_steps=((pl.col("res_max") > TELEPORT_M) & good).sum(),
            gdev_steps=((pl.col("gdiff_dev").abs() > GDEV_FT) & good).sum(),
            wind_steps=((pl.col("wind_dev") > WIND_KT) & good).sum(),
            cluster_steps=pl.col("in_cluster").sum(),
            mlat_jump_max=pl.col("mlat_jump").max(),
        )
        .with_columns(
            is_control=pl.col("region").is_in(list(CONTROLS)),
            cruise=pl.col("alt_med") > 10_000,
            t_start=pl.col("win") * WIN_STEPS * 30,
        )
    )
