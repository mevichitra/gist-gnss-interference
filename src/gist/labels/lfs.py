"""Labeling functions over window aggregates (PLAN §6.2).

LFs never look at the region name, so labels cannot leak geography. ``HELD_OUT`` LFs are
excluded from training labels and used as independent evaluation signals (§8.4).
"""

from __future__ import annotations

import numpy as np
import polars as pl

from gist.labels.label_model import ABSTAIN, JAMMED, NORMAL, SPOOFED

C = pl.col


def _vote(cond: pl.Expr, label: int) -> pl.Expr:
    return pl.when(cond.fill_null(False)).then(label).otherwise(ABSTAIN)


LFS: dict[str, pl.Expr] = {
    # --- jamming: integrity collapse / position loss on modern transponders at cruise
    "nic_zero": _vote(C("modern") & C("cruise") & (C("nic_zero_frac") >= 0.3), JAMMED),
    "nic_drop": _vote(C("modern") & C("cruise") & C("nic_hi_any") & (C("nic_low_frac") >= 0.3), JAMMED),
    "pos_loss": _vote(C("cruise") & (C("adsb_gap_max") >= 90) & (C("nic_low_frac") >= 0.1), JAMMED),
    "cell_jam": _vote(C("modern") & (C("cell_nic_low") >= 0.3) & (C("nic_low_frac") >= 0.3), JAMMED),
    # --- spoofing: high reported integrity but physically inconsistent
    "teleport": _vote(C("teleport_steps") >= 1, SPOOFED),
    "gdiff": _vote(C("gdev_steps") >= 2, SPOOFED),
    "cluster": _vote(C("cluster_steps") >= 1, SPOOFED),
    "mlat": _vote(C("mlat_jump_max") > 10_000, SPOOFED),
    # --- normal
    "clean": _vote(
        C("nic_all_hi") & (C("teleport_steps") == 0) & (C("gdev_steps") == 0) & (C("cluster_steps") == 0)
        & (C("n_steps") >= 10),
        NORMAL,
    ),
    "clean_lowcell": _vote(C("modern") & (C("cell_nic_low") < 0.05) & (C("nic_low_frac") == 0)
                           & (C("teleport_steps") == 0) & (C("gdev_steps") == 0), NORMAL),
}

HELD_OUT = ("mlat",)
TRAIN_LFS = tuple(k for k in LFS if k not in HELD_OUT)


def apply_lfs(windows: pl.DataFrame) -> pl.DataFrame:
    return windows.with_columns(**{f"lf_{k}": v.cast(pl.Int8) for k, v in LFS.items()})


def vote_matrix(df: pl.DataFrame, names=TRAIN_LFS) -> np.ndarray:
    return df.select([f"lf_{k}" for k in names]).to_numpy().astype(np.int64)
