"""Build steps + labelled windows for every parsed day; fit one label model across days.

  data_store/steps/<day>.parquet     30 s step features
  data_store/windows/<day>.parquet   window aggregates + LF votes + label-model posteriors
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import polars as pl

from gist.features.physics import build_steps
from gist.features.windows import window_table
from gist.labels.label_model import ABSTAIN, LabelModel
from gist.labels.lfs import apply_lfs, vote_matrix

POINTS = Path("data_store/points")
STEPS = Path("data_store/steps")
WINDOWS = Path("data_store/windows")
MODEL = Path("data_store/label_model.pkl")


def build_day(day: str) -> Path:
    STEPS.mkdir(parents=True, exist_ok=True)
    WINDOWS.mkdir(parents=True, exist_ok=True)
    out = WINDOWS / f"{day}.parquet"
    if out.exists():
        return out
    steps = build_steps(POINTS / day).with_columns(day=pl.lit(day))
    steps.write_parquet(STEPS / f"{day}.parquet", compression="zstd")
    apply_lfs(window_table(steps)).with_columns(day=pl.lit(day)).write_parquet(out, compression="zstd")
    return out


def fit_and_label(max_rows: int = 3_000_000, seed: int = 0) -> LabelModel:
    files = sorted(WINDOWS.glob("*.parquet"))
    w = pl.concat([pl.read_parquet(f) for f in files], how="vertical_relaxed")
    votes = vote_matrix(w)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(votes))[:max_rows]
    lm = LabelModel().fit(votes[idx])
    MODEL.write_bytes(pickle.dumps(lm))
    for f in files:
        d = pl.read_parquet(f)
        v = vote_matrix(d)
        p = lm.predict_proba(v)
        d.with_columns(
            p_normal=p[:, 0], p_jam=p[:, 1], p_spoof=p[:, 2],
            covered=pl.Series((v != ABSTAIN).any(1)), label=pl.Series(p.argmax(1)).cast(pl.Int8),
        ).write_parquet(f, compression="zstd")
    return lm


def main(days: list[str]) -> None:
    ready = sorted(p.name for p in POINTS.iterdir() if (p / "_SUCCESS").exists())
    for day in days or ready:
        print(build_day(day), flush=True)
    lm = fit_and_label()
    print("LF accuracies:", np.round(lm.lf_accuracy(), 3))


if __name__ == "__main__":
    main(sys.argv[1:])
