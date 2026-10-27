"""Aggregate GIST predictions into a compact JSON for the interactive demo map.

Per (day, region, ~22 km H3 cell, hour): aircraft-windows, mean jam / spoof probability, and
the gpsjam-style baseline (share of windows with NIC ≤ 6) for side-by-side comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h3
import polars as pl

from gist.data.regions import REGIONS


def aggregate(run: Path, day: str) -> dict:
    pred = pl.read_parquet(run / f"pred_{day}.parquet")
    wins = pl.read_parquet(f"data_store/windows/{day}.parquet", columns=["hex", "win", "nic_low_frac"])
    # win = unix_time // 600 → hour of day (UTC)
    df = pred.join(wins, on=["hex", "win"], how="left").with_columns(hour=((pl.col("win") % 144) // 6).cast(pl.Int16))
    df = df.with_columns(cell=pl.struct("lat", "lon").map_elements(
        lambda r: h3.latlng_to_cell(r["lat"], r["lon"], 4), return_dtype=pl.Utf8))
    agg = (df.group_by("region", "cell", "hour")
           .agg(n=pl.len(), jam=pl.col("p_jam").mean(), spoof=pl.col("p_spoof").mean(),
                nic_low=(pl.col("nic_low_frac").fill_null(0) >= 0.3).mean())
           .filter(pl.col("n") >= 2))
    cells = {}
    for c in agg["cell"].unique().to_list():
        cells[c] = [[round(lo, 3), round(la, 3)] for la, lo in h3.cell_to_boundary(c)]
    rows = [[r["region"], r["cell"], r["hour"], r["n"], round(r["jam"], 3), round(r["spoof"], 3), round(r["nic_low"], 3)]
            for r in agg.iter_rows(named=True)]
    em_file = run / f"emitters_{day}.parquet"
    emitters = []
    if em_file.exists():
        em = pl.read_parquet(em_file).with_columns(hour=((pl.col("win") % 144) // 6))
        emitters = [[r["region"], int(r["hour"]), round(r["lat"], 3), round(r["lon"], 3), round(r["jam_steps"])]
                    for r in em.iter_rows(named=True)]
    summary = (df.group_by("region").agg(n=pl.len(), jam=(pl.col("p_jam") > 0.5).mean(), spoof=(pl.col("p_spoof") > 0.5).mean(),
                                          nic_low=(pl.col("nic_low_frac").fill_null(0) >= 0.3).mean())
               .sort("region").to_dicts())
    return {"cells": cells, "rows": rows, "emitters": emitters, "summary": summary}


def clip_ring(ring, x0, x1, y0, y1):
    """Sutherland–Hodgman clip of a polygon ring to an axis-aligned box."""
    pts = [tuple(p) for p in ring]
    for axis, bound, keep_greater in ((0, x0, True), (0, x1, False), (1, y0, True), (1, y1, False)):
        if not pts:
            break
        inside = (lambda p: p[axis] >= bound) if keep_greater else (lambda p: p[axis] <= bound)
        out = []
        for i, cur in enumerate(pts):
            prev = pts[i - 1]
            if inside(cur):
                if not inside(prev):
                    out.append(_cross(prev, cur, axis, bound))
                out.append(cur)
            elif inside(prev):
                out.append(_cross(prev, cur, axis, bound))
        pts = out
    return pts


def _cross(a, b, axis, bound):
    t = (bound - a[axis]) / (b[axis] - a[axis])
    return (a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1]))


def land_by_region(margin: float = 2.0, step: int = 1) -> dict:
    """Natural Earth land rings clipped (by bbox overlap) to each region, lightly thinned."""
    from gist.viz.figures import land_rings

    rings = land_rings()
    out = {}
    for name, (la0, la1, lo0, lo1) in REGIONS.items():
        keep = []
        for r in rings:
            if r[:, 0].max() < lo0 - margin or r[:, 0].min() > lo1 + margin or r[:, 1].max() < la0 - margin or r[:, 1].min() > la1 + margin:
                continue
            clipped = clip_ring(r, lo0 - margin, lo1 + margin, la0 - margin, la1 + margin)
            if len(clipped) < 3:
                continue
            pts = clipped[::step] if len(clipped) > 40 else clipped
            keep.append([[round(float(x), 2), round(float(y), 2)] for x, y in pts])
        out[name] = keep
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=Path("viz_build/demo_data.json"))
    a = ap.parse_args()
    data = {"regions": {k: list(v) for k, v in REGIONS.items()}, "land": land_by_region(), "days": {}}
    for d in a.days:
        data["days"][d] = aggregate(a.run, d)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(data, separators=(",", ":")))
    print(a.out, a.out.stat().st_size // 1024, "KB")
    tpl = Path("src/gist/viz/demo_template.html").read_text()
    page = Path("viz_build/gist_demo.html")
    page.write_text(tpl.replace("/*__DEMO_DATA__*/null", a.out.read_text()))
    print(page, page.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
