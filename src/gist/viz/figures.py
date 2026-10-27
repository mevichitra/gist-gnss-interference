"""Paper figures, all generated from run outputs (PLAN §9.3).

  uv run python -m gist.viz.figures --run runs/gist_final_s0 --out paper/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h3
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.collections import PolyCollection

from gist.data.regions import REGIONS

LAND = Path("viz_build/land-50m.json")
plt.rcParams.update({"font.size": 8, "axes.titlesize": 9, "figure.dpi": 200, "savefig.bbox": "tight"})


def land_rings() -> list[np.ndarray]:
    """Decode Natural Earth land (world-atlas TopoJSON, quantized + delta-encoded arcs)."""
    topo = json.loads(LAND.read_text())
    sx, sy = topo["transform"]["scale"]
    tx, ty = topo["transform"]["translate"]
    arcs = []
    for arc in topo["arcs"]:
        a = np.cumsum(np.array(arc, dtype=float), 0)
        arcs.append(np.c_[a[:, 0] * sx + tx, a[:, 1] * sy + ty])

    def ring(idx):
        pts = [arcs[i] if i >= 0 else arcs[~i][::-1] for i in idx]
        return np.concatenate(pts)

    rings = []
    for geom in topo["objects"]["land"]["geometries"]:
        polys = geom["arcs"] if geom["type"] == "MultiPolygon" else [geom["arcs"]]
        for poly in polys:
            rings.append(ring(poly[0]))
    return rings


_RINGS = None


def draw_land(ax, box):
    global _RINGS
    _RINGS = _RINGS or land_rings()
    la0, la1, lo0, lo1 = box
    pc = PolyCollection([r for r in _RINGS if (r[:, 0].max() > lo0 and r[:, 0].min() < lo1 and r[:, 1].max() > la0 and r[:, 1].min() < la1)],
                        facecolor="#e9e6df", edgecolor="#9a958a", linewidth=0.3, zorder=0)
    ax.add_collection(pc)
    ax.set_xlim(lo0, lo1)
    ax.set_ylim(la0, la1)
    ax.set_aspect(1 / np.cos(np.radians((la0 + la1) / 2)))
    ax.set_facecolor("#f7fafc")
    ax.tick_params(labelsize=6)


def hex_layer(ax, df: pl.DataFrame, col: str, cmap: str, vmax: float = 1.0, min_n: int = 3):
    agg = df.group_by("cell").agg(v=pl.col(col).mean(), n=pl.len()).filter(pl.col("n") >= min_n)
    polys, vals = [], []
    for c, v in zip(agg["cell"].to_list(), agg["v"].to_list()):
        b = np.array(h3.cell_to_boundary(c))
        polys.append(b[:, ::-1])
        vals.append(v)
    pc = PolyCollection(polys, array=np.array(vals), cmap=cmap, clim=(0, vmax), edgecolor="none", alpha=0.85, zorder=1)
    ax.add_collection(pc)
    return pc


def case_study(run: Path, day: str, region: str, out: Path) -> None:
    pred = pl.read_parquet(run / f"pred_{day}.parquet").filter(pl.col("region") == region)
    wins = pl.read_parquet(f"data_store/windows/{day}.parquet", columns=["hex", "win", "nic_low_frac"])
    df = pred.join(wins, on=["hex", "win"], how="left").with_columns(
        cell=pl.struct("lat", "lon").map_elements(lambda r: h3.latlng_to_cell(r["lat"], r["lon"], 4), return_dtype=pl.Utf8),
        nic_low=(pl.col("nic_low_frac").fill_null(0) >= 0.3).cast(pl.Float64),
    )
    box = REGIONS[region]
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6))
    for ax, (col, title, cmap) in zip(axes, (("nic_low", "Baseline: share NIC ≤ 6", "Greys"),
                                             ("p_jam", "GIST: P(jammed)", "Oranges"),
                                             ("p_spoof", "GIST: P(spoofed)", "Purples"))):
        draw_land(ax, box)
        pc = hex_layer(ax, df, col, cmap)
        ax.set_title(title)
        fig.colorbar(pc, ax=ax, fraction=0.04, pad=0.02)
    em_file = run / f"emitters_{day}.parquet"
    if em_file.exists():
        em = pl.read_parquet(em_file).filter(pl.col("region") == region)
        if em.height:
            axes[1].scatter(em["lon"], em["lat"], s=10, marker="x", c="#7a1f00", linewidths=0.8, zorder=3, label="jammer estimates")
            axes[1].legend(loc="lower left", fontsize=6, frameon=False)
    fig.suptitle(f"{region} · {day} · all 10-min windows", fontsize=9)
    fig.savefig(out / f"case_{region}_{day}.pdf")
    fig.savefig(out / f"case_{region}_{day}.png")
    plt.close(fig)


def loc_cdf(errors: dict[str, list[float]], out: Path, name: str) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for k, v in errors.items():
        v = np.sort(v)
        ax.plot(v, np.arange(1, len(v) + 1) / len(v), label=k, lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("localization error [km]")
    ax.set_ylabel("fraction of scenes")
    ax.grid(alpha=0.3, which="both", lw=0.4)
    ax.legend(fontsize=6, frameon=False)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("paper/figures"))
    ap.add_argument("--cases", nargs="+", default=[], help="day:REGION pairs")
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    for c in a.cases:
        day, region = c.split(":")
        case_study(a.run, day, region, a.out)
        print("figure", day, region)
    for f in sorted(a.run.glob("locerr_*.json")):
        loc_cdf(json.loads(f.read_text()), a.out, f.stem)
        print("figure", f.stem)


if __name__ == "__main__":
    main()
