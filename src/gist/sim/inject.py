"""Inject synthetic jammers / spoofers into real clean traffic at the step-feature level (PLAN §7).

Working on step tables (not raw points) keeps the simulated data in exactly the space the
models consume; neighbor and cluster features are recomputed after injection so cross-aircraft
evidence reacts to the emitter as it would in real data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import polars as pl

from gist.features.physics import add_cells, neighbor_features, spoof_clusters
from gist.sim import propagation as P

DERIVED = ("slot", "band", "n_neigh", "n_neigh_right", "nic_low_share", "n_band", "gdiff_med", "wu_med", "wv_med",
           "gdiff_dev", "wind_dev", "in_cluster")


@dataclass
class Emitter:
    kind: str            # "jam" | "spoof"
    lat: float
    lon: float
    height_m: float
    eirp_dbw: float
    mode: str = "none"   # spoof trajectory: static | circle | drift
    fake_lat: float = 0.0
    fake_lon: float = 0.0
    fake_alt_off_ft: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def sample_emitter(rng: np.random.Generator, lat0: float, lon0: float, radius_m: float = 250_000) -> Emitter:
    kind = "jam" if rng.random() < 0.6 else "spoof"
    la, lo = P.dest_point(lat0, lon0, rng.uniform(0, 360), radius_m * np.sqrt(rng.random()))
    airborne = rng.random() < 0.15
    h = rng.uniform(1_000, 10_000) if airborne else rng.uniform(2, 50)
    # 100 W – 100 kW jammers, 10 W – 10 kW spoofers (Liu et al. 2022 estimate ~1 kW for KDEN)
    eirp = rng.uniform(20, 50) if kind == "jam" else rng.uniform(10, 40)
    em = Emitter(kind, float(la), float(lo), float(h), float(eirp))
    if kind == "spoof":
        em.mode = str(rng.choice(["static", "circle", "drift"]))
        fla, flo = P.dest_point(la, lo, rng.uniform(0, 360), rng.uniform(20_000, 300_000))
        em.fake_lat, em.fake_lon = float(fla), float(flo)
        em.fake_alt_off_ft = float(rng.choice([-1, 1]) * rng.uniform(1_000, 5_000))
    return em


def _received(df: pl.DataFrame, em: Emitter) -> tuple[np.ndarray, np.ndarray]:
    lat = df["lat_any"].to_numpy().astype(float)
    lon = df["lon_any"].to_numpy().astype(float)
    h_rx = np.nan_to_num(df["alt_baro"].to_numpy().astype(float), nan=0.0) * P.FT
    g = P.haversine_m(lat, lon, em.lat, em.lon)
    pr = P.received_power_dbw(em.eirp_dbw, g, em.height_m, h_rx)
    los = P.line_of_sight(g, em.height_m, h_rx)
    return np.where(los, pr, -np.inf), g


def inject(steps: pl.DataFrame, em: Emitter, rng: np.random.Generator, tol_sigma_db: float = 3.0) -> pl.DataFrame:
    """Return a copy of ``steps`` with the emitter applied and a ``truth`` column (0/1/2 per step)."""
    df = steps.drop([c for c in DERIVED if c in steps.columns])
    pr, _ = _received(df, em)
    hexes = df["hex"].to_numpy()
    uniq, inv = np.unique(hexes, return_inverse=True)
    tol = P.JAM_TOLERANCE_DBW + rng.normal(0, tol_sigma_db, len(uniq))[inv]   # receiver-to-receiver spread
    margin = pr - tol
    n = df.height
    cols = {c: df[c].to_numpy().astype(float).copy() for c in (
        "nic_min", "nic_max", "nacp_min", "lat", "lon", "alt_geom", "gdiff", "wu", "wv",
        "res_max", "adsb_gap_max", "stale_frac", "gs", "track")}
    truth = np.zeros(n, dtype=np.int8)

    if em.kind == "jam":
        strong = margin > 0
        moderate = (margin > -6) & ~strong
        lost = strong & (rng.random(n) < 1 / (1 + np.exp(-margin / 3)))
        cols["nic_min"][strong] = 0
        cols["nic_max"][strong] = rng.integers(0, 2, strong.sum())
        cols["nacp_min"][strong] = 0
        cols["nic_min"][moderate] = rng.integers(1, 7, moderate.sum())
        cols["nic_max"][moderate] = np.maximum(cols["nic_min"][moderate], rng.integers(1, 8, moderate.sum()))
        for c in ("lat", "lon", "gs", "track", "wu", "wv", "res_max"):
            cols[c][lost] = np.nan
        cols["alt_geom"][strong] = np.nan
        cols["gdiff"][strong] = np.nan
        cols["adsb_gap_max"][lost] = rng.uniform(30, 120, lost.sum())
        cols["stale_frac"][lost] = 1.0
        truth[strong | moderate] = 1
    else:
        captured_step = margin > -5   # spoofers need less power than denial to capture a receiver
        # Once captured, an aircraft stays captured for the rest of the scene.
        steps_arr = df["step"].to_numpy()
        order = np.lexsort((steps_arr, inv))
        cap = np.zeros(n, dtype=bool)
        cap_sorted = captured_step[order]
        inv_sorted = inv[order]
        running = np.zeros(len(uniq), dtype=bool)
        onset = np.zeros(n, dtype=bool)
        for k in range(n):   # n is a scene (~10^4 rows): a plain loop is fine
            i = inv_sorted[k]
            if cap_sorted[k] and not running[i]:
                running[i] = True
                onset[order[k]] = True
            cap[order[k]] = running[i]
        t = (steps_arr - steps_arr.min()) * 30.0
        if em.mode == "static":
            fla = np.full(n, em.fake_lat) + rng.normal(0, 0.002, n)
            flo = np.full(n, em.fake_lon) + rng.normal(0, 0.002, n)
            fgs = rng.uniform(0, 5, n)
        elif em.mode == "circle":
            ang = (t / 600.0) * 2 * np.pi
            fla, flo = P.dest_point(em.fake_lat, em.fake_lon, np.degrees(ang) % 360, 10_000.0)
            fgs = np.full(n, 2 * np.pi * 10_000 / 600 / 0.514444)
        else:  # drift: true position plus a growing offset (~200 m/s)
            br = np.degrees(np.arctan2(em.fake_lon - em.lon, em.fake_lat - em.lat)) % 360
            lat_t = np.nan_to_num(df["lat_any"].to_numpy().astype(float))
            lon_t = np.nan_to_num(df["lon_any"].to_numpy().astype(float))
            fla, flo = P.dest_point(lat_t, lon_t, np.full(n, br), 200.0 * t + 5_000)
            fgs = np.nan_to_num(cols["gs"]) + 389.0
        jump = P.haversine_m(np.nan_to_num(cols["lat"]), np.nan_to_num(cols["lon"]), fla, flo)
        cols["lat"][cap] = fla[cap]
        cols["lon"][cap] = flo[cap]
        cols["res_max"][cap] = np.where(onset, jump, rng.uniform(0, 500, n))[cap]
        dgs = fgs - np.nan_to_num(cols["gs"])
        trk = np.radians(np.nan_to_num(cols["track"]))
        cols["wu"][cap] = (cols["wu"] + dgs * np.sin(trk))[cap]
        cols["wv"][cap] = (cols["wv"] + dgs * np.cos(trk))[cap]
        cols["gs"][cap] = fgs[cap]
        cols["alt_geom"][cap] = (df["alt_baro"].to_numpy() + em.fake_alt_off_ft)[cap]
        cols["gdiff"][cap] = em.fake_alt_off_ft
        cols["nic_min"][cap] = np.maximum(cols["nic_min"][cap], 8)
        cols["nic_max"][cap] = np.maximum(cols["nic_max"][cap], 8)
        truth[cap] = 2

    if em.kind == "spoof":
        cols["lat_any"] = np.where(cap, cols["lat"], df["lat_any"].to_numpy().astype(float))
        cols["lon_any"] = np.where(cap, cols["lon"], df["lon_any"].to_numpy().astype(float))
    out = df.with_columns(
        [pl.Series(c, v, dtype=df.schema[c]) if df.schema[c].is_float() else pl.Series(c, np.round(v), strict=False).cast(df.schema[c], strict=False)
         for c, v in cols.items()]
    ).with_columns(truth=pl.Series(truth))
    out = out.with_columns([pl.col(c).fill_nan(None) for c in cols if out.schema[c].is_float()])
    # Spoofed positions move cells; neighbor consensus and clusters are recomputed from reported data.
    if em.kind == "spoof":
        out = add_cells(out.drop("cell3", "cell4"))
    out = neighbor_features(out)
    return spoof_clusters(out)
