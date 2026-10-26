"""Parse readsb trace JSONs (adsb.lol globe_history) into a filtered Parquet point table.

Trace point layout (wiedehopf/readsb README-json.md#trace-jsons):
  0 dt, 1 lat, 2 lon, 3 alt_baro|"ground"|null, 4 gs, 5 track, 6 flags, 7 baro_rate,
  8 details|null, 9 source type, 10 alt_geom, 11 geom_rate, 12 ias, 13 roll
The details object is attached only to some points; its integrity fields are forward-filled
along the trace, and ``details_age`` records how stale they are.
"""

from __future__ import annotations

import argparse
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import polars as pl

from gist.data.regions import assign_region

DETAIL_KEYS = ("nic", "nac_p", "nac_v", "sil", "gva", "sda", "nic_baro", "version", "rc",
               "tas", "mag_heading", "true_heading", "ws", "wd", "oat", "category")
FLAG_STALE, FLAG_NEWLEG, FLAG_GEOM_VR, FLAG_GEOM_ALT = 1, 2, 4, 8


def _load(path: Path) -> dict:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def parse_trace(path: Path) -> dict[str, list] | None:
    d = _load(path)
    pts = d.get("trace") or []
    if not pts:
        return None
    t0 = float(d["timestamp"])
    cols: dict[str, list] = {k: [] for k in (
        "t", "lat", "lon", "alt_baro", "on_ground", "gs", "track", "flags", "baro_rate",
        "src", "alt_geom", "geom_rate", "ias", "roll", "details_age", *DETAIL_KEYS)}
    cur = {k: None for k in DETAIL_KEYS}
    last_detail_t = None
    for p in pts:
        t = t0 + p[0]
        det = p[8] if len(p) > 8 else None
        if det:
            for k in DETAIL_KEYS:
                if k in det:
                    cur[k] = det[k]
            last_detail_t = t
        alt = p[3]
        cols["t"].append(t)
        cols["lat"].append(p[1])
        cols["lon"].append(p[2])
        cols["on_ground"].append(alt == "ground")
        cols["alt_baro"].append(None if alt == "ground" else alt)
        cols["gs"].append(p[4])
        cols["track"].append(p[5])
        cols["flags"].append(p[6])
        cols["baro_rate"].append(p[7])
        cols["src"].append(p[9] if len(p) > 9 else None)
        cols["alt_geom"].append(p[10] if len(p) > 10 else None)
        cols["geom_rate"].append(p[11] if len(p) > 11 else None)
        cols["ias"].append(p[12] if len(p) > 12 else None)
        cols["roll"].append(p[13] if len(p) > 13 else None)
        cols["details_age"].append(None if last_detail_t is None else t - last_detail_t)
        for k in DETAIL_KEYS:
            cols[k].append(cur[k])
    cols["hex"] = [d["icao"]] * len(pts)
    cols["type"] = [d.get("t")] * len(pts)
    return cols


SCHEMA = {
    "hex": pl.Utf8, "type": pl.Utf8, "t": pl.Float64, "lat": pl.Float32, "lon": pl.Float32,
    "alt_baro": pl.Float32, "on_ground": pl.Boolean, "gs": pl.Float32, "track": pl.Float32,
    "flags": pl.Int16, "baro_rate": pl.Float32, "src": pl.Utf8, "alt_geom": pl.Float32,
    "geom_rate": pl.Float32, "ias": pl.Float32, "roll": pl.Float32, "details_age": pl.Float32,
    "nic": pl.Int8, "nac_p": pl.Int8, "nac_v": pl.Int8, "sil": pl.Int8, "gva": pl.Int8,
    "sda": pl.Int8, "nic_baro": pl.Int8, "version": pl.Int8, "rc": pl.Float32, "tas": pl.Float32,
    "mag_heading": pl.Float32, "true_heading": pl.Float32, "ws": pl.Float32, "wd": pl.Float32,
    "oat": pl.Float32, "category": pl.Utf8,
}


def _frame(cols: dict[str, list]) -> pl.DataFrame:
    return pl.DataFrame({k: cols[k] for k in SCHEMA}, schema=SCHEMA, strict=False)


def parse_shard(files: list[Path], out: Path) -> int:
    """Parse a list of trace files, keep airborne points inside regions/controls, write Parquet."""
    frames = []
    for f in files:
        try:
            cols = parse_trace(f)
        except (OSError, ValueError, KeyError):
            continue
        if cols is None:
            continue
        lat = np.asarray(cols["lat"], dtype=float)
        lon = np.asarray(cols["lon"], dtype=float)
        reg = assign_region(lat, lon)
        if not (reg != "").any():
            continue
        df = _frame(cols).with_columns(region=pl.Series(reg.astype(str)))
        frames.append(df.filter((pl.col("region") != "") & ~pl.col("on_ground")))
    if not frames:
        return 0
    df = pl.concat(frames, how="vertical_relaxed")
    df.write_parquet(out, compression="zstd")
    return df.height


def parse_day(day_dir: Path, out_dir: Path, workers: int = 6, shard_size: int = 2000) -> int:
    files = sorted(day_dir.glob("traces/*/trace_full_*.json"))
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = [files[i:i + shard_size] for i in range(0, len(files), shard_size)]
    with ProcessPoolExecutor(workers) as ex:
        futs = [ex.submit(parse_shard, s, out_dir / f"part-{i:04d}.parquet") for i, s in enumerate(shards)]
        return sum(f.result() for f in futs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("day_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    a = ap.parse_args()
    print(parse_day(a.day_dir, a.out_dir), "rows")


if __name__ == "__main__":
    main()
