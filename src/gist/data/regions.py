"""Regions of interest (PLAN §5.2). Boxes are (lat_min, lat_max, lon_min, lon_max)."""

from __future__ import annotations

import numpy as np

REGIONS: dict[str, tuple[float, float, float, float]] = {
    "BAL": (53.0, 61.5, 9.0, 31.0),     # Baltic Sea, Kaliningrad, Gulf of Finland
    "EMED": (29.0, 37.5, 27.0, 37.5),   # Cyprus, Lebanon, Israel, Egypt
    "BLK": (40.5, 47.5, 27.0, 42.0),    # Black Sea, Romania, Turkey coast
    "MEA": (24.0, 38.0, 38.0, 58.0),    # Iraq, Iran, Persian Gulf
    "INPK": (24.0, 33.5, 68.0, 78.0),   # India-Pakistan border
    "MYN": (10.0, 28.5, 92.0, 101.5),   # Myanmar
}

# Negative controls: regions without reported persistent interference.
CONTROLS: dict[str, tuple[float, float, float, float]] = {
    "CEU": (46.0, 52.0, 2.0, 15.0),     # France/Germany/Benelux/Alps
    "CUS": (35.0, 45.0, -100.0, -85.0),  # US mid-west
    "AUS": (-38.0, -27.0, 140.0, 154.0),  # SE Australia
}

ALL = {**REGIONS, **CONTROLS}


def assign_region(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Return the region id for each point, or "" when it is in no box (first match wins)."""
    out = np.full(lat.shape, "", dtype=object)
    for name, (la0, la1, lo0, lo1) in ALL.items():
        m = (out == "") & (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
        out[m] = name
    return out
