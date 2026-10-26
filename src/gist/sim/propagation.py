"""Geometry and RF propagation shared by the simulator, baselines and the localization head.

Everything here is written against an ``xp`` array namespace so the same code runs in NumPy
(simulation, NLS baseline) and in PyTorch (differentiable localization head).
"""

from __future__ import annotations

import math

import numpy as np

EARTH_R = 6_371_000.0          # mean Earth radius [m]
K_FACTOR = 4.0 / 3.0           # standard atmospheric refraction for radio horizon
L1_HZ = 1_575.42e6
C = 299_792_458.0
L1_LAMBDA = C / L1_HZ
FT = 0.3048
JAM_TOLERANCE_DBW = -115.0     # Liu et al. 2022: max tolerable jamming power (C/N0 threshold 25 dB-Hz)


def _xp(a):
    try:
        import torch

        if isinstance(a, torch.Tensor):
            return torch
    except ImportError:  # pragma: no cover
        pass
    return np


def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance [m]; inputs in degrees, broadcastable."""
    xp = _xp(lat1)
    lat1, lon1, lat2, lon2 = (x * (math.pi / 180) for x in (lat1, lon1, lat2, lon2))
    a = xp.sin((lat2 - lat1) / 2) ** 2 + xp.cos(lat1) * xp.cos(lat2) * xp.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_R * xp.arcsin(xp.sqrt(xp.clip(a, 0.0, 1.0)))


def radio_horizon_m(h1_m, h2_m):
    """Max line-of-sight ground distance between two antennas at heights h1, h2 [m] (4/3 Earth)."""
    xp = _xp(h1_m)
    re = K_FACTOR * EARTH_R
    return xp.sqrt(2 * re * xp.clip(h1_m, 0, None)) + xp.sqrt(2 * re * xp.clip(h2_m, 0, None))


def slant_range_m(ground_m, h1_m, h2_m):
    """Straight-line distance between two points given ground distance and heights (flat approx + dh)."""
    xp = _xp(ground_m)
    return xp.sqrt(ground_m**2 + (h2_m - h1_m) ** 2)


def elevation_deg(ground_m, h_tx_m, h_rx_m):
    """Elevation of the transmitter as seen from the receiver (negative = below the aircraft)."""
    xp = _xp(ground_m)
    # Account for Earth curvature drop over the ground distance.
    drop = ground_m**2 / (2 * K_FACTOR * EARTH_R)
    return xp.arctan2(h_tx_m - h_rx_m - drop, xp.clip(ground_m, 1.0, None)) * (180 / math.pi)


def antenna_gain_db(elev_deg):
    """Coarse GNSS patch-antenna gain pattern vs elevation (RTCA DO-301-like envelope).

    Top-mounted aircraft antennas strongly attenuate signals from below: ~ -10 dBi at the
    horizon falling to ~ -30 dBi straight below. Smooth so it stays differentiable.
    """
    xp = _xp(elev_deg)
    e = xp.clip(elev_deg, -90.0, 90.0)
    above = -10.0 + 13.0 * xp.sin(xp.clip(e, 0, 90) * (math.pi / 180))       # -10 .. +3 dBi
    below = -10.0 - 20.0 * xp.sin(xp.clip(-e, 0, 90) * (math.pi / 180))      # -10 .. -30 dBi
    return xp.where(e >= 0, above, below)


def received_power_dbw(eirp_dbw, ground_m, h_tx_m, h_rx_m, use_pattern: bool = True):
    """Friis received jamming power at the aircraft GNSS antenna [dBW]."""
    xp = _xp(ground_m)
    d = xp.clip(slant_range_m(ground_m, h_tx_m, h_rx_m), 1.0, None)
    fspl_db = 20 * xp.log10(4 * math.pi * d / L1_LAMBDA)
    gain = antenna_gain_db(elevation_deg(ground_m, h_tx_m, h_rx_m)) if use_pattern else 0.0
    return eirp_dbw - fspl_db + gain


def line_of_sight(ground_m, h_tx_m, h_rx_m):
    return ground_m <= radio_horizon_m(h_tx_m, h_rx_m)


def dest_point(lat, lon, bearing_deg, dist_m):
    """Point reached travelling ``dist_m`` from (lat, lon) along ``bearing_deg`` (NumPy)."""
    la, lo, b = np.radians(lat), np.radians(lon), np.radians(bearing_deg)
    d = dist_m / EARTH_R
    la2 = np.arcsin(np.sin(la) * np.cos(d) + np.cos(la) * np.sin(d) * np.cos(b))
    lo2 = lo + np.arctan2(np.sin(b) * np.sin(d) * np.cos(la), np.cos(d) - np.sin(la) * np.sin(la2))
    return np.degrees(la2), (np.degrees(lo2) + 540) % 360 - 180
