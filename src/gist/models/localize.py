"""Classical jammer-localization baselines (PLAN §8.1 #6).

All take per-aircraft observations for one scene: true-ish positions (lat, lon), height [m],
and an ``affected`` score in [0, 1] (hard NIC-based labels for the baselines, learned
probabilities for GIST's hybrid head).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

from gist.sim import propagation as P


def centroid(lat, lon, affected) -> tuple[float, float]:
    w = np.asarray(affected, float)
    if w.sum() == 0:
        return float(np.mean(lat)), float(np.mean(lon))
    return float(np.average(lat, weights=w)), float(np.average(lon, weights=w))


def _grid(lat, lon, half_deg=4.0, n=81):
    la0, lo0 = np.mean(lat), np.mean(lon)
    dlo = half_deg / max(np.cos(np.radians(la0)), 0.2)
    gla, glo = np.meshgrid(np.linspace(la0 - half_deg, la0 + half_deg, n), np.linspace(lo0 - dlo, lo0 + dlo, n), indexing="ij")
    return gla.ravel(), glo.ravel()


def horizon_score(lat, lon, h_m, affected, tx_h=10.0) -> tuple[float, float]:
    """Schäfer-style: pick the grid point whose radio horizon covers the affected set best
    while covering as few unaffected aircraft as possible (soft F1 over aircraft)."""
    gla, glo = _grid(lat, lon)
    a = np.asarray(affected, float)
    g = P.haversine_m(gla[:, None], glo[:, None], np.asarray(lat)[None], np.asarray(lon)[None])
    # Jamming range is bounded by both horizon and power; use the affected set's own extent.
    reach = np.minimum(P.radio_horizon_m(np.asarray(h_m)[None], tx_h), np.percentile(g[:, a > 0.5].min(0), 90) * 3 if (a > 0.5).any() else 4e5)
    inside = (g <= reach).astype(float)
    tp = (inside * a).sum(1)
    fp = (inside * (1 - a)).sum(1)
    fn = ((1 - inside) * a).sum(1)
    f1 = 2 * tp / np.clip(2 * tp + fp + fn, 1e-9, None)
    # Tie-break towards the point closest to the affected aircraft.
    dist_aff = (g * a).sum(1) / np.clip(a.sum(), 1e-9, None)
    best = np.lexsort((dist_aff, -f1))[0]
    return float(gla[best]), float(glo[best])


def nls(lat, lon, h_m, affected, tx_h=10.0, temp_db=3.0) -> tuple[float, float, float]:
    """Liu et al.-style nonlinear least squares, as a logistic likelihood over affected / not.

    Model: P(affected_i | x) = σ((Pr_i(x) − τ) / T), Pr from Friis + antenna pattern + LOS.
    Optimise (lat, lon, EIRP) from the best of a coarse grid, then L-BFGS.
    """
    lat, lon, h_m, a = (np.asarray(x, float) for x in (lat, lon, h_m, affected))

    def nll(theta):
        la, lo, eirp = theta
        g = P.haversine_m(lat, lon, la, lo)
        pr = P.received_power_dbw(eirp, g, tx_h, h_m)
        pr = np.where(P.line_of_sight(g, tx_h, h_m), pr, pr - 40)
        z = (pr - P.JAM_TOLERANCE_DBW) / temp_db
        logp = -np.logaddexp(0, -z)
        log1p = -np.logaddexp(0, z)
        return -(a * logp + (1 - a) * log1p).sum()

    gla, glo = _grid(lat, lon, n=41)
    best = None
    for eirp in (10.0, 20.0, 30.0, 40.0, 50.0):
        vals = [nll((x, y, eirp)) for x, y in zip(gla, glo)]
        i = int(np.argmin(vals))
        if best is None or vals[i] < best[0]:
            best = (vals[i], (gla[i], glo[i], eirp))
    res = minimize(nll, np.array(best[1]), method="L-BFGS-B",
                   bounds=[(best[1][0] - 2, best[1][0] + 2), (best[1][1] - 3, best[1][1] + 3), (0, 70)])
    return float(res.x[0]), float(res.x[1]), float(res.x[2])
