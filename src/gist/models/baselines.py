"""Detection baselines (PLAN §8.1), evaluated on the same scene files as GIST.

Each baseline maps a scene batch to per-node scores [N, 3] (normal, jam, spoof) so the
metric code in ``gist.train.train.evaluate`` can be reused unchanged.
  gpsjam  — share of aircraft in the same H3-res4-like 22 km cell with NIC ≤ 6 (jam score only)
  iforest — Isolation Forest on window aggregates; anomaly score used for both jam and spoof
  lstm_ae — LSTM autoencoder reconstruction error, same use
  rules   — the training labeling functions' own decision (upper bound for "copying the LFs")
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest

from gist.train.tensors import STEP_FEATURES

FEAT = {name: i for i, (name, _, _) in enumerate(STEP_FEATURES)}


def window_aggregates(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """x [N, T, F], mask [N, T] → [N, 3F] (mean, min, max over present steps)."""
    m = mask[..., None]
    cnt = np.maximum(m.sum(1), 1)
    mean = (x * m).sum(1) / cnt
    big = np.where(m, x, np.inf).min(1)
    small = np.where(m, x, -np.inf).max(1)
    big[~np.isfinite(big)] = 0
    small[~np.isfinite(small)] = 0
    return np.concatenate([mean, big, small], 1)


def gpsjam_scores(x, mask, xy_node, cell_km: float = 22.0) -> np.ndarray:
    nic_low = ((x[..., FEAT["nic_min"]] * 11 <= 6.5) & mask).sum(1) >= 2
    cell = np.floor(xy_node / cell_km).astype(int)
    keys = cell[:, 0] * 100_000 + cell[:, 1]
    share = np.zeros(len(keys))
    for k in np.unique(keys):
        sel = keys == k
        share[sel] = nic_low[sel].mean()
    s = np.zeros((len(keys), 3))
    s[:, 1] = share
    s[:, 0] = 1 - share
    return s


class IForestBaseline:
    def fit(self, scenes):
        X = np.concatenate([window_aggregates(s["x"], s["mask"]) for s in scenes])
        self.m = IsolationForest(n_estimators=200, random_state=0).fit(X)
        return self

    def scores(self, x, mask, xy_node):
        a = -self.m.score_samples(window_aggregates(x, mask))
        a = (a - a.min()) / max(a.max() - a.min(), 1e-9)
        return np.stack([1 - a, a, a], 1)


class LSTMAE(nn.Module):
    def __init__(self, f: int, h: int = 64):
        super().__init__()
        self.enc = nn.LSTM(f, h, batch_first=True)
        self.dec = nn.LSTM(h, h, batch_first=True)
        self.out = nn.Linear(h, f)

    def forward(self, x):
        _, (hn, _) = self.enc(x)
        z = hn[-1].unsqueeze(1).expand(-1, x.shape[1], -1)
        y, _ = self.dec(z)
        return self.out(y)


class LSTMAEBaseline:
    def fit(self, scenes, epochs: int = 5, dev="cpu"):
        xs = np.concatenate([s["x"] for s in scenes])
        ms = np.concatenate([s["mask"] for s in scenes])
        # Fit on windows the label model considers normal — it learns "normal" behaviour.
        ys = np.concatenate([np.nan_to_num(s["y_soft"]).argmax(1) for s in scenes])
        keep = ys == 0
        x = torch.tensor(xs[keep], dtype=torch.float32)
        m = torch.tensor(ms[keep])
        self.model = LSTMAE(x.shape[-1])
        opt = torch.optim.Adam(self.model.parameters(), 1e-3)
        for _ in range(epochs):
            perm = torch.randperm(len(x))
            for i in range(0, len(x), 512):
                idx = perm[i:i + 512]
                rec = self.model(x[idx])
                loss = (((rec - x[idx]) ** 2).mean(-1) * m[idx]).sum() / m[idx].sum().clamp(min=1)
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def scores(self, x, mask, xy_node):
        xt = torch.tensor(x, dtype=torch.float32)
        err = (((self.model(xt) - xt) ** 2).mean(-1) * torch.tensor(mask)).sum(1) / torch.tensor(mask).sum(1).clamp(min=1)
        a = err.numpy()
        a = a / max(np.percentile(a, 99), 1e-9)
        a = np.clip(a, 0, 1)
        return np.stack([1 - a, a, a], 1)
