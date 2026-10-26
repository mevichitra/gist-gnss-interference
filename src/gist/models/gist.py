"""GIST: trajectory Transformer + spatial graph attention + classification / step / localization heads.

Shapes: B scenes, N nodes (aircraft-windows), T steps, F features, D model width.
Plain PyTorch (no PyG) so it runs on Apple MPS.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as Fn

from gist.sim import propagation as P

GRID_HALF_KM = 400.0
GRID_N = 21                      # 40 km cells (coarse; refined by soft-argmax)
EIRP_CANDIDATES = (20.0, 30.0, 40.0, 50.0)


class TrajEncoder(nn.Module):
    def __init__(self, f_in: int, d: int = 96, layers: int = 3, heads: int = 4, t_max: int = 20, dropout: float = 0.1):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(f_in, d), nn.GELU(), nn.Linear(d, d))
        self.pos = nn.Parameter(torch.randn(t_max, d) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)

    def forward(self, x, mask):
        """x [M, T, F], mask [M, T] → step tokens [M, T, D], window embedding [M, D]."""
        m, t, _ = x.shape
        h = self.inp(x) + self.pos[:t]
        h = torch.cat([self.cls.expand(m, -1, -1), h], 1)
        pad = torch.cat([torch.zeros(m, 1, dtype=torch.bool, device=x.device), ~mask], 1)
        h = self.enc(h, src_key_padding_mask=pad)
        return h[:, 1:], h[:, 0]


class SpatialAttention(nn.Module):
    """Dense multi-head attention between aircraft of a scene with a learned distance bias
    and a hard radius mask (GATv2-style edge features: dx, dy, distance, Δaltitude)."""

    def __init__(self, d: int = 96, heads: int = 4, radius_km: float = 400.0, dropout: float = 0.1):
        super().__init__()
        self.h, self.dk, self.radius = heads, d // heads, radius_km
        self.qkv = nn.Linear(d, 3 * d)
        self.edge = nn.Sequential(nn.Linear(4, 32), nn.GELU(), nn.Linear(32, heads))
        self.out = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, z, xy, alt, node_mask):
        """z [B, N, D], xy [B, N, 2] km, alt [B, N] km, node_mask [B, N]."""
        b, n, d = z.shape
        q, k, v = self.qkv(self.norm(z)).view(b, n, 3, self.h, self.dk).unbind(2)
        att = torch.einsum("bihk,bjhk->bhij", q, k) / math.sqrt(self.dk)
        rel = xy[:, None, :, :] - xy[:, :, None, :]
        dist = rel.norm(dim=-1, keepdim=True)
        dalt = (alt[:, None, :] - alt[:, :, None]).unsqueeze(-1)
        e = self.edge(torch.cat([rel / 100, dist / 100, dalt / 5], -1))     # [B, N, N, H]
        att = att + e.permute(0, 3, 1, 2)
        valid = (dist.squeeze(-1) <= self.radius) & node_mask[:, None, :] & node_mask[:, :, None]
        att = att.masked_fill(~valid[:, None], float("-inf"))
        att = torch.nan_to_num(att.softmax(-1), nan=0.0)
        o = torch.einsum("bhij,bjhk->bihk", att, v).reshape(b, n, d)
        z = z + self.out(o)
        return z + self.ff(z)


class PhysicsLocalizer(nn.Module):
    """Differentiable radio-horizon/Friis likelihood over a grid, mixed with a learned heatmap.

    For each grid cell g and EIRP e: LL(g, e) = Σ_steps w·[p log σ(z) + (1−p) log σ(−z)],
    z = (Pr(g, e) − τ) / T. p = predicted per-step jamming probability.
    """

    def __init__(self, d: int = 96, use_physics: bool = True, use_learned: bool = True):
        super().__init__()
        lin = torch.linspace(-GRID_HALF_KM, GRID_HALF_KM, GRID_N)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        self.register_buffer("grid", torch.stack([gx.flatten(), gy.flatten()], -1))     # [G, 2]
        self.register_buffer("eirp", torch.tensor(EIRP_CANDIDATES))
        self.log_temp = nn.Parameter(torch.tensor(math.log(3.0)))
        self.tau_off = nn.Parameter(torch.tensor(-3.0))            # learned offset to the −115 dBW tolerance
        self.mix = nn.Parameter(torch.tensor(1.0))
        self.use_physics, self.use_learned = use_physics, use_learned
        self.q = nn.Linear(2, d)
        self.k = nn.Linear(d + 2, d)

    def forward(self, step_p, pos_km, h_m, mask, z_nodes, xy_nodes, node_mask, stride: int = 4):
        b = step_p.shape[0]
        logits = torch.zeros(b, self.grid.shape[0], device=step_p.device)
        if self.use_physics:
            p = step_p[:, :, ::stride].clamp(1e-4, 1 - 1e-4)
            w = mask[:, :, ::stride].float()
            pos = pos_km[:, :, ::stride]
            h = h_m[:, :, ::stride]
            g = self.grid.view(1, -1, 1, 1, 2)
            dist_m = (pos.unsqueeze(1) - g).norm(dim=-1) * 1000.0                     # [B, G, N, T]
            temp = self.log_temp.exp()
            lls = []
            for e in self.eirp:
                pr = P.received_power_dbw(e, dist_m, torch.tensor(10.0, device=dist_m.device), h.unsqueeze(1))
                los = dist_m <= P.radio_horizon_m(h.unsqueeze(1), torch.tensor(10.0, device=dist_m.device))
                pr = torch.where(los, pr, pr - 40.0)
                z = (pr - (P.JAM_TOLERANCE_DBW + self.tau_off)) / temp
                ll = (w.unsqueeze(1) * (p.unsqueeze(1) * Fn.logsigmoid(z) + (1 - p.unsqueeze(1)) * Fn.logsigmoid(-z))).sum((-1, -2))
                lls.append(ll)
            ll = torch.stack(lls, -1).logsumexp(-1)                                     # [B, G]
            ll = ll - ll.mean(-1, keepdim=True)
            logits = logits + ll / (1 + w.sum((1, 2)).unsqueeze(-1).sqrt())
        if self.use_learned:
            q = self.q(self.grid / GRID_HALF_KM)                                         # [G, D]
            k = self.k(torch.cat([z_nodes, xy_nodes / GRID_HALF_KM], -1))                 # [B, N, D]
            s = torch.einsum("gd,bnd->bgn", q, k) / math.sqrt(q.shape[-1])
            s = s.masked_fill(~node_mask[:, None, :], float("-inf"))
            jam_w = step_p.max(-1).values                                                 # node jam evidence
            learned = (s.softmax(-1) * jam_w[:, None, :]).sum(-1)
            logits = logits + self.mix * learned * 10
        return logits

    def soft_argmax(self, logits):
        return (logits.softmax(-1).unsqueeze(-1) * self.grid).sum(1)


class GIST(nn.Module):
    def __init__(self, f_in: int, d: int = 96, enc_layers: int = 3, graph_layers: int = 3,
                 use_graph: bool = True, use_physics: bool = True, use_learned_loc: bool = True):
        super().__init__()
        self.enc = TrajEncoder(f_in, d, enc_layers)
        self.graph = nn.ModuleList([SpatialAttention(d) for _ in range(graph_layers)]) if use_graph else nn.ModuleList()
        self.node_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 3))
        self.step_head = nn.Sequential(nn.LayerNorm(2 * d), nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 3))
        self.loc = PhysicsLocalizer(d, use_physics, use_learned_loc)
        self.recon = nn.Linear(d, f_in)   # used by masked pretraining

    def encode(self, x, mask):
        b, n, t, f = x.shape
        steps, win = self.enc(x.view(b * n, t, f), mask.view(b * n, t))
        return steps.view(b, n, t, -1), win.view(b, n, -1)

    def forward(self, batch, localize: bool = True):
        x, mask, node_mask = batch["x"], batch["mask"], batch["node_mask"]
        steps, z = self.encode(x, mask)
        alt_km = (batch["h_m"] * mask).sum(-1) / mask.sum(-1).clamp(min=1) / 1000
        for layer in self.graph:
            z = layer(z, batch["xy_node"], alt_km, node_mask)
        node_logits = self.node_head(z)
        step_logits = self.step_head(torch.cat([steps, z.unsqueeze(2).expand_as(steps)], -1))
        out = {"node_logits": node_logits, "step_logits": step_logits, "z": z}
        if localize:
            step_p = step_logits.softmax(-1)[..., 1] * mask
            out["loc_logits"] = self.loc(step_p, batch["pos_km"], batch["h_m"], mask, z, batch["xy_node"], node_mask)
        return out
