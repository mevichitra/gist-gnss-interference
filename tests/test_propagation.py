import numpy as np
import torch

from gist.sim import propagation as P


def test_haversine_one_degree_lat():
    assert abs(P.haversine_m(0.0, 0.0, 1.0, 0.0) - 111_195) < 50


def test_radio_horizon_cruise_vs_ground():
    # 10 km aircraft vs 10 m jammer: roughly 412 km + 13 km.
    h = P.radio_horizon_m(np.array(10_000.0), np.array(10.0))
    assert 400_000 < h < 440_000


def test_liu_1kw_jammer_reaches_tolerance_near_100km():
    # Liu et al. 2022 estimate Pt ~ 1000 W from points affected out to ~O(100 km).
    p = P.received_power_dbw(30.0, np.array(100_000.0), 10.0, 10_000.0, use_pattern=False)
    assert p > P.JAM_TOLERANCE_DBW


def test_torch_and_numpy_agree_and_differentiable():
    g = torch.tensor([50_000.0], requires_grad=True)
    pt = P.received_power_dbw(torch.tensor(20.0), g, torch.tensor(10.0), torch.tensor(9_000.0))
    pn = P.received_power_dbw(20.0, np.array([50_000.0]), 10.0, 9_000.0)
    assert abs(pt.item() - pn[0]) < 1e-3
    pt.sum().backward()
    assert g.grad is not None and g.grad.item() < 0


def test_dest_point_roundtrip():
    la, lo = P.dest_point(54.7, 20.5, 90.0, 100_000.0)
    assert abs(P.haversine_m(54.7, 20.5, la, lo) - 100_000) < 1
