import torch


def test_mps_available():
    assert torch.backends.mps.is_available()
