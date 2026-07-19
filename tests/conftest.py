from __future__ import annotations

import pytest
import torch


@pytest.fixture
def weather() -> dict[str, torch.Tensor]:
    dtype = torch.float64
    t_air = torch.tensor([22.0, 30.0, 37.0], dtype=dtype)
    saturation = 611.0 * torch.exp(17.502 * t_air / (240.97 + t_air))
    return {
        "ca": torch.tensor([280.0, 420.0, 800.0], dtype=dtype),
        "O2": torch.full((3,), 210.0, dtype=dtype),
        "tAir": t_air,
        "ea": saturation * torch.tensor([0.45, 0.65, 0.8], dtype=dtype),
        "pressure": torch.full((3,), 98400.0, dtype=dtype),
        "wind": torch.tensor([0.8, 3.0, 6.0], dtype=dtype),
        "PAR": torch.tensor([80.0, 300.0, 520.0], dtype=dtype),
        "long": torch.full((3,), 716.873119944625, dtype=dtype),
        "NIR": torch.tensor([25.0, 90.0, 145.0], dtype=dtype),
        "controlTemp": t_air,
    }
