"""Parameter defaults, constraints, and differentiable transformations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

import torch
from torch import Tensor


DEFAULT_PARAMETERS: Final[dict[str, float]] = {
    "vcmax25": 50.0,
    "vpmax25": 110.0,
    "jmax25": 350.0,
    "vpr25": 80.0,
    "sco25": 2590.0,
    "Ko25": 450.0,
    "Kc25": 650.0,
    "Kp25": 80.0,
    "gbs": 0.003,
    "gm25": 3.0,
    "alpha": 0.0,
    "theta": 0.7,
    "x": 0.4,
    "rd25": 0.05,
    "go": 0.08,
    "g1": 3.0,
    "leafDim": 0.08,
    "cForced": 0.004322,
    "cFree": 0.001631,
    "s": 0.71,
}


Transform = Literal["positive", "bounded"]


@dataclass(frozen=True)
class ParameterSpec:
    transform: Transform
    lower: float = 0.0
    upper: float | None = None


PARAMETER_SPECS: Final[dict[str, ParameterSpec]] = {
    "vcmax25": ParameterSpec("positive"),
    "vpmax25": ParameterSpec("positive"),
    "jmax25": ParameterSpec("positive"),
    "vpr25": ParameterSpec("positive"),
    "sco25": ParameterSpec("positive"),
    "Ko25": ParameterSpec("positive"),
    "Kc25": ParameterSpec("positive"),
    "Kp25": ParameterSpec("positive"),
    "gbs": ParameterSpec("positive"),
    "gm25": ParameterSpec("positive"),
    "alpha": ParameterSpec("bounded", 0.0, 1.0),
    "theta": ParameterSpec("bounded", 0.05, 0.999),
    "x": ParameterSpec("bounded", 0.001, 0.999),
    "rd25": ParameterSpec("bounded", 1.0e-5, 0.5),
    "go": ParameterSpec("positive"),
    "g1": ParameterSpec("positive"),
    "leafDim": ParameterSpec("positive"),
    "cForced": ParameterSpec("positive"),
    "cFree": ParameterSpec("positive"),
    "s": ParameterSpec("positive"),
}


PARAMETER_BOUNDS: Final[dict[str, tuple[float, float | None]]] = {
    name: (spec.lower, spec.upper) for name, spec in PARAMETER_SPECS.items()
}


def to_raw(value: Tensor, spec: ParameterSpec) -> Tensor:
    """Map a physical value to the unconstrained optimizer coordinate."""

    if spec.transform == "positive":
        return torch.log(value - spec.lower)

    assert spec.upper is not None
    fraction = (value - spec.lower) / (spec.upper - spec.lower)
    return torch.logit(fraction)


def from_raw(raw: Tensor, spec: ParameterSpec) -> Tensor:
    """Map an optimizer coordinate to a valid physical value."""

    if spec.transform == "positive":
        return spec.lower + torch.exp(raw)

    assert spec.upper is not None
    return spec.lower + (spec.upper - spec.lower) * torch.sigmoid(raw)
