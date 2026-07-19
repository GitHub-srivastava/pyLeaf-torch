"""Tensor outputs returned by :class:`DifferentiableLeaf`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from torch import Tensor


@dataclass(frozen=True)
class SolverDiagnostics:
    """Detached numerical diagnostics for each weather row."""

    converged: Tensor
    iterations: Tensor
    residual_norm: Tensor
    jacobian_condition: Tensor
    line_search_failures: Tensor
    at_state_bound: Tensor


@dataclass(frozen=True)
class LeafOutput:
    """Differentiable physical outputs plus detached solver diagnostics."""

    state: Mapping[str, Tensor]
    mass: Mapping[str, Tensor]
    energy: Mapping[str, Tensor]
    diagnostics: SolverDiagnostics
    residual: Tensor
    limitation: Mapping[str, Tensor]

    def select(self, name: str) -> Tensor:
        """Return an output by name from state, mass, or energy mappings."""

        for group in (self.state, self.mass, self.energy):
            if name in group:
                return group[name]
        raise KeyError(f"Unknown leaf output: {name}")
