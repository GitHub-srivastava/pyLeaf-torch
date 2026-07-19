"""Differentiable C4 leaf gas-exchange and energy-balance model."""

from .data import OutputFrames, simulate_dataframe, weather_from_dataframe
from .model import DifferentiableLeaf
from .outputs import LeafOutput, SolverDiagnostics
from .parameters import DEFAULT_PARAMETERS, PARAMETER_BOUNDS
from .solver import SolverOptions

__all__ = [
    "DEFAULT_PARAMETERS",
    "PARAMETER_BOUNDS",
    "DifferentiableLeaf",
    "LeafOutput",
    "OutputFrames",
    "SolverDiagnostics",
    "SolverOptions",
    "simulate_dataframe",
    "weather_from_dataframe",
]

__version__ = "0.1.0"
