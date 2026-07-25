"""Differentiable C4 leaf gas-exchange and energy-balance model."""

from .calibration import (
    CalibrationOptions,
    CalibrationResult,
    CurveGroup,
    IdentifiabilityReport,
    RatioRegularizer,
    build_curve_models,
    collect_parameters,
    fit_curve_group,
    identifiability_report,
)
from .data import OutputFrames, simulate_dataframe, weather_from_dataframe
from .model import Leaf, LeafBiochemistry
from .outputs import LeafOutput, SolverDiagnostics
from .parameters import DEFAULT_PARAMETERS, PARAMETER_BOUNDS
from .solver import SolverOptions

__all__ = [
    "DEFAULT_PARAMETERS",
    "PARAMETER_BOUNDS",
    "CalibrationOptions",
    "CalibrationResult",
    "CurveGroup",
    "IdentifiabilityReport",
    "Leaf",
    "LeafBiochemistry",
    "LeafOutput",
    "OutputFrames",
    "RatioRegularizer",
    "SolverDiagnostics",
    "SolverOptions",
    "build_curve_models",
    "collect_parameters",
    "fit_curve_group",
    "identifiability_report",
    "simulate_dataframe",
    "weather_from_dataframe",
]

__version__ = "0.1.0"
