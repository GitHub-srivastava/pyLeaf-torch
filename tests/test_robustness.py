from __future__ import annotations

import pytest
import torch

from pyleaf_torch import Leaf, SolverOptions


def test_difficult_forcing_rows_converge_with_staged_multistart() -> None:
    dtype = torch.float64
    torch.manual_seed(123)
    rows = 200

    def uniform(lower: float, upper: float) -> torch.Tensor:
        return lower + (upper - lower) * torch.rand(rows, dtype=dtype)

    t_air = uniform(2.0, 50.0)
    saturation = 611.0 * torch.exp(17.502 * t_air / (240.97 + t_air))
    weather = {
        "ca": uniform(100.0, 1500.0),
        "O2": uniform(180.0, 250.0),
        "tAir": t_air,
        "ea": saturation * uniform(0.05, 1.0),
        "pressure": uniform(70000.0, 105000.0),
        "wind": 10.0 ** uniform(-3.0, 1.0),
        "PAR": uniform(0.0, 700.0),
        "long": uniform(300.0, 900.0),
        "NIR": uniform(0.0, 300.0),
        "controlTemp": uniform(2.0, 60.0),
    }
    difficult = torch.tensor(
        [51, 75, 80, 86, 95, 99, 144, 152, 168, 176, 178, 193]
    )
    subset = {name: value[difficult] for name, value in weather.items()}
    model = Leaf(
        trainable=(),
        mode="hard",
        energy_balance=False,
        solver_options=SolverOptions(max_iterations=100, residual_tolerance=1.0e-8),
    )
    with torch.no_grad():
        output = model(subset)
    assert bool(output.diagnostics.converged.all())
    assert float(output.diagnostics.residual_norm.max()) < 1.0e-8


def test_zero_wind_has_clear_validation_error(weather) -> None:
    calm = {name: value.clone() for name, value in weather.items()}
    calm["wind"][0] = 0.0
    model = Leaf(trainable=(), energy_balance=False)
    with pytest.raises(ValueError, match="zero wind"):
        model(calm)


def test_residual_rejects_multidimensional_state_mapping(weather) -> None:
    model = Leaf(trainable=(), energy_balance=False)
    malformed = {
        name: torch.ones((3, 1), dtype=torch.float64) for name in model.STATE_NAMES
    }
    with pytest.raises(ValueError, match="scalar or one-dimensional"):
        model.equilibrium_residual(malformed, weather)


@pytest.mark.parametrize(
    "options",
    (
        {"initial_damping": -1.0},
        {"minimum_damping": 1.0, "maximum_damping": 0.1},
        {"damping_decrease": 1.0},
        {"damping_increase": 1.0},
        {"implicit_regularization": float("nan")},
        {"residual_tolerance": float("nan")},
        {"max_iterations": 2.5},
    ),
)
def test_invalid_solver_options_are_rejected(options) -> None:
    with pytest.raises(ValueError):
        SolverOptions(**options)


def test_unused_vpr25_cannot_be_marked_trainable() -> None:
    with pytest.raises(ValueError, match="do not use it"):
        Leaf(trainable=("vpr25",))
