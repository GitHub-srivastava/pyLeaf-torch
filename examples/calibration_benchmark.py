"""Compare autograd calibration with derivative-free Powell optimization."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize

from pyleaf_torch import DifferentiableLeaf, SolverOptions


FIT_NAMES = ("vcmax25", "jmax25", "g1")
TRUTH = {"vcmax25": 62.0, "jmax25": 285.0, "g1": 4.2}
START = {"vcmax25": 32.0, "jmax25": 500.0, "g1": 1.6}
PHYSICAL_BOUNDS = ((5.0, 150.0), (20.0, 800.0), (0.2, 12.0))
RAW_BOUNDS = tuple((np.log(lower), np.log(upper)) for lower, upper in PHYSICAL_BOUNDS)


def synthetic_weather(rows: int) -> dict[str, torch.Tensor]:
    dtype = torch.float64
    phase = torch.linspace(0.0, 1.0, rows, dtype=dtype)
    t_air = 18.0 + 20.0 * phase
    saturation = 611.0 * torch.exp(17.502 * t_air / (240.97 + t_air))
    return {
        "ca": 220.0 + 780.0 * torch.remainder(phase * 1.7, 1.0),
        "O2": torch.full((rows,), 210.0, dtype=dtype),
        "tAir": t_air,
        "ea": saturation * (0.4 + 0.4 * torch.remainder(phase * 2.3, 1.0)),
        "pressure": torch.full((rows,), 98400.0, dtype=dtype),
        "wind": 0.5 + 5.0 * torch.remainder(phase * 1.3, 1.0),
        "PAR": 15.0 + 525.0 * torch.remainder(phase * 2.1, 1.0),
        "long": torch.full((rows,), 716.87, dtype=dtype),
        "NIR": 150.0 * torch.remainder(phase * 2.1, 1.0),
        "controlTemp": t_air,
    }


def standardized_loss(output, target_a, target_gs) -> torch.Tensor:
    return ((output.mass["aNet"] - target_a) / 10.0).square().mean() + (
        (output.state["gs"] - target_gs) / 0.1
    ).square().mean()


def require_converged(output, label: str) -> None:
    if not bool(output.diagnostics.converged.all()):
        rows = torch.nonzero(~output.diagnostics.converged).flatten().tolist()
        raise RuntimeError(f"{label} has nonconverged row indices: {rows}")


def raw_vector(model: DifferentiableLeaf) -> np.ndarray:
    return np.array(
        [float(model.raw_parameters[name].detach()) for name in FIT_NAMES], dtype=float
    )


def set_raw_vector(model: DifferentiableLeaf, vector: np.ndarray) -> None:
    with torch.no_grad():
        for name, value in zip(FIT_NAMES, vector, strict=True):
            model.raw_parameters[name].copy_(
                torch.as_tensor(value, dtype=model.state_scale.dtype)
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--evaluations", type=int, default=120)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.rows < 3:
        parser.error("--rows must be at least 3")
    if args.evaluations < 2:
        parser.error("--evaluations must be at least 2")

    torch.manual_seed(7)
    weather = synthetic_weather(args.rows)
    solver = SolverOptions(max_iterations=100, residual_tolerance=1.0e-7)
    truth_model = DifferentiableLeaf(
        TRUTH,
        trainable=(),
        mode="smooth",
        energy_balance=False,
        solver_options=solver,
    )
    with torch.no_grad():
        truth_output = truth_model(weather)
        require_converged(truth_output, "synthetic truth")
        target_a = truth_output.mass["aNet"].detach()
        target_gs = truth_output.state["gs"].detach()

    gradient_model = DifferentiableLeaf(
        START,
        trainable=FIT_NAMES,
        mode="smooth",
        energy_balance=False,
        solver_options=solver,
    )
    optimizer = torch.optim.Adam(gradient_model.parameters(), lr=0.03)
    with torch.no_grad():
        initial_gradient_output = gradient_model(weather)
        require_converged(initial_gradient_output, "gradient initial point")
        initial_gradient_loss = float(
            standardized_loss(initial_gradient_output, target_a, target_gs)
        )
    gradient_start = time.perf_counter()
    gradient_trace: list[float] = []
    gradient_best_loss = initial_gradient_loss
    gradient_best_raw = raw_vector(gradient_model)
    for _ in range(args.evaluations - 1):
        optimizer.zero_grad()
        gradient_output = gradient_model(weather)
        loss = standardized_loss(gradient_output, target_a, target_gs)
        loss.backward()
        loss_value = float(loss.detach())
        if loss_value < gradient_best_loss:
            gradient_best_loss = loss_value
            gradient_best_raw = raw_vector(gradient_model)
        torch.nn.utils.clip_grad_norm_(gradient_model.parameters(), max_norm=10.0)
        optimizer.step()
        with torch.no_grad():
            for name, (lower, upper) in zip(FIT_NAMES, RAW_BOUNDS, strict=True):
                gradient_model.raw_parameters[name].clamp_(lower, upper)
        gradient_trace.append(loss_value)
    with torch.no_grad():
        final_gradient_output = gradient_model(weather)
        require_converged(final_gradient_output, "gradient final point")
        final_gradient_loss = float(
            standardized_loss(final_gradient_output, target_a, target_gs)
        )
    if final_gradient_loss < gradient_best_loss:
        gradient_best_loss = final_gradient_loss
        gradient_best_raw = raw_vector(gradient_model)
    gradient_seconds = time.perf_counter() - gradient_start

    powell_model = DifferentiableLeaf(
        START,
        trainable=FIT_NAMES,
        mode="smooth",
        energy_balance=False,
        solver_options=solver,
    )
    initial_raw = raw_vector(powell_model)
    powell_trace: list[float] = []
    powell_invalid_evaluations = 0
    with torch.no_grad():
        initial_powell_output = powell_model(weather)
        require_converged(initial_powell_output, "Powell initial point")
        initial_powell_loss = float(
            standardized_loss(initial_powell_output, target_a, target_gs)
        )
    powell_best_loss = initial_powell_loss
    powell_best_raw = initial_raw.copy()

    def objective(vector: np.ndarray) -> float:
        nonlocal powell_best_loss, powell_best_raw, powell_invalid_evaluations
        set_raw_vector(powell_model, vector)
        with torch.no_grad():
            candidate_output = powell_model(weather)
            if not bool(candidate_output.diagnostics.converged.all()):
                powell_invalid_evaluations += 1
                value = 1.0e6 + float(candidate_output.diagnostics.residual_norm.max())
            else:
                value = float(
                    standardized_loss(candidate_output, target_a, target_gs)
                )
        powell_trace.append(value)
        if value < powell_best_loss:
            powell_best_loss = value
            powell_best_raw = vector.copy()
        return value

    powell_start = time.perf_counter()
    powell_result = minimize(
        objective,
        initial_raw,
        method="Powell",
        bounds=RAW_BOUNDS,
        options={"maxfev": args.evaluations, "disp": False},
    )
    powell_seconds = time.perf_counter() - powell_start
    powell_returned_parameters = {
        name: float(np.exp(value))
        for name, value in zip(FIT_NAMES, powell_result.x, strict=True)
    }
    set_raw_vector(powell_model, powell_best_raw)

    result = {
        "warning": (
            "This synthetic local benchmark measures parameter calibration, not a "
            "guarantee of forward-solver or real-data convergence."
        ),
        "truth": TRUTH,
        "start": START,
        "gradient": {
            "evaluations": args.evaluations,
            "optimizer_updates": args.evaluations - 1,
            "initial_loss": initial_gradient_loss,
            "last_evaluated_loss": final_gradient_loss,
            "best_evaluated_loss": gradient_best_loss,
            "best_loss_reduction_percent": 100.0
            * (initial_gradient_loss - gradient_best_loss)
            / initial_gradient_loss,
            "seconds": gradient_seconds,
            "maximum_final_residual": float(
                final_gradient_output.diagnostics.residual_norm.max()
            ),
            "maximum_final_jacobian_condition": float(
                final_gradient_output.diagnostics.jacobian_condition.max()
            ),
            "last_parameters": {
                name: gradient_model.parameter_report()[name] for name in FIT_NAMES
            },
            "best_parameters": {
                name: float(np.exp(value))
                for name, value in zip(FIT_NAMES, gradient_best_raw, strict=True)
            },
        },
        "powell": {
            "evaluations": len(powell_trace),
            "initial_loss": initial_powell_loss,
            "optimizer_returned_loss": float(powell_result.fun),
            "best_evaluated_loss": powell_best_loss,
            "best_loss_reduction_percent": 100.0
            * (initial_powell_loss - powell_best_loss)
            / initial_powell_loss,
            "seconds": powell_seconds,
            "success": bool(powell_result.success),
            "invalid_equilibrium_evaluations": powell_invalid_evaluations,
            "optimizer_returned_parameters": powell_returned_parameters,
            "best_parameters": {
                name: powell_model.parameter_report()[name] for name in FIT_NAMES
            },
        },
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
