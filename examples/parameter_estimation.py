"""Multi-leaf A-Ci parameter estimation from measured Ci, matching standard
practice: naive per-curve fits vs. a PhoTorch-style shared/ratio-regularized
joint fit, on synthetic data with a known truth.

Real A-Ci curve parameter estimation is driven by measured intercellular CO2
(``Ci``), not ambient CO2 (``Ca``): a gas-exchange instrument already reports
``Ci`` (back-calculated from the measured An/gs), so biochemical parameters
are fit by comparing predicted vs. measured assimilation at that given ``Ci``
directly, without re-solving a stomatal-conductance model. This script
generates "measured" data the way a real gas-exchange system would --
simulating each leaf with the full, `ca`-driven :class:`~pyleaf_torch.Leaf`
model (with its own, leaf-specific stomatal behavior) -- and then fits
against only the resulting ``Ci``, ``tLeaf``, ``PAR``, ``O2``, and ``An``
using :class:`~pyleaf_torch.LeafBiochemistry`. The fitting code never sees
``ca``, ``go``, or ``g1``: you don't need a leaf's stomatal parameters to fit
its biochemistry from measured Ci.

The default fit targets ``vcmax25``, ``vpmax25``, and ``rd25`` --
deliberately not ``jmax25``: a single A-Ci curve is run at one (saturating)
PAR, so it constrains electron transport only as the realized rate ``J`` at
that light level, not the light-response curvature needed to separate out
``Jmax25`` (that needs a PAR sweep / A-Q curve). Fitting ``jmax25`` from
Ci-only data would silently reintroduce an equifinality problem instead of
avoiding one.

This demonstrates the equifinality problem `pyleaf_torch.calibration` targets
for the parameters an A-Ci curve *can* identify: `vpmax25` and `vcmax25` are
still frequently confounded on a single curve (identifiability_report below
routinely flags |correlation| > 0.9). Each leaf's curve starts from its own
independently random-perturbed initial guess (``START_RANGES``), so naive
per-curve fits are free to drift to different, individually-plausible
combinations; a cross-leaf ratio regularizer (adapted from PhoTorch's
Jmax-Vcmax correlation penalty) that assumes capacities co-scale across
leaves of the same kind pulls those independent fits back toward a shared,
more biologically consistent configuration ("joint").

A second, separate, minimal demonstration afterward switches on finite
mesophyll conductance (`finite_gm=True`) for a single curve, showing the new
`cm` state and `gm25` fit -- this is off by default because a standard A-Ci
curve at one temperature does not, by itself, identify `gm25` from `vpmax25`
well either (see MODEL_NOTES.md); it is included here only to show the
mechanism, not as a recommended default.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from pyleaf_torch import (
    CalibrationOptions,
    CurveGroup,
    Leaf,
    LeafBiochemistry,
    RatioRegularizer,
    SolverOptions,
    fit_curve_group,
    identifiability_report,
)


TRUTH_VCMAX25 = (40.0, 55.0, 65.0, 80.0)
VPMAX_RATIO = 2.2
TRUTH_RD_RATIO = 0.04
# Every curve draws its own starting point from these ranges (independent per
# curve, matching how a real multi-leaf calibration actually starts: each
# leaf gets its own rough initial guess, not an identical one). A shared
# starting point tends to make every curve move almost identically even
# without regularization, which would understate the effect being
# demonstrated.
START_RANGES = {
    "vcmax25": (20.0, 45.0),
    "vpmax25": (45.0, 95.0),
    "rd25": (0.01, 0.03),
}
FIT_NAMES = ("vcmax25", "vpmax25", "rd25")


def synthetic_truth(num_leaves: int, seed: int) -> list[dict[str, float]]:
    generator = torch.Generator().manual_seed(seed)

    def jitter() -> float:
        return float(1.0 + 0.05 * (2.0 * torch.rand((), generator=generator) - 1.0))

    truth: list[dict[str, float]] = []
    for index in range(num_leaves):
        vcmax25 = TRUTH_VCMAX25[index % len(TRUTH_VCMAX25)]
        truth.append(
            {
                "vcmax25": vcmax25,
                "vpmax25": vcmax25 * VPMAX_RATIO * jitter(),
                "rd25": TRUTH_RD_RATIO * jitter(),
            }
        )
    return truth


def synthesize_measured_curve(
    leaf_truth: dict[str, float],
    rows: int,
    solver: SolverOptions,
    dtype: torch.dtype,
    *,
    finite_gm: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Simulate one leaf with the full ca-driven Leaf model and return what a
    gas-exchange instrument would actually report: Ci, tLeaf, PAR, O2, and An.
    """

    t_air = torch.full((rows,), 28.0, dtype=dtype)
    saturation = 611.0 * torch.exp(17.502 * t_air / (240.97 + t_air))
    weather = {
        "ca": torch.linspace(80.0, 1400.0, rows, dtype=dtype),
        "O2": torch.full((rows,), 210.0, dtype=dtype),
        "tAir": t_air,
        "ea": saturation * 0.6,
        "pressure": torch.full((rows,), 98400.0, dtype=dtype),
        "wind": torch.full((rows,), 2.0, dtype=dtype),
        "PAR": torch.full((rows,), 1500.0, dtype=dtype),  # saturating light
        "long": torch.full((rows,), 716.87, dtype=dtype),
        "NIR": torch.full((rows,), 100.0, dtype=dtype),
        "controlTemp": t_air,
    }
    leaf = Leaf(
        leaf_truth,
        trainable=(),
        mode="smooth",
        energy_balance=False,
        finite_gm=finite_gm,
        solver_options=solver,
        dtype=dtype,
    )
    with torch.no_grad():
        output = leaf(weather)
    if not bool(output.diagnostics.converged.all()):
        raise RuntimeError("Synthetic truth simulation did not converge")

    conditions = {
        "ci": output.state["ci"].detach(),
        "tLeaf": output.state["tLeaf"].detach(),
        "PAR": weather["PAR"],
        "O2": weather["O2"],
    }
    return conditions, output.mass["aNet"].detach()


def build_curve_group(
    truth: list[dict[str, float]], rows_per_leaf: int, solver: SolverOptions, dtype: torch.dtype
) -> CurveGroup:
    condition_parts: list[dict[str, torch.Tensor]] = []
    observed_parts: list[torch.Tensor] = []
    for leaf_truth in truth:
        conditions, observed_a = synthesize_measured_curve(leaf_truth, rows_per_leaf, solver, dtype)
        condition_parts.append(conditions)
        observed_parts.append(observed_a)

    conditions = {
        name: torch.cat([part[name] for part in condition_parts]) for name in condition_parts[0]
    }
    curve_id = torch.repeat_interleave(
        torch.arange(len(truth), dtype=torch.int64), rows_per_leaf
    )
    return CurveGroup(
        conditions=conditions, observed_aNet=torch.cat(observed_parts), curve_id=curve_id
    )


def recovery_error(
    fitted: dict[str, list[float]], truth: list[dict[str, float]]
) -> dict[str, float]:
    errors: dict[str, float] = {}
    for name in fitted:
        percent_errors = [
            100.0 * abs(fitted[name][index] - truth[index][name]) / truth[index][name]
            for index in range(len(truth))
        ]
        errors[name] = sum(percent_errors) / len(percent_errors)
    return errors


def finite_gm_demo(solver: SolverOptions, dtype: torch.dtype, iterations: int) -> dict:
    """Minimal single-curve demonstration that the finite-gm switch works.

    Not a full naive-vs-joint study: a standard single-temperature A-Ci curve
    does not, by itself, identify gm25 well either (see the identifiability
    report below), so this is shown only as a mechanism check, not a
    recommended default workflow.
    """

    truth = {"vcmax25": 55.0, "vpmax25": 120.0, "gm25": 3.0, "rd25": 0.04}
    rows = 10
    conditions, observed_a = synthesize_measured_curve(
        {**truth}, rows=rows, solver=solver, dtype=dtype, finite_gm=True
    )
    curves = CurveGroup(
        conditions=conditions, observed_aNet=observed_a, curve_id=torch.zeros(rows, dtype=torch.int64)
    )
    options = CalibrationOptions(iterations=iterations, learning_rate=0.15, lr_step=20, lr_gamma=0.5)
    result = fit_curve_group(
        curves,
        per_curve=("vcmax25", "vpmax25", "gm25"),
        fixed={"rd25": 0.04},
        initial={"vcmax25": 35.0, "vpmax25": 70.0, "gm25": 6.0},
        finite_gm=True,
        mode="smooth",
        solver_options=solver,
        dtype=dtype,
        options=options,
    )
    report = identifiability_report(
        result.curve_models[0],
        curves.curve_conditions(0),
        curves.curve_observed_aNet(0),
        ("vcmax25", "vpmax25", "gm25"),
    )
    return {
        "truth": truth,
        "fitted": {name: values[0] for name, values in result.per_curve_parameters.items()},
        "best_loss": result.best_loss,
        "flagged_pairs": report.flagged_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--leaves", type=int, default=4)
    parser.add_argument("--rows-per-leaf", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--skip-finite-gm-demo", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.leaves < 2:
        parser.error("--leaves must be at least 2 for cross-curve regularization to mean anything")
    if args.rows_per_leaf < 8:
        parser.error(
            "--rows-per-leaf must be at least 8: real A-Ci curve protocols use "
            "8-10+ CO2 points per curve to resolve the Rubisco-limited, transition, "
            "and CO2-saturated regions"
        )

    torch.manual_seed(args.seed)
    dtype = torch.float64
    solver = SolverOptions(max_iterations=100, residual_tolerance=1.0e-7)
    truth = synthetic_truth(args.leaves, args.seed)
    curves = build_curve_group(truth, args.rows_per_leaf, solver, dtype)

    options = CalibrationOptions(
        iterations=args.iterations,
        learning_rate=0.15,
        lr_step=25,
        lr_gamma=0.5,
        restarts=1,
        seed=args.seed,
        grad_clip_norm=None,
    )

    naive = fit_curve_group(
        curves,
        per_curve=FIT_NAMES,
        start_ranges=START_RANGES,
        mode="smooth",
        solver_options=solver,
        dtype=dtype,
        options=options,
    )
    joint = fit_curve_group(
        curves,
        per_curve=FIT_NAMES,
        start_ranges=START_RANGES,
        ratio_regularizers=(RatioRegularizer("vpmax25", "vcmax25", weight=12.0),),
        mode="smooth",
        solver_options=solver,
        dtype=dtype,
        options=options,
    )

    naive_error = recovery_error(naive.per_curve_parameters, truth)
    joint_error = recovery_error(joint.per_curve_parameters, truth)

    naive_reports = [
        identifiability_report(
            naive.curve_models[index],
            curves.curve_conditions(index),
            curves.curve_observed_aNet(index),
            ("vpmax25", "vcmax25"),
        )
        for index in range(args.leaves)
    ]
    joint_reports = [
        identifiability_report(
            joint.curve_models[index],
            curves.curve_conditions(index),
            curves.curve_observed_aNet(index),
            ("vpmax25", "vcmax25"),
        )
        for index in range(args.leaves)
    ]

    result = {
        "warning": (
            "This synthetic local demonstration measures cross-curve equifinality "
            "reduction, not a guarantee of recovery on real, noisy data."
        ),
        "truth": truth,
        "naive": {
            "best_loss": naive.best_loss,
            "mean_absolute_percent_error": naive_error,
            "per_curve_parameters": naive.per_curve_parameters,
            "flagged_pairs_by_leaf": [report.flagged_pairs for report in naive_reports],
        },
        "joint_regularized": {
            "best_loss": joint.best_loss,
            "mean_absolute_percent_error": joint_error,
            "per_curve_parameters": joint.per_curve_parameters,
            "flagged_pairs_by_leaf": [report.flagged_pairs for report in joint_reports],
        },
    }
    if not args.skip_finite_gm_demo:
        result["finite_gm_demo"] = finite_gm_demo(solver, dtype, iterations=args.iterations)

    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
