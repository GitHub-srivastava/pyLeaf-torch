from __future__ import annotations

import pytest
import torch

from pyleaf_torch import (
    CalibrationOptions,
    CurveGroup,
    LeafBiochemistry,
    RatioRegularizer,
    SolverOptions,
    build_curve_models,
    collect_parameters,
    fit_curve_group,
    identifiability_report,
)


DTYPE = torch.float64
SOLVER = SolverOptions(max_iterations=100, residual_tolerance=1.0e-7)


def _curve_conditions(rows: int, ci_lo: float, ci_hi: float, t_leaf_value: float) -> dict[str, torch.Tensor]:
    return {
        "ci": torch.linspace(ci_lo, ci_hi, rows, dtype=DTYPE),
        "tLeaf": torch.full((rows,), t_leaf_value, dtype=DTYPE),
        "PAR": torch.full((rows,), 1200.0, dtype=DTYPE),
        "O2": torch.full((rows,), 210.0, dtype=DTYPE),
    }


def _two_curve_group(
    truths: tuple[dict[str, float], dict[str, float]], rows: int = 6
) -> CurveGroup:
    c0 = _curve_conditions(rows, 150.0, 900.0, 25.0)
    c1 = _curve_conditions(rows, 150.0, 900.0, 25.0)
    conditions = {name: torch.cat([c0[name], c1[name]]) for name in c0}
    curve_id = torch.cat([torch.zeros(rows, dtype=torch.int64), torch.ones(rows, dtype=torch.int64)])

    observed = []
    for truth, c in zip(truths, (c0, c1), strict=True):
        model = LeafBiochemistry(truth, trainable=(), mode="smooth", solver_options=SOLVER, dtype=DTYPE)
        with torch.no_grad():
            output = model(c)
        assert bool(output.diagnostics.converged.all())
        observed.append(output.mass["aNet"].detach())

    return CurveGroup(conditions=conditions, observed_aNet=torch.cat(observed), curve_id=curve_id)


@pytest.fixture
def conditions() -> dict[str, torch.Tensor]:
    return _curve_conditions(3, 200.0, 800.0, 25.0)


def test_curve_group_requires_matching_row_counts(conditions) -> None:
    with pytest.raises(ValueError, match="one entry per condition row"):
        CurveGroup(
            conditions=conditions,
            observed_aNet=torch.zeros(2, dtype=DTYPE),
            curve_id=torch.zeros(3, dtype=torch.int64),
        )


def test_curve_group_requires_integer_curve_id(conditions) -> None:
    with pytest.raises(ValueError, match="integer"):
        CurveGroup(
            conditions=conditions,
            observed_aNet=torch.zeros(3, dtype=DTYPE),
            curve_id=torch.zeros(3, dtype=DTYPE),
        )


def test_curve_group_num_curves_and_curve_conditions(conditions) -> None:
    curve_id = torch.tensor([5, 5, 9], dtype=torch.int64)
    group = CurveGroup(conditions=conditions, observed_aNet=torch.zeros(3, dtype=DTYPE), curve_id=curve_id)
    assert group.num_curves == 2
    assert group.curve_conditions(0)["ci"].shape == (2,)
    assert group.curve_conditions(1)["ci"].shape == (1,)


def test_build_curve_models_ties_shared_and_separates_per_curve() -> None:
    group = CurveGroup(
        conditions=_curve_conditions(3, 200.0, 800.0, 25.0),
        observed_aNet=torch.zeros(3, dtype=DTYPE),
        curve_id=torch.tensor([0, 1, 1], dtype=torch.int64),
    )
    models = build_curve_models(
        group, shared=("x",), per_curve=("vcmax25",), mode="smooth", solver_options=SOLVER, dtype=DTYPE
    )
    assert len(models) == 2
    assert models[0].raw_parameters["x"] is models[1].raw_parameters["x"]
    assert models[0].raw_parameters["vcmax25"] is not models[1].raw_parameters["vcmax25"]

    with torch.no_grad():
        models[0].raw_parameters["x"].add_(0.01)
    assert torch.equal(models[0].raw_parameters["x"], models[1].raw_parameters["x"])


def test_build_curve_models_rejects_shared_per_curve_overlap() -> None:
    group = CurveGroup(
        conditions=_curve_conditions(2, 200.0, 800.0, 25.0),
        observed_aNet=torch.zeros(2, dtype=DTYPE),
        curve_id=torch.tensor([0, 1], dtype=torch.int64),
    )
    with pytest.raises(ValueError, match="cannot be both"):
        build_curve_models(group, shared=("vcmax25",), per_curve=("vcmax25",))


def test_collect_parameters_dedups_tied_parameters() -> None:
    group = CurveGroup(
        conditions=_curve_conditions(2, 200.0, 800.0, 25.0),
        observed_aNet=torch.zeros(2, dtype=DTYPE),
        curve_id=torch.tensor([0, 1], dtype=torch.int64),
    )
    models = build_curve_models(group, shared=("vcmax25",), per_curve=("jmax25",))
    parameters = collect_parameters(models)
    assert len(parameters) == 3  # shared vcmax25 once, jmax25 for each of 2 curves


def test_ratio_regularizer_penalty_matches_hand_computation() -> None:
    left = LeafBiochemistry({"vcmax25": 50.0, "vpmax25": 100.0}, trainable=())
    right = LeafBiochemistry({"vcmax25": 50.0, "vpmax25": 150.0}, trainable=())
    regularizer = RatioRegularizer("vpmax25", "vcmax25", weight=2.0)
    penalty = regularizer.penalty([left, right])

    import math

    log_ratios = (math.log(100.0 / 50.0), math.log(150.0 / 50.0))
    mean = sum(log_ratios) / 2.0
    expected = 2.0 * sum((value - mean) ** 2 for value in log_ratios) / 2.0
    assert float(penalty) == pytest.approx(expected, rel=1.0e-9)


def test_ratio_regularizer_penalty_zero_for_single_model() -> None:
    only = LeafBiochemistry({"vcmax25": 50.0, "vpmax25": 100.0}, trainable=())
    regularizer = RatioRegularizer("vpmax25", "vcmax25")
    assert float(regularizer.penalty([only])) == 0.0


def test_fit_curve_group_reduces_loss_and_shapes_result() -> None:
    truths = (
        {"vcmax25": 55.0, "vpmax25": 120.0, "rd25": 0.04},
        {"vcmax25": 65.0, "vpmax25": 140.0, "rd25": 0.04},
    )
    curves = _two_curve_group(truths)
    options = CalibrationOptions(iterations=60, learning_rate=0.08, lr_step=25, lr_gamma=0.5, restarts=1)

    result = fit_curve_group(
        curves,
        per_curve=("vcmax25", "vpmax25"),
        fixed={"rd25": 0.04},
        initial={"vcmax25": 30.0, "vpmax25": 70.0},
        mode="smooth",
        solver_options=SOLVER,
        dtype=DTYPE,
        options=options,
    )

    assert result.loss_trace[-1] < 0.25 * result.loss_trace[0]
    assert result.best_loss <= min(result.loss_trace)
    assert set(result.per_curve_parameters) == {"vcmax25", "vpmax25"}
    for values in result.per_curve_parameters.values():
        assert len(values) == 2
    assert result.shared_parameters == {}
    assert result.restart_losses == (result.best_loss,)

    # Fitted per-curve vcmax25 should move meaningfully toward each curve's
    # own truth rather than staying stuck near the shared bad start.
    fitted_vcmax = result.per_curve_parameters["vcmax25"]
    assert abs(fitted_vcmax[0] - 55.0) < abs(30.0 - 55.0)
    assert abs(fitted_vcmax[1] - 65.0) < abs(30.0 - 65.0)


def test_shared_parameter_fit_is_identical_across_curves() -> None:
    truths = (
        {"vcmax25": 55.0, "vpmax25": 120.0, "rd25": 0.04},
        {"vcmax25": 55.0, "vpmax25": 120.0, "rd25": 0.04},
    )
    curves = _two_curve_group(truths)
    options = CalibrationOptions(iterations=40, learning_rate=0.08, lr_step=20, lr_gamma=0.5, restarts=1)

    result = fit_curve_group(
        curves,
        shared=("vcmax25",),
        fixed={"vpmax25": 120.0, "rd25": 0.04},
        initial={"vcmax25": 30.0},
        mode="smooth",
        solver_options=SOLVER,
        dtype=DTYPE,
        options=options,
    )
    assert set(result.shared_parameters) == {"vcmax25"}
    assert result.per_curve_parameters == {}
    # Both curves were tied to the same underlying tensor, so their reported
    # physical values must match exactly.
    for model in result.curve_models[1:]:
        assert model.parameter_report()["vcmax25"] == pytest.approx(
            result.curve_models[0].parameter_report()["vcmax25"]
        )


def test_ratio_regularizer_pulls_cross_curve_ratios_together() -> None:
    truths = (
        {"vcmax25": 50.0, "vpmax25": 100.0, "rd25": 0.04},
        {"vcmax25": 70.0, "vpmax25": 140.0, "rd25": 0.04},
    )
    curves = _two_curve_group(truths)
    start = {"vcmax25": 30.0, "vpmax25": 45.0}
    options = CalibrationOptions(iterations=60, learning_rate=0.1, lr_step=20, lr_gamma=0.5, restarts=1)

    def ratio_spread(result) -> float:
        ratios = [
            vp / vc
            for vp, vc in zip(
                result.per_curve_parameters["vpmax25"], result.per_curve_parameters["vcmax25"], strict=True
            )
        ]
        return max(ratios) - min(ratios)

    naive = fit_curve_group(
        curves,
        per_curve=("vcmax25", "vpmax25"),
        fixed={"rd25": 0.04},
        initial=start,
        mode="smooth",
        solver_options=SOLVER,
        dtype=DTYPE,
        options=options,
    )
    regularized = fit_curve_group(
        curves,
        per_curve=("vcmax25", "vpmax25"),
        fixed={"rd25": 0.04},
        initial=start,
        ratio_regularizers=(RatioRegularizer("vpmax25", "vcmax25", weight=8.0),),
        mode="smooth",
        solver_options=SOLVER,
        dtype=DTYPE,
        options=options,
    )

    assert ratio_spread(regularized) < ratio_spread(naive)


def test_identifiability_report_flags_weakly_separated_pair() -> None:
    truth = {"vcmax25": 55.0, "vpmax25": 120.0, "gm25": 3.0, "rd25": 0.04}
    conditions = _curve_conditions(1, 20.0, 20.0, 25.0)
    truth_model = LeafBiochemistry(
        truth, trainable=(), mode="smooth", finite_gm=True, solver_options=SOLVER, dtype=DTYPE
    )
    with torch.no_grad():
        output = truth_model(conditions)
    assert bool(output.diagnostics.converged.all())
    observed = output.mass["aNet"].detach()

    model = LeafBiochemistry(
        truth,
        trainable=("vpmax25", "gm25", "vcmax25"),
        mode="smooth",
        finite_gm=True,
        solver_options=SOLVER,
        dtype=DTYPE,
    )
    report = identifiability_report(
        model, conditions, observed, ("vpmax25", "gm25", "vcmax25"), threshold=0.7
    )
    assert report.correlation.shape == (3, 3)
    assert torch.allclose(report.correlation.diagonal(), torch.ones(3, dtype=DTYPE))
    flagged_names = {frozenset(pair[:2]) for pair in report.flagged_pairs}
    assert frozenset({"vpmax25", "gm25"}) in flagged_names


def test_identifiability_report_rejects_unknown_parameter(conditions) -> None:
    model = LeafBiochemistry({"vcmax25": 55.0}, trainable=("vcmax25",), mode="smooth", solver_options=SOLVER, dtype=DTYPE)
    with torch.no_grad():
        observed = model(conditions).mass["aNet"].detach()
    with pytest.raises(KeyError):
        identifiability_report(model, conditions, observed, ("vcmax25", "jmax25"))
