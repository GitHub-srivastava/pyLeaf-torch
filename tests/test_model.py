from __future__ import annotations

import pytest
import torch

from pyleaf_torch import DifferentiableLeaf, SolverOptions


def test_fixed_temperature_equilibrium_and_identities(weather) -> None:
    model = DifferentiableLeaf(
        trainable=(),
        mode="hard",
        energy_balance=False,
        solver_options=SolverOptions(max_iterations=80, residual_tolerance=1.0e-7),
    )
    with torch.no_grad():
        output = model(weather)
    assert bool(output.diagnostics.converged.all())
    assert float(output.diagnostics.residual_norm.max()) < 1.0e-7
    assert torch.all(torch.isfinite(output.mass["aNet"]))
    assert torch.allclose(output.state["tLeaf"], weather["controlTemp"], atol=1.0e-8)
    assert torch.allclose(output.mass["aGross"], output.mass["aNet"] + output.mass["rd"])
    assert torch.allclose(output.mass["rd"], output.mass["rbs"] + output.mass["rm"])
    assert torch.allclose(
        output.energy["radiation"], weather["PAR"] + weather["NIR"] + weather["long"]
    )


def test_energy_balance_converges(weather) -> None:
    model = DifferentiableLeaf(
        trainable=(),
        mode="hard",
        energy_balance=True,
        solver_options=SolverOptions(max_iterations=100, residual_tolerance=1.0e-7),
    )
    with torch.no_grad():
        output = model(weather)
    assert bool(output.diagnostics.converged.all())
    assert float(output.diagnostics.residual_norm.max()) < 1.0e-7
    assert bool(((output.state["tLeaf"] >= 2.0) & (output.state["tLeaf"] <= 60.0)).all())
    assert float(output.energy["residual"].abs().max()) < 1.0e-5


def test_smooth_mode_has_finite_nonzero_parameter_gradients(weather) -> None:
    names = ("vcmax25", "jmax25", "g1")
    model = DifferentiableLeaf(
        trainable=names,
        mode="smooth",
        energy_balance=False,
        solver_options=SolverOptions(max_iterations=80, residual_tolerance=1.0e-8),
    )
    output = model(weather)
    loss = output.mass["aNet"].mean() + 5.0 * output.state["gs"].mean()
    loss.backward()
    for name in names:
        gradient = model.raw_parameters[name].grad
        assert gradient is not None
        assert bool(torch.isfinite(gradient))
        assert float(gradient.abs()) > 1.0e-8


def test_vpr25_is_structurally_unused(weather) -> None:
    low = DifferentiableLeaf(
        {"vpr25": 20.0}, trainable=(), mode="hard", energy_balance=False
    )
    high = DifferentiableLeaf(
        {"vpr25": 200.0}, trainable=(), mode="hard", energy_balance=False
    )
    with torch.no_grad():
        low_output = low(weather)
        high_output = high(weather)
    # atol is near float64 machine epsilon: multi-threaded reduction ordering in
    # the iterative solver can perturb the last bit even when, as here, vpr25
    # never enters the computation graph.
    assert torch.allclose(low_output.mass["aNet"], high_output.mass["aNet"], rtol=0.0, atol=1.0e-12)
    assert torch.allclose(low_output.state["gs"], high_output.state["gs"], rtol=0.0, atol=1.0e-12)


def test_implicit_gradient_matches_central_difference(weather) -> None:
    one_row = {name: value[1:2] for name, value in weather.items()}
    model = DifferentiableLeaf(
        trainable=("vcmax25",),
        mode="smooth",
        energy_balance=False,
        solver_options=SolverOptions(max_iterations=100, residual_tolerance=1.0e-10),
    )
    output = model(one_row)
    value = output.mass["aNet"].sum()
    value.backward()
    autograd_value = float(model.raw_parameters["vcmax25"].grad)

    raw = model.raw_parameters["vcmax25"]
    original = raw.detach().clone()
    epsilon = 1.0e-5
    with torch.no_grad():
        raw.copy_(original + epsilon)
        plus = float(model(one_row).mass["aNet"])
        raw.copy_(original - epsilon)
        minus = float(model(one_row).mass["aNet"])
        raw.copy_(original)
    finite_difference = (plus - minus) / (2.0 * epsilon)
    assert autograd_value == pytest.approx(finite_difference, rel=3.0e-3, abs=3.0e-4)


def test_grad_and_no_grad_forward_values_are_identical(weather) -> None:
    model = DifferentiableLeaf(
        trainable=("vcmax25",), mode="smooth", energy_balance=False
    )
    graph_output = model(weather)
    with torch.no_grad():
        inference_output = model(weather)
    for name in graph_output.mass:
        assert torch.allclose(
            graph_output.mass[name], inference_output.mass[name], rtol=1.0e-12, atol=1.0e-12
        )
    for name in graph_output.state:
        assert torch.allclose(
            graph_output.state[name], inference_output.state[name], rtol=1.0e-12, atol=1.0e-12
        )


def test_inference_mode_is_supported(weather) -> None:
    model = DifferentiableLeaf(trainable=("vcmax25",), energy_balance=False)
    with torch.no_grad():
        expected = model(weather)
    with torch.inference_mode():
        observed = model(weather)
    assert torch.allclose(observed.mass["aNet"], expected.mass["aNet"], rtol=1.0e-12)
    assert torch.allclose(observed.state["tLeaf"], expected.state["tLeaf"], rtol=1.0e-12)


def test_invalid_root_is_reported_instead_of_differentiated(weather) -> None:
    model = DifferentiableLeaf(
        trainable=("vcmax25",),
        energy_balance=False,
        solver_options=SolverOptions(max_iterations=1, residual_tolerance=1.0e-14),
    )
    with pytest.raises(RuntimeError, match="converged interior root"):
        model(weather)
    with torch.no_grad():
        output = model(weather)
    assert not bool(output.diagnostics.converged.all())


def test_hard_mode_enforces_residual_stomatal_conductance(weather) -> None:
    dark = {name: value[:1].clone() for name, value in weather.items()}
    dark["PAR"].zero_()
    dark["NIR"].zero_()
    model = DifferentiableLeaf(trainable=(), mode="hard", energy_balance=False)
    with torch.no_grad():
        output = model(dark)
    assert float(output.state["gs"]) >= model.parameter_report()["go"]


@pytest.mark.parametrize(
    ("name", "value"),
    (("vcmax25", -1.0), ("theta", 2.0), ("rd25", 0.8)),
)
def test_invalid_parameter_values_are_rejected(name, value) -> None:
    with pytest.raises(ValueError, match=name):
        DifferentiableLeaf({name: value}, trainable=())


def test_bounded_endpoint_is_fixed_only() -> None:
    fixed = DifferentiableLeaf({"alpha": 0.0}, trainable=())
    assert fixed.parameter_report()["alpha"] == 0.0
    with pytest.raises(ValueError, match="Trainable alpha"):
        DifferentiableLeaf({"alpha": 0.0}, trainable=("alpha",))
