"""Multi-curve photosynthetic parameter estimation.

Adapted from PhoTorch (Lei, Rizzo & Bailey 2025, *Photosynthesis Research*;
github.com/GEMINI-Breeding/photorch), which fits the C3 FvCB model in PyTorch
and fights equifinality (many parameter sets explaining one curve equally
well) with three ideas this module ports to the C4
:class:`~pyleaf_torch.LeafBiochemistry` model:

1. ``onefit``-style multi-curve fitting: curves in a group either share one
   set of capacity parameters, or keep their own while sharing shape
   parameters across the group. Pooling curves is what constrains a shared
   parameter that a single curve cannot identify. See :func:`build_curve_models`
   (parameter tying) and :func:`fit_curve_group`.
2. A cross-curve ratio/correlation penalty (PhoTorch pulls Jmax25-Vcmax25
   toward a target correlation) that discourages independent per-curve
   drift. See :class:`RatioRegularizer`.
3. Adam with a step-decayed learning rate and best-loss checkpointing across
   iterations, rather than trusting the final iterate.

This module deliberately does *not* port PhoTorch's ReLU-penalty box bounds:
``pyleaf_torch.parameters`` already gives every parameter a hard,
transform-based feasible region (log or logit coordinates), which is strictly
better than a tunable penalty weight. What PhoTorch adds that this repo
lacked is the multi-curve/shared-parameter machinery and the anti-drift
regularizer, both implemented here on top of the existing
``LeafBiochemistry`` API rather than by touching the validated core equations.

Fitting is ``ci``-driven, not ``ca``-driven: matching how A-Ci curve parameter
estimation is actually done, :class:`CurveGroup` carries measured
intercellular CO2 (``ci``), leaf temperature, PAR, and O2 per row, and
predicted assimilation is compared against measured assimilation at that
given ``ci`` — no stomatal-conductance model is fit or required. See
:class:`~pyleaf_torch.LeafBiochemistry` for why this differs from driving the
full :class:`~pyleaf_torch.Leaf` model with ambient CO2 (``ca``).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .model import LeafBiochemistry, Mode
from .parameters import DEFAULT_PARAMETERS, PARAMETER_SPECS
from .solver import SolverOptions


@dataclass(frozen=True)
class CurveGroup:
    """Rows from one or more measured A-Ci curves sharing a fitting problem.

    ``conditions`` holds the driving variables :class:`~pyleaf_torch.LeafBiochemistry`
    needs directly from measurement: ``ci`` (intercellular CO2), ``tLeaf``,
    ``PAR``, and ``O2``. ``curve_id`` labels which physical curve (e.g. a
    distinct A-Ci sweep, or a distinct leaf) each row belongs to; labels need
    not be contiguous or start at zero. Every condition field and observation
    tensor must be one-dimensional with one entry per row. ``observed_gs`` is
    accepted for API symmetry with earlier ``ca``-driven fitting but is not
    meaningful here (:class:`LeafBiochemistry` does not model stomatal
    conductance) — leave it unset.
    """

    conditions: Mapping[str, Tensor]
    observed_aNet: Tensor
    curve_id: Tensor
    observed_gs: Tensor | None = None

    def __post_init__(self) -> None:
        if not self.conditions:
            raise ValueError("conditions must be non-empty")
        lengths = {name: int(value.shape[0]) for name, value in self.conditions.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Condition fields must share one row count, got {lengths}")
        (rows,) = set(lengths.values())
        if tuple(self.curve_id.shape) != (rows,):
            raise ValueError("curve_id must be one-dimensional with one entry per condition row")
        if self.curve_id.dtype not in (torch.int32, torch.int64):
            raise ValueError("curve_id must be an integer tensor")
        if tuple(self.observed_aNet.shape) != (rows,):
            raise ValueError(
                "observed_aNet must be one-dimensional with one entry per condition row"
            )
        if self.observed_gs is not None and tuple(self.observed_gs.shape) != (rows,):
            raise ValueError(
                "observed_gs must be one-dimensional with one entry per condition row"
            )

    @property
    def unique_curve_ids(self) -> Tensor:
        return torch.unique(self.curve_id, sorted=True)

    @property
    def num_curves(self) -> int:
        return int(self.unique_curve_ids.numel())

    def _mask(self, curve_index: int) -> Tensor:
        return self.curve_id == self.unique_curve_ids[curve_index]

    def curve_conditions(self, curve_index: int) -> dict[str, Tensor]:
        mask = self._mask(curve_index)
        return {name: value[mask] for name, value in self.conditions.items()}

    def curve_observed_aNet(self, curve_index: int) -> Tensor:
        return self.observed_aNet[self._mask(curve_index)]

    def curve_observed_gs(self, curve_index: int) -> Tensor | None:
        if self.observed_gs is None:
            return None
        return self.observed_gs[self._mask(curve_index)]


def build_curve_models(
    curves: CurveGroup,
    *,
    shared: Sequence[str] = (),
    per_curve: Sequence[str] = (),
    fixed: Mapping[str, float] | None = None,
    mode: Mode = "smooth",
    finite_gm: bool = False,
    solver_options: SolverOptions | None = None,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str | None = None,
) -> list[LeafBiochemistry]:
    """Build one :class:`LeafBiochemistry` per curve, tying ``shared`` names.

    Tying is done by assigning the *same* ``nn.Parameter`` object into every
    curve model's ``raw_parameters``, so gradients from every curve
    accumulate onto one tensor and a single optimizer step updates all of
    them consistently (weight tying, the mechanism PhoTorch calls
    ``onefit``). ``per_curve`` names stay independent per curve model.
    """

    overlap = sorted(set(shared) & set(per_curve))
    if overlap:
        raise ValueError(f"Parameter(s) cannot be both shared and per-curve: {', '.join(overlap)}")
    trainable = tuple(dict.fromkeys((*shared, *per_curve)))
    base_parameters = dict(fixed or {})
    models = [
        LeafBiochemistry(
            base_parameters,
            trainable=trainable,
            mode=mode,
            finite_gm=finite_gm,
            solver_options=solver_options,
            dtype=dtype,
            device=device,
        )
        for _ in range(curves.num_curves)
    ]
    for name in shared:
        shared_parameter = models[0].raw_parameters[name]
        for model in models[1:]:
            model.raw_parameters[name] = shared_parameter
    return models


def collect_parameters(models: Sequence[LeafBiochemistry]) -> list[nn.Parameter]:
    """Deduplicate trainable parameters across curve models by identity.

    Required before handing parameters to an optimizer: tied (shared) names
    are the same object across models and must only be optimized once.
    """

    seen: set[int] = set()
    parameters: list[nn.Parameter] = []
    for model in models:
        for parameter in model.raw_parameters.values():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append(parameter)
    return parameters


@dataclass(frozen=True)
class RatioRegularizer:
    """Penalize per-curve capacity ratios drifting apart across a group.

    Adapted from PhoTorch's Jmax25-Vcmax25 cross-curve correlation penalty.
    With ``target_log_ratio=None`` (the default) this shrinks every curve's
    ``log(numerator) - log(denominator)`` toward the group's own mean ratio,
    i.e. it minimizes the cross-curve variance of the ratio without asserting
    a literature value. Pass ``target_log_ratio`` to instead shrink toward an
    externally known ratio (e.g. from an independent measurement or the
    literature).
    """

    numerator: str
    denominator: str
    weight: float = 1.0
    target_log_ratio: float | None = None

    def penalty(self, models: Sequence[LeafBiochemistry]) -> Tensor:
        if len(models) < 2:
            return models[0].state_scale.new_zeros(())
        numerator_values = torch.stack(
            [model.physical_parameters()[self.numerator] for model in models]
        )
        denominator_values = torch.stack(
            [model.physical_parameters()[self.denominator] for model in models]
        )
        log_ratio = torch.log(numerator_values) - torch.log(denominator_values)
        if self.target_log_ratio is None:
            center = log_ratio.mean()
        else:
            center = torch.as_tensor(self.target_log_ratio, dtype=log_ratio.dtype)
        return self.weight * (log_ratio - center).square().mean()


@dataclass(frozen=True)
class CalibrationOptions:
    """Optimizer hyperparameters for :func:`fit_curve_group`.

    Defaults (Adam, step-decayed learning rate) mirror PhoTorch's own
    optimizer choice for the FvCB model.
    """

    iterations: int = 800
    learning_rate: float = 0.05
    lr_step: int = 200
    lr_gamma: float = 0.7
    restarts: int = 1
    seed: int | None = None
    grad_clip_norm: float | None = 10.0

    def __post_init__(self) -> None:
        if not isinstance(self.iterations, int) or self.iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not isinstance(self.lr_step, int) or self.lr_step < 1:
            raise ValueError("lr_step must be a positive integer")
        if not math.isfinite(self.lr_gamma) or self.lr_gamma <= 0:
            raise ValueError("lr_gamma must be finite and positive")
        if not isinstance(self.restarts, int) or self.restarts < 1:
            raise ValueError("restarts must be a positive integer")
        if self.grad_clip_norm is not None and (
            not math.isfinite(self.grad_clip_norm) or self.grad_clip_norm <= 0
        ):
            raise ValueError("grad_clip_norm must be None or finite and positive")


@dataclass(frozen=True)
class CalibrationResult:
    """Fit outcome for the best-scoring restart, plus the full restart record.

    ``restart_*`` fields carry every restart's outcome (not just the best
    one) so callers can inspect the spread of recovered parameters across
    independent starts: a wide spread despite similar losses is itself an
    equifinality signal, complementing :func:`identifiability_report`.
    """

    curve_models: tuple[LeafBiochemistry, ...]
    shared_parameters: dict[str, float]
    per_curve_parameters: dict[str, list[float]]
    loss_trace: list[float]
    best_loss: float
    restart_losses: tuple[float, ...]
    restart_shared_parameters: tuple[dict[str, float], ...]
    restart_per_curve_parameters: tuple[dict[str, list[float]], ...]


def _sample_start(
    name: str,
    bounds: tuple[float, float],
    generator: torch.Generator | None,
    dtype: torch.dtype,
) -> float:
    lower, upper = bounds
    if not (math.isfinite(lower) and math.isfinite(upper)) or lower >= upper:
        raise ValueError(
            f"start_ranges[{name!r}] must be a finite (lower, upper) pair with lower < upper"
        )
    spec = PARAMETER_SPECS[name]
    draw = (
        torch.rand((), generator=generator, dtype=dtype)
        if generator is not None
        else torch.rand((), dtype=dtype)
    )
    if spec.transform == "positive":
        if lower <= 0.0:
            raise ValueError(
                f"start_ranges[{name!r}] lower bound must be positive for a "
                "positive-transform parameter"
            )
        return math.exp(math.log(lower) + float(draw) * (math.log(upper) - math.log(lower)))
    return lower + float(draw) * (upper - lower)


def _seed_initial_values(
    models: Sequence[LeafBiochemistry],
    *,
    shared: Sequence[str],
    per_curve: Sequence[str],
    initial: Mapping[str, float] | None,
    start_ranges: Mapping[str, tuple[float, float]] | None,
    generator: torch.Generator | None,
) -> None:
    initial = dict(initial or {})
    start_ranges = dict(start_ranges or {})
    dtype = models[0].state_scale.dtype
    for name in shared:
        if name in start_ranges:
            value = _sample_start(name, start_ranges[name], generator, dtype)
        elif name in initial:
            value = initial[name]
        else:
            continue
        models[0].set_physical_parameter_(name, value)
    for name in per_curve:
        for model in models:
            if name in start_ranges:
                value = _sample_start(name, start_ranges[name], generator, dtype)
            elif name in initial:
                value = initial[name]
            else:
                continue
            model.set_physical_parameter_(name, value)


def _group_loss(
    models: Sequence[LeafBiochemistry],
    curve_data: Sequence[tuple[Mapping[str, Tensor], Tensor, Tensor | None]],
    ratio_regularizers: Sequence[RatioRegularizer],
    aNet_scale: float,
    gs_scale: float,
) -> Tensor:
    total = models[0].state_scale.new_zeros(())
    for model, (conditions, target_a, target_gs) in zip(models, curve_data, strict=True):
        output = model(conditions)
        total = total + ((output.mass["aNet"] - target_a) / aNet_scale).square().mean()
        if target_gs is not None:
            total = total + ((output.state["gs"] - target_gs) / gs_scale).square().mean()
    for regularizer in ratio_regularizers:
        total = total + regularizer.penalty(models)
    return total


def fit_curve_group(
    curves: CurveGroup,
    *,
    shared: Sequence[str] = (),
    per_curve: Sequence[str] = (),
    fixed: Mapping[str, float] | None = None,
    initial: Mapping[str, float] | None = None,
    start_ranges: Mapping[str, tuple[float, float]] | None = None,
    ratio_regularizers: Sequence[RatioRegularizer] = (),
    mode: Mode = "smooth",
    finite_gm: bool = False,
    solver_options: SolverOptions | None = None,
    dtype: torch.dtype = torch.float64,
    aNet_scale: float = 10.0,
    gs_scale: float = 0.1,
    options: CalibrationOptions | None = None,
) -> CalibrationResult:
    """Fit a group of ``ci``-driven curves jointly with PhoTorch-style tying.

    ``shared`` parameter names are tied (one value fit across every curve in
    the group, via :func:`build_curve_models`); ``per_curve`` names get an
    independent value per curve. ``ratio_regularizers`` additionally shrink
    named per-curve pairs toward a common ratio to resist equifinal drift.
    With ``options.restarts > 1`` and ``start_ranges`` supplied for the
    parameters of interest, every restart begins from an independent random
    draw within that range; inspect ``result.restart_per_curve_parameters``
    to see whether independent starts converge to consistent estimates.

    A single A-Ci curve constrains electron transport only as the realized
    rate ``J`` at its (usually saturating) measurement PAR, not the
    light-response curvature needed to separate out ``jmax25`` — that needs a
    PAR sweep. Fitting ``jmax25`` from `ci`-only data would reintroduce an
    equifinality problem rather than avoid one; prefer ``vcmax25``,
    ``vpmax25``, and ``rd25`` as the default per-curve target set.

    Like the rest of this package's gradient path, a candidate parameter set
    that leaves any row unconverged or at a state bound makes the forward
    pass raise ``RuntimeError`` instead of returning an invalid gradient (see
    ``LeafBiochemistry.forward``); that exception propagates out of this loop
    unmodified, mirroring ``examples/calibration_benchmark.py``.
    """

    options = options or CalibrationOptions()
    shared = tuple(dict.fromkeys(shared))
    per_curve = tuple(dict.fromkeys(per_curve))
    overlap = sorted(set(shared) & set(per_curve))
    if overlap:
        raise ValueError(f"Parameter(s) cannot be both shared and per-curve: {', '.join(overlap)}")
    unknown = sorted((set(shared) | set(per_curve)) - set(DEFAULT_PARAMETERS))
    if unknown:
        raise KeyError(f"Unknown pyLeaf parameter(s): {', '.join(unknown)}")

    generator = torch.Generator().manual_seed(options.seed) if options.seed is not None else None

    restart_records: list[dict[str, object]] = []
    for _ in range(options.restarts):
        models = build_curve_models(
            curves,
            shared=shared,
            per_curve=per_curve,
            fixed=fixed,
            mode=mode,
            finite_gm=finite_gm,
            solver_options=solver_options,
            dtype=dtype,
        )
        _seed_initial_values(
            models,
            shared=shared,
            per_curve=per_curve,
            initial=initial,
            start_ranges=start_ranges,
            generator=generator,
        )

        curve_data = [
            (curves.curve_conditions(i), curves.curve_observed_aNet(i), curves.curve_observed_gs(i))
            for i in range(curves.num_curves)
        ]
        parameters = collect_parameters(models)
        optimizer = torch.optim.Adam(parameters, lr=options.learning_rate)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=options.lr_step, gamma=options.lr_gamma
        )

        loss_trace: list[float] = []
        best_loss = math.inf
        best_state = [parameter.detach().clone() for parameter in parameters]
        for _ in range(options.iterations):
            optimizer.zero_grad()
            loss = _group_loss(models, curve_data, ratio_regularizers, aNet_scale, gs_scale)
            loss.backward()
            if options.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, max_norm=options.grad_clip_norm)
            loss_value = float(loss.detach())
            loss_trace.append(loss_value)
            if loss_value < best_loss:
                best_loss = loss_value
                best_state = [parameter.detach().clone() for parameter in parameters]
            optimizer.step()
            scheduler.step()

        with torch.no_grad():
            for parameter, value in zip(parameters, best_state, strict=True):
                parameter.copy_(value)

        restart_records.append(
            {
                "models": models,
                "shared_parameters": {
                    name: models[0].parameter_report()[name] for name in shared
                },
                "per_curve_parameters": {
                    name: [model.parameter_report()[name] for model in models]
                    for name in per_curve
                },
                "loss_trace": loss_trace,
                "best_loss": best_loss,
            }
        )

    best_record = min(restart_records, key=lambda record: record["best_loss"])
    return CalibrationResult(
        curve_models=tuple(best_record["models"]),
        shared_parameters=best_record["shared_parameters"],
        per_curve_parameters=best_record["per_curve_parameters"],
        loss_trace=best_record["loss_trace"],
        best_loss=best_record["best_loss"],
        restart_losses=tuple(record["best_loss"] for record in restart_records),
        restart_shared_parameters=tuple(
            record["shared_parameters"] for record in restart_records
        ),
        restart_per_curve_parameters=tuple(
            record["per_curve_parameters"] for record in restart_records
        ),
    )


@dataclass(frozen=True)
class IdentifiabilityReport:
    """Gauss-Newton parameter correlation diagnostic for one curve fit."""

    parameter_names: tuple[str, ...]
    correlation: Tensor
    flagged_pairs: tuple[tuple[str, str, float], ...]


def identifiability_report(
    model: LeafBiochemistry,
    conditions: Mapping[str, Tensor],
    observed_aNet: Tensor,
    parameter_names: Sequence[str],
    *,
    observed_gs: Tensor | None = None,
    aNet_scale: float = 10.0,
    gs_scale: float = 0.1,
    threshold: float = 0.9,
    ridge: float = 1.0e-6,
) -> IdentifiabilityReport:
    """Diagnose equifinality among ``parameter_names`` for one curve fit.

    Builds the residual Jacobian ``d(aNet[, gs])/d(raw trainable params)`` at
    the model's current point via autograd, forms the Gauss-Newton
    approximate parameter covariance ``pinv(J^T J)``, and flags pairs whose
    correlation exceeds ``threshold``. This operationalizes the correlated
    groups this repo's README already documents for a single curve/leaf fit
    (e.g. ``{vcmax25, rd25}``, ``{vpmax25, gbs, x}``, and now
    ``{vpmax25, gm25}``): a flagged pair here means this curve's data cannot
    tell those two parameters apart, and a multi-curve
    :func:`fit_curve_group` (shared/tied parameters plus
    :class:`RatioRegularizer`) is the tool for resolving it rather than
    fitting more parameters into one curve.

    Diagnoses one model against one curve's data; for a multi-curve ensemble,
    call this once per curve model, or use ``result.restart_per_curve_parameters``
    from :func:`fit_curve_group` to see cross-restart spread instead.
    """

    names = tuple(dict.fromkeys(parameter_names))
    missing = sorted(set(names) - set(model.raw_parameters))
    if missing:
        raise KeyError(f"Not trainable on this model: {', '.join(missing)}")
    parameters = [model.raw_parameters[name] for name in names]

    output = model(conditions)
    residual_terms = [(output.mass["aNet"] - observed_aNet) / aNet_scale]
    if observed_gs is not None:
        residual_terms.append((output.state["gs"] - observed_gs) / gs_scale)
    residual = torch.cat(residual_terms)

    rows: list[Tensor] = []
    length = int(residual.shape[0])
    for index in range(length):
        grads = torch.autograd.grad(
            residual[index],
            parameters,
            retain_graph=index + 1 < length,
            allow_unused=True,
        )
        rows.append(
            torch.stack(
                [
                    grad if grad is not None else torch.zeros((), dtype=residual.dtype)
                    for grad in grads
                ]
            )
        )
    jacobian = torch.stack(rows, dim=0).detach()

    gram = jacobian.T @ jacobian
    scale = gram.diagonal().abs().amax().clamp_min(1.0)
    gram = gram + ridge * scale * torch.eye(len(names), dtype=gram.dtype)
    covariance = torch.linalg.pinv(gram)
    variance = covariance.diagonal().clamp_min(1.0e-300)
    normalizer = variance.sqrt()
    correlation = (covariance / (normalizer[:, None] * normalizer[None, :])).clamp(-1.0, 1.0)

    flagged: list[tuple[str, str, float]] = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            value = float(correlation[i, j])
            if abs(value) >= threshold:
                flagged.append((names[i], names[j], value))
    flagged.sort(key=lambda item: -abs(item[2]))

    return IdentifiabilityReport(
        parameter_names=names, correlation=correlation, flagged_pairs=tuple(flagged)
    )


__all__ = [
    "CalibrationOptions",
    "CalibrationResult",
    "CurveGroup",
    "IdentifiabilityReport",
    "RatioRegularizer",
    "build_curve_models",
    "collect_parameters",
    "fit_curve_group",
    "identifiability_report",
]
