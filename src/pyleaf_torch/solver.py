"""Scaled damped root solver used by the differentiable model."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


ResidualFunction = Callable[[Tensor], Tensor]


@dataclass(frozen=True)
class SolverOptions:
    """Configuration for the batched Levenberg--Marquardt equilibrium solve."""

    max_iterations: int = 80
    max_backtracking: int = 10
    residual_tolerance: float = 1.0e-7
    initial_damping: float = 1.0e-4
    minimum_damping: float = 1.0e-10
    maximum_damping: float = 1.0e10
    damping_decrease: float = 0.3
    damping_increase: float = 10.0
    implicit_regularization: float = 1.0e-10

    def __post_init__(self) -> None:
        if not isinstance(self.max_iterations, int) or self.max_iterations < 1:
            raise ValueError("max_iterations must be positive")
        if not isinstance(self.max_backtracking, int) or self.max_backtracking < 1:
            raise ValueError("max_backtracking must be positive")
        if not math.isfinite(self.residual_tolerance) or self.residual_tolerance <= 0:
            raise ValueError("residual_tolerance must be finite and positive")
        positive_fields = {
            "initial_damping": self.initial_damping,
            "minimum_damping": self.minimum_damping,
            "maximum_damping": self.maximum_damping,
            "damping_decrease": self.damping_decrease,
            "damping_increase": self.damping_increase,
        }
        for name, value in positive_fields.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not math.isfinite(self.implicit_regularization) or self.implicit_regularization < 0:
            raise ValueError("implicit_regularization must be finite and non-negative")
        if not self.minimum_damping <= self.initial_damping <= self.maximum_damping:
            raise ValueError(
                "damping values must satisfy minimum <= initial <= maximum"
            )
        if self.damping_decrease >= 1.0:
            raise ValueError("damping_decrease must be less than 1")
        if self.damping_increase <= 1.0:
            raise ValueError("damping_increase must be greater than 1")


@dataclass(frozen=True)
class RootResult:
    x: Tensor
    converged: Tensor
    iterations: Tensor
    residual_norm: Tensor
    jacobian: Tensor
    jacobian_condition: Tensor
    line_search_failures: Tensor
    at_state_bound: Tensor


def batched_jacobian(residual: Tensor, x: Tensor) -> Tensor:
    """Jacobian for independent rows in a batched residual calculation."""

    rows: list[Tensor] = []
    width = residual.shape[-1]
    for index in range(width):
        gradient = torch.autograd.grad(
            residual[:, index].sum(),
            x,
            retain_graph=index + 1 < width,
            create_graph=False,
        )[0]
        rows.append(gradient)
    return torch.stack(rows, dim=1)


def _norm(residual: Tensor) -> Tensor:
    return torch.linalg.vector_norm(residual, ord=float("inf"), dim=-1)


def _merit(residual: Tensor) -> Tensor:
    return residual.square().sum(dim=-1)


def solve_root(
    residual_function: ResidualFunction,
    initial_x: Tensor,
    lower_bound: Tensor,
    upper_bound: Tensor,
    options: SolverOptions,
) -> RootResult:
    """Solve independent scaled systems without retaining the iteration graph."""

    x = torch.minimum(torch.maximum(initial_x.detach(), lower_bound), upper_bound)
    batch, width = x.shape
    damping = torch.full(
        (batch,), options.initial_damping, dtype=x.dtype, device=x.device
    )
    iterations = torch.zeros(batch, dtype=torch.int64, device=x.device)
    failures = torch.zeros(batch, dtype=torch.int64, device=x.device)
    converged = torch.zeros(batch, dtype=torch.bool, device=x.device)
    identity = torch.eye(width, dtype=x.dtype, device=x.device).expand(batch, -1, -1)

    with torch.enable_grad():
        for _ in range(options.max_iterations):
            x_leaf = x.detach().requires_grad_(True)
            residual = residual_function(x_leaf)
            residual_norm = _norm(residual).detach()
            newly_active = ~converged
            iterations = iterations + newly_active.to(iterations.dtype)
            converged = converged | (residual_norm <= options.residual_tolerance)
            if bool(torch.all(converged)):
                x = x_leaf.detach()
                break

            jacobian = batched_jacobian(residual, x_leaf).detach()
            augmented_matrix = torch.cat(
                (jacobian, torch.sqrt(damping)[:, None, None] * identity), dim=-2
            )
            augmented_right = torch.cat(
                (residual.detach().unsqueeze(-1), torch.zeros_like(residual).unsqueeze(-1)),
                dim=-2,
            )
            step = torch.linalg.lstsq(augmented_matrix, augmented_right).solution.squeeze(-1)
            invalid_step = ~torch.isfinite(step).all(dim=-1)
            safe_step = torch.where(invalid_step[:, None], torch.zeros_like(step), step)

            best_x = x
            best_merit = _merit(residual.detach())
            accepted = torch.zeros(batch, dtype=torch.bool, device=x.device)
            factor = torch.ones(batch, dtype=x.dtype, device=x.device)
            for _ in range(options.max_backtracking):
                candidate = x - factor[:, None] * safe_step
                candidate = torch.minimum(torch.maximum(candidate, lower_bound), upper_bound)
                candidate_residual = residual_function(candidate)
                candidate_merit = _merit(candidate_residual).detach()
                improves = (
                    (candidate_merit < best_merit)
                    & torch.isfinite(candidate_merit)
                    & ~invalid_step
                    & ~converged
                )
                best_x = torch.where(improves[:, None], candidate.detach(), best_x)
                best_merit = torch.where(improves, candidate_merit, best_merit)
                accepted = accepted | improves
                factor = factor * 0.5

            stalled = ~accepted & ~converged
            failures = failures + stalled.to(failures.dtype)
            x = torch.where(accepted[:, None], best_x, x)
            damping = torch.where(
                accepted,
                torch.clamp(
                    damping * options.damping_decrease,
                    min=options.minimum_damping,
                    max=options.maximum_damping,
                ),
                torch.clamp(
                    damping * options.damping_increase,
                    min=options.minimum_damping,
                    max=options.maximum_damping,
                ),
            )

        final_x = x.detach().requires_grad_(True)
        final_residual = residual_function(final_x)
        final_jacobian = batched_jacobian(final_residual, final_x).detach()
        final_norm = _norm(residual_function(x.detach())).detach()

    converged = final_norm <= options.residual_tolerance
    finite_jacobian = torch.isfinite(final_jacobian).all(dim=(-2, -1))
    safe_jacobian = torch.where(
        finite_jacobian[:, None, None], final_jacobian, identity
    )
    condition = torch.linalg.cond(safe_jacobian).detach()
    condition = torch.where(
        finite_jacobian, condition, torch.full_like(condition, float("inf"))
    )
    bound_epsilon = 100.0 * torch.finfo(x.dtype).eps
    at_bound = (
        ((x - lower_bound).abs() <= bound_epsilon)
        | ((x - upper_bound).abs() <= bound_epsilon)
    ).any(dim=-1)
    return RootResult(
        x=x.detach(),
        converged=converged,
        iterations=iterations,
        residual_norm=final_norm,
        jacobian=final_jacobian,
        jacobian_condition=condition,
        line_search_failures=failures,
        at_state_bound=at_bound.detach(),
    )


def select_better_root(primary: RootResult, alternate: RootResult) -> RootResult:
    """Select the lower-residual result independently for every batch row."""

    use_alternate = alternate.residual_norm < primary.residual_norm

    def choose(left: Tensor, right: Tensor) -> Tensor:
        mask = use_alternate
        while mask.ndim < left.ndim:
            mask = mask.unsqueeze(-1)
        return torch.where(mask, right, left)

    return RootResult(
        x=choose(primary.x, alternate.x),
        converged=choose(primary.converged, alternate.converged),
        iterations=choose(primary.iterations, alternate.iterations),
        residual_norm=choose(primary.residual_norm, alternate.residual_norm),
        jacobian=choose(primary.jacobian, alternate.jacobian),
        jacobian_condition=choose(
            primary.jacobian_condition, alternate.jacobian_condition
        ),
        line_search_failures=choose(
            primary.line_search_failures, alternate.line_search_failures
        ),
        at_state_bound=choose(primary.at_state_bound, alternate.at_state_bound),
    )


def implicit_root(
    residual_function: ResidualFunction,
    result: RootResult,
    options: SolverOptions,
) -> Tensor:
    """Attach the implicit-function gradient to a detached root estimate.

    The forward value remains exactly the numerical root. A zero-valued gradient
    surrogate borrows the derivative of one Newton correction, giving
    ``dx/dp = -J_x^-1 J_p`` at a converged interior root without storing solver
    iterations. The model's public forward rejects nonconverged and bound-active
    rows before this function, and an unsolvable implicit Jacobian raises here.
    """

    x_leaf = result.x.detach().requires_grad_(True)
    residual = residual_function(x_leaf)
    width = x_leaf.shape[-1]
    identity = torch.eye(width, dtype=x_leaf.dtype, device=x_leaf.device)
    jacobian = result.jacobian + options.implicit_regularization * identity
    direct = torch.linalg.solve_ex(jacobian, residual.unsqueeze(-1))
    use_fallback = (direct.info != 0) | ~torch.isfinite(direct.result).all(dim=(-2, -1))
    if bool(use_fallback.any()):
        rows = torch.nonzero(use_fallback, as_tuple=False).flatten().tolist()
        raise RuntimeError(
            "Implicit Jacobian solve failed for row indices "
            f"{rows}; gradients are unavailable for this equilibrium."
        )
    correction = direct.result
    corrected = x_leaf - correction.squeeze(-1)

    # Keep the numerical root as the exact forward value while borrowing only
    # the implicit correction's derivative. Gradients at failed or bound-active
    # rows are intentionally masked because the interior IFT assumptions fail.
    valid_gradient = result.converged & ~result.at_state_bound
    gradient_surrogate = corrected - corrected.detach()
    gradient_surrogate = torch.where(
        valid_gradient[:, None], gradient_surrogate, torch.zeros_like(gradient_surrogate)
    )
    return result.x + gradient_surrogate
