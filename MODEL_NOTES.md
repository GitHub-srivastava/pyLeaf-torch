# Model and numerical notes

## Coupled state equations

The new implementation solves the physical state

```text
z = [A_net, C_bs, C_i, g_s, C_b, T_leaf]
```

from six residuals:

```text
R_A   = A - (1 - GammaStar / Cbs) * Cbs*a1/(Cbs+b1) + rd
R_Cbs = Cbs - Ci - (Vp - A - rm) / gbs
R_Ci  = Ci - Cb + 1.6*A/gs
gs*   = max(go, go + g1*A*(eb/ei)/(Cb-Gamma))
R_gs  = gs - gs*
R_Cb  = Cb - ca + 1.37*A/(gb*Mb)
R_T   = PAR + NIR + long - 0.506*A - H - LE - emission
```

In fixed-temperature mode, `R_T = T_leaf - controlTemp`. The biochemical
temperature responses, electron transport, limitation rates, boundary layer,
stomatal coupling, compensation point, and energy fluxes are recomputed from the
current state during every residual evaluation.

Smooth mode replaces the hard maximum in `gs*` with a narrow smooth maximum.
This floor intentionally corrects the legacy code's inconsistent intended floor:
the old Jacobian branch tried to clamp `gs`, but its residual did not.

The solver uses state scales `[10, 1000, 100, 0.1, 100, 10]` and residual scales
`[10, 1000, 100, 0.1, 100, 100]` (temperature residual scale `10` in fixed mode).
It solves an augmented damped least-squares system (avoiding `J.T @ J`), applies
state bounds, and uses squared-L2 backtracking consistent with that step.
Convergence is based on the infinity norm of the complete scaled residual.
Rows that fail from the legacy-like assimilation start are retried from three
additional assimilation guesses; the lowest-residual result is retained.

## Implicit differentiation

After the detached numerical solve returns `z*`, the model forms a zero-valued
gradient surrogate from one differentiable Newton correction while treating the
equilibrium Jacobian as constant. Its forward value remains exactly `z*`, and its
backward derivative is the implicit-function result

```text
dz/dp = -(dR/dz)^-1 (dR/dp).
```

This avoids gradients that depend on early-stopping branches and avoids retaining
the graph for every solver iteration. It assumes a locally unique, interior,
well-converged root. Training raises instead of returning a gradient for a
nonconverged, bound-active, or unsolvable-Jacobian row. This construction supports
correct first derivatives only; higher-order autograd is not supported because
the equilibrium Jacobian is detached.

## Why exact legacy equality is not expected

The legacy code uses a nested Picard/Newton/GEKKO process. Its hand-written CO2
Jacobian freezes quantities that depend on the current state; its stomatal-floor
branch changes Jacobian entries without changing the corresponding residual; and
its final reported derived rates can precede the last state or temperature update.
The inner stopping metric mixes unscaled changes in `aNet`, `ci`, and `gs`, while
the outer loop checks only temperature change.

The new solver preserves the equilibrium equations and constants but corrects
those numerical inconsistencies. Therefore:

- use the hard mode and `examples/compare_models.py` to quantify output changes;
- compare limitation regimes as well as values;
- inspect the new full-equation residual;
- do not make a converged corrected solution imitate an unconverged/stale legacy
  output merely to reduce a parity metric.

The energy solver is also different. Legacy GEKKO minimizes a squared energy
residual with `2 <= T_leaf <= 60`. The new model solves the coupled bounded
equilibrium. If no zero exists inside those bounds, it reports a nonzero residual
and a bound-active diagnostic rather than calling the row converged.

## Differentiability limits

Hard mode is piecewise differentiable. Gradients are discontinuous at:

- forced/free-convection ties;
- CO2/light limitation ties;
- the literal `vpCO2 <= 100` cap;
- the `gs >= go` stomatal floor;
- zero virtual-temperature difference;
- active temperature or other state bounds.

Smooth mode rounds these transitions with unit-specific widths, which improves
gradient continuity at the cost of a small local model change. Invalid physics
and structural non-identifiability are not repaired by smoothing.

## Calibration recommendations

Start with a controlled-temperature experiment and a small identifiable subset,
for example `vcmax25`, `vpmax25`, `jmax25`, `rd25`, `go`, and `g1`, while observing
both assimilation and stomatal conductance. Add energy balance and thermal/water
flux observations later. Use diverse forcing conditions, train/held-out splits,
multiple starts, physical residual checks, and sensitivity singular values.

Do not fit `vpr25` without first changing and scientifically validating the model:
the current equations never use it. Do not silently interpret the literal `100`
PEP cap as `vpr25`; that would be a new biological assumption.

Legacy `flag`, `Terror`, `Aerror`, `Cierror`, and `Gserror` values are iteration
diagnostics, not physical outputs. The Torch DataFrame puts explicitly named
scaled equation components in its diagnostics table instead; the comparison
harness excludes legacy diagnostic columns from parity metrics.
