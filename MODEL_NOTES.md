# Model and numerical notes

## Coupled state equations

`Leaf` solves the physical state

```text
z = [A_net, C_bs, C_i, g_s, C_b, T_leaf]          (finite_gm=False, the default)
z = [A_net, C_bs, C_i, C_m, g_s, C_b, T_leaf]      (finite_gm=True)
```

from six or seven residuals:

```text
R_A   = A - (1 - GammaStar / Cbs) * Cbs*a1/(Cbs+b1) + rd
R_Cbs = Cbs - Cm - (Vp - A - rm) / gbs   (Cm = Ci when finite_gm=False)
R_Ci  = Ci - Cb + 1.6*A/gs
R_Cm  = Cm - Ci + A/gm                   (only when finite_gm=True)
gs*   = max(go, go + g1*A*(eb/ei)/(Cb-Gamma))
R_gs  = gs - gs*
R_Cb  = Cb - ca + 1.37*A/(gb*Mb)
R_T   = PAR + NIR + long - 0.506*A - H - LE - emission
```

`finite_gm=False` (the default, and the model's original behavior before
mesophyll conductance was added) assumes `C_m = C_i`: PEP carboxylation and
the bundle-sheath leak term draw directly from `C_i`, with no coupled `C_m`
state — the equations above collapse to exactly six residuals, bit-identical
to every equation this model used prior to the `gm25`/`cm` addition.

`finite_gm=True` adds `C_m`, the CO2 partial pressure at the PEP-carboxylation
site in the mesophyll cytosol, separated from `C_i` (intercellular airspace)
by a finite mesophyll conductance `gm`. `R_Cm` follows the same
`Cc = Ci - A/gm` form used for mesophyll conductance in C3 (FvCB) models,
since at steady state the net-assimilated CO2 flux is exactly what crosses
the `Ci -> Cm` boundary (bundle-sheath leakage `L = gbs*(Cbs - Cm)`
recirculates internally and does not re-cross it). Unlike an earlier version
of this note, `gm` **does** have a temperature response: a peaked Arrhenius
function identical in form to `Vcmax`/`Jmax`/`Vpmax`
(`_temperature_response` in `model.py`), with literal constants adapted from
Bernacchi et al. (2002, *Plant, Cell & Environment* 25:851-858) — the
standard C3 `gm(T)` reference used by most FvCB fitting tools. C4-specific
`gm(T)` data is sparse, so this is a documented placeholder assumption, not a
measured C4 value; treat it the same way as the deliberately-unused `vpr25`
when it is not the thing you are studying. `gm25` and `Cm` are computed
harmlessly but never used in the residual system when `finite_gm=False`.

In fixed-temperature mode, `R_T = T_leaf - controlTemp`. The biochemical
temperature responses, electron transport, limitation rates, boundary layer,
stomatal coupling, compensation point, and energy fluxes are recomputed from the
current state during every residual evaluation.

Smooth mode replaces the hard maximum in `gs*` with a narrow smooth maximum,
so the `gs >= go` floor is enforced consistently in both the residual and its
Jacobian.

With `finite_gm=False`, the solver uses state scales `[10, 1000, 100, 0.1, 100, 10]`
and residual scales `[10, 1000, 100, 0.1, 100, 100]`; with `finite_gm=True`,
`[10, 1000, 100, 100, 0.1, 100, 10]` and `[10, 1000, 100, 100, 0.1, 100, 100]`
(the `C_m` slot inserted after `C_i`; temperature residual scale `10` in
fixed mode either way).
It solves an augmented damped least-squares system (avoiding `J.T @ J`), applies
state bounds, and uses squared-L2 backtracking consistent with that step.
Convergence is based on the infinity norm of the complete scaled residual.
Rows that fail from the default `aNet = 0.1*ca` assimilation start are retried
from three additional physiology-informed assimilation guesses; the
lowest-residual result is retained.

## Ci-driven biochemistry (`LeafBiochemistry`)

`Leaf` is driven by ambient CO2 (`ca`) and solves stomatal conductance itself.
Real A-Ci curve parameter estimation instead starts from *measured*
intercellular CO2 (`Ci`, already back-calculated by the gas-exchange
instrument from measured `An`/`gs`) and fits biochemical parameters by
comparing predicted vs. measured `An` at that given `Ci` — without
re-deriving `Ci` from a stomatal model. `LeafBiochemistry` implements this
reduced problem: `Ci`, `T_leaf`, `PAR`, and `O2` are direct per-row inputs
(not solved), and the coupled state shrinks to

```text
z = [A_net, C_bs]              (finite_gm=False)
z = [A_net, C_bs, C_m]         (finite_gm=True)
```

with `R_A` and `R_Cbs` (and `R_Cm` when finite) identical in form to `Leaf`'s
above, and no `R_Ci`/`gs*`/`R_gs`/`R_Cb`/`R_T` at all — no stomatal
conductance, boundary layer, or energy balance is modeled. Both classes share
the CO2/light-limited assimilation equations (electron transport, `Vc`,
`Vp`, `Gamma_C`, `Gamma`, temperature response) through a common
`_biochemistry_core` method in `model.py`, so the equations are guaranteed
identical between the two model classes rather than hand-duplicated;
`tests/test_model.py::test_leaf_biochemistry_matches_leaf_given_leafs_own_ci`
cross-checks this directly by feeding `Leaf`'s own solved `Ci` back into
`LeafBiochemistry` and confirming matching `A_net`/`C_bs`.

A single A-Ci curve constrains electron transport only as the realized rate
`J` at its (usually saturating) measurement PAR, not the light-response
curvature needed to separate out `Jmax25` — that needs a PAR sweep (A-Q
curve), which this model does not yet combine with an A-Ci sweep in one fit.
`pyleaf_torch.calibration`'s default fit target is therefore `vcmax25`,
`vpmax25`, and `rd25`, not `jmax25`.

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

## Energy-balance convergence

The model solves the coupled bounded equilibrium with `2 <= T_leaf <= 60`. If no
zero exists inside those bounds, it reports a nonzero residual and a
bound-active diagnostic (`output.diagnostics.at_state_bound`) rather than
calling the row converged.

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

`gm25` (when `finite_gm=True`) joins `vpmax25` and `gbs` as a newly correlated
group: all three shape how CO2 supply into the C4 cycle limits `Vp`, so a
single A-Ci curve at one temperature generally cannot separate them; it is
therefore off by default (`finite_gm=False`, the original infinite-`gm`
model) and should only be turned on with the expectation of needing more
data/iterations or independent priors to resolve it — see
`examples/parameter_estimation.py`'s `finite_gm_demo` for a worked example
where `gm25` is recovered poorly (~44% error) despite `vcmax25`/`vpmax25`
recovering well, exactly because of this correlation.
`pyleaf_torch.calibration` (see the README's "Parameter estimation" section)
implements multi-curve/shared-parameter fitting and cross-curve ratio
regularization, adapted from PhoTorch, specifically to make grouped fits
tractable instead of fitting one curve at a time — and fits against measured
`Ci` (via `LeafBiochemistry`), not `Ca`, matching standard practice.

Do not fit `vpr25` without first changing and scientifically validating the model:
the current equations never use it. Do not silently interpret the literal `100`
PEP cap as `vpr25`; that would be a new biological assumption.

The Torch DataFrame puts explicitly named scaled equation components
(`scaled_residual_aNet`, `scaled_residual_cbs`, ...) plus `converged`,
`iterations`, `residual_norm`, `jacobian_condition`, `line_search_failures`,
and `at_state_bound` in its diagnostics table for exactly this purpose.
