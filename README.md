# pyLeaf Torch

`src/pyleaf_torch/` is a differentiable C4 leaf gas-exchange and
energy-balance model implemented with PyTorch. `Leaf` solves the coupled
equilibrium (`aNet`, `cbs`, `ci`, `gs`, `cb`, `tLeaf`, plus `cm` when
`finite_gm=True`) with a scaled damped root solver and differentiates the
converged solution with the implicit-function theorem.
`LeafBiochemistry` is a reduced, `ci`-driven variant (no stomata/boundary
layer/energy balance) for fitting measured A-Ci curve data — see "Parameter
estimation" below.

Measured residual, stress-grid, gradient, and calibration results are in
[VALIDATION.md](VALIDATION.md).

## What differentiability does—and does not—help

PyTorch makes gradients available for parameter calibration. That normally makes
Adam or L-BFGS much more evaluation-efficient than derivative-free fitting when
the selected parameters are identifiable and the initial point is in the basin
of the correct solution.

Differentiability alone does **not** make the forward leaf-state equations
converge. Forward robustness here comes from a separate numerical redesign:

1. `aNet`, `cbs`, `ci`, `gs`, `cb`, `tLeaf` (and `cm` when `finite_gm=True`)
   are solved as one equilibrium.
2. Every residual evaluation refreshes all dependent rates and conductances.
3. State and residual components are scaled before a damped least-squares step.
4. Backtracking uses the squared-L2 merit minimized by that step.
5. Difficult rows are retried from physiology-informed assimilation starts, and
   the lowest-residual equilibrium is retained.
6. The result reports convergence, residual norm, iteration count, Jacobian
   condition number, line-search failures, and active state bounds.
7. Gradients use an implicit correction at the converged root, rather than
   backpropagating through a variable number of solver iterations.

This distinction matters: gradient-based parameter fitting can still fail with
unidentifiable parameters, a singular equilibrium Jacobian, poor data coverage,
or hard regime switches.

Implicit differentiation is intentionally first-order. The detached equilibrium
Jacobian gives correct first derivatives, but Hessians and other higher-order
derivatives through the equilibrium are not supported.

## Install

Python 3.10 or newer is required. From this repository:

```bash
python -m venv .venv
# Windows
.venv\Scripts\python -m pip install -e ".[all]"
# macOS/Linux
.venv/bin/python -m pip install -e ".[all]"
```

For the Torch tensor model alone, `pip install -e .` is enough. The `all` extra
adds pandas/Excel adapters, Matplotlib plotting, SciPy benchmark, and tests.
The model defaults to `torch.float64` because the physical variables span several
orders of magnitude.

## Tensor API

```python
import pandas as pd
import torch

from pyleaf_torch import Leaf, weather_from_dataframe

frame = pd.read_excel("examples/data/Input.xlsx")
weather = weather_from_dataframe(frame)

model = Leaf(
    trainable=("vcmax25", "vpmax25", "jmax25", "rd25", "go", "g1"),
    mode="smooth",
    dtype=torch.float64,
)
output = model(weather)

loss = (output.mass["aNet"] - observations).square().mean()
loss.backward()
print(model.raw_parameters["vcmax25"].grad)
print(output.diagnostics.residual_norm.max())
```

Trainable parameters use internal log or logit coordinates so an optimizer
cannot make positive capacities negative or push fractions out of their physical
domain. Use `model.parameter_report()` to read values in physical units.

For reporting with pandas:

```python
from pyleaf_torch import simulate_dataframe

tensor_output, frames = simulate_dataframe(model, frame)
print(frames.mass[["aNet", "transpiration"]])
print(frames.diagnostics)
```

The DataFrame conversion deliberately detaches tensors. Fit against the tensor
members of `tensor_output`, not against `frames`.

## Hard and smooth physics modes

- `mode="hard"` uses exact `minimum`, `maximum`, and absolute-value switches.
  It is the right mode for final scientific evaluation and is differentiable
  almost everywhere, but its gradient changes discontinuously at regime ties.
- `mode="smooth"` uses narrow, unit-specific smooth transitions. It is easier to
  optimize near limitation boundaries but slightly changes outputs near a tie.

A practical workflow is to calibrate in smooth mode, reduce the learning rate,
then evaluate/refine in hard mode. Always report which mode was used.

Both modes enforce `gs >= go` exactly, by construction of the smooth/hard
maximum used for the stomatal floor.

## Response curve plotting

To generate A-Ci and A-Q response curves from a weather workbook, create an
isolated environment and install the plotting dependency set once:

```bash
git clone <repository-url>
cd pyLeaf-torch
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-plot.txt
.\.venv\Scripts\python.exe examples\plot_response_curves.py
```

On macOS or Linux:

```bash
.venv/bin/python -m pip install -r requirements-plot.txt
.venv/bin/python examples/plot_response_curves.py
```

The last row of `examples/data/Input.xlsx` supplies the fixed conditions. The
script performs controlled ambient-CO2 and irradiance sweeps, saves a PNG
figure, and writes both curves (including solver status) as CSV files under
`curve_output/`. Run it with `--help` to see sweep, input-row,
parameter-JSON, and output options. If it is launched with an interpreter that
does not have the required packages, it reports every missing dependency and
the exact installation command instead of failing on an import traceback.

## Calibration convergence benchmark

Run the seeded synthetic comparison with the same post-initial loss-evaluation
cap and physical parameter bounds for both optimizers:

```bash
python examples/calibration_benchmark.py --rows 12 --evaluations 40
```

It fits `vcmax25`, `jmax25`, and `g1` using Torch/Adam and SciPy/Powell from the
same start. This is a reproducible local demonstration, not proof that one
optimizer wins for every dataset. The script reports the best valid incumbent
within each cap because Powell can stop midway through a line search. A serious
study should repeat many starts and seeds, use held-out weather conditions, match
both evaluation and wall-time budgets, and analyze the sensitivity matrix rank.

## Parameter estimation

`src/pyleaf_torch/calibration.py` is a multi-curve photosynthetic parameter
estimation framework, adapted from
[PhoTorch](https://doi.org/10.1007/s11120-025-01136-7) (Lei, Rizzo & Bailey
2025, *Photosynthesis Research*), which fits the C3 FvCB model in PyTorch and
explicitly targets **equifinality**: many parameter combinations can fit one
curve equally well. It ports two ideas from PhoTorch on top of this repo's
existing bounded-transform parameters (which are already a hard,
penalty-free feasible region — stronger than PhoTorch's own ReLU-penalty
bounds, so that part wasn't ported):

- **Multi-curve/shared-parameter fitting** (PhoTorch's `onefit`): tie a
  parameter across every curve in a group, or let each curve keep its own
  value while other parameters stay shared. `build_curve_models` implements
  this by assigning the same `nn.Parameter` object into multiple
  `LeafBiochemistry` instances (weight tying), so a single optimizer step
  updates every tied curve consistently.
- **Cross-curve ratio regularization** (PhoTorch shrinks Jmax25-Vcmax25
  toward a target correlation): `RatioRegularizer` shrinks a per-curve
  capacity ratio (e.g. `vpmax25/vcmax25`) toward the group's own mean ratio,
  discouraging independent per-curve drift without requiring a hard-coded
  literature target.

Fitting is driven by **measured `ci`, not `ca`**, via `LeafBiochemistry` — the
reduced model described above. This matches how A-Ci curve parameter
estimation is actually done: a gas-exchange instrument already reports `ci`
(back-calculated from measured `An`/`gs`), so predicted assimilation is
compared against measured assimilation at that given `ci` directly, without
fitting a stomatal-conductance model at all. `CurveGroup` carries `ci`,
`tLeaf`, `PAR`, and `O2` per row (not full weather).

```python
import torch
from pyleaf_torch import CurveGroup, CalibrationOptions, RatioRegularizer, fit_curve_group

curves = CurveGroup(conditions=conditions, observed_aNet=observed_aNet, curve_id=curve_id)
result = fit_curve_group(
    curves,
    per_curve=("vcmax25", "vpmax25", "rd25"),  # not jmax25 -- see below
    ratio_regularizers=(RatioRegularizer("vpmax25", "vcmax25"),),
    options=CalibrationOptions(iterations=300, restarts=4, seed=0),
)
print(result.per_curve_parameters, result.best_loss)
```

The default fit target is `vcmax25`, `vpmax25`, and `rd25` — deliberately
**not** `jmax25`: a single A-Ci curve runs at one (usually saturating) PAR,
so it constrains electron transport only as the realized rate `J` at that
light level, not the light-response curvature needed to separate out
`Jmax25` (that needs a PAR sweep / A-Q curve). Fitting `jmax25` from
`ci`-only data would silently reintroduce an equifinality problem instead of
avoiding one. `gm25` (mesophyll conductance, `LeafBiochemistry(finite_gm=True)`)
is available but off by default for the same reason — see "Important model
findings" below and `examples/parameter_estimation.py`'s `finite_gm_demo`.

`fit_curve_group` uses Adam with a step-decayed learning rate and best-loss
checkpointing across iterations (mirroring PhoTorch's own optimizer choice),
and supports multiple random restarts (`options.restarts` with
`start_ranges`) so `result.restart_per_curve_parameters` can be inspected for
cross-restart spread — a wide spread despite similar losses is itself an
equifinality signal.

`identifiability_report` complements this: it builds the Gauss-Newton
parameter correlation matrix for one curve's fit from the residual Jacobian,
flagging pairs above a threshold. It measures the local curvature of a single
curve's own likelihood, so a flagged pair stays flagged regardless of the
regularizer — the regularizer's contribution is external information about
*where* the fit should land, not a change to what one curve's data alone can
separate. Run `python examples/parameter_estimation.py` for an end-to-end
demonstration: synthetic multi-leaf A-Ci curves (generated realistically by
simulating the full `ca`-driven `Leaf` model per leaf and reading off only
the resulting `ci`/`An`, exactly as a gas-exchange instrument would) fit
independently versus with shared ratio regularization, comparing
recovered-parameter error against the synthetic truth. In one seeded run
(numbers in [VALIDATION.md](VALIDATION.md)), regularization cut mean
`vcmax25` error from 46% to 7.6% by reining in one leaf whose naive fit let
`vcmax25` run away to 219 (truth 80) while still fitting that leaf's data
well — the equifinality failure mode this framework targets.

## Important model findings

- The equations remain C4-only. The `plant` column is validated but never used;
  the DataFrame adapter rejects values other than `4` explicitly.
- `vpr25` is stored but unused in the current equations, so its output sensitivity
  is exactly zero. The literal PEP-regeneration cap of `100` is preserved rather
  than silently replacing it with `vpr25`; attempts to train it are rejected.
- A single PAR curve such as `examples/data/Input.xlsx` cannot identify the full
  parameter set. In particular, `jmax25`, `theta`, and `x`; `vpmax25`, `gbs`, and
  `x`; `vcmax25` and `rd25`; `go` and `g1`; and, when `finite_gm=True`,
  `vpmax25` and `gm25` can be strongly correlated.
- `gm25` (mesophyll conductance from the intercellular airspace to the
  PEP-carboxylation site) is **opt-in** via `finite_gm=True` on `Leaf` or
  `LeafBiochemistry` (default `False`). When off (the default), the model
  behaves exactly as it always has: PEP carboxylation draws directly from
  `ci` (the infinite-`gm` assumption), with no `cm` state and no `gm25`
  sensitivity. When on, a coupled state `cm` is added
  (`Cm = Ci - aNet/gm`, `gm` temperature-scaled like `vcmax`/`jmax`/`vpmax`
  via a peaked-Arrhenius response adapted from Bernacchi et al. 2002 — a
  documented C3-literature placeholder, not a measured C4 value); see
  [MODEL_NOTES.md](MODEL_NOTES.md) for the full residual system.
- Hard minima give little or no information about an inactive capacity. Vary PAR,
  CO2, temperature, humidity, and wind, and measure more than `aNet` when fitting.
- A singular/ill-conditioned equilibrium Jacobian makes implicit gradients
  unreliable. Inspect `output.diagnostics.jacobian_condition` and residuals.
- `gbForced` and `gbFree` use the m/s boundary-layer convection convention, while
  `gb` is converted to mol m^-2 s^-1. `L` is bundle-sheath leak flux, and `cb` is
  leaf-surface CO2.
- Wind must be strictly positive. An exact zero-wind, zero-buoyancy state has no
  boundary transfer and makes the gas-transport equations undefined, so it is
  rejected with a clear validation error.

See [MODEL_NOTES.md](MODEL_NOTES.md) for equations, numerical design, and
interpretation of diagnostics.

## Test

```bash
python -m pytest
```

Tests cover physical identities, hard/smooth equilibrium, energy balance,
nonzero parameter gradients, the zero influence of `vpr25`, an implicit-gradient
finite-difference check, inference mode, staged multi-start, invalid-root/
parameter handling, solver options, the pandas adapter, the opt-in
mesophyll-conductance state (including that `LeafBiochemistry` reproduces
`Leaf`'s own solved `ci` exactly, confirming the shared biochemistry core),
and the multi-curve calibration framework (weight tying, ratio regularization,
and the identifiability diagnostic).

## Repository layout

```text
src/pyleaf_torch/                   differentiable Torch package
src/pyleaf_torch/calibration.py     multi-curve parameter estimation framework
examples/plot_response_curves.py    A-Ci and A-Q response curves
examples/calibration_benchmark.py   autograd vs. derivative-free calibration
examples/parameter_estimation.py    multi-curve, ratio-regularized parameter estimation
examples/data/Input.xlsx            sample weather workbook
tests/                              solver, gradient, and adapter tests
MODEL_NOTES.md                      numerical/scientific design notes
```
