# pyLeaf Torch

This repository contains both versions needed for a controlled comparison:

- `legacy/pyLeaf.py` is a byte-for-byte snapshot of the current NumPy/GEKKO
  implementation at upstream commit `e96a34cae637687e4f6d4e0d55aaa27bf4381aa3`.
- `src/pyleaf_torch/` is a new C4 leaf model implemented with PyTorch. It solves
  the coupled equilibrium with a scaled damped root solver and differentiates
  the converged solution with the implicit-function theorem.

The existing upstream repository was not edited. The legacy snapshot has SHA-256
`d14cd68d3ae90dcfd2d993ef379b4ef7a06f21b55941ad801fa3e95e55797efd`,
and a test protects it from accidental changes.

Measured parity, residual, stress-grid, gradient, and calibration results are in
[VALIDATION.md](VALIDATION.md).

## What differentiability does—and does not—help

PyTorch makes gradients available for parameter calibration. That normally makes
Adam or L-BFGS much more evaluation-efficient than derivative-free fitting when
the selected parameters are identifiable and the initial point is in the basin
of the correct solution.

Differentiability alone does **not** make the forward leaf-state equations
converge. Forward robustness here comes from a separate numerical redesign:

1. `aNet`, `cbs`, `ci`, `gs`, `cb`, and `tLeaf` are solved as one equilibrium.
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
adds pandas/Excel adapters, GEKKO legacy comparison, SciPy benchmark, and tests.
The model defaults to `torch.float64` because the physical variables span several
orders of magnitude.

## Tensor API

```python
import pandas as pd
import torch

from pyleaf_torch import DifferentiableLeaf, weather_from_dataframe

frame = pd.read_excel("legacy/Input.xlsx")
weather = weather_from_dataframe(frame)

model = DifferentiableLeaf(
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

Both modes enforce `gs >= go`. This intentionally corrects the legacy code's
inconsistent stomatal-floor Jacobian branch and changes some dark or negative-
assimilation outputs.

## Compare the two implementations

The comparison script runs the frozen GEKKO code and the new hard or smooth
equilibrium on the same workbook. It writes side-by-side values, absolute and
relative differences, solver diagnostics, and limitation regimes.

```bash
python examples/compare_models.py --mode hard
```

Outputs go to `comparison_output/`. The new solver intentionally corrects stale
state dependencies and convergence checks, so agreement should be judged along
with the new equation residual—not by forcing exact equality with a legacy state
that stopped early.

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

## Important model findings

- The equations remain C4-only. The legacy `plant` column is validated but never
  used; the new DataFrame adapter rejects values other than `4` explicitly.
- `vpr25` is stored but unused in the current equations, so its output sensitivity
  is exactly zero. The literal PEP-regeneration cap of `100` is preserved rather
  than silently replacing it with `vpr25`; attempts to train it are rejected.
- A single PAR curve such as `legacy/Input.xlsx` cannot identify the full
  parameter set. In particular, `jmax25`, `theta`, and `x`; `vpmax25`, `gbs`, and
  `x`; `vcmax25` and `rd25`; and `go` and `g1` can be strongly correlated.
- Hard minima give little or no information about an inactive capacity. Vary PAR,
  CO2, temperature, humidity, and wind, and measure more than `aNet` when fitting.
- A singular/ill-conditioned equilibrium Jacobian makes implicit gradients
  unreliable. Inspect `output.diagnostics.jacobian_condition` and residuals.
- `gbForced` and `gbFree` retain the legacy m/s convention, while `gb` is converted
  to mol m^-2 s^-1. `L` is bundle-sheath leak flux, and `cb` is leaf-surface CO2.
- Wind must be strictly positive. An exact zero-wind, zero-buoyancy state has no
  boundary transfer and makes the gas-transport equations undefined, so it is
  rejected with a clear validation error.

See [MODEL_NOTES.md](MODEL_NOTES.md) for equations, compatibility differences,
and interpretation of diagnostics.

## Test

```bash
python -m pytest
```

Tests cover every frozen legacy hash, physical identities, hard/smooth equilibrium,
energy balance, nonzero parameter gradients, the zero influence of `vpr25`, an
implicit-gradient finite-difference check, inference mode, staged multi-start,
invalid-root/parameter handling, solver options, and the pandas adapter.

## Repository layout

```text
legacy/                       frozen current NumPy/GEKKO implementation
src/pyleaf_torch/             differentiable Torch package
examples/compare_models.py    output comparison harness
examples/calibration_benchmark.py
tests/                        solver, gradient, adapter, and provenance tests
MODEL_NOTES.md                numerical/scientific design notes
```
