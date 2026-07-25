# Validation results

Results through the "Automated suite" section were produced in the
repository's isolated environment on 2026-07-19 with Python 3.12.4,
PyTorch 2.13.0+cpu, NumPy 2.5.1, and pandas 3.0.3. The `gm25`/`cm` physics and
parameter-estimation results below were produced on 2026-07-25 in the same
environment (identical package versions).

## Broad forward-convergence stress grid

A deterministic seed-123 grid sampled 200 controlled-temperature rows across
`ca=100..1500`, `O2=180..250`, `tAir=2..50 deg C`, relative vapor pressure
`0.05..1.0`, pressure `70..105 kPa`, wind `0.001..10 m/s`, PAR `0..700`, longwave
`300..900`, NIR `0..300`, and `controlTemp=2..60 deg C`.

With hard physics, `max_iterations=100`, and tolerance `1e-8`, the final staged
multi-start solver converged **200/200** rows; the largest scaled residual was
`6.769e-9`. The original single `aNet=0.1*ca` start converged only 188/200 on the
same grid. This improvement comes from the solver design, not differentiability
by itself.

## Implicit gradient check

`tests/test_model.py::test_implicit_gradient_matches_central_difference` compares
the autograd derivative of `aNet` with respect to the unconstrained `vcmax25`
coordinate against a central finite difference. Additional audit checks covered
nine parameters plus `ca`, PAR, and air-temperature forcings in fixed-temperature
and energy modes; the worst observed relative error was about `7e-8`. These are
first-order gradients only. Separate tests require finite nonzero gradients for
`vcmax25`, `jmax25`, and `g1`, and verify exact zero output influence from unused
`vpr25`.

## Seeded calibration example

Command:

```bash
python examples/calibration_benchmark.py --rows 12 --evaluations 40
```

Synthetic truth: `vcmax25=62`, `jmax25=285`, `g1=4.2`. Shared start:
`vcmax25=32`, `jmax25=500`, `g1=1.6`. Both arms use the same smooth model,
observations, physical bounds, initial diagnostic, and cap of 40 subsequent loss
evaluations. Adam uses 39 parameter updates plus a final evaluation; Powell uses
40 function evaluations. The best evaluated incumbent is reported for both.

| Optimizer | Initial loss | Best loss | Reduction | Time on validation machine |
|---|---:|---:|---:|---:|
| Torch autograd + Adam | 3.44820 | 0.06788 | 98.03% | 47.9 s |
| SciPy Powell | 3.44820 | 0.09925 | 97.12% | 79.2 s |

Adam achieved the lower loss and shorter wall time in this run. Both estimates
retained large `vcmax25`/`jmax25` errors even while fitting outputs well,
demonstrating their confounding under this small design. Best-incumbent `g1` was
`3.898` with autograd and `3.786` with Powell versus truth `4.2`. Powell reached
the evaluation cap mid-search (`success=false`) and one trial equilibrium was
invalid and penalized; its best valid incumbent is the reported result.

This benchmark supports a narrow conclusion: exact gradients can improve local
calibration efficiency. It does not guarantee easy convergence or parameter
identifiability. A scientific benchmark should use more conditions, observations,
seeds, starts, held-out rows, and matched wall-clock as well as evaluation caps.

## Automated suite

```text
26 passed
```

The suite covers fixed-temperature and energy equilibria, physical identities,
implicit and finite gradients, unused-parameter sensitivity, inference mode,
parameter/solver validation, difficult-row multi-start recovery, and DataFrame
input/output behavior.

## Mesophyll conductance (`gm25`/`cm`), now opt-in, and `LeafBiochemistry`

Mesophyll conductance was revised after initial review: `finite_gm` is now a
constructor flag on both `Leaf` and the new `LeafBiochemistry` class,
**defaulting to `False`** (the original infinite-`gm` equations, bit-for-bit
what this model used before `gm25`/`cm` were added — no `cm` state, PEP
carboxylation draws from `ci` directly). `finite_gm=True` adds the 7th
coupled state `cm` (`Cm = Ci - aNet/gm`), with `gm` now temperature-scaled by
a peaked-Arrhenius response (adapted from Bernacchi et al. 2002) rather than
held flat.

`LeafBiochemistry` is a new, separate, `ci`-driven model (no stomata/boundary
layer/energy balance) for fitting measured A-Ci curve data directly, sharing
its CO2/light-limited assimilation equations with `Leaf` through one
`_biochemistry_core` method rather than duplicating them.

```text
python -m pytest tests/test_model.py tests/test_robustness.py tests/test_dataframe.py
38 passed in 28.96s
```

All 26 original tests pass unmodified against the new `finite_gm=False`
default (confirming the revert is exact), plus: `gm25` bounds validation; a
`finite_gm=False` sanity check that `cm` is absent from `STATE_NAMES` and
output; `Cm` staying within `~1e-2` of `Ci` when `gm25` is very large (`1e4`,
`finite_gm=True`); `Ci - Cm` matching `aNet/gm` exactly to `1e-6`; a finite
nonzero `aNet`/`gm25` gradient in smooth mode; `gm`'s temperature response
being peaked/nonzero-sensitivity; `LeafBiochemistry` converging from direct
`ci`/`tLeaf`/`PAR`/`O2` inputs; **`LeafBiochemistry` reproducing `Leaf`'s own
solved `aNet`/`cbs`/`cm` to `1e-6` when fed `Leaf`'s own solved `ci`** (both
`finite_gm` branches — the strongest available check that the shared
biochemistry core is truly equivalent between the two classes); missing/
invalid condition-field validation; and a nonzero `vcmax25` gradient on
`LeafBiochemistry`.

`python examples/calibration_benchmark.py --rows 12 --evaluations 40` and
`python examples/plot_response_curves.py` were both re-run as regression
checks and are unaffected (both use `Leaf` with the `finite_gm=False`
default).

## Multi-curve, `ci`-driven calibration framework (`pyleaf_torch.calibration`)

```text
python -m pytest tests/test_calibration.py
13 passed in 37.84s
```

`calibration.py` now fits `LeafBiochemistry` (driven by measured `ci`, `tLeaf`,
`PAR`, `O2`) rather than the full `ca`-driven `Leaf`, matching standard A-Ci
curve fitting practice: predicted assimilation is compared against measured
assimilation at a given `ci` directly, with no stomatal-conductance model
fit. `CurveGroup.conditions` replaced `CurveGroup.weather`. Switching to the
much smaller `LeafBiochemistry` state (2-3 variables vs. `Leaf`'s 6-7) also
cut this suite's runtime from ~13 minutes to well under a minute, since each
equilibrium solve needs far fewer autograd passes per Levenberg-Marquardt
iteration.

13 tests cover the same ground as before (`CurveGroup` validation and
per-curve masking; `build_curve_models` weight tying and its validation;
`collect_parameters` deduplication; `RatioRegularizer.penalty` against a
hand-computed value; `fit_curve_group` reducing loss and shaping its result
for both per-curve and fully-shared parameter groups; `RatioRegularizer`
measurably reducing cross-curve ratio spread on a synthetic case with a
genuinely shared truth ratio; `identifiability_report` flagging a
deliberately weakly-identified `{vpmax25, gm25}` pair, now at `ci=20`,
correlation `-0.99` against a `0.7` threshold, plus its unknown-parameter
error path), updated for the `conditions`/`LeafBiochemistry` API.

## Seeded parameter estimation example

Command:

```bash
python examples/parameter_estimation.py
```

Four synthetic leaves, `vcmax25` truth `40`/`55`/`65`/`80`, `vpmax25` truth a
noisy `2.2x` ratio of `vcmax25` (`+-5%` jitter, seed `11`) so that PhoTorch-
style ratio regularization is fitting a relationship that genuinely holds in
the synthetic truth; `rd25` truth independently noisy around `0.04` (left
unregularized, matching the model's own built-in `rd25`-as-ratio-of-`vcmax`
mitigation instead). Each leaf is simulated realistically with the full,
`ca`-driven `Leaf` model (10 points per curve, saturating PAR) and only the
resulting `ci`/`tLeaf`/`PAR`/`O2`/`aNet` are kept for fitting — `ca`, `go`,
and `g1` are discarded, exactly as a real gas-exchange dataset would only
report `ci` and `An`. Each leaf's curve starts from its own independently
random-perturbed initial guess (`START_RANGES`) rather than a shared start,
so naive per-curve fits are free to drift independently. 150 Adam
iterations, `learning_rate=0.15`; naive and joint (ratio-regularized) fits
share the same seed and starting draws for a matched comparison. Fit target
is `vcmax25`/`vpmax25`/`rd25` — not `jmax25`, since a single, one-PAR A-Ci
curve does not identify it (see README/MODEL_NOTES.md).

| Parameter | Naive mean abs. % error | Joint (regularized) mean abs. % error |
|---|---:|---:|
| `vcmax25` | 45.98% | 7.59% |
| `vpmax25` | 2.10% | 7.06% |
| `rd25` (unregularized) | 52.59% | 76.40% |

This is an honest, non-cherry-picked result: the naive fit let one leaf's
`vcmax25` run away to `219.0` (truth `80.0`, a 174% error) while still fitting
that leaf's own data well (`vpmax25=180.6` vs. truth `177.5`) — a textbook
equifinality failure where `vcmax25` was locally underdetermined for that
leaf's conditions. Tying `vpmax25/vcmax25` across leaves via
`RatioRegularizer` reined that leaf's `vcmax25` back to `96.7` (21% error),
which is why the *mean* `vcmax25` error drops so sharply (46%→7.6%). The
trade-off is visible too: `vpmax25`'s own mean error got slightly worse
(2.1%→7.1%), since regularization pulls it along with the correction to the
outlier leaf. `rd25`, left unregularized here, is not meaningfully identified
by either fit within this iteration budget in this run.

A second, separate, minimal demonstration in the same script switches on
`finite_gm=True` for a single 10-point curve (`vcmax25=55`, `vpmax25=120`,
`gm25=3.0` truth, this time genuinely simulated with `finite_gm=True` so
`gm25` actually shapes the data): `vcmax25` recovered to `55.08` (<1% error),
`vpmax25` to `114.7` (~4% error), but `gm25` only to `4.32` (~44% error), with
`identifiability_report` flagging `{vpmax25, gm25}` at correlation `-0.92` —
consistent with, and a direct illustration of, why `finite_gm` defaults to
off and why fitting it well needs more than one ordinary A-Ci curve.
