# Validation results

These results were produced in the repository's isolated environment on
2026-07-19 with Python 3.12.4, PyTorch 2.13.0+cpu, NumPy 2.5.1, and pandas 3.0.3.

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
