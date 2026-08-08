# The Differentiable Forward Leaf Model

How `Leaf` turns weather into a full leaf state, why that requires solving a
coupled equilibrium rather than evaluating a formula, and how the equations are
written so PyTorch autograd can differentiate straight through the solve.

Physics background is in `MODEL_NOTES.md`; this document is about the *mechanism*
— what happens on a forward pass and why each expression takes the form it does.
Everything here concerns `Leaf` (the `ca`-driven coupled model in
`src/pyleaf_torch/model.py:377`), not the A-Ci fitting path.

---

## 1. What a forward pass computes

You give `Leaf.forward` nine weather fields for a batch of rows
(`model.py:398-408`):

```text
ca        ambient CO2                 µmol mol⁻¹
O2        ambient O2                  mmol mol⁻¹
tAir      air temperature             °C
ea        air vapour pressure         Pa
pressure  atmospheric pressure        Pa
wind      wind speed                  m s⁻¹
PAR       absorbed PAR                W m⁻²   (converted to µmol with ×4.57)
long      absorbed longwave           W m⁻²
NIR       absorbed near-infrared      W m⁻²
```

and you get back the leaf's steady state: assimilation, the CO2 chain from air to
bundle sheath, stomatal conductance, leaf temperature, and the full energy budget
(`LeafOutput`, `outputs.py:23`).

The point of interest is that **none of these outputs can be computed directly**.

---

## 2. Why it is a root-find, not an evaluation

The leaf's variables define each other in a cycle. Trace it:

```text
       aNet  ──needs──▶  cbs, ci        (biochemistry: CO2 at the carboxylation sites)
        ▲                  │
        │                  ▼
        │                 ci  ──needs──▶  gs, cb    (Fick's law through the stomata)
        │                                  │
        │                                  ▼
        │                                 gs  ──needs──▶  aNet, eb, cb, Gamma
        │                                  │              (stomata open in proportion
        │                                  │               to assimilation)
        │                                  ▼
        │                                 cb  ──needs──▶  aNet, gb
        │                                                  │
        │                                                  ▼
        │                                                 gb  ──needs──▶  tLeaf
        │                                                      (free convection depends
        │                                                       on the leaf-air ΔT)
        │                                                        │
        └────────────── tLeaf ──needs──▶ LE ──needs──▶ gs, gb, ei(tLeaf), aNet
```

Three interlocking loops:

- **Carbon loop** — `aNet` sets the CO2 drawdown that sets `ci` and `cbs`, which
  set `aNet`.
- **Stomatal loop** — `gs` responds to `aNet`, but `gs` controls the CO2 supply
  that produces `aNet`.
- **Energy loop** — `tLeaf` drives `ei`, latent heat, and every temperature
  response; but latent heat depends on `gs` and `gb`, which depend on `tLeaf`.

There is no ordering of these that lets you evaluate them once. So the model
states them as a residual system that the true state must zero:

```text
R(z, p, w) = 0
```

`z` is the unknown state, `p` the parameters, `w` the weather.
`forward` finds `z*` numerically. That framing is what later makes the
differentiability clean — see §7.

---

## 3. The state vector and the six residuals

```text
z = [aNet, cbs, ci, gs, cb, tLeaf]           finite_gm=False  (default)
z = [aNet, cbs, ci, cm, gs, cb, tLeaf]       finite_gm=True
```

declared at `model.py:451-455`, assembled in `_raw_residual` (`model.py:611`):

| # | Residual | Expression | Physical statement |
|---|---|---|---|
| 1 | `R_A` | `aNet − (1 − Γ*/cbs)·cbs·a1/(cbs+b1) + rd` | net assimilation equals gross carboxylation, discounted for photorespiration, minus day respiration |
| 2 | `R_cbs` | `cbs − site_co2 − (vp − aNet − rm)/gbs` | bundle-sheath CO2 balance: the C4 pump delivers `vp`, consumption plus leakage removes the rest |
| 3 | `R_ci` | `ci − cb + 1.6·aNet/gs` | Fick's law through the stomatal pore; `1.6` is the H₂O/CO₂ diffusivity ratio |
| 3b | `R_cm` | `cm − ci + aNet/gm` | mesophyll diffusion (only when `finite_gm=True`) |
| 4 | `R_gs` | `gs − gs*` | stomata sit at their Ball-Berry-type target |
| 5 | `R_cb` | `cb − ca + 1.37·aNet/(gb·Mb)` | boundary-layer CO2 drawdown; `1.37` is the ratio in the turbulent layer |
| 6 | `R_T` | `radiation − 0.506·aNet − H − LE − emission` | leaf energy balance closes |

With `energy_balance=False`, residual 6 becomes `tLeaf − controlTemp`
(`model.py:637-640`) — the leaf temperature is pinned to a measured value and the
row still solves the other five equations.

The CO2 chain reads as a series of resistances, which is exactly what residuals
5 → 3 → 3b → 2 encode:

```text
ca ──(boundary layer, gb·Mb)──▶ cb ──(stomata, gs)──▶ ci ──(mesophyll, gm)──▶ cm ──(C4 pump, vp)──▶ cbs
```

`cbs` ends up far above ambient — that is the whole point of C4 photosynthesis.
In the worked example of §8 it reaches ~6200 µmol mol⁻¹ against `ca = 400`.

---

## 4. Inside one residual evaluation

Every solver iteration calls `_derived` (`model.py:505`), which rebuilds the
entire leaf from the current state guess. Nothing is cached between iterations —
that is deliberate, and it is what keeps the residual a pure function of
`(z, p, w)` and therefore differentiable.

### 4.1 Temperature responses — `_temperature_response` (`model.py:196`)

Three different functional forms, chosen per quantity:

```python
Ko = Ko25 · 1.2^((T−25)/10)          # Q10, gentle
Kc = Kc25 · 2.1^((T−25)/10)          # Q10
Kp = Kp25 · 2.1^((T−25)/10)          # Q10
```

Capacities use a **peaked Arrhenius** (`model.py:207-226`) — activation up to an
optimum, then deactivation as enzymes denature:

```text
V(T) = V25 · exp(Ha·(Tk−298.15)/(298.15·R·Tk)) · (1+exp((298.15·ΔS−Hd)/(298.15·R)))
                                                / (1+exp((Tk·ΔS−Hd)/(Tk·R)))
```

applied to `jmax` (Ha=77900, ΔS=627, Hd=191929), `vcmax` (67294, 472, 144568),
`vpmax` (70373, 376, 117910), and `gm` (49600, 1400, 437400). The specificity
factor `sco` uses a plain *decreasing* Arrhenius (`model.py:238-242`), so
`Γ*` rises with temperature — photorespiration worsens in heat.

Then `rd = rd25 · vcmax` (`model.py:244`), with `rm = rbs = 0.5·rd`: respiration
is expressed as a *fraction of Rubisco capacity*, not an independent rate, and is
split evenly between mesophyll and bundle sheath.

### 4.2 Electron transport (`model.py:281-291`)

```python
phi_ps2 = 0.352 + 0.022·T − 3.4·T²/10000       # PSII quantum yield, peaks near ~32 °C
light   = PAR · 4.57 · phi_ps2 / 2             # W m⁻² → µmol, split across two photosystems
J       = (light + jmax − √((light+jmax)² − 4·θ·light·jmax)) / (2θ)
```

The last line is the smaller root of the standard non-rectangular hyperbola. It
is solved **in closed form** rather than iteratively — one of several places
where an analytic solution is chosen specifically so autograd gets an exact
derivative instead of one accumulated through an inner loop.

### 4.3 C4 assimilation — `_biochemistry_core` (`model.py:260`)

Bundle-sheath O2 rises with assimilation if `alpha > 0` (O2 evolved inside the
sheath); with the default `alpha = 0` it is just ambient:

```python
obs   = alpha·aNet/(0.047·gbs·1000) + O2
Γ*    = gammaStarRatio · obs                # gammaStarRatio = 1/(2·sco)
```

Rubisco carboxylation, either CO2- or light-limited:

```python
vc_co2   = cbs·vcmax / (cbs + Kc·(1 + obs/Ko))        # Michaelis-Menten in cbs
vc_light = cbs·(1−x)·J / (3·cbs + 7·Γ*)               # electron-transport share (1−x)
vc       = min(vc_co2, vc_light)
```

PEP carboxylation (the C4 pump), with the literal regeneration cap:

```python
vp_light = x·J/2                                      # fraction x of J goes to PEP
vp_co2   = min(cm·vpmax/(cm + Kp), 100)               # 100 is a literal cap, NOT vpr25
vp       = min(vp_light, vp_co2)
```

The `100` is a hard-coded PEP-regeneration ceiling. `vpr25` exists in the
parameter table but is **deliberately never used**; `model.py:63-67` refuses to
make it trainable so nobody silently reinterprets the literal as a fitted value.

#### The `a1` / `b1` construction — why it exists

Residual 1 does not use `vc` directly. It uses a single rational form
(`model.py:315-320`):

```python
w  = enzyme_weight                                     # 1 → CO2-limited, 0 → light-limited
a1 = w·vcmax        + (1−w)·((1−x)·J/3)
b1 = w·(Kc(1+O/Ko)) + (1−w)·(7·Γ*/3)

carboxylation = cbs·a1/(cbs + b1)
```

Substitute the two extremes and you recover the branches exactly:

```text
w = 1:  cbs·vcmax/(cbs + Kc(1+O/Ko))            = vc_co2      ✔
w = 0:  cbs·(1−x)J/3 /(cbs + 7Γ*/3)
      = cbs·(1−x)J   /(3cbs + 7Γ*)              = vc_light    ✔
```

So `a1`/`b1` is a **unified Michaelis-Menten form that reproduces whichever limit
applies**, expressed as one smooth rational function of `cbs`. That matters for
the solver: `cbs` is an unknown being iterated, and a single differentiable
rational expression gives it a well-behaved derivative, instead of a `min` whose
derivative in `cbs` jumps at the crossover.

#### Compensation points

`Gamma_C` (`model.py:322-329`) is the bundle-sheath CO2 compensation point:

```text
Gamma_C = (Γ* + Kc(1+O/Ko)·rd/vcmax) / (1 − rd/vcmax)
```

`Gamma` (`model.py:331-350`) is the leaf-level compensation point that the
stomatal model needs, and it is the positive root of

```text
gbs·Γ² + [gbs·(Kp − Gamma_C) + vpmax − rm]·Γ − [Gamma_C·Kp·gbs + rm·Kp] = 0
```

The root extraction here is written in a specific numerically-stable way — see
§6.3, it is not incidental.

### 4.4 Boundary layer (`model.py:522-556`)

```python
Mb  = 0.5·(1+s)²/(1+s²)                                # stomatal-ratio factor for the two faces
conv = P/(8.309·Tk)                                    # m s⁻¹ → mol m⁻² s⁻¹
gb_forced = cForced·Tk^0.56·((Tk+120)·(u/d/P))^0.5     # wind-driven
gb_free   = cFree·Tlk^0.56·(((Tlk+120)/P)^0.5·(|ΔTv|/d)^0.25)   # buoyancy-driven
gb = max(gb_free, gb_forced) · conv
```

Free convection is driven by the **virtual** temperature difference — leaf minus
air, each corrected for the density effect of water vapour
(`model.py:539-543`) — because humid air is lighter than dry air, and near-zero
leaf-air ΔT the moisture term can flip the direction of buoyancy.

That creates a sub-circularity: `gb_free` needs the boundary-layer vapour
pressure `eb`, and `eb` needs `gb`. It is resolved by a **fixed 8-iteration
loop** (`model.py:538-556`), not a convergence test. Fixed trip count means the
unrolled autograd graph has static structure and a well-defined derivative; a
`while` loop keyed on a tolerance would make the gradient depend on where the
inner loop happened to stop.

```python
eb = (gs·ei + gb·Mb·ea) / (gs + gb·Mb)     # vapour pressure at the leaf surface
g  = gs·gb·Mb / (gs + gb·Mb)               # stomata and boundary layer in series
```

### 4.5 Stomatal target (`model.py:560-570`)

```python
gs* = max( go,  go + g1·aNet·(eb/ei)/(cb − Gamma) )
```

A Ball-Berry-Leuning form: conductance scales with assimilation, humidity at the
leaf surface (`eb/ei`), and the inverse of the CO2 drawdown above the
compensation point. The `max` enforces the cuticular floor `go`, which matters
at night when `aNet < 0` would otherwise drive `gs` negative.

### 4.6 Energy balance (`model.py:572-582`)

```python
λ_molar   = (2500 − 2.36·tLeaf)·18                      # J mol⁻¹, temperature-dependent
H         = 29.3·(gb·0.924)·(tLeaf − tAir)              # sensible heat, both faces
LE        = λ_molar·(g/P)·(ei − ea)                     # latent heat
emission  = 2·0.94·σ·Tlk⁴                               # longwave out, both faces
radiation = PAR + NIR + long
R_T       = radiation − 0.506·aNet − H − LE − emission
```

`0.506·aNet` is the energy stored in photochemistry (J per µmol CO2 fixed) — a
small term, but it is what makes the carbon and energy loops share a variable in
both directions.

---

## 5. Solving it

`forward` (`model.py:853`) runs a batched Levenberg-Marquardt solve
(`solver.py:95`) with every row independent.

**Scaling.** `_scaled_residual` (`model.py:649`) divides the state by
`[10, 1000, 100, 0.1, 100, 10]` and the residuals by
`[10, 1000, 100, 0.1, 100, 100]`. Without this, `cbs ≈ 6000` and `gs ≈ 0.3` differ
by four orders of magnitude and the Jacobian is hopelessly ill-conditioned. This
is a diagonal linear change of variables, so it conditions the solve without
altering the solution or the gradient.

**Initial guess.** Physiology-informed rather than arbitrary
(`model.py:668-685`):

```text
aNet = 0.1·ca      cbs = 10·ca      ci = 0.7·ca
gs = 0.1           cb = 0.8·ca      tLeaf = tAir
```

**Bounds.** `[−20, 150]` for `aNet`, `[0.1, 20000]` for `cbs`, and `tLeaf`
narrowed to `[2, 60]` °C under energy balance (`model.py:687-693`).

**Restarts.** If any row fails, `forward` retries it from three alternative
assimilation guesses — light-scaled `0.08·PAR − 2`, then `0.03·ca`, then zero —
keeping the lowest-residual result per row (`model.py:880-898`,
`select_better_root` at `solver.py:206`). Rows that already converged keep their
answer.

**Convergence** is the infinity norm of the *complete scaled* residual against
`1e-7` (`model.py:709-712`), so no equation can hide behind another.

---

## 6. How the equations are written to stay differentiable

This is the part your question was really about. Being written in `torch` is
necessary but nowhere near sufficient — a function can be perfectly well-defined
and still hand back `NaN` or zero gradients. Six deliberate constructions:

### 6.1 Nothing breaks the graph

No `.item()`, no NumPy round-trip, no `math.exp` on a tensor, no Python `if` on a
tensor's *value*. State is taken apart with `unbind` (`model.py:512-515`) and
residuals reassembled with `torch.stack` (`model.py:642-647`) — both
differentiable. No in-place mutation of a graph tensor anywhere in the residual.

The one Python-level branch that exists — `if self.mode == "hard"` — switches on a
*configuration flag*, not on data, so the graph structure is fixed at
construction.

### 6.2 Safe denominators

```python
# model.py:172-175
def _safe_denominator(value, epsilon=1e-10):
    sign = torch.where(value >= 0, ones, −ones)
    return torch.where(value.abs() < epsilon, sign·epsilon, value)
```

Used on every divisor that can legitimately approach zero: `gs + gb·Mb` in the
`eb` and `g` expressions (`model.py:534, 556`), the virtual-temperature
denominators (`model.py:539-543`), and `cb − Gamma` in the stomatal target
(`model.py:566`). That last one is the dangerous case — at the compensation point
`cb − Gamma → 0` and `gs*` would blow up.

The `torch.where` form is doing real work here. It is a *selection*: gradient
flows through whichever branch was taken, and the epsilon branch is a constant
with zero derivative — never `inf`. Adding `+eps` instead would bias the physics
everywhere; `clamp` would leave a kink and lose the sign. Preserving `sign` keeps
the function odd about zero, so the gradient does not spuriously flip.

### 6.3 Clamped discriminants and a stable quadratic root

```python
# model.py:286-288 and 341-344
torch.clamp(discriminant, min=torch.finfo(dtype).tiny)
```

`d(√x)/dx = 1/(2√x)`, which is `inf` at zero and `NaN` below it. Roundoff can push
a mathematically non-negative discriminant slightly negative, so both the electron-
transport and `Gamma` discriminants are clamped to the smallest positive normal.
This is the single most common source of `NaN` in naively written photosynthesis
models.

The `Gamma` root then avoids catastrophic cancellation (`model.py:346-350`):

```python
gamma_standard = (−b + √D)/(2a)
gamma_rational = (−2c)/safe_denominator(b + √D)
gamma = torch.where(b >= 0, gamma_rational, gamma_standard)
```

When `b > 0` and `4ac ≪ b²`, `−b + √D` subtracts two nearly equal numbers. The
*value* loses precision; the *derivative* loses it much faster, because
differentiation amplifies relative error. Selecting the algebraically equivalent
form that **adds** rather than subtracts keeps both accurate. With the default
parameters `b` is positive in normal conditions, so the rational branch is the
one that actually runs.

### 6.4 Smooth switches — `mode="smooth"`

The model is full of `min`/`max`: CO2 vs light limitation (twice), the `100` PEP
cap, forced vs free convection, the `gs ≥ go` floor, and `|ΔTv|`. `torch.minimum`
is differentiable almost everywhere, but its derivative jumps discontinuously at
the tie and is only a subgradient there.

`mode="smooth"` (`model.py:177-190`) swaps in softened versions:

```python
_minimum(l, r, τ) = −τ·logaddexp(−l/τ, −r/τ)
_maximum(l, r, τ) =  τ·logaddexp( l/τ,  r/τ)
_absolute(v, ε)   =  √(v² + ε²)
```

`logaddexp` rather than `log(exp+exp)` because `τ = 0.05` puts `l/τ` in the
hundreds, where the naive form overflows.

τ is chosen per unit: `0.05` for assimilation rates in µmol m⁻² s⁻¹
(`model.py:303, 308, 309`), `1e-6` for conductances in m s⁻¹ (`model.py:552`),
`1e-5` for the `gs` floor (`model.py:569`), `0.01` for `Gamma_C`
(`model.py:329`), `1e-3` for the virtual temperature difference
(`model.py:548`). The bias at an exact tie is about `τ·ln2` and decays
exponentially away from it — physically negligible, numerically decisive.
`MODEL_NOTES.md:53-55` notes the one that matters most: smoothing the `gs*`
maximum keeps the floor consistent between the residual *and* its Jacobian.

### 6.5 The limitation switch as a sigmoid weight

The subtlest case, and a real dead end rather than merely a kink. In hard mode
`enzyme_weight` is a boolean cast to float, whose gradient is **identically zero
everywhere** (`model.py:311-314`):

```python
if self.mode == "hard":
    enzyme_weight = (vc_co2 < vc_light).to(dtype)          # ∂/∂anything ≡ 0
else:
    enzyme_weight = torch.sigmoid((vc_light − vc_co2)/0.05)
```

The sigmoid recovers the boolean as `τ → 0` but carries a genuine derivative with
respect to *both* branches near the crossover — so a change that shifts which
process limits the leaf produces a gradient reflecting that shift. This is why
gradient-based work should use `mode="smooth"` while faithful forward simulation
keeps the default `"hard"`.

### 6.6 Fixed-length inner loop

Covered in §4.4: the 8-pass boundary-layer loop has no data-dependent exit. It
lives *inside* the residual, so autograd differentiates through it normally when
the Jacobian is assembled.

---

## 7. Getting gradients through the solve

Everything above makes `R(z, p, w)` differentiable. The remaining problem is that
`z*` comes out of an *iterative solve*, and backpropagating through 5-80 LM
iterations would be expensive, memory-hungry, and — worse — would produce
gradients that depend on the stopping decision.

### 7.1 The solve is detached

```python
# model.py:865-872
detached_weather    = {k: v.detach() for k, v in prepared.items()}
detached_parameters = self.physical_parameters(detach=True)
root = solve_root(lambda x: self._scaled_residual(x, detached_weather, detached_parameters), ...)
```

The iteration history never enters the loss graph. Autograd *is* used inside the
solver, but only locally, to build `∂R/∂z` at the current iterate
(`solver.py:71-84`) — and with `create_graph=False`, so it leaves no residue. The
`.sum()` trick there exploits row independence to get all per-row Jacobians in
`width` backward passes instead of `batch × width`.

### 7.2 The implicit function theorem

At the root, `R(z*, p) = 0` holds *identically* in `p`. Differentiate totally:

```text
∂R/∂z · dz*/dp + ∂R/∂p = 0      ⟹      dz*/dp = −(∂R/∂z)⁻¹ · (∂R/∂p)
```

This needs only that `∂R/∂z` be invertible at the root. It says nothing about how
the root was found — which is precisely what licenses throwing the iterations
away. The gradient is a property of the *equilibrium*, not of the algorithm.

### 7.3 The zero-valued surrogate

`implicit_root` (`solver.py:233`) never forms `∂R/∂p` explicitly. It takes one
differentiable Newton step from the detached root with the Jacobian held constant:

```python
# solver.py:247-261
x_leaf   = result.x.detach().requires_grad_(True)
residual = residual_function(x_leaf)        # live p and w — this carries the graph
jacobian = result.jacobian + reg·I          # detached, constant
corrected = x_leaf − solve(jacobian, residual)
```

Since `x_leaf` is detached, `∂corrected/∂p` picks up only the `p`-dependence of
`residual`, with the constant Jacobian acting as `J_z⁻¹` — exactly the IFT
expression. Then:

```python
# solver.py:267-271
gradient_surrogate = corrected − corrected.detach()   # value ≡ 0, gradient ≡ dz*/dp
return result.x + gradient_surrogate
```

`corrected − corrected.detach()` is numerically zero forward and carries the full
derivative backward. The returned tensor's **value is bit-identical to the
numerical root**, its **gradient is the implicit-function result**. That exact
equality is asserted by `tests/test_model.py:105`, and the gradient itself is
checked against central differences at `tests/test_model.py:79`.

If nothing requires grad, the whole implicit path is skipped
(`model.py:900-904`) — pure forward simulation pays nothing for this machinery.

### 7.4 Where gradients are refused

The IFT assumes a locally unique, **interior**, converged root. When that fails,
the model raises rather than returning a plausible-looking wrong number
(`model.py:904-912`):

- **not converged** → `R ≠ 0`, so the differentiated identity does not hold;
- **at a state bound** (`solver.py:189-193`) → the root is a clamped boundary
  point, not a stationary solution — common when the energy balance has no zero
  inside `2-60 °C`;
- **singular Jacobian** (`solver.py:252-259`) → the IFT precondition is violated
  outright.

The cost of the detached Jacobian is that **second derivatives are unavailable**:
`∂²/∂p²` would silently omit the term from `J_z`'s own `p`-dependence.

### 7.5 Outputs are recomputed, not read off

`_build_output` (`model.py:695-708`) feeds the differentiable `scaled_state` back
through `_derived`, so every reported quantity is rebuilt rather than harvested
from the solve. An output like `transpiration = LE/λ_molar·1e6` therefore
receives gradient by two routes — through `z*` (implicit) and through the explicit
appearance of `p` and `w` in `_derived` — and autograd sums them into the correct
total derivative. Diagnostics are all detached (`model.py:778-785`); the
`limitation` flags are booleans and carry no gradient by construction.

---

## 8. A worked forward pass

Single row, defaults, `finite_gm=False`, energy balance on:

```text
ca = 400   PAR = 400 W m⁻²   tAir = 25 °C   RH = 60 %   wind = 2 m s⁻¹   P = 101325 Pa
```

Solved state (5 LM iterations, residual norm 6.2e-8, Jacobian condition 26.8):

| Quantity | Value | |
|---|---|---|
| `aNet` | 45.47 | µmol m⁻² s⁻¹ |
| `cbs` | 6197.7 | µmol mol⁻¹ — the C4 pump concentrating CO2 ~15× above ambient |
| `ci` | 117.3 | strong drawdown, characteristic of C4 |
| `cb` | 353.6 | boundary-layer CO2 |
| `gs` | 0.308 | mol m⁻² s⁻¹ |
| `tLeaf` | 28.30 | °C — 3.3 K above air |
| `vc` / `vp` | 48.78 / 65.16 | Rubisco / PEP carboxylation |
| `J` | 325.8 | µmol e⁻ m⁻² s⁻¹ |
| `Gamma` | 1.32 | leaf CO2 compensation point |
| `H` / `LE` / `emission` | 123.2 / 210.5 / 880.2 | W m⁻², against 1236.9 absorbed |

Limitation flags: `vcCO2Limited=True`, `vpCO2Limited=False`,
`forcedConvection=True`, `vpRegenerationCapped=False`, `stomatalFloor=False` —
Rubisco-limited carboxylation with a light-limited C4 pump, wind-dominated
boundary layer.

Sanity-check the energy budget: `1236.9 − 0.506×45.47 − 123.2 − 210.5 − 880.2 ≈ 0`.
That equation was *solved*, not evaluated — which is the whole story of §2.

---

## 9. What differentiability buys the forward model

Because the forward pass is differentiable in its *inputs* as well as its
parameters, you get exact analytic sensitivities without finite differences:

```python
import torch
from pyleaf_torch import Leaf

dtype = torch.float64
ca   = torch.tensor([400.0], dtype=dtype, requires_grad=True)
par  = torch.tensor([400.0], dtype=dtype, requires_grad=True)
tair = torch.tensor([25.0],  dtype=dtype, requires_grad=True)
sat  = 611.0 * torch.exp(17.502 * tair / (240.97 + tair))

weather = {
    "ca": ca, "PAR": par, "tAir": tair,
    "O2": torch.tensor([210.0], dtype=dtype),
    "ea": (sat * 0.6).detach(),
    "pressure": torch.tensor([101325.0], dtype=dtype),
    "wind": torch.tensor([2.0], dtype=dtype),
    "long": torch.tensor([716.873119944625], dtype=dtype),
    "NIR": torch.tensor([120.0], dtype=dtype),
}

model = Leaf(mode="smooth")
aNet = model(weather).mass["aNet"].sum()
print(torch.autograd.grad(aNet, [ca, par, tair]))
```

```text
∂aNet/∂ca    = 0.007328   µmol m⁻² s⁻¹ per µmol mol⁻¹   (CO2-saturated, as expected for C4)
∂aNet/∂PAR   = 0.032485   per W m⁻²                     (still light-responsive at 400 W m⁻²)
∂aNet/∂tAir  = 0.65197    per K                         (below the thermal optimum)
```

Central differences on the same point give `0.0073279` and `0.0324850` — agreement
to ~8 significant figures, confirming the implicit gradient.

The two sensitivities together are readable against the flags from §8. `∂aNet/∂ca`
is small because the C4 pump has already saturated Rubisco at `cbs ≈ 6200`, so
adding ambient CO2 buys little. `∂aNet/∂PAR` is comparatively large — not because
Rubisco is light-limited (`vcCO2Limited=True` says it is not), but because the PEP
pump is (`vpCO2Limited=False`): more light raises `J`, which raises `vp`, which
raises `cbs` through `R_cbs`, which finally raises `aNet` through `R_A`. That
gradient path runs through three coupled residuals, and getting it by hand would
mean differentiating the whole equilibrium — which is precisely the work §7 does
automatically.

Use `float64` (the default, `model.py:426`) for anything gradient-related; the
implicit correction loses precision quickly in `float32`.

---

## 10. Known limits of the forward path

| Limit | Cause | Mitigation |
|---|---|---|
| Kinked gradients at regime ties | `torch.minimum` / boolean limitation switch | `mode="smooth"` |
| No second derivatives | equilibrium Jacobian detached in `implicit_root` | first-order methods only |
| Raises on non-converged rows | IFT precondition violated | alternative starts already automatic; inspect under `torch.no_grad()` |
| Raises at active state bounds | root not interior — typically energy balance with no zero in `2-60 °C` | check `diagnostics.at_state_bound` |
| `vpr25` unused | the literal `100` PEP cap is a different assumption | do not fit it without changing the model |

Smoothing repairs gradient continuity. It does not repair invalid physics, and it
does not make a structurally non-identifiable parameter identifiable.
