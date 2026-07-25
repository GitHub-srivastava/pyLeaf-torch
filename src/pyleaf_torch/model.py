"""Differentiable coupled-equilibrium implementation of the pyLeaf equations."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any, Literal

import torch
from torch import Tensor, nn

from .outputs import LeafOutput, SolverDiagnostics
from .parameters import (
    DEFAULT_PARAMETERS,
    PARAMETER_SPECS,
    from_raw,
    to_raw,
)
from .solver import SolverOptions, implicit_root, select_better_root, solve_root


Mode = Literal["hard", "smooth"]


class DifferentiableLeaf(nn.Module):
    """C4 leaf model with a batched Torch root solve and implicit gradients.

    Parameters are represented in valid physical domains. Only names listed in
    ``trainable`` become :class:`torch.nn.Parameter` objects; all others are
    fixed buffers. The tensor-only forward path is differentiable with respect
    to trainable parameters and tensor weather forcings.

    ``mode="hard"`` retains exact limitation switches and is piecewise
    differentiable. ``mode="smooth"`` rounds switch surfaces slightly to make
    calibration gradients continuous near regime changes.
    """

    WEATHER_COLUMNS = (
        "ca",
        "O2",
        "tAir",
        "ea",
        "pressure",
        "wind",
        "PAR",
        "long",
        "NIR",
    )

    STATE_NAMES = ("aNet", "cbs", "ci", "gs", "cb", "tLeaf")

    def __init__(
        self,
        parameters: Mapping[str, float] | None = None,
        *,
        trainable: Iterable[str] = (
            "vcmax25",
            "vpmax25",
            "jmax25",
            "rd25",
            "go",
            "g1",
        ),
        mode: Mode = "hard",
        energy_balance: bool | None = None,
        solver_options: SolverOptions | None = None,
        dtype: torch.dtype = torch.float64,
        device: torch.device | str | None = None,
        boundary_iterations: int = 8,
    ) -> None:
        super().__init__()
        if mode not in ("hard", "smooth"):
            raise ValueError("mode must be 'hard' or 'smooth'")
        if boundary_iterations < 1:
            raise ValueError("boundary_iterations must be positive")

        supplied = dict(parameters or {})
        parameter_energy_switch = supplied.pop("energySwitch", None)
        unknown = sorted(set(supplied) - set(DEFAULT_PARAMETERS))
        if unknown:
            raise KeyError(f"Unknown pyLeaf parameter(s): {', '.join(unknown)}")

        values = dict(DEFAULT_PARAMETERS)
        values.update({name: float(value) for name, value in supplied.items()})
        trainable_names = tuple(dict.fromkeys(trainable))
        unknown_trainable = sorted(set(trainable_names) - set(DEFAULT_PARAMETERS))
        if unknown_trainable:
            raise KeyError(
                f"Unknown trainable parameter(s): {', '.join(unknown_trainable)}"
            )
        if "vpr25" in trainable_names:
            raise ValueError(
                "vpr25 cannot be trainable because the preserved pyLeaf equations "
                "do not use it; the literal PEP-regeneration cap remains 100"
            )

        if energy_balance is None:
            energy_balance = (
                True if parameter_energy_switch is None else bool(parameter_energy_switch)
            )

        for name, value in values.items():
            spec = PARAMETER_SPECS[name]
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if spec.transform == "positive" and value <= spec.lower:
                raise ValueError(f"{name} must be greater than {spec.lower}")
            if spec.transform == "bounded" and (
                value < spec.lower or value > spec.upper
            ):
                raise ValueError(
                    f"{name} must be between {spec.lower} and {spec.upper}"
                )
            if (
                name in trainable_names
                and spec.transform == "bounded"
                and (value <= spec.lower or value >= spec.upper)
            ):
                raise ValueError(
                    f"Trainable {name} must be strictly between "
                    f"{spec.lower} and {spec.upper}"
                )

        self.mode: Mode = mode
        self.energy_balance = bool(energy_balance)
        self.solver_options = solver_options or SolverOptions()
        self.boundary_iterations = boundary_iterations
        self.raw_parameters = nn.ParameterDict()
        self._trainable_names = trainable_names

        factory = {"dtype": dtype, "device": device}
        for name, value in values.items():
            physical = torch.tensor(value, **factory)
            spec = PARAMETER_SPECS[name]
            if not bool(torch.isfinite(physical)):
                raise ValueError(f"{name} is not finite in the requested dtype")
            if spec.transform == "positive" and bool(physical <= spec.lower):
                raise ValueError(f"{name} is not positive in the requested dtype")
            if name in trainable_names and spec.transform == "bounded":
                assert spec.upper is not None
                fraction = (physical - spec.lower) / (spec.upper - spec.lower)
                margin = torch.finfo(physical.dtype).eps
                if bool(fraction < margin) or bool(fraction > 1.0 - margin):
                    raise ValueError(
                        f"Trainable {name} is too close to a transform boundary "
                        f"for dtype {physical.dtype}"
                    )
            if name in trainable_names:
                raw = to_raw(physical, spec)
                self.raw_parameters[name] = nn.Parameter(raw)
            else:
                self.register_buffer(f"_fixed_{name}", physical)

        self.register_buffer(
            "state_scale",
            torch.tensor([10.0, 1000.0, 100.0, 0.1, 100.0, 10.0], **factory),
        )
        self.register_buffer(
            "state_lower",
            torch.tensor([-20.0, 0.1, 0.1, 1.0e-4, 0.1, -20.0], **factory),
        )
        self.register_buffer(
            "state_upper",
            torch.tensor([150.0, 20000.0, 5000.0, 5.0, 5000.0, 70.0], **factory),
        )

    @property
    def trainable_parameter_names(self) -> tuple[str, ...]:
        return self._trainable_names

    def physical_parameters(self, *, detach: bool = False) -> dict[str, Tensor]:
        """Return all parameters in physical units."""

        values: dict[str, Tensor] = {}
        for name in DEFAULT_PARAMETERS:
            if name in self.raw_parameters:
                value = from_raw(self.raw_parameters[name], PARAMETER_SPECS[name])
            else:
                value = getattr(self, f"_fixed_{name}")
            values[name] = value.detach() if detach else value
        return values

    def parameter_report(self) -> dict[str, float]:
        """Return detached physical parameters for logging or serialization."""

        return {
            name: float(value.detach().cpu())
            for name, value in self.physical_parameters().items()
        }

    def set_physical_parameter_(self, name: str, value: float) -> None:
        """Update one parameter in place, preserving its constraint transform."""

        if name not in DEFAULT_PARAMETERS:
            raise KeyError(f"Unknown pyLeaf parameter: {name}")
        reference = self.state_scale
        physical = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
        spec = PARAMETER_SPECS[name]
        if not bool(torch.isfinite(physical)):
            raise ValueError(f"{name} must be finite")
        if spec.transform == "positive" and bool(physical <= spec.lower):
            raise ValueError(f"{name} must be greater than {spec.lower}")
        if spec.transform == "bounded":
            outside = bool(physical < spec.lower) or bool(physical > spec.upper)
            endpoint_trainable = name in self.raw_parameters and (
                bool(physical <= spec.lower) or bool(physical >= spec.upper)
            )
            if outside or endpoint_trainable:
                qualifier = "strictly " if name in self.raw_parameters else ""
                raise ValueError(
                    f"{name} must be {qualifier}between {spec.lower} and {spec.upper}"
                )
        with torch.no_grad():
            if name in self.raw_parameters:
                self.raw_parameters[name].copy_(to_raw(physical, spec))
            else:
                getattr(self, f"_fixed_{name}").copy_(physical)

    def _prepare_weather(self, weather: Mapping[str, Any]) -> dict[str, Tensor]:
        required = list(self.WEATHER_COLUMNS)
        if not self.energy_balance:
            required.append("controlTemp")
        missing = [name for name in required if name not in weather]
        if missing:
            raise KeyError(f"Missing weather field(s): {', '.join(missing)}")

        tensors: list[Tensor] = []
        for name in required:
            value = torch.as_tensor(
                weather[name], dtype=self.state_scale.dtype, device=self.state_scale.device
            )
            if value.ndim == 0:
                value = value.reshape(1)
            if value.ndim != 1:
                raise ValueError(f"Weather field {name!r} must be scalar or one-dimensional")
            tensors.append(value)
        tensors = list(torch.broadcast_tensors(*tensors))
        prepared = dict(zip(required, tensors, strict=True))

        detached = {name: value.detach() for name, value in prepared.items()}
        if any(not bool(torch.isfinite(value).all()) for value in detached.values()):
            raise ValueError("Weather fields must contain only finite values")
        if bool((detached["pressure"] <= 0).any()):
            raise ValueError("Weather field 'pressure' must be positive")
        if bool((detached["wind"] <= 0).any()):
            raise ValueError(
                "Weather field 'wind' must be positive; zero wind can create "
                "an undefined zero-transfer boundary state"
            )
        if bool((detached["ea"] < 0).any()):
            raise ValueError("Weather field 'ea' must be non-negative")
        return prepared

    @staticmethod
    def _safe_denominator(value: Tensor, epsilon: float = 1.0e-10) -> Tensor:
        sign = torch.where(value >= 0, torch.ones_like(value), -torch.ones_like(value))
        return torch.where(value.abs() < epsilon, sign * epsilon, value)

    def _minimum(self, left: Tensor, right: Tensor, tau: float) -> Tensor:
        if self.mode == "hard":
            return torch.minimum(left, right)
        return -tau * torch.logaddexp(-left / tau, -right / tau)

    def _maximum(self, left: Tensor, right: Tensor, tau: float) -> Tensor:
        if self.mode == "hard":
            return torch.maximum(left, right)
        return tau * torch.logaddexp(left / tau, right / tau)

    def _absolute(self, value: Tensor, epsilon: float) -> Tensor:
        if self.mode == "hard":
            return value.abs()
        return torch.sqrt(value.square() + epsilon**2)

    @staticmethod
    def _saturation_vapour_pressure(temperature: Tensor) -> Tensor:
        return 611.0 * torch.exp(17.502 * temperature / (240.97 + temperature))

    def _temperature_response(
        self, temperature: Tensor, parameters: Mapping[str, Tensor]
    ) -> dict[str, Tensor]:
        kelvin = temperature + 273.15
        reference_kelvin = 298.15
        gas_constant = 8.314

        Ko = parameters["Ko25"] * 1.2 ** ((temperature - 25.0) / 10.0)
        Kc = parameters["Kc25"] * 2.1 ** ((temperature - 25.0) / 10.0)
        Kp = parameters["Kp25"] * 2.1 ** ((temperature - 25.0) / 10.0)

        def peaked_response(
            base: Tensor, activation: float, entropy: float, deactivation: float
        ) -> Tensor:
            activation_term = torch.exp(
                activation
                * (kelvin - reference_kelvin)
                / (reference_kelvin * gas_constant * kelvin)
            )
            reference_deactivation = 1.0 + torch.exp(
                torch.as_tensor(
                    (reference_kelvin * entropy - deactivation)
                    / (reference_kelvin * gas_constant),
                    dtype=kelvin.dtype,
                    device=kelvin.device,
                )
            )
            current_deactivation = 1.0 + torch.exp(
                (kelvin * entropy - deactivation) / (kelvin * gas_constant)
            )
            return base * activation_term * reference_deactivation / current_deactivation

        jmax = peaked_response(parameters["jmax25"], 77900.0, 627.0, 191929.0)
        vcmax = peaked_response(parameters["vcmax25"], 67294.0, 472.0, 144568.0)
        vpmax = peaked_response(parameters["vpmax25"], 70373.0, 376.0, 117910.0)
        sco = parameters["sco25"] * 1.0e-3 * torch.exp(
            -55900.0
            * (kelvin - reference_kelvin)
            / (reference_kelvin * gas_constant * kelvin)
        )
        gamma_star_ratio = 1.0 / (2.0 * sco)
        rd = parameters["rd25"] * vcmax
        return {
            "ei": self._saturation_vapour_pressure(temperature),
            "Ko": Ko,
            "Kc": Kc,
            "Kp": Kp,
            "jmax": jmax,
            "vcmax": vcmax,
            "vpmax": vpmax,
            "gammaStarRatio": gamma_star_ratio,
            "rd": rd,
            "rm": 0.5 * rd,
            "rbs": 0.5 * rd,
        }

    def _derived(
        self,
        state: Tensor,
        weather: Mapping[str, Tensor],
        parameters: Mapping[str, Tensor],
    ) -> dict[str, Tensor]:
        a_net, cbs, ci, gs, cb, t_leaf = state.unbind(dim=-1)
        response = self._temperature_response(t_leaf, parameters)
        temperature_kelvin = weather["tAir"] + 273.15
        leaf_kelvin = t_leaf + 273.15
        mb = 0.5 * (1.0 + parameters["s"]) ** 2 / (1.0 + parameters["s"] ** 2)
        conversion_gb = weather["pressure"] / (8.309 * temperature_kelvin)

        gb_forced = parameters["cForced"] * temperature_kelvin**0.56 * (
            (temperature_kelvin + 120.0)
            * (weather["wind"] / parameters["leafDim"] / weather["pressure"])
        ) ** 0.5

        gb_mps = gb_forced
        gb = gb_mps * conversion_gb
        eb = (gs * response["ei"] + gb * mb * weather["ea"]) / self._safe_denominator(
            gs + gb * mb
        )
        gb_free = torch.zeros_like(gb_forced)
        for _ in range(self.boundary_iterations):
            virtual_difference = leaf_kelvin / self._safe_denominator(
                1.0 - 0.378 * eb / weather["pressure"]
            ) - temperature_kelvin / self._safe_denominator(
                1.0 - 0.378 * weather["ea"] / weather["pressure"]
            )
            gb_free = parameters["cFree"] * leaf_kelvin**0.56 * (
                ((leaf_kelvin + 120.0) / weather["pressure"]) ** 0.5
                * (
                    self._absolute(virtual_difference, 1.0e-3)
                    / parameters["leafDim"]
                )
                ** 0.25
            )
            gb_mps = self._maximum(gb_free, gb_forced, 1.0e-6)
            gb = gb_mps * conversion_gb
            eb = (
                gs * response["ei"] + gb * mb * weather["ea"]
            ) / self._safe_denominator(gs + gb * mb)

        total_conductance = gs * gb * mb / self._safe_denominator(gs + gb * mb)
        phi_ps2 = 0.352 + 0.022 * t_leaf - 3.4 * t_leaf.square() / 10000.0
        light = weather["PAR"] * 4.57 * phi_ps2 / 2.0
        electron_discriminant = (light + response["jmax"]) ** 2 - (
            4.0 * parameters["theta"] * light * response["jmax"]
        )
        electron_discriminant = torch.clamp(
            electron_discriminant, min=torch.finfo(state.dtype).tiny
        )
        electron_transport = (
            light + response["jmax"] - torch.sqrt(electron_discriminant)
        ) / (2.0 * parameters["theta"])

        obs = (
            parameters["alpha"] * a_net / (0.047 * parameters["gbs"] * 1000.0)
            + weather["O2"]
        )
        gamma_star = response["gammaStarRatio"] * obs
        vc_light = cbs * (1.0 - parameters["x"]) * electron_transport / (
            3.0 * cbs + 7.0 * gamma_star
        )
        vc_co2 = cbs * response["vcmax"] / (
            cbs + response["Kc"] * (1.0 + obs / response["Ko"])
        )
        vc = self._minimum(vc_co2, vc_light, 0.05)

        vp_light = parameters["x"] * electron_transport / 2.0
        vp_co2_uncapped = ci * response["vpmax"] / (ci + response["Kp"])
        hundred = torch.full_like(vp_co2_uncapped, 100.0)
        vp_co2 = self._minimum(vp_co2_uncapped, hundred, 0.05)
        vp = self._minimum(vp_light, vp_co2, 0.05)

        if self.mode == "hard":
            enzyme_weight = (vc_co2 < vc_light).to(state.dtype)
        else:
            enzyme_weight = torch.sigmoid((vc_light - vc_co2) / 0.05)
        a1 = enzyme_weight * response["vcmax"] + (1.0 - enzyme_weight) * (
            (1.0 - parameters["x"]) * electron_transport / 3.0
        )
        b1 = enzyme_weight * (
            response["Kc"] * (1.0 + obs / response["Ko"])
        ) + (1.0 - enzyme_weight) * (7.0 * gamma_star / 3.0)

        gamma_c = (
            gamma_star
            + response["Kc"]
            * (1.0 + obs / response["Ko"])
            * response["rd"]
            / response["vcmax"]
        ) / (1.0 - response["rd"] / response["vcmax"])
        gamma_c = self._maximum(gamma_c, torch.zeros_like(gamma_c), 0.01)

        quadratic_a = parameters["gbs"]
        quadratic_b = (
            parameters["gbs"] * (response["Kp"] - gamma_c)
            + response["vpmax"]
            - response["rm"]
        )
        quadratic_c = -(
            gamma_c * response["Kp"] * parameters["gbs"]
            + response["rm"] * response["Kp"]
        )
        gamma_discriminant = torch.clamp(
            quadratic_b.square() - 4.0 * quadratic_a * quadratic_c,
            min=torch.finfo(state.dtype).tiny,
        )
        gamma_sqrt = torch.sqrt(gamma_discriminant)
        gamma_standard = (-quadratic_b + gamma_sqrt) / (2.0 * quadratic_a)
        gamma_rational = (-2.0 * quadratic_c) / self._safe_denominator(
            quadratic_b + gamma_sqrt
        )
        gamma = torch.where(quadratic_b >= 0.0, gamma_rational, gamma_standard)

        stomatal_unbounded = (
            parameters["go"]
            + parameters["g1"]
            * a_net
            * eb
            / response["ei"]
            / self._safe_denominator(cb - gamma)
        )
        stomatal_target = self._maximum(
            stomatal_unbounded, parameters["go"].expand_as(stomatal_unbounded), 1.0e-5
        )

        latent_heat_molar = (2500.0 - 2.36 * t_leaf) * 18.0
        sensible = 29.3 * (gb * 0.924) * (t_leaf - weather["tAir"])
        latent = (
            latent_heat_molar
            * total_conductance
            / weather["pressure"]
            * (response["ei"] - weather["ea"])
        )
        emission = 2.0 * 0.94 * 5.67e-8 * leaf_kelvin**4
        radiation = weather["PAR"] + weather["NIR"] + weather["long"]
        energy_residual = radiation - 0.506 * a_net - sensible - latent - emission

        carboxylation_factor = 1.0 - response["gammaStarRatio"] * obs / cbs
        return {
            **response,
            "aNet": a_net,
            "cbs": cbs,
            "ci": ci,
            "gs": gs,
            "cb": cb,
            "tLeaf": t_leaf,
            "Mb": mb,
            "eb": eb,
            "gbForced": gb_forced,
            "gbFree": gb_free,
            "gb": gb,
            "g": total_conductance,
            "J": electron_transport,
            "obs": obs,
            "GammaStar": gamma_star,
            "Gamma_C": gamma_c,
            "Gamma": gamma,
            "vcLight": vc_light,
            "vcCO2": vc_co2,
            "vc": vc,
            "vpLight": vp_light,
            "vpCO2": vp_co2,
            "vpCO2Uncapped": vp_co2_uncapped,
            "vp": vp,
            "a1": a1,
            "b1": b1,
            "enzymeWeight": enzyme_weight,
            "stomatalUnbounded": stomatal_unbounded,
            "stomatalTarget": stomatal_target,
            "acCO2": carboxylation_factor * vc_co2 - response["rd"],
            "acLight": carboxylation_factor * vc_light - response["rd"],
            "apCO2": carboxylation_factor * vp_co2 - response["rd"],
            "apLight": carboxylation_factor * vp_light - response["rd"],
            "H": sensible,
            "LE": latent,
            "emission": emission,
            "radiation": radiation,
            "energyResidual": energy_residual,
            "latentHeatMolar": latent_heat_molar,
        }

    def _raw_residual(
        self,
        state: Tensor,
        weather: Mapping[str, Tensor],
        parameters: Mapping[str, Tensor],
        derived: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        values = self._derived(state, weather, parameters) if derived is None else derived
        r_a = values["aNet"] - (
            1.0 - values["GammaStar"] / values["cbs"]
        ) * (values["cbs"] * values["a1"] / (values["cbs"] + values["b1"])) + values[
            "rd"
        ]
        r_cbs = values["cbs"] - values["ci"] - (
            values["vp"] - values["aNet"] - values["rm"]
        ) / parameters["gbs"]
        r_ci = values["ci"] - values["cb"] + 1.6 * values["aNet"] / values["gs"]
        r_gs = values["gs"] - values["stomatalTarget"]
        r_cb = (
            values["cb"]
            - weather["ca"]
            + 1.37
            * values["aNet"]
            / (values["gb"] * values["Mb"])
        )
        if self.energy_balance:
            r_temperature = values["energyResidual"]
        else:
            r_temperature = values["tLeaf"] - weather["controlTemp"]
        return torch.stack((r_a, r_cbs, r_ci, r_gs, r_cb, r_temperature), dim=-1)

    def _scaled_residual(
        self,
        scaled_state: Tensor,
        weather: Mapping[str, Tensor],
        parameters: Mapping[str, Tensor],
    ) -> Tensor:
        state = scaled_state * self.state_scale
        raw = self._raw_residual(state, weather, parameters)
        residual_scale = torch.tensor(
            [10.0, 1000.0, 100.0, 0.1, 100.0, 100.0],
            dtype=state.dtype,
            device=state.device,
        )
        if not self.energy_balance:
            residual_scale = residual_scale.clone()
            residual_scale[-1] = 10.0
        return raw / residual_scale

    def _initial_state(self, weather: Mapping[str, Tensor]) -> Tensor:
        temperature = (
            weather["tAir"] if self.energy_balance else weather["controlTemp"]
        )
        state = torch.stack(
            (
                0.1 * weather["ca"],
                10.0 * weather["ca"],
                0.7 * weather["ca"],
                torch.full_like(weather["ca"], 0.1),
                0.8 * weather["ca"],
                temperature,
            ),
            dim=-1,
        )
        return state / self.state_scale

    def _state_bounds(self) -> tuple[Tensor, Tensor]:
        lower = self.state_lower.clone()
        upper = self.state_upper.clone()
        if self.energy_balance:
            lower[-1] = 2.0
            upper[-1] = 60.0
        return lower / self.state_scale, upper / self.state_scale

    def _build_output(
        self,
        scaled_state: Tensor,
        weather: Mapping[str, Tensor],
        parameters: Mapping[str, Tensor],
        *,
        iterations: Tensor,
        jacobian_condition: Tensor,
        line_search_failures: Tensor,
        at_state_bound: Tensor,
    ) -> LeafOutput:
        state_tensor = scaled_state * self.state_scale
        values = self._derived(state_tensor, weather, parameters)
        scaled_residual = self._scaled_residual(scaled_state, weather, parameters)
        residual_norm = torch.linalg.vector_norm(
            scaled_residual.detach(), ord=float("inf"), dim=-1
        )
        converged = residual_norm <= self.solver_options.residual_tolerance

        state = {
            "flag": (~converged).to(state_tensor.dtype),
            "cbs": values["cbs"],
            "obs": values["obs"],
            "ci": values["ci"],
            "cb": values["cb"],
            "gs": values["gs"],
            "tLeaf": values["tLeaf"],
            "ei": values["ei"],
            "eb": values["eb"],
            "gbForced": values["gbForced"],
            "gbFree": values["gbFree"],
            "gb": values["gb"],
            "g": values["g"],
            "Ko": values["Ko"],
            "Kc": values["Kc"],
            "Kp": values["Kp"],
            "gammaStar": values["gammaStarRatio"],
            "GammaStar": values["GammaStar"],
            "Gamma": values["Gamma"],
            "Gamma_C": values["Gamma_C"],
        }
        mass = {
            "rd": values["rd"],
            "rbs": values["rbs"],
            "rm": values["rm"],
            "J": values["J"],
            "aNet": values["aNet"],
            "aGross": values["aNet"] + values["rd"],
            "acCO2": values["acCO2"],
            "acLight": values["acLight"],
            "apCO2": values["apCO2"],
            "apLight": values["apLight"],
            "L": parameters["gbs"] * (values["cbs"] - values["ci"]),
            "vp": values["vp"],
            "vc": values["vc"],
            "vpLight": values["vpLight"],
            "vpCO2": values["vpCO2"],
            "vcLight": values["vcLight"],
            "vcCO2": values["vcCO2"],
            "vpmax": values["vpmax"],
            "vcmax": values["vcmax"],
            "jmax": values["jmax"],
            "transpiration": values["LE"] / values["latentHeatMolar"] * 1.0e6,
        }
        energy = {
            "H": values["H"],
            "LE": values["LE"],
            "emission": values["emission"],
            "radiation": values["radiation"],
            "residual": values["energyResidual"],
        }
        limitation = {
            "enzymeWeight": values["enzymeWeight"],
            "vcCO2Limited": (values["vcCO2"] < values["vcLight"]),
            "vpCO2Limited": (values["vpCO2"] < values["vpLight"]),
            "forcedConvection": (values["gbForced"] >= values["gbFree"]),
            "vpRegenerationCapped": (values["vpCO2Uncapped"] >= 100.0),
            "stomatalFloor": (values["stomatalUnbounded"] <= parameters["go"]),
        }
        diagnostics = SolverDiagnostics(
            converged=converged.detach(),
            iterations=iterations.detach(),
            residual_norm=residual_norm.detach(),
            jacobian_condition=jacobian_condition.detach(),
            line_search_failures=line_search_failures.detach(),
            at_state_bound=at_state_bound.detach(),
        )
        return LeafOutput(
            state=state,
            mass=mass,
            energy=energy,
            diagnostics=diagnostics,
            residual=scaled_residual,
            limitation=limitation,
        )

    def equilibrium_residual(
        self,
        state: Tensor | Mapping[str, Any],
        weather: Mapping[str, Any],
        *,
        scaled: bool = True,
    ) -> Tensor:
        """Evaluate all coupled equations at a supplied physical state.

        This is useful for checking externally generated states. State columns
        are ordered as ``aNet, cbs, ci, gs, cb, tLeaf`` when a tensor is
        supplied. The operation remains differentiable.
        """

        prepared = self._prepare_weather(weather)
        if isinstance(state, Mapping):
            missing = [name for name in self.STATE_NAMES if name not in state]
            if missing:
                raise KeyError(f"Missing state field(s): {', '.join(missing)}")
            components = [
                torch.as_tensor(
                    state[name],
                    dtype=self.state_scale.dtype,
                    device=self.state_scale.device,
                )
                for name in self.STATE_NAMES
            ]
            components = [value.reshape(1) if value.ndim == 0 else value for value in components]
            invalid_dimensions = [
                name
                for name, value in zip(self.STATE_NAMES, components, strict=True)
                if value.ndim != 1
            ]
            if invalid_dimensions:
                raise ValueError(
                    "State fields must be scalar or one-dimensional; invalid: "
                    + ", ".join(invalid_dimensions)
                )
            components = list(torch.broadcast_tensors(*components))
            state_tensor = torch.stack(components, dim=-1)
        else:
            state_tensor = torch.as_tensor(
                state, dtype=self.state_scale.dtype, device=self.state_scale.device
            )
            if state_tensor.ndim == 1:
                state_tensor = state_tensor.unsqueeze(0)
            if state_tensor.ndim != 2 or state_tensor.shape[-1] != len(self.STATE_NAMES):
                raise ValueError("State tensor must have shape [rows, 6]")
        if state_tensor.shape[0] != prepared["ca"].shape[0]:
            raise ValueError("State and weather row counts must match")

        parameters = self.physical_parameters()
        if scaled:
            return self._scaled_residual(state_tensor / self.state_scale, prepared, parameters)
        return self._raw_residual(state_tensor, prepared, parameters)

    def forward(self, weather: Mapping[str, Any]) -> LeafOutput:
        if torch.is_inference_mode_enabled():
            # Autodiff is needed internally to assemble the equilibrium Jacobian
            # even though the returned inference result carries no graph.
            with torch.inference_mode(False), torch.no_grad():
                ordinary_weather = {
                    name: value.clone() if isinstance(value, Tensor) else value
                    for name, value in weather.items()
                }
                return self.forward(ordinary_weather)

        prepared = self._prepare_weather(weather)
        detached_weather = {name: value.detach() for name, value in prepared.items()}
        detached_parameters = self.physical_parameters(detach=True)
        initial = self._initial_state(detached_weather)
        lower, upper = self._state_bounds()

        detached_residual = lambda x: self._scaled_residual(
            x, detached_weather, detached_parameters
        )
        root = solve_root(
            detached_residual,
            initial,
            lower,
            upper,
            self.solver_options,
        )
        if not bool(root.converged.all()):
            alternative_assimilation = (
                torch.clamp(0.08 * detached_weather["PAR"] - 2.0, min=-10.0, max=100.0),
                0.03 * detached_weather["ca"],
                torch.zeros_like(detached_weather["ca"]),
            )
            for assimilation in alternative_assimilation:
                alternative_initial = initial.clone()
                alternative_initial[:, 0] = assimilation / self.state_scale[0]
                alternate = solve_root(
                    detached_residual,
                    alternative_initial,
                    lower,
                    upper,
                    self.solver_options,
                )
                root = select_better_root(root, alternate)
                if bool(root.converged.all()):
                    break

        live_parameters = self.physical_parameters()
        gradient_source = any(
            parameter.requires_grad for parameter in self.raw_parameters.values()
        ) or any(value.requires_grad for value in prepared.values())
        if torch.is_grad_enabled() and gradient_source:
            invalid_gradient = ~root.converged | root.at_state_bound
            if bool(invalid_gradient.any()):
                rows = torch.nonzero(invalid_gradient, as_tuple=False).flatten().tolist()
                raise RuntimeError(
                    "Implicit gradients require a converged interior root; "
                    f"invalid row indices: {rows}. Use torch.no_grad() to inspect "
                    "the flagged numerical output."
                )
            live_residual = lambda x: self._scaled_residual(
                x, prepared, live_parameters
            )
            scaled_state = implicit_root(live_residual, root, self.solver_options)
        else:
            scaled_state = root.x

        return self._build_output(
            scaled_state,
            prepared,
            live_parameters,
            iterations=root.iterations,
            jacobian_condition=root.jacobian_condition,
            line_search_failures=root.line_search_failures,
            at_state_bound=root.at_state_bound,
        )
