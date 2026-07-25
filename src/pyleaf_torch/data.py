"""Optional pandas adapters kept outside the differentiable tensor core."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping

import torch
from torch import Tensor

from .outputs import LeafOutput

if TYPE_CHECKING:
    import pandas as pd

    from .model import Leaf


@dataclass(frozen=True)
class OutputFrames:
    state: "pd.DataFrame"
    mass: "pd.DataFrame"
    energy: "pd.DataFrame"
    diagnostics: "pd.DataFrame"
    limitation: "pd.DataFrame"


def weather_from_dataframe(
    frame: "pd.DataFrame", *, dtype: torch.dtype = torch.float64
) -> dict[str, Tensor]:
    """Convert numeric DataFrame columns to one-dimensional CPU tensors."""

    required = {
        "plant",
        "ca",
        "O2",
        "tAir",
        "ea",
        "pressure",
        "wind",
        "PAR",
        "long",
        "NIR",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Missing weather column(s): {', '.join(missing)}")
    if not bool((frame["plant"] == 4).all()):
        raise ValueError(
            "The available pyLeaf equations are C4-only; every 'plant' value must be 4"
        )

    columns = required | ({"controlTemp"} if "controlTemp" in frame.columns else set())
    result: dict[str, Tensor] = {}
    for name in sorted(columns):
        numeric = frame[name]
        if not bool(numeric.notna().all()):
            if name == "controlTemp":
                continue
            raise ValueError(f"Weather column {name!r} contains missing values")
        try:
            result[name] = torch.as_tensor(numeric.to_numpy(copy=True), dtype=dtype)
        except (TypeError, ValueError) as error:
            raise ValueError(f"Weather column {name!r} must be numeric") from error
    return result


def _frame(mapping: Mapping[str, Tensor]) -> "pd.DataFrame":
    import pandas as pd

    return pd.DataFrame(
        {
            name: value.detach().cpu().numpy()
            for name, value in mapping.items()
        }
    )


def output_to_dataframes(output: LeafOutput) -> OutputFrames:
    """Detach an output explicitly and construct reporting DataFrames."""

    import pandas as pd

    diagnostics_values: dict[str, Any] = {
            "converged": output.diagnostics.converged.cpu().numpy(),
            "iterations": output.diagnostics.iterations.cpu().numpy(),
            "residual_norm": output.diagnostics.residual_norm.cpu().numpy(),
            "jacobian_condition": output.diagnostics.jacobian_condition.cpu().numpy(),
            "line_search_failures": output.diagnostics.line_search_failures.cpu().numpy(),
            "at_state_bound": output.diagnostics.at_state_bound.cpu().numpy(),
    }
    residual_names = (
        ("aNet", "cbs", "ci", "cm", "gs", "cb", "tLeaf")
        if output.residual.shape[-1] == 7
        else ("aNet", "cbs", "ci", "gs", "cb", "tLeaf")
    )
    diagnostics_values.update(
        {
            f"scaled_residual_{name}": output.residual[:, index].detach().cpu().numpy()
            for index, name in enumerate(residual_names)
        }
    )
    diagnostics = pd.DataFrame(diagnostics_values)
    return OutputFrames(
        state=_frame(output.state),
        mass=_frame(output.mass),
        energy=_frame(output.energy),
        diagnostics=diagnostics,
        limitation=_frame(output.limitation),
    )


def simulate_dataframe(
    model: "Leaf", frame: "pd.DataFrame"
) -> tuple[LeafOutput, OutputFrames]:
    """Run a DataFrame and return both graph-carrying tensors and detached tables."""

    weather = weather_from_dataframe(frame, dtype=model.state_scale.dtype)
    if not model.energy_balance and "controlTemp" not in weather:
        raise ValueError(
            "Fixed-temperature mode requires a non-missing 'controlTemp' column"
        )
    output = model(weather)
    return output, output_to_dataframes(output)
