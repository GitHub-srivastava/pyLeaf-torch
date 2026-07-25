"""Plot pyLeaf-torch A-Ci and A-Q response curves.

One input-workbook row supplies the fixed environmental conditions. Two
controlled sweeps are constructed and passed to the differentiable model:
ambient CO2 for A-Ci, and irradiance for A-Q.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

# Allow this example to run directly from a source checkout without requiring
# an editable installation of pyleaf-torch first.
REPOSITORY = Path(__file__).resolve().parents[1]
SOURCE = REPOSITORY / "src"
if str(SOURCE) not in sys.path:
    sys.path.insert(0, str(SOURCE))

REQUIRED_PACKAGES = ("numpy", "pandas", "torch", "matplotlib", "openpyxl")
missing_packages = [
    package
    for package in REQUIRED_PACKAGES
    if importlib.util.find_spec(package) is None
]
if missing_packages:
    requirements = REPOSITORY / "requirements-plot.txt"
    raise SystemExit(
        "Missing plotting dependencies: "
        + ", ".join(missing_packages)
        + "\nInstall them with the same Python interpreter:\n  "
        + f'{sys.executable} -m pip install -r "{requirements}"'
    )

import numpy as np
import pandas as pd
import torch

from pyleaf_torch import Leaf, simulate_dataframe


PAR_TO_Q = 4.57  # micromol photons per joule, also used inside pyLeaf


def curve_input(
    base: pd.Series,
    values: np.ndarray,
    column: str,
    *,
    scale_nir: bool = False,
) -> pd.DataFrame:
    """Repeat one forcing row and replace the requested sweep variable."""
    frame = pd.DataFrame([base.to_dict()] * len(values))
    frame[column] = values
    if scale_nir:
        base_par = float(base["PAR"])
        if base_par <= 0.0:
            raise ValueError("The base row must have PAR > 0 to scale NIR")
        frame["NIR"] = values * float(base["NIR"]) / base_par
    return frame


def run_curve(
    weather: pd.DataFrame,
    parameters: dict[str, Any],
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the differentiable model and return its state and diagnostics."""
    model = Leaf(
        parameters=parameters,
        trainable=(),
        mode=mode,
        dtype=torch.float64,
    )
    _, frames = simulate_dataframe(model, weather)

    result = pd.DataFrame(
        {
            "aNet": frames.mass["aNet"],
            "ci": frames.state["ci"],
        }
    )
    return result, frames.diagnostics


def parse_parameters(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, dict):
        raise ValueError("The parameter JSON must contain one object")
    return values


def save_plot(aci: pd.DataFrame, aq: pd.DataFrame, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)

    axes[0].plot(aci["ci"], aci["aNet"], "-", lw=2, label="pyLeaf-torch")
    axes[0].set_xlabel(r"Intercellular CO$_2$, $C_i$ [$\mu$mol mol$^{-1}$]")
    axes[0].set_ylabel(r"Net assimilation, $A$ [$\mu$mol m$^{-2}$ s$^{-1}$]")
    axes[0].set_title(r"A--$C_i$ response")

    axes[1].plot(aq["Q"], aq["aNet"], "-", lw=2, label="pyLeaf-torch")
    axes[1].set_xlabel(r"Photon flux, $Q$ [$\mu$mol photons m$^{-2}$ s$^{-1}$]")
    axes[1].set_ylabel(r"Net assimilation, $A$ [$\mu$mol m$^{-2}$ s$^{-1}$]")
    axes[1].set_title("A--Q response")

    for axis in axes:
        axis.axhline(0.0, color="0.75", lw=0.8)
        axis.grid(alpha=0.2)
        axis.legend(frameon=False)

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=REPOSITORY / "examples" / "data" / "Input.xlsx"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY / "curve_output"
    )
    parser.add_argument(
        "--base-row",
        type=int,
        default=-1,
        help="zero-based workbook row used for fixed conditions (default: last)",
    )
    parser.add_argument("--points", type=int, default=31)
    parser.add_argument("--ca-min", type=float, default=50.0)
    parser.add_argument("--ca-max", type=float, default=1200.0)
    parser.add_argument("--par-min", type=float, default=0.0)
    parser.add_argument("--par-max", type=float, default=600.0)
    parser.add_argument("--mode", choices=("hard", "smooth"), default="hard")
    parser.add_argument(
        "--parameters",
        type=Path,
        help="optional JSON object of parameter overrides",
    )
    parser.add_argument("--dpi", type=int, default=200)
    args = parser.parse_args()

    if args.points < 2:
        parser.error("--points must be at least 2")
    if args.ca_min <= 0.0 or args.ca_max <= args.ca_min:
        parser.error("require 0 < --ca-min < --ca-max")
    if args.par_min < 0.0 or args.par_max <= args.par_min:
        parser.error("require 0 <= --par-min < --par-max")

    source = pd.read_excel(args.input)
    try:
        base = source.iloc[args.base_row].copy()
    except IndexError as error:
        parser.error(f"--base-row {args.base_row} is outside a {len(source)}-row input")
        raise AssertionError from error
    parameters = parse_parameters(args.parameters)

    ca_values = np.linspace(args.ca_min, args.ca_max, args.points)
    par_values = np.linspace(args.par_min, args.par_max, args.points)
    aci_weather = curve_input(base, ca_values, "ca")
    aq_weather = curve_input(base, par_values, "PAR", scale_nir=True)

    aci_result, diag_aci = run_curve(aci_weather, parameters, args.mode)
    aq_result, diag_aq = run_curve(aq_weather, parameters, args.mode)

    aci = pd.DataFrame(
        {
            "ca": ca_values,
            "ci": aci_result["ci"],
            "aNet": aci_result["aNet"],
            "converged": diag_aci["converged"],
            "residual_norm": diag_aci["residual_norm"],
        }
    )
    aq = pd.DataFrame(
        {
            "PAR": par_values,
            "Q": par_values * PAR_TO_Q,
            "NIR": aq_weather["NIR"],
            "ci": aq_result["ci"],
            "aNet": aq_result["aNet"],
            "converged": diag_aq["converged"],
            "residual_norm": diag_aq["residual_norm"],
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aci.to_csv(args.output_dir / "A_Ci_curve.csv", index=False)
    aq.to_csv(args.output_dir / "A_Q_curve.csv", index=False)
    figure = args.output_dir / "A_Ci_A_Q_curves.png"
    save_plot(aci, aq, figure, args.dpi)

    print(f"Base input row: {args.base_row} ({args.input})")
    print(
        "Solver status: converged "
        f"{int(aci['converged'].sum() + aq['converged'].sum())}/{2 * args.points} "
        "curve points"
    )
    print(f"Wrote figure and curve data to {args.output_dir}")


if __name__ == "__main__":
    main()
