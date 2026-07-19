"""Compare pyLeaf and pyLeaf-torch A-Ci and A-Q response curves.

One input-workbook row supplies the fixed environmental conditions. Two
controlled sweeps are constructed and passed unchanged to both models:
ambient CO2 for A-Ci, and irradiance for A-Q.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from compare_models import load_legacy
from pyleaf_torch import DifferentiableLeaf, simulate_dataframe


REPOSITORY = Path(__file__).resolve().parents[1]
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


def run_pair(
    weather: pd.DataFrame,
    parameters: dict[str, Any],
    mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run both implementations and return legacy, Torch, and diagnostics."""
    legacy = load_legacy().Leaf(parameters)
    legacy.SeriesSolver(weather)

    torch_model = DifferentiableLeaf(
        parameters=parameters,
        trainable=(),
        mode=mode,
        dtype=torch.float64,
    )
    _, torch_frames = simulate_dataframe(torch_model, weather)

    legacy_result = pd.DataFrame(
        {
            "aNet": legacy.LeafMassFlux["aNet"],
            "ci": legacy.LeafState["ci"],
            "flagged": legacy.LeafState["flag"].astype(bool),
        }
    )
    torch_result = pd.DataFrame(
        {
            "aNet": torch_frames.mass["aNet"],
            "ci": torch_frames.state["ci"],
        }
    )
    return legacy_result, torch_result, torch_frames.diagnostics


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

    axes[0].plot(
        aci["legacy_ci"], aci["legacy_aNet"], "o-", ms=3.5, label="pyLeaf"
    )
    axes[0].plot(
        aci["torch_ci"], aci["torch_aNet"], "--", lw=2, label="pyLeaf-torch"
    )
    axes[0].set_xlabel(r"Intercellular CO$_2$, $C_i$ [$\mu$mol mol$^{-1}$]")
    axes[0].set_ylabel(r"Net assimilation, $A$ [$\mu$mol m$^{-2}$ s$^{-1}$]")
    axes[0].set_title(r"A--$C_i$ response")

    axes[1].plot(aq["Q"], aq["legacy_aNet"], "o-", ms=3.5, label="pyLeaf")
    axes[1].plot(aq["Q"], aq["torch_aNet"], "--", lw=2, label="pyLeaf-torch")
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
        "--input", type=Path, default=REPOSITORY / "legacy" / "Input.xlsx"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY / "curve_comparison_output"
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
        help="optional JSON object of parameter overrides, used by both models",
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

    legacy_aci, torch_aci, diag_aci = run_pair(aci_weather, parameters, args.mode)
    legacy_aq, torch_aq, diag_aq = run_pair(aq_weather, parameters, args.mode)

    aci = pd.DataFrame(
        {
            "ca": ca_values,
            "legacy_ci": legacy_aci["ci"],
            "legacy_aNet": legacy_aci["aNet"],
            "torch_ci": torch_aci["ci"],
            "torch_aNet": torch_aci["aNet"],
            "legacy_flagged": legacy_aci["flagged"],
            "torch_converged": diag_aci["converged"],
            "torch_residual_norm": diag_aci["residual_norm"],
        }
    )
    aq = pd.DataFrame(
        {
            "PAR": par_values,
            "Q": par_values * PAR_TO_Q,
            "NIR": aq_weather["NIR"],
            "legacy_ci": legacy_aq["ci"],
            "legacy_aNet": legacy_aq["aNet"],
            "torch_ci": torch_aq["ci"],
            "torch_aNet": torch_aq["aNet"],
            "legacy_flagged": legacy_aq["flagged"],
            "torch_converged": diag_aq["converged"],
            "torch_residual_norm": diag_aq["residual_norm"],
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aci.to_csv(args.output_dir / "A_Ci_comparison.csv", index=False)
    aq.to_csv(args.output_dir / "A_Q_comparison.csv", index=False)
    figure = args.output_dir / "A_Ci_A_Q_comparison.png"
    save_plot(aci, aq, figure, args.dpi)

    print(f"Base input row: {args.base_row} ({args.input})")
    print(
        "Solver status: "
        f"legacy flagged {int(aci['legacy_flagged'].sum() + aq['legacy_flagged'].sum())}, "
        f"Torch converged {int(aci['torch_converged'].sum() + aq['torch_converged'].sum())}"
        f"/{2 * args.points} curve points"
    )
    print(f"Wrote figure and curve data to {args.output_dir}")


if __name__ == "__main__":
    main()
