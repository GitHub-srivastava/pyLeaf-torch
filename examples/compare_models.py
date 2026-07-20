"""Run the frozen GEKKO implementation and the Torch equilibrium side by side."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import time
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import torch

from pyleaf_torch import (
    DifferentiableLeaf,
    simulate_dataframe,
    weather_from_dataframe,
)


REPOSITORY = Path(__file__).resolve().parents[1]
LEGACY_HASH = "d14cd68d3ae90dcfd2d993ef379b4ef7a06f21b55941ad801fa3e95e55797efd"
LEGACY_DIAGNOSTIC_COLUMNS = {"flag", "Terror", "Aerror", "Cierror", "Gserror"}


def load_legacy() -> ModuleType:
    source = REPOSITORY / "legacy" / "pyLeaf.py"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != LEGACY_HASH:
        raise RuntimeError(
            f"Frozen legacy source changed: expected {LEGACY_HASH}, observed {digest}"
        )
    spec = importlib.util.spec_from_file_location("pyleaf_frozen_legacy", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def comparison_table(legacy: pd.DataFrame, torch_frame: pd.DataFrame) -> pd.DataFrame:
    common = [
        name
        for name in legacy.columns
        if name in torch_frame.columns and name not in LEGACY_DIAGNOSTIC_COLUMNS
    ]
    result: dict[str, np.ndarray] = {}
    for name in common:
        baseline = legacy[name].to_numpy(dtype=float)
        candidate = torch_frame[name].to_numpy(dtype=float)
        result[f"legacy_{name}"] = baseline
        result[f"torch_{name}"] = candidate
        result[f"abs_diff_{name}"] = np.abs(candidate - baseline)
        result[f"rel_diff_{name}"] = np.abs(candidate - baseline) / (
            np.abs(baseline) + 1.0e-12
        )
    return pd.DataFrame(result)


def metrics(legacy: pd.DataFrame, torch_frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    common = [
        name
        for name in legacy.columns
        if name in torch_frame.columns and name not in LEGACY_DIAGNOSTIC_COLUMNS
    ]
    for name in common:
        baseline = legacy[name].to_numpy(dtype=float)
        candidate = torch_frame[name].to_numpy(dtype=float)
        finite = np.isfinite(baseline) & np.isfinite(candidate)
        if not finite.any():
            continue
        difference = candidate[finite] - baseline[finite]
        result[name] = {
            "mae": float(np.mean(np.abs(difference))),
            "rmse": float(np.sqrt(np.mean(difference**2))),
            "max_abs": float(np.max(np.abs(difference))),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=REPOSITORY / "legacy" / "Input.xlsx"
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY / "comparison_output"
    )
    parser.add_argument("--mode", choices=("hard", "smooth"), default="hard")
    parser.add_argument("--rows", type=int, default=None)
    args = parser.parse_args()

    weather = pd.read_excel(args.input)
    if args.rows is not None:
        weather = weather.iloc[: args.rows].copy()

    legacy_module = load_legacy()
    legacy_leaf = legacy_module.Leaf()
    legacy_start = time.perf_counter()
    legacy_leaf.SeriesSolver(weather)
    legacy_seconds = time.perf_counter() - legacy_start
    legacy_frames = {
        "state": legacy_leaf.LeafState,
        "mass": legacy_leaf.LeafMassFlux,
        "energy": legacy_leaf.LeafEnergyFlux,
    }

    model = DifferentiableLeaf(trainable=(), mode=args.mode, dtype=torch.float64)
    torch_start = time.perf_counter()
    _, torch_frames = simulate_dataframe(model, weather)
    torch_seconds = time.perf_counter() - torch_start
    new_frames = {
        "state": torch_frames.state,
        "mass": torch_frames.mass,
        "energy": torch_frames.energy,
    }
    tensor_weather = weather_from_dataframe(weather, dtype=torch.float64)
    legacy_state = {
        "aNet": torch.as_tensor(legacy_frames["mass"]["aNet"].to_numpy(copy=True)),
        "cbs": torch.as_tensor(legacy_frames["state"]["cbs"].to_numpy(copy=True)),
        "ci": torch.as_tensor(legacy_frames["state"]["ci"].to_numpy(copy=True)),
        "gs": torch.as_tensor(legacy_frames["state"]["gs"].to_numpy(copy=True)),
        "cb": torch.as_tensor(legacy_frames["state"]["cb"].to_numpy(copy=True)),
        "tLeaf": torch.as_tensor(legacy_frames["state"]["tLeaf"].to_numpy(copy=True)),
    }
    with torch.no_grad():
        legacy_residual = model.equilibrium_residual(legacy_state, tensor_weather)
        legacy_residual_norm = torch.linalg.vector_norm(
            legacy_residual, ord=float("inf"), dim=-1
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {
        "mode": args.mode,
        "rows": len(weather),
        "legacy_sha256": LEGACY_HASH,
        "runtime_seconds": {
            "legacy_gekko": legacy_seconds,
            "pytorch": torch_seconds,
        },
        "legacy_solver": {
            "flagged_rows": int(legacy_frames["state"]["flag"].sum()),
            "note": "Legacy flags and iteration deltas are not parity metrics.",
        },
        "legacy_state_under_coupled_equations": {
            "rows_below_torch_tolerance": int(
                (legacy_residual_norm <= model.solver_options.residual_tolerance).sum()
            ),
            "median_residual_norm": float(legacy_residual_norm.median()),
            "maximum_residual_norm": float(legacy_residual_norm.max()),
        },
        "torch_solver": {
            "converged_rows": int(torch_frames.diagnostics["converged"].sum()),
            "maximum_residual_norm": float(
                torch_frames.diagnostics["residual_norm"].max()
            ),
            "maximum_iterations": int(torch_frames.diagnostics["iterations"].max()),
        },
        "groups": {},
    }
    for group in ("state", "mass", "energy"):
        comparison_table(legacy_frames[group], new_frames[group]).to_csv(
            args.output_dir / f"{group}_comparison.csv", index=False
        )
        summary["groups"][group] = metrics(legacy_frames[group], new_frames[group])
    torch_frames.diagnostics.to_csv(
        args.output_dir / "torch_solver_diagnostics.csv", index=False
    )
    torch_frames.limitation.to_csv(
        args.output_dir / "torch_limitation_regimes.csv", index=False
    )
    pd.DataFrame(
        {
            **{
                f"residual_{name}": legacy_residual[:, index].cpu().numpy()
                for index, name in enumerate(model.STATE_NAMES)
            },
            "residual_norm": legacy_residual_norm.cpu().numpy(),
        }
    ).to_csv(args.output_dir / "legacy_coupled_residual.csv", index=False)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nModel runtimes")
    print(f"  Legacy GEKKO model: {legacy_seconds:.6f} seconds")
    print(f"  PyTorch model:      {torch_seconds:.6f} seconds")
    print(json.dumps(summary["torch_solver"], indent=2))
    print(f"Wrote comparison files to {args.output_dir}")


if __name__ == "__main__":
    main()
