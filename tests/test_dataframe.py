from __future__ import annotations

import pandas as pd
import pytest

from pyleaf_torch import Leaf, simulate_dataframe


def test_dataframe_adapter(weather) -> None:
    frame = pd.DataFrame({name: value.numpy() for name, value in weather.items()})
    frame.insert(0, "plant", 4)
    model = Leaf(trainable=(), energy_balance=False)
    output, frames = simulate_dataframe(model, frame)
    assert len(frames.mass) == len(frame)
    assert list(frames.mass.columns)[4] == "aNet"
    assert frames.diagnostics["converged"].all()
    assert output.mass["aNet"].shape == (len(frame),)


def test_dataframe_rejects_c3(weather) -> None:
    frame = pd.DataFrame({name: value.numpy() for name, value in weather.items()})
    frame.insert(0, "plant", [4, 3, 4])
    model = Leaf(trainable=(), energy_balance=False)
    with pytest.raises(ValueError, match="C4-only"):
        simulate_dataframe(model, frame)
