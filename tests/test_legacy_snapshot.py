from __future__ import annotations

import hashlib
from pathlib import Path


def test_all_listed_legacy_files_are_byte_for_byte_snapshots() -> None:
    legacy = Path(__file__).resolve().parents[1] / "legacy"
    for line in (legacy / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        expected, filename = line.split(maxsplit=1)
        observed = hashlib.sha256((legacy / filename).read_bytes()).hexdigest()
        assert observed == expected, filename
