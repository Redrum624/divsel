"""Golden-fixture conformance reader — the Python half of "divsel reproduces
``test-assets/golden-selection.json``".

The fixture file is the cross-language contract (``docs/CONFORMANCE.md``): the
Aura (Python) and limbic (TypeScript) ports prove conformance against the same
file with the same rules this test applies:

* ``expected_selected`` matches EXACTLY (list equality, order included);
* every float field agrees within ``f_rel * max(1, |expected|)``, with
  ``f_rel`` read from the file's own ``tolerance`` block;
* ``expected_stage`` matches exactly.

The generator is ``python/tools/gen_golden.py``; the Rust-side reader is
``crates/divsel/tests/golden.rs``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from divsel import gist_select_full

_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = _ROOT / "test-assets" / "golden-selection.json"
# The fixture's generator is the marker for "the checkout that owns the fixture".
# Where it is present, a missing fixture is a hard error, never a skip.
_GENERATOR = _ROOT / "python" / "tools" / "gen_golden.py"


def _load_golden() -> dict | None:
    """The fixture, or ``None`` when it is genuinely not there to be read.

    Loading at import time is what makes the 22 cases separate parametrised
    tests, but it also means a missing file is a *collection* error that takes
    down every test in this module, ``test_golden_header`` included. Only an
    installed copy of this suite can hit that -- the fixture lives at the
    repository root -- so it becomes a skip there and stays fatal here.
    """
    try:
        with open(GOLDEN_PATH, encoding="utf-8") as fh:
            return json.load(fh)
    except FileNotFoundError:
        if _GENERATOR.exists():
            raise
        return None


GOLDEN = _load_golden()
CASES = GOLDEN["cases"] if GOLDEN is not None else []
F_REL = GOLDEN["tolerance"]["f_rel"] if GOLDEN is not None else 0.0

pytestmark = pytest.mark.skipif(
    GOLDEN is None,
    reason=f"{GOLDEN_PATH} is not present: the golden fixture ships with the "
    "repository (and the sdist), not with the wheel",
)


def _close(actual: float, expected: float) -> bool:
    """The conformance float rule: |actual - expected| <= f_rel * max(1, |expected|)."""
    return abs(actual - expected) <= F_REL * max(1.0, abs(expected))


def test_golden_header() -> None:
    assert GOLDEN is not None
    assert GOLDEN["schema"] == 1
    assert GOLDEN["generator"].startswith("divsel ")
    assert GOLDEN["paper"] == "arXiv:2405.18754v3"
    assert GOLDEN["tolerance"]["selected"] == "exact"
    assert len(GOLDEN["cases"]) == 22


@pytest.mark.parametrize("case", CASES, ids=[c["name"] for c in CASES])
def test_golden_case(case: dict) -> None:
    vectors = np.ascontiguousarray(np.array(case["vectors"], dtype=np.float32))
    utilities = case["utilities"]
    if case["utility"] == "linear" and utilities is not None:
        utilities = np.array(utilities, dtype=np.float64)

    out = gist_select_full(
        vectors,
        utilities,
        k=case["k"],
        lam=case["lam"],
        eps=case["eps"],
        metric=case["metric"],
        utility=case["utility"],
        exhaustive_thresholds=case["exhaustive_thresholds"],
        diameter=case["diameter"],
        diameter_sweeps=case["diameter_sweeps"],
    )

    ctx = f"case {case['name']!r} — {case['note']}"
    assert out["selected"] == case["expected_selected"], f"{ctx}: selected differs"
    assert out["stage"] == case["expected_stage"], f"{ctx}: stage differs"
    for field, expected in (
        ("f_value", case["expected_f"]),
        ("g_value", case["expected_g"]),
        ("div", case["expected_div"]),
        ("threshold", case["expected_threshold"]),
        ("d_max", case["expected_d_max"]),
    ):
        assert _close(out[field], expected), (
            f"{ctx}: {field} = {out[field]!r} differs from expected {expected!r} "
            f"beyond {F_REL} * max(1, |expected|)"
        )
