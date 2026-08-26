"""Unit coverage for ``python/tools/gen_golden.py``, the golden-fixture generator.

1079 lines whose only gate was the end-to-end ``--check`` in one CI cell: a
fidelity gap inside one of its reference reimplementations shows up as a fixture
that pins the wrong number, and ``--check`` cannot see that -- it compares the
generator against itself.

The reimplementations tested here (``thresholds_f32``, ``_approx_diameter``,
``sweep_ceiling``) exist to predict what the *library* does, so each one is
checked against the library or against the core's documented rule, never against
another copy of itself.
"""

from __future__ import annotations

import functools
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from divsel import gist_select_full

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@functools.lru_cache(maxsize=1)
def _gen():
    path = TOOLS / "gen_golden.py"
    if not path.exists():  # an installed copy of this suite has no tools/
        pytest.skip(f"{path} is not present")
    spec = importlib.util.spec_from_file_location("_gen_golden", path)
    module = importlib.util.module_from_spec(spec)
    # `Hand` is a dataclass with `from __future__ import annotations`, and
    # dataclasses resolves those through `sys.modules[cls.__module__]`, so the
    # module has to be registered before its body runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


F32_EPS = float(np.finfo(np.float32).eps)


def test_thresholds_f32_matches_the_cores_grid_sizes():
    gen = _gen()
    # gist.rs: |D| = 1 + floor(log_{1+eps}(bound)); 32 at the default eps with
    # the paper's 2/eps ceiling, 39 with the 4/eps one approx mode uses.
    assert len(gen.thresholds_f32(1.0, 0.1, 2.0 / 0.1)) == 32
    assert len(gen.thresholds_f32(1.0, 0.1, 4.0 / 0.1)) == 39
    assert len(gen.thresholds_f32(1.0, 1e-3, 2.0 / 1e-3)) == 7605

    # The grid is ascending, deduplicated, and starts at eps * d_max / 2.
    grid = gen.thresholds_f32(12.0, 0.1, 20.0)
    assert grid == sorted(grid)
    assert len(set(grid)) == len(grid)
    assert grid[0] == float(np.float32(0.1 * 12.0 / 2.0))


def test_thresholds_f32_stops_where_the_core_stops():
    """The core's eps floor, replicated.

    ``gist.rs::thresholds_with_bound`` returns an empty grid below
    ``f32::EPSILON``: multiplying by ``1 + eps`` can no longer separate two
    consecutive ``f32`` entries there, and below ``2**-53`` it does not advance
    at all. Without the same guard this generator's loop diverges from the core
    and, at a small enough eps, never terminates.
    """
    gen = _gen()
    assert gen.thresholds_f32(1.0, F32_EPS / 2.0, 4.0) == []
    assert gen.thresholds_f32(1.0, 1e-30, 4.0) == []
    assert gen.thresholds_f32(1.0, 0.0, 4.0) == []
    assert gen.thresholds_f32(1.0, float("nan"), 4.0) == []
    # ... and f32::EPSILON itself is inside the range, so a tiny ceiling still
    # yields entries (a big one would be 1.4e8 of them).
    assert len(gen.thresholds_f32(1.0, F32_EPS, 1.0)) == 1


def test_a_ceiling_below_one_yields_no_thresholds():
    gen = _gen()
    for bound in (-1.0, 0.0, 0.5):
        assert gen.thresholds_f32(1.0, 0.1, bound) == []


def test_the_sweep_ceiling_widens_under_approx():
    """The prediction ceiling has to follow the core's, per diameter mode.

    ``verify_hand`` predicted every hand case's winning threshold with ``2/eps``,
    including under ``diameter="approx"`` where the core uses ``4/eps`` -- so a
    hand case in approx mode whose threshold lands above ``~d_hat`` would be
    reported as "no grid threshold in the interval".
    """
    gen = _gen()
    assert gen.sweep_ceiling("exact") == 2.0
    assert gen.sweep_ceiling("approx") == 4.0
    with pytest.raises(ValueError):
        gen.sweep_ceiling("guess")

    d_hat, eps = 12.0, 0.1
    narrow = gen.thresholds_f32(d_hat, eps, gen.sweep_ceiling("exact") / eps)
    wide = gen.thresholds_f32(d_hat, eps, gen.sweep_ceiling("approx") / eps)
    assert narrow[-1] < d_hat < wide[-1]
    assert wide[: len(narrow)] == narrow


# A 12x3 set on which the farthest-point walk needs four double sweeps to
# converge -- the same fixture test_api.py uses to pin the sweep default.
_SWEEP_SENSITIVE = [
    [0.42473456, 0.53754765, -0.65292674],
    [-0.495182, 0.76948655, -0.13159879],
    [0.39689574, -0.192833, 1.8935074],
    [-1.3601785, -0.45732597, 0.49488983],
    [-0.23497039, 0.33717105, -1.7059377],
    [1.992929, -0.9904514, 0.55411506],
    [-0.29105875, 0.18395717, 0.650892],
    [-0.45368975, 2.3688433, -0.32602257],
    [-0.5333204, -1.0139397, -1.6846485],
    [-0.49256995, -1.9239715, -0.8081258],
    [1.9454916, 0.95573187, 1.4676777],
    [-0.5040848, 1.3585472, -1.523295],
]


def test_approx_diameter_reimplementation_matches_the_library():
    gen = _gen()
    vectors = [[float(np.float32(x)) for x in row] for row in _SWEEP_SENSITIVE]
    x = np.ascontiguousarray(np.array(vectors, dtype=np.float32))
    dist = gen._dist_matrix(vectors, "euclidean")

    seen = set()
    for sweeps in (0, 1, 2, 3, 4, 7):
        library = gist_select_full(
            x, k=1, metric="euclidean", diameter="approx", diameter_sweeps=sweeps
        )["d_max"]
        replica = gen._approx_diameter(dist, len(vectors), sweeps)
        assert replica == pytest.approx(library, rel=1e-6), f"sweeps={sweeps}"
        seen.add(round(library, 6))
    # Guards the guard: on a fixture where every sweep count agrees, the loop
    # above would compare one number to itself six times.
    assert len(seen) >= 4

    # Fewer than two points is 0.0, matching Points::diameter.
    assert gen._approx_diameter([[0.0]], 1, 3) == 0.0


def test_verify_hand_uses_the_widened_ceiling_under_approx():
    """``sweep_ceiling("approx")`` has to be what ``verify_hand`` predicts with.

    No fixture reaches it: the only ``diameter="approx"`` case (case 20) carries
    no ``Hand``, so ``verify_hand`` returns before the grid is ever built and
    the widened ``4/eps`` ceiling is unexercised end to end. This drives the
    function directly with a synthetic case, on an interval that lies *above*
    the exact ceiling: under ``approx`` the grid reaches it, under ``exact`` it
    does not -- so a regression that dropped the widening turns from silent into
    a ``no grid threshold in`` failure.
    """
    gen = _gen()

    exact_grid = gen.thresholds_f32(1.0, 0.1, gen.sweep_ceiling("exact") / np.float32(0.1))
    approx_grid = gen.thresholds_f32(1.0, 0.1, gen.sweep_ceiling("approx") / np.float32(0.1))
    assert approx_grid[: len(exact_grid)] == exact_grid, "the grids must be nested"
    assert len(approx_grid) > len(exact_grid), "the widening must add entries"

    lo, winner = exact_grid[-1], approx_grid[-1]
    hand = gen.Hand(
        selected=[0, 1],
        f=1.0,
        g=1.0,
        div=1.0,
        stage="sweep",
        d_max=1.0,
        interval=(lo, winner),
    )
    case = {"name": "synthetic_approx", "eps": 0.1, "diameter": "approx", "_hand": hand}
    out = {
        "selected": [0, 1],
        "stage": "sweep",
        "f_value": 1.0,
        "g_value": 1.0,
        "div": 1.0,
        "d_max": 1.0,
        "threshold": winner,
    }

    gen.verify_hand(case, out)  # the widened ceiling reaches `winner`

    # The same interval under the paper's `2/eps` ceiling contains no grid
    # entry at all, which is exactly what predicting an approx case with the
    # exact ceiling would do.
    with pytest.raises(SystemExit, match="no grid threshold in"):
        gen.verify_hand({**case, "diameter": "exact"}, out)


def test_margin_check_accepts_the_generated_case_and_rejects_a_wrong_selection():
    gen = _gen()
    case = gen.hand_cases()[0]
    out = gen.run_library(case)

    ok, reason, gap = gen.margin_check(case, out)
    assert ok, reason
    assert gap >= 1e-4

    # A selection that is not the argmax must not clear the margin rules --
    # reported WITH its own brute-force f value, so the reimplementation-drift
    # guard (which rejects any selection whose f does not match the library's
    # reported one) cannot be what rejects it. Handing the argmax's f_value
    # along with a different selection only ever exercised that guard, which is
    # not what rules (a)/(b)/(c) do.
    values, _ = gen._brute_f(case)
    wrong_selected = [0, 1]
    assert wrong_selected != out["selected"], "the fixture must disagree here"
    wrong = dict(out)
    wrong["selected"] = wrong_selected
    wrong["f_value"] = values[frozenset(wrong_selected)]

    ok, reason, _ = gen.margin_check(case, wrong)
    assert not ok
    assert "drifts" not in reason, f"the drift guard fired, not a margin rule: {reason}"
    # Rule (b): [0, 1] and [2, 3] both score 2.5 on this case.
    assert "f value shared with" in reason, reason

    # The drift guard is still there, and still rejects an inconsistent pair.
    inconsistent = dict(out)
    inconsistent["selected"] = wrong_selected
    ok, reason, _ = gen.margin_check(case, inconsistent)
    assert not ok and "drifts" in reason


def test_dy_rejects_values_the_fixture_rules_forbid():
    gen = _gen()
    assert gen.dy(0.5) == 0.5
    assert gen.dy(-4.0) == -4.0
    for bad in (0.1, 5.0, -4.5, 1.0 / 3.0):
        with pytest.raises(AssertionError):
            gen.dy(bad)
    assert math.isclose(gen.dy(1 / 64), 1 / 64)
