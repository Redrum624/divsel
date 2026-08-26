"""Golden-fixture conformance reader — the Python half of "divsel reproduces
``test-assets/golden-selection.json``".

The fixture file is the cross-language contract (``docs/CONFORMANCE.md``): the
Aura (Python) and limbic (TypeScript) ports prove conformance against the same
file with the same rules this test applies:

* ``expected_selected`` matches EXACTLY (list equality, order included);
* ``expected_stage`` matches exactly;
* ``expected_g``, ``expected_div``, ``expected_threshold`` and
  ``expected_d_max`` agree within ``tol(expected) = f_rel * max(1, |expected|)``,
  with ``f_rel`` read from the file's own ``tolerance`` block;
* ``expected_f`` agrees within ``tol(expected_g) + lam * tol(expected_div)``.
  ``f = g + lam * div`` is derived, not primitive, so its bound is derived too:
  an error in ``div`` reaches ``f`` multiplied by ``lam``, which a bound
  relative to ``|f|`` misses whenever ``g`` dominates ``f`` while ``div`` is
  small -- the near-duplicate cosine regime, where ``1 - a.b`` cancels and the
  absolute error stays at the ``ulp(1)`` scale. See "Why ``f``'s bound is
  derived instead of relative to ``f``" in ``docs/CONFORMANCE.md``.

One caveat on that table, which the rules above do **not** encode because no
tolerance can: ``expected_threshold`` under ``stage == "sweep"`` is a *selected
grid entry*, not a measured quantity, so its error is quantized to a factor
``1 + eps`` and its bound is never the thing that decides. What decides is
rule 2's fold, and a tie there can in principle be broken differently by an f32
kernel and a float64 port. It cannot be on these 22 -- that property is pinned
by ``the_reported_threshold_is_never_decided_by_a_breakable_tie`` in
``crates/divsel/tests/golden.rs``, on the Rust side only, because it needs
``thresholds``/``greedy_independent_set``/``eval_g`` and the Python extension
exports just ``gist_select`` and ``gist_select_full``. The two readers still
apply the *same* conformance rules; the extra property test is not one of them.
"`expected_threshold` is a selected grid entry" in ``docs/CONFORMANCE.md`` says
what a port's own harness does about the mode off the 22.

The generator is ``python/tools/gen_golden.py``; the Rust-side reader is
``crates/divsel/tests/golden.rs``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

from divsel import gist_select_full

_ROOT = Path(__file__).resolve().parents[2]
GOLDEN_PATH = _ROOT / "test-assets" / "golden-selection.json"


def _require_golden(value: str | None) -> bool:
    """``DIVSEL_REQUIRE_GOLDEN`` set to anything but empty or ``0``.

    The same override the Rust reader honours (``crates/divsel/tests/golden.rs``)
    and the same one CI sets: a gate that can decide it has nothing to gate is
    not a gate.
    """
    return value is not None and value != "" and value != "0"


def _owns_the_fixture(root: Path) -> bool:
    """Does `root` look like the repository the fixture lives in?

    Several markers, none of which may be the only coupling. Keying the whole
    decision on ``python/tools/gen_golden.py`` -- an unrelated file a refactor
    is free to move -- meant that moving it while the fixture was also
    unreachable turned the 22-case contract into ``2 skipped``, exit 0, with
    nothing checked. This mirrors ``missing_fixture_policy`` in
    ``crates/divsel/tests/golden.rs``; the two readers must agree.
    """
    if (root / ".git").exists():
        return True
    if (root / "test-assets").is_dir():
        return True
    if (root / "python" / "tools" / "gen_golden.py").exists():
        return True
    try:
        return "[workspace]" in (root / "Cargo.toml").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        # `UnicodeDecodeError` is a ValueError, not an OSError. The Rust twin
        # reads this with `read_to_string(...).is_ok_and(...)`, which absorbs
        # invalid UTF-8 and skips; catching only `OSError` here made the same
        # tree raise out of `_load_golden` at import time and take down
        # collection of this whole module. The two readers must agree.
        return False


def _missing_fixture_is_fatal(required: bool, root: Path) -> bool:
    """A missing fixture is a failure in any tree that owns it, or on demand."""
    return required or _owns_the_fixture(root)


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
        if _missing_fixture_is_fatal(
            _require_golden(os.environ.get("DIVSEL_REQUIRE_GOLDEN")), _ROOT
        ):
            raise
        return None


GOLDEN = _load_golden()
CASES = GOLDEN["cases"] if GOLDEN is not None else []
F_REL = GOLDEN["tolerance"]["f_rel"] if GOLDEN is not None else 0.0

# Applied per test rather than as a module-level `pytestmark`, so that the two
# policy tests below -- which decide when that skip is even allowed -- run
# everywhere, including in a tree with no fixture in it.
needs_the_fixture = pytest.mark.skipif(
    GOLDEN is None,
    reason=f"{GOLDEN_PATH} is not present: the golden fixture ships with the "
    "repository (and the sdist), not with the wheel",
)


def _tol(expected: float) -> float:
    """The per-field tolerance budget: ``tol(x) = f_rel * max(1, |x|)``.

    The ``max(1, .)`` floor is the load-bearing half for a distance: a cosine
    distance near zero has no relative accuracy left -- it is ``1 - a.b``, a
    cancellation -- but its absolute error stays bounded by a few ``ulp(1)``.
    """
    return F_REL * max(1.0, abs(expected))


def _close(actual: float, expected: float, bound: float) -> bool:
    """The conformance float rule: |actual - expected| <= bound.

    ``bound`` is ``_tol`` of the field itself for every primitive field, and
    ``_tol(g) + lam * _tol(div)`` for the derived ``f``.
    """
    return abs(actual - expected) <= bound


@needs_the_fixture
def test_the_f_bound_carries_lam_times_the_div_budget() -> None:
    """The ``f`` bound carries ``lam`` times ``div``'s budget, and that is load-bearing.

    This pins the fix for the tolerance defect the 2026-08-26 differential
    found: ``f = g + lam * div``, so an error in ``div`` reaches ``f``
    multiplied by ``lam``, and a bound relative to ``|f|`` misses it whenever
    ``g`` dominates ``f`` while ``div`` is small. The numbers are the worked
    case in ``docs/CONFORMANCE.md`` -- two near-duplicate cosine rows at
    ``lam = 64``, where ``1 - a.b`` cancels and the f32 distance carries an
    absolute error at the ``ulp(1)`` scale. Twin of
    ``the_f_bound_carries_lam_times_the_div_budget`` in
    ``crates/divsel/tests/golden.rs``; the two readers must agree.
    """
    lam = 64.0
    # What divsel reports (f32 distance kernel).
    expected_f = 2.0066680908203125
    expected_g = 2.0
    expected_div = 1.0418891906738281e-4
    # What a float64 port reports for the SAME selection and stage: the same
    # algorithm, a wider distance arithmetic -- and the more accurate side.
    port_f = 2.0066829063008953

    # The old, lam-independent rule rejected it. That was the contract defect:
    # it failed a correct port at high lam.
    assert not _close(port_f, expected_f, _tol(expected_f))

    # The derived rule accepts it, and uses under a quarter of the budget.
    bound = _tol(expected_g) + lam * _tol(expected_div)
    assert _close(port_f, expected_f, bound)
    assert abs(port_f - expected_f) < 0.25 * bound

    # It still rejects an error a real bug would produce. The smallest discrete
    # step this fixture family's objective can take is 1/64 (dyadic weights),
    # 237x the bound; the generator's own robustness margin (1e-4 relative) is
    # 3x it even in this worst regime.
    assert not _close(expected_f + 1.0 / 64.0, expected_f, bound)
    assert not _close(expected_f + 1e-4 * max(1.0, abs(expected_f)), expected_f, bound)

    # At lam == 0 the bound is exactly g's: rule 18 makes `f` equal `g` there.
    assert _tol(expected_g) + 0.0 * _tol(expected_div) == _tol(expected_g)


def test_the_require_golden_override_is_read_the_way_ci_sets_it() -> None:
    assert not _require_golden(None)
    assert not _require_golden("")
    assert not _require_golden("0")
    assert _require_golden("1")
    assert _require_golden("true")


def test_a_missing_fixture_only_skips_outside_the_repository_that_owns_it(tmp_path) -> None:
    """The Python half of the policy the Rust reader already pins.

    The regression: keying the decision on ``python/tools/gen_golden.py`` alone
    meant that moving the generator -- a refactor nothing forbids -- while the
    fixture was also unreachable made ``pytest python/tests/test_golden.py``
    report ``2 skipped``, exit 0, with zero of the 22 conformance cases checked,
    and it stayed 0 even with ``DIVSEL_REQUIRE_GOLDEN=1`` because only the Rust
    reader read that variable.
    """
    empty = tmp_path / "empty"
    empty.mkdir()
    # Nothing at all: an installed copy of this suite, a wheel, a sdist-less tree.
    assert not _missing_fixture_is_fatal(False, empty)
    # ... unless CI asked for the gate explicitly.
    assert _missing_fixture_is_fatal(True, empty)

    for marker in ("test-assets", ".git", "python/tools/gen_golden.py", "Cargo.toml"):
        root = tmp_path / marker.replace("/", "_").replace(".", "_")
        root.mkdir()
        assert not _missing_fixture_is_fatal(False, root), marker
        if marker in ("test-assets", ".git"):
            (root / marker).mkdir()
        elif marker == "Cargo.toml":
            (root / marker).write_text("[workspace]\nmembers = []\n", encoding="utf-8")
        else:
            (root / "python" / "tools").mkdir(parents=True)
            (root / marker).write_text("# generator\n", encoding="utf-8")
        assert _missing_fixture_is_fatal(False, root), marker

    # A package-root manifest is not a workspace manifest.
    package = tmp_path / "package"
    package.mkdir()
    (package / "Cargo.toml").write_text('[package]\nname = "divsel"\n', encoding="utf-8")
    assert not _missing_fixture_is_fatal(False, package)

    # A manifest that is not valid UTF-8 is not a workspace manifest either --
    # and the two readers "must agree" (the docstring on `_owns_the_fixture`).
    # The Rust twin reads it with `read_to_string(...).is_ok_and(...)`, which
    # absorbs invalid UTF-8 and returns `Missing::Skip`; this reader caught only
    # `OSError`, so the same tree raised `UnicodeDecodeError` -- a ValueError --
    # out of `_load_golden` at import time and took down collection of this
    # whole module, `test_golden_header` and both policy tests included.
    mojibake = tmp_path / "mojibake"
    mojibake.mkdir()
    (mojibake / "Cargo.toml").write_bytes(b"[workspace]\nname = \xff\xfe\n")
    assert not _missing_fixture_is_fatal(False, mojibake)

    # And where the fixture IS reachable, the tree it sits in must be one the
    # policy calls an owner -- a missing file there is a broken checkout, never
    # a skip. (An installed copy of this suite has neither, and legitimately
    # skips; that is the one situation the skip exists for.)
    if GOLDEN_PATH.exists():
        assert _missing_fixture_is_fatal(False, _ROOT)


@needs_the_fixture
def test_golden_header() -> None:
    assert GOLDEN is not None
    assert GOLDEN["schema"] == 1
    assert GOLDEN["generator"].startswith("divsel ")
    assert GOLDEN["paper"] == "arXiv:2405.18754v3"
    assert GOLDEN["tolerance"]["selected"] == "exact"
    assert len(GOLDEN["cases"]) == 22


@needs_the_fixture
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
    # `f` is derived from `g` and `div` (`f = g + lam * div`), so its budget is
    # the sum of theirs with `lam` applied to `div`'s -- the same `lam` the
    # objective applies. At `lam == 0` it collapses to `_tol(g)`, which is what
    # rule 18 promises: `f == g` exactly, even for an infinite `div`.
    for field, expected, bound in (
        ("f_value", case["expected_f"],
         _tol(case["expected_g"]) + case["lam"] * _tol(case["expected_div"])),
        ("g_value", case["expected_g"], _tol(case["expected_g"])),
        ("div", case["expected_div"], _tol(case["expected_div"])),
        ("threshold", case["expected_threshold"], _tol(case["expected_threshold"])),
        ("d_max", case["expected_d_max"], _tol(case["expected_d_max"])),
    ):
        assert _close(out[field], expected, bound), (
            f"{ctx}: {field} = {out[field]!r} differs from expected {expected!r} "
            f"by {abs(out[field] - expected)!r}, beyond the conformance bound {bound!r}"
        )
