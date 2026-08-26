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


def _close(actual: float, expected: float) -> bool:
    """The conformance float rule: |actual - expected| <= f_rel * max(1, |expected|)."""
    return abs(actual - expected) <= F_REL * max(1.0, abs(expected))


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
