"""Smoke coverage for the two benchmark scripts, which nothing else imports.

``bench/compare.py`` and ``bench/gist_select_readme_table.py`` produce the
numbers in ``docs/benchmarks/README.md`` and the README's comparison claims, and
until now no test loaded either of them: a syntax error, a renamed method or an
argument parsed the wrong way would have been found by hand, at benchmark time.

These tests do not *run* a benchmark -- a single cell is minutes and gigabytes.
They load each module and exercise the pure argument/plan helpers, which is
where the two defects this file was written for lived: ``--rows 0`` silently
selecting the 2,000,000-point row, and the n >= 1M cells being labelled with a
configuration that did not run.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

BENCH = Path(__file__).resolve().parents[2] / "bench"


def _load(name: str):
    path = BENCH / f"{name}.py"
    if not path.exists():  # an installed copy of this suite has no bench/
        pytest.skip(f"{path} is not present")
    spec = importlib.util.spec_from_file_location(f"_bench_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --------------------------------------------------------------------------- #
# bench/gist_select_readme_table.py                                           #
# --------------------------------------------------------------------------- #


def test_readme_table_rows_are_validated():
    table = _load("gist_select_readme_table")
    assert table.parse_rows("1,2,3,4") == [1, 2, 3, 4]
    assert table.parse_rows(" 2 , 3 ") == [2, 3]

    # `--rows 0` used to index ROWS[-1]: the 2,000,000-point row (~1 GB of
    # float32, 1015 s in docs/benchmarks/README.md) for a caller who asked for
    # nothing. Out of range, not row 4.
    for bad in ("0", "-1", "5", "", "x", "1,,2"):
        with pytest.raises(SystemExit) as info:
            table.parse_rows(bad)
        assert "--rows" in str(info.value)


def test_readme_table_rows_cover_the_readme_table():
    table = _load("gist_select_readme_table")
    assert [row[0] for row in table.ROWS] == [10_000, 100_000, 500_000, 2_000_000]
    assert table.parse_rows(",".join(str(i + 1) for i in range(len(table.ROWS)))) == [
        1,
        2,
        3,
        4,
    ]


# --------------------------------------------------------------------------- #
# bench/compare.py                                                            #
# --------------------------------------------------------------------------- #


def test_compare_method_registry_agrees_with_its_name_tuple():
    compare = _load("compare")
    assert tuple(compare.METHODS) == compare.METHOD_NAMES
    assert "divsel" in compare.METHODS
    assert "divsel[diameter=approx]" in compare.METHODS


def test_compare_large_cells_are_approx_diameter_only():
    compare = _load("compare")
    cells = compare.cells_for(SimpleNamespace(all=True, large=True))
    large = [c for c in cells if c[0] >= 1_000_000]
    assert large, "--all --large must add the n = 1M cells"
    # Every 1M cell runs the approximate diameter, which is why `main` labels
    # those rows `divsel[diameter=approx]` rather than plain `divsel`.
    assert {c[4] for c in large} == {"approx"}
    assert {c[3] for c in large} == {"linear"}
    assert {c[4] for c in cells if c[0] < 1_000_000} == {"exact"}


def test_compare_small_cells_come_from_the_explicit_grid():
    compare = _load("compare")
    args = SimpleNamespace(
        all=False, large=False, utility="linear", n=[100], dim=[8], k=[5]
    )
    assert compare.cells_for(args) == [(100, 8, 5, "linear", "exact")]
    # The 1M rule is not tied to --large: any n >= 1M switches the mode.
    args.n = [1_000_000]
    assert compare.cells_for(args) == [(1_000_000, 8, 5, "linear", "approx")]
