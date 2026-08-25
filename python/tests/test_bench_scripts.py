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
import json
import os
import subprocess
import sys
import threading
import time
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


# --------------------------------------------------------------------------- #
# bench/compare.py: the n >= 1M relabelling, the reporters, the timeout kill   #
# --------------------------------------------------------------------------- #


def test_every_requested_method_gets_exactly_one_row_at_1m():
    """The relabelling branch, which had no test and dropped a row.

    ``cells_for`` forces ``diameter="approx"`` at ``n >= 1M`` (R-G21), so both
    divsel spellings mean the same cell there. The second one used to be skipped
    with a bare ``continue``: ``--all --large`` with the default methods wrote 6
    rows where 7 were asked for, and nothing in the JSON said why. Every label
    the caller asked for gets one entry now, and the ``not run`` reason names a
    label that actually appears in the output.
    """
    compare = _load("compare")
    methods = list(compare.METHOD_NAMES)

    # Below 1M nothing is rewritten: the two divsel spellings are different runs.
    plan = compare.resolve_methods(methods, 10_000)
    assert [label for label, _, _ in plan] == methods
    assert [effective for _, effective, _ in plan] == methods
    assert {reason for _, _, reason in plan} == {""}

    plan = compare.resolve_methods(methods, 1_000_000)
    assert [label for label, _, _ in plan] == methods, "a requested method vanished"
    ran = [(label, effective) for label, effective, _ in plan if effective is not None]
    assert ran == [("divsel", "divsel[diameter=approx]")]
    for label, effective, reason in plan:
        if effective is None:
            assert reason, f"{label} was not run and no reason was recorded"
            # The reason must not name a label the output never carries.
            assert "divsel only" not in reason
    dropped = dict((label, reason) for label, effective, reason in plan if effective is None)
    assert "same cell" in dropped["divsel[diameter=approx]"]

    # Asking for the approx spelling alone still runs it, under its own label.
    assert compare.resolve_methods(["divsel[diameter=approx]"], 1_000_000) == [
        ("divsel[diameter=approx]", "divsel[diameter=approx]", "")
    ]


def test_markdown_table_renders_ok_and_not_run_rows():
    compare = _load("compare")
    ok = {
        "n": 10, "dim": 2, "k": 2, "utility": "linear", "method": "divsel",
        "status": "ok", "wall_median_s": 1.5, "wall_runs_s": [1.4, 1.5, 1.6],
        "peak_rss_mib": 12.0, "baseline_rss_mib": 8.0,
        "eval_f": 3.25, "eval_g": 1.25, "eval_div": 2.0, "eval_size": 2,
        "library_reported": {"f": 3.25, "stage": "sweep", "diameter": "exact"},
    }
    skipped = {
        "n": 1_000_000, "dim": 2, "k": 2, "utility": "linear", "method": "mmr",
        "status": "not run", "reason": "n = 1M is run for divsel[diameter=approx] only (R-G21)",
    }
    table = compare.markdown_table([ok, skipped])
    lines = table.splitlines()
    assert len(lines) == 4  # header, separator, two rows
    assert "1.500 (1.400-1.600)" in lines[2]
    assert "stage=sweep, diameter=exact" in lines[2]
    assert "not run" in lines[3]
    assert lines[3].startswith("| 1000000 |")
    # A row that did not run still carries its reason, and the empty numeric
    # cells render as "-" rather than raising on the missing keys.
    assert "R-G21" in lines[3]
    assert lines[3].count("|") == lines[2].count("|")


def test_merge_into_updates_by_key_and_truncates_long_selections(tmp_path):
    compare = _load("compare")
    out = tmp_path / "nested" / "results.json"
    base = {"n": 10, "dim": 2, "k": 2, "utility": "linear", "status": "ok"}

    first = compare.merge_into(out, {"host": "a"}, [{**base, "method": "divsel", "eval_f": 1.0}])
    assert out.exists() and len(first["results"]) == 1

    second = compare.merge_into(
        out,
        {"host": "b"},
        [
            {**base, "method": "divsel", "eval_f": 2.0},  # same key: replaces
            {**base, "method": "mmr", "selected": list(range(250))},  # new key
        ],
    )
    assert second["meta"] == {"host": "b"}
    by_method = {r["method"]: r for r in second["results"]}
    assert set(by_method) == {"divsel", "mmr"}
    assert by_method["divsel"]["eval_f"] == 2.0
    assert len(by_method["mmr"]["selected"]) == 100
    assert by_method["mmr"]["selected_truncated"] is True
    # Rows are ordered by the registry, not by arrival.
    order = [compare.METHOD_NAMES.index(r["method"]) for r in second["results"]]
    assert order == sorted(order)
    # It round-trips through the file, which is what --out promises.
    assert json.loads(out.read_text(encoding="utf-8")) == second


def _spawn_parent_with_a_child(tmp_path):
    """A process that spawns one child and waits, both holding our stdout pipe."""
    ready = tmp_path / "ready"
    code = (
        "import pathlib, subprocess, sys, time\n"
        "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        f"pathlib.Path({str(ready)!r}).write_text('up')\n"
        "time.sleep(300)\n"
    )
    session = {} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        **session,
    )
    deadline = time.monotonic() + 60
    while not ready.exists() and time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError("the helper process exited early")
        time.sleep(0.05)
    assert ready.exists(), "the helper never started its child"
    return proc


def _pipe_closes_within(proc, seconds: float) -> bool:
    """True when every holder of `proc`'s stdout pipe is gone within `seconds`.

    The child inherits the parent's stdout, so the read only returns once BOTH
    have exited -- which is exactly the question "did the descendants die too?".
    """
    done = threading.Event()

    def drain():
        try:
            proc.stdout.read()
        finally:
            done.set()

    threading.Thread(target=drain, daemon=True).start()
    return done.wait(seconds)


def test_kill_tree_takes_the_workers_descendants_too(tmp_path):
    """``subprocess.run(..., timeout=)`` reaps only the direct child.

    A ``gist-sampling`` cell runs joblib/loky with ``n_jobs=-1``; on a hard
    timeout its workers used to stay resident, holding the multi-GiB working
    sets the harness reports, while the next cell's wall clock and peak RSS were
    measured against them.
    """
    compare = _load("compare")

    # The control: killing only the direct process leaves the child holding the
    # pipe. Without this the assertion below could pass on a helper that never
    # spawned anything.
    control_dir = tmp_path / "control"
    control_dir.mkdir()
    control = _spawn_parent_with_a_child(control_dir)
    try:
        control.kill()
        control.wait(timeout=30)
        assert not _pipe_closes_within(control, 5), (
            "the helper's child did not outlive a plain kill(); the test below "
            "would prove nothing"
        )
    finally:
        compare.kill_tree(control)

    proc = _spawn_parent_with_a_child(tmp_path)
    compare.kill_tree(proc)
    assert _pipe_closes_within(proc, 60), (
        "a descendant of the worker survived kill_tree"
    )
