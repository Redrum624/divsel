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
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

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


def _pid_is_alive(pid: int) -> bool:
    """Cross-platform "is this pid still running", for pids we do not own."""
    if os.name == "nt":
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/NH"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # pragma: no cover - alive, owned by someone else
        return True
    return True


def _force_kill(pid: int) -> None:
    """Kill one pid directly, for a descendant `kill_tree` can no longer reach."""
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/PID", str(pid)], capture_output=True, check=False
        )
        return
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):  # pragma: no cover
        pass


def _reap(compare, proc, child_pid=None) -> None:
    """Leave nothing behind: the tree, then the orphan, then the pipe.

    Order matters. `kill_tree` walks from the parent, so it is a no-op once the
    parent has been reaped -- which is exactly the state the control block below
    creates on purpose -- and the grandchild then has to be killed by pid. The
    stdout pipe is closed last, after its holders are gone, so a drain thread's
    `read()` returns instead of blocking on a pipe nobody will ever close.
    """
    compare.kill_tree(proc)
    if child_pid is not None and _pid_is_alive(child_pid):
        _force_kill(child_pid)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover - the OS refused a kill
        pass
    if proc.stdout is not None and not proc.stdout.closed:
        try:
            proc.stdout.close()
        except (OSError, ValueError):  # pragma: no cover - a reader owns it
            pass


def _spawn_parent_with_a_child(tmp_path, ready_timeout: float = 60.0):
    """A process that spawns one child and waits, both holding our stdout pipe.

    Returns `(proc, child_pid)`: the child's pid travels back through the ready
    file, because a caller that reaps `proc` first (the control below does) can
    no longer reach the child through it. On any failure the helper and whatever
    it already spawned are killed here -- the caller's `finally` is only entered
    once this function has returned, so a readiness check that fails on a loaded
    runner used to leak both processes for 300 seconds.
    """
    ready = tmp_path / "ready"
    code = (
        "import pathlib, subprocess, sys, time\n"
        "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
        f"pathlib.Path({str(ready)!r}).write_text(str(child.pid))\n"
        "time.sleep(300)\n"
    )
    session = {} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        **session,
    )
    try:
        deadline = time.monotonic() + ready_timeout
        while not ready.exists() and time.monotonic() < deadline:
            if proc.poll() is not None:
                raise AssertionError("the helper process exited early")
            time.sleep(0.05)
        assert ready.exists(), "the helper never started its child"
        return proc, int(ready.read_text())
    except BaseException:
        # Nothing has been handed to the caller, so this is the only cleanup
        # there will ever be: `kill_tree` takes the helper and, while it is
        # still alive, everything under it; the ready file names the child if it
        # got that far.
        pid = int(ready.read_text()) if ready.exists() else None
        _reap(_load("compare"), proc, pid)
        raise


def _pipe_closes_within(proc, seconds: float):
    """`(closed, reader)`: did every holder of `proc`'s stdout pipe go away?

    The child inherits the parent's stdout, so the read only returns once BOTH
    have exited -- which is exactly the question "did the descendants die too?".
    The reader thread is returned so a caller can join it after the cleanup that
    lets it finish: it is a daemon, but until `read()` returns it holds the pipe
    and a reference to the tree for the rest of the session.
    """
    done = threading.Event()

    def drain():
        try:
            proc.stdout.read()
        finally:
            done.set()

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    return done.wait(seconds), thread


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
    control, control_child = _spawn_parent_with_a_child(control_dir)
    try:
        control.kill()
        control.wait(timeout=30)
        closed, drain = _pipe_closes_within(control, 5)
        assert not closed, (
            "the helper's child did not outlive a plain kill(); the test below "
            "would prove nothing"
        )
    finally:
        # `kill_tree(control)` alone cannot clean this up: `control` has been
        # reaped, so `taskkill /T` has no pid to walk from and `os.getpgid`
        # raises ProcessLookupError into a `pass`. The orphan this block creates
        # on purpose has to be killed by pid, or every run of `pytest
        # python/tests` leaves a 300-second sleeper and its pipe behind.
        _reap(compare, control, control_child)
    drain.join(30)
    assert not drain.is_alive(), (
        "the control's descendants outlived the test: something still holds the "
        "pipe, so the tree this test orphaned on purpose is still running"
    )
    assert not _pid_is_alive(control_child), "the control's child was left running"

    proc, child = _spawn_parent_with_a_child(tmp_path)
    try:
        compare.kill_tree(proc)
        closed, drain = _pipe_closes_within(proc, 60)
        assert closed, "a descendant of the worker survived kill_tree"
    finally:
        _reap(compare, proc, child)
    drain.join(30)


def test_a_helper_that_never_reports_ready_is_not_left_behind(tmp_path, monkeypatch):
    """The helper's own failure path is the only cleanup it will ever get.

    `_spawn_parent_with_a_child` raises *before* returning, so the caller's
    `finally` is never entered: on a loaded runner, where the helper needs
    longer than the readiness deadline, the helper and the 300-second sleeper
    under it were both left running, holding a pipe nobody reads. Forcing the
    deadline to zero reproduces that without waiting a minute for it.
    """
    _load("compare")  # skips the test where bench/ is not present

    spawned = []
    real_popen = subprocess.Popen

    def recording(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        spawned.append(proc)
        return proc

    monkeypatch.setattr(subprocess, "Popen", recording)

    with pytest.raises(AssertionError, match="never started its child"):
        _spawn_parent_with_a_child(tmp_path, ready_timeout=0.0)

    # `kill_tree` shells out to taskkill on Windows, which is a Popen too.
    helpers = [pr for pr in spawned if pr.args and pr.args[0] == sys.executable]
    assert len(helpers) == 1, "the helper is the only interpreter this test starts"
    assert helpers[0].returncode is not None, "the helper was left running"
    assert helpers[0].stdout is None or helpers[0].stdout.closed

    ready = tmp_path / "ready"
    if ready.exists():  # it got as far as spawning the sleeper
        pid = int(ready.read_text())
        deadline = time.monotonic() + 30
        while _pid_is_alive(pid) and time.monotonic() < deadline:
            time.sleep(0.05)
        assert not _pid_is_alive(pid), "the helper's child outlived the failure"


def test_run_cell_reports_a_timeout_and_leaves_no_reader_behind(monkeypatch):
    """The timeout branch of ``run_cell``: nothing drove it before.

    Two things are checked. (1) The record: `status`, a `reason` that says the
    descendants went too, and `worker_wall_s`, which every other status carries
    and this one used to drop. (2) The pipes: CPython's Windows
    `Popen._communicate` leaves its two reader threads running and the fds open
    on a `TimeoutExpired` ("in case the user calls communicate again"), so a
    branch that kills the tree and returns without a second `communicate()`
    leaks a blocked thread per timed-out cell -- inside the process that is
    measuring peak RSS.
    """
    compare = _load("compare")
    monkeypatch.setattr(compare, "TIMEOUT_GRACE_S", 1)
    args = SimpleNamespace(runs=1, timeout=0)

    before = threading.active_count()
    # 200 000 points with the exact O(n^2) diameter cannot finish in a second.
    r = compare.run_cell("divsel", 200_000, 64, 5, "linear", args, "exact")

    assert r["status"] == "timeout", r
    assert "descendants" in r["reason"]
    assert r["worker_wall_s"] > 0.0
    assert (r["n"], r["dim"], r["k"]) == (200_000, 64, 5)

    deadline = time.monotonic() + 30
    while threading.active_count() > before and time.monotonic() < deadline:
        time.sleep(0.05)
    assert threading.active_count() <= before, (
        "a pipe reader outlived the timed-out cell"
    )


def test_run_cell_drives_a_real_worker_end_to_end():
    """``run_cell`` + ``worker`` on the smallest cell that means anything.

    Neither had a test: the driver spawns ``compare.py --worker`` and parses the
    JSON line it prints, and a renamed key or a mis-built command line would only
    show up in a benchmark run. 60 points in 4 dimensions is milliseconds.
    """
    compare = _load("compare")
    args = SimpleNamespace(runs=1, timeout=60)
    r = compare.run_cell("divsel", 60, 4, 3, "linear", args, "exact")

    assert r["status"] == "ok", r.get("reason")
    assert (r["n"], r["dim"], r["k"], r["utility"], r["method"]) == (60, 4, 3, "linear", "divsel")
    assert len(r["selected"]) == 3 == r["eval_size"]
    assert len(set(r["selected"])) == 3
    assert r["worker_wall_s"] > 0.0
    assert len(r["wall_runs_s"]) == args.runs
    # The driver's own evaluator and the library must agree on f(S).
    assert r["self_check_abs_diff"] < 1e-5
    assert r["library_reported"]["diameter"] == "exact"


def test_readme_table_main_validates_rows_before_importing_gist_select(monkeypatch):
    """``--rows`` is parsed before the third-party import, and must stay that way.

    A typo should not require ``gist-select`` to be installed before the caller
    is told about it -- which is also what makes this arm of ``main`` testable
    anywhere.
    """
    table = _load("gist_select_readme_table")
    monkeypatch.setattr(sys, "argv", ["gist_select_readme_table.py", "--rows", "9"])
    with pytest.raises(SystemExit) as info:
        table.main()
    assert "--rows" in str(info.value)
    assert "out of range" in str(info.value)


def test_a_run_that_measures_nothing_says_so_and_fails(tmp_path, capsys):
    """A header-only table, exit 0, is the "degenerate table, green, zero
    measurements" failure ``assemble_matrix.py`` was hardened against on the CI
    side -- and ``test_assemble_matrix_with_no_cells_says_so_and_fails`` forbids
    there.

    Measured before this: ``compare.py --methods "" --n 100 --dim 4 --k 2``
    printed two header lines and exited 0, and so did ``--n ""``. With ``--out``
    it still reached ``merge_into``, whose unconditional ``doc["meta"] = meta``
    restamped a published results file's environment block having measured
    nothing at all.
    """
    compare = _load("compare")
    published = tmp_path / "results.json"
    compare.merge_into(
        published,
        {"machine": "the machine that measured it", "date": "2026-08-22"},
        [{"n": 10, "dim": 2, "k": 2, "utility": "linear", "method": "divsel", "eval_f": 1.0}],
    )
    before = json.loads(published.read_text(encoding="utf-8"))

    for argv in (
        ["--methods", "", "--n", "100", "--dim", "4", "--k", "2"],
        ["--n", "", "--dim", "4", "--k", "2"],
        ["--dim", "", "--n", "100", "--k", "2"],
        ["--k", "", "--n", "100", "--dim", "4"],
    ):
        code = compare.main([*argv, "--out", str(published)])
        text = capsys.readouterr().out
        assert code != 0, argv
        assert "| n | dim |" not in text, f"a degenerate table was printed anyway: {argv}"
        assert "no cells" in text.lower() or "no methods" in text.lower(), argv
        # The published file is untouched -- meta included.
        assert json.loads(published.read_text(encoding="utf-8")) == before, argv


def test_run_cell_kills_the_worker_tree_on_any_exception_not_just_a_timeout(tmp_path):
    """``run_cell`` guarded only ``subprocess.TimeoutExpired``.

    Any other exception out of ``proc.communicate`` -- ``KeyboardInterrupt``,
    ``MemoryError``, ``OSError`` -- left the worker and its joblib/loky
    descendants alive with no ``try/finally`` around the ``Popen``. Ctrl-C
    during a 100k facility-location cell was exactly the condition
    ``kill_tree``'s own docstring says corrupts the next cell's wall clock and
    peak RSS.
    """
    compare = _load("compare")
    proc = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(600)"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    killed = []
    real_kill_tree = compare.kill_tree

    def spy(p):
        killed.append(p)
        real_kill_tree(p)

    class Args:
        runs = 1
        timeout = 60
        worker = False

    try:
        with mock.patch.object(compare, "kill_tree", spy), mock.patch.object(
            compare.subprocess, "Popen", lambda *a, **kw: proc
        ), mock.patch.object(
            proc, "communicate", mock.Mock(side_effect=KeyboardInterrupt("ctrl-c"))
        ):
            with pytest.raises(KeyboardInterrupt):
                compare.run_cell("divsel", 10, 2, 2, "linear", Args(), "exact")
        assert killed == [proc], "the worker tree was not killed on the way out"
        assert proc.wait(timeout=30) is not None, "the worker is still running"
    finally:
        real_kill_tree(proc)
        proc.wait(timeout=30)
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None and not pipe.closed:
                pipe.close()
