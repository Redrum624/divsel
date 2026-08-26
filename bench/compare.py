#!/usr/bin/env python
"""divsel against the incumbents: one harness, one evaluator, identical inputs.

    python bench/compare.py --n 2000 --dim 64 --k 5                         # smoke
    python bench/compare.py --n 10000 --dim 384 --k 10,50,100 --out docs/benchmarks/results-2026-08-22.json
    python bench/compare.py --n 100000 --dim 384 --k 10 --utility linear --out docs/benchmarks/results-2026-08-22.json
    python bench/compare.py --all [--large] --out ...

What is compared
----------------
* ``divsel``                      -- ``divsel.gist_select_full``, the library under test, with the exact
                                     O(n^2) diameter the paper uses (``diameter="exact"``, the default).
* ``divsel[diameter=approx]``     -- the same call with ``diameter="approx"`` (farthest-point double sweeps,
                                     estimate in ``[d_max/2, d_max]``), the documented option for large n.
* ``gist-select``               -- PyPI ``gist-select``: ``gist.gist(points, utility, distance, k, lam, eps,
                                     n_jobs=1, seed)``, the README's documented call. Linear utility only;
                                     its package ships no facility-location utility.
* ``gist-select[n_jobs=-1]``      -- the same call with ``n_jobs=-1`` (the README's "parallel threshold
                                     sweep", joblib threads).
* ``gist-sampling``               -- git ``musubi-labs/gist-sampling``: ``GISTSelector(n_samples, metric,
                                     epsilon, lambda_diversity, random_state).fit(X)``, documented defaults
                                     (``mode="auto"``, which is the approximate kNN mode above n = 2000).
                                     Facility location only; it has no linear utility.
* ``gist-sampling[mode=exact]``   -- the same selector with ``mode="exact"`` (dense O(n^2) distance matrix).
* ``mmr``                         -- a naive numpy MMR baseline, defined in this file.

Inputs are generated inside every worker from ``np.random.default_rng(0)``: float32 Gaussian rows,
L2-normalised, and float64 weights uniform in [0, 1). Every library gets the same array.

Every library's selection is scored by ONE evaluator (``evaluate``) with divsel's definitions:
``f = g(S) + lam * div(S)``, ``div`` = minimum pairwise cosine distance (``d_max`` when ``|S| <= 1``),
Linear ``g = sum(w[S])``, FacilityLocation ``g = sum_i max_{j in S} max(0, 1 - dist(i, j) / scale)`` with
``scale = 1.0`` for cosine. The evaluator runs in float64 from the same float32 rows. Where a library reports
its own objective it is recorded next to the evaluator's number; for divsel the two are compared as a
self-check.

Protocol
--------
Each (cell, method) runs in a fresh subprocess so that peak RSS is per method and a crash or a hang in one
library cannot take the harness down. Inside the worker: one warm-up call (this also absorbs numba JIT and
BLAS thread start-up), then ``--runs`` timed calls with ``time.perf_counter``; the median is reported with the
min and max. A warm-up longer than ``--timeout`` seconds ends the cell as ``timeout``. Peak RSS is
``psutil.Process().memory_info().peak_wset`` on Windows and ``resource.getrusage(RUSAGE_SELF).ru_maxrss``
elsewhere -- process-wide, so it includes the interpreter, numpy, the generated data and the library's
imports; the value after data generation and before the first call is reported as ``baseline``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import importlib.metadata as _md
import json
import os
import platform
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

SEED = 0
LAM = 1.0
EPS = 0.1
METRIC = "cosine"
MMR_LAMBDA = 0.5
UTILITIES = ("linear", "facility_location")
METHOD_NAMES = (
    "divsel",
    "divsel[diameter=approx]",
    "gist-select",
    "gist-select[n_jobs=-1]",
    "gist-sampling",
    "gist-sampling[mode=exact]",
    "mmr",
)
# The diameter the shared evaluator would need for |S| <= 1 is an O(n^2) scan; above this n it is skipped.
EXACT_DIAMETER_MAX_N = 200_000
# R-G21: facility location is measured at n = 10k only. Its marginal is O(n * dim), so one call at 10k
# already takes minutes (docs/benchmarks/README.md); the explicit --n path enforces the same ceiling --all does.
FACILITY_LOCATION_MAX_N = 10_000


class Unavailable(Exception):
    """A method cannot run this cell for a known, stated reason."""


# --------------------------------------------------------------------------------------------------
# Inputs
# --------------------------------------------------------------------------------------------------


def make_data(n: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """float32 Gaussian rows, L2-normalised in place, and float64 weights in [0, 1). Seeded, so every
    worker regenerates the identical arrays."""
    rng = np.random.default_rng(SEED)
    x = rng.standard_normal((n, dim), dtype=np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    x = np.ascontiguousarray(x, dtype=np.float32)
    w = rng.random(n)
    return x, w


# --------------------------------------------------------------------------------------------------
# The shared evaluator
# --------------------------------------------------------------------------------------------------


def cosine_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """divsel's cosine distance on L2-normalised rows: ``clamp(1 - a.b, 0, 2)``, here in float64."""
    return np.clip(1.0 - a.astype(np.float64) @ b.astype(np.float64).T, 0.0, 2.0)


def exact_cosine_diameter(x: np.ndarray, block: int = 2048) -> float:
    """``max_{u,v} dist(u, v)`` by a blockwise O(n^2) scan; only needed when ``|S| <= 1``."""
    n = x.shape[0]
    if n < 2:
        return 0.0
    x64 = x.astype(np.float64)
    min_dot = np.inf
    for start in range(0, n, block):
        dots = x64[start : start + block] @ x64.T
        min_dot = min(min_dot, float(dots.min()))
    return float(np.clip(1.0 - min_dot, 0.0, 2.0))


def evaluate(
    x: np.ndarray, w: np.ndarray, selected: list[int], utility: str, lam: float = LAM
) -> dict:
    """``f = g(S) + lam * div(S)`` with divsel's definitions, in float64, for any library's selection."""
    sel = [int(i) for i in selected]
    if len(set(sel)) != len(sel):
        raise ValueError(f"selection has duplicate indices: {sel}")
    xs = x[sel]
    if utility == "linear":
        g = float(np.sum(w[sel])) if sel else 0.0
    elif utility == "facility_location":
        if sel:
            sim = np.maximum(0.0, 1.0 - cosine_dist(x, xs) / 1.0)  # scale = 1.0 under cosine
            g = float(sim.max(axis=1).sum())
        else:
            g = 0.0
    else:
        raise ValueError(utility)
    if len(sel) >= 2:
        d = cosine_dist(xs, xs)
        iu = np.triu_indices(len(sel), k=1)
        div = float(d[iu].min())
        div_note = None
    elif x.shape[0] <= EXACT_DIAMETER_MAX_N:
        div = exact_cosine_diameter(x)
        div_note = "|S| <= 1: div = exact diameter d_max"
    else:
        div = None
        div_note = f"|S| <= 1 at n > {EXACT_DIAMETER_MAX_N}: diameter scan skipped"
    # `lam == 0` contributes a literal 0.0, exactly as the core does
    # (docs/CONFORMANCE.md rule 18): `div` can be +inf and `0.0 * inf` is nan.
    f = None if div is None else (g if lam == 0.0 else g + lam * div)
    return {"f": f, "g": g, "div": div, "size": len(sel), "div_note": div_note}


# --------------------------------------------------------------------------------------------------
# Methods. Each returns (selected indices, what the library itself reported).
# --------------------------------------------------------------------------------------------------


def run_divsel(x, w, k, utility, opts):
    import divsel

    weights = w if utility == "linear" else None
    out = divsel.gist_select_full(
        x,
        weights,
        k=k,
        lam=LAM,
        eps=EPS,
        metric=METRIC,
        utility=utility,
        diameter=opts.get("diameter", "exact"),
    )
    reported = {
        "f": out["f_value"],
        "g": out["g_value"],
        "div": out["div"],
        "stage": out["stage"],
        "threshold": out["threshold"],
        "d_max": out["d_max"],
        "diameter": opts.get("diameter", "exact"),
    }
    return list(out["selected"]), reported


def run_divsel_approx(x, w, k, utility, opts):
    return run_divsel(x, w, k, utility, {**opts, "diameter": "approx"})


def _gist_select(x, w, k, utility, n_jobs):
    import gist  # PyPI gist-select 0.1.0: src/gist/algorithm.py

    if utility != "linear":
        raise Unavailable(
            "gist-select ships no facility-location utility (gist/objectives.py exports "
            "LinearUtility, CoverageFunction, SubmodularFunction only)"
        )
    res = gist.gist(
        points=x,
        utility=gist.LinearUtility(w),
        distance=gist.CosineDistance(),
        k=k,
        lam=LAM,
        eps=EPS,
        n_jobs=n_jobs,
        seed=SEED,
    )
    reported = {
        "f": float(res.objective_value),
        "g": float(res.utility_value),
        "div": float(res.diversity),
        "n_jobs": n_jobs,
    }
    return [int(i) for i in res.indices], reported


def run_gist_select(x, w, k, utility, opts):
    return _gist_select(x, w, k, utility, n_jobs=1)


def run_gist_select_threads(x, w, k, utility, opts):
    return _gist_select(x, w, k, utility, n_jobs=-1)


def _gist_sampling(x, w, k, utility, mode):
    from gist_sampling import GISTSelector  # git musubi-labs/gist-sampling: selectors/gist_selector.py

    if utility != "facility_location":
        raise Unavailable(
            'gist-sampling supports utility="facility_location" only '
            '(selectors/gist_selector.py: UtilityType = Literal["facility_location"])'
        )
    kwargs = dict(
        n_samples=k,
        utility="facility_location",
        metric=METRIC,
        epsilon=EPS,
        lambda_diversity=LAM,
        random_state=SEED,
    )
    if mode is not None:
        kwargs["mode"] = mode
    sel = GISTSelector(**kwargs).fit(x)
    reported = {
        "f": float(sel.objective_value_),
        "div": float(sel.diversity_),
        "mode_used": sel.mode_used_,
        "similarity": "rbf (its own g; scored below with divsel's FL definition)",
    }
    return [int(i) for i in sel.selected_indices_], reported


def run_gist_sampling(x, w, k, utility, opts):
    return _gist_sampling(x, w, k, utility, mode=None)


def run_gist_sampling_exact(x, w, k, utility, opts):
    return _gist_sampling(x, w, k, utility, mode="exact")


def run_mmr(x, w, k, utility, opts):
    """Naive MMR: greedily pick ``argmax lam * rel(v) - (1 - lam) * max_{s in S} sim(v, s)`` with
    ``lam = 0.5``, ``rel`` = the Linear weights (already in [0, 1)), ``sim`` = cosine similarity on the
    normalised rows, and ``max sim`` taken as 0 for the empty selection. The same ``rel`` is used for the
    facility-location cells: MMR has no notion of ``g``; it is scored by the shared evaluator like the rest."""
    n = x.shape[0]
    rel = w.astype(np.float64)
    max_sim = np.zeros(n)
    chosen = np.zeros(n, dtype=bool)
    selected: list[int] = []
    for _ in range(min(k, n)):
        score = MMR_LAMBDA * rel - (1.0 - MMR_LAMBDA) * max_sim
        score[chosen] = -np.inf
        v = int(np.argmax(score))
        selected.append(v)
        chosen[v] = True
        np.maximum(max_sim, (x @ x[v]).astype(np.float64), out=max_sim)
    return selected, {"lambda": MMR_LAMBDA, "rel": "linear weights"}


METHODS = {
    "divsel": run_divsel,
    "divsel[diameter=approx]": run_divsel_approx,
    "gist-select": run_gist_select,
    "gist-select[n_jobs=-1]": run_gist_select_threads,
    "gist-sampling": run_gist_sampling,
    "gist-sampling[mode=exact]": run_gist_sampling_exact,
    "mmr": run_mmr,
}
if tuple(METHODS) != METHOD_NAMES:  # not an assert: that would vanish under `python -O`
    raise RuntimeError(f"METHODS {tuple(METHODS)} and METHOD_NAMES {METHOD_NAMES} disagree")


# --------------------------------------------------------------------------------------------------
# Environment record
# --------------------------------------------------------------------------------------------------


def peak_rss_mib() -> float | None:
    try:
        if sys.platform == "win32":
            import psutil

            return psutil.Process().memory_info().peak_wset / 2**20
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return rss / 2**20 if sys.platform == "darwin" else rss / 2**10
    except Exception:  # noqa: BLE001 -- the number is optional, the run is not
        return None


def _dist_version(name: str) -> str | None:
    try:
        return _md.version(name)
    except _md.PackageNotFoundError:
        return None


def _git_commit_of(dist_name: str) -> str | None:
    try:
        raw = _md.distribution(dist_name).read_text("direct_url.json")
        return json.loads(raw).get("vcs_info", {}).get("commit_id") if raw else None
    except Exception:  # noqa: BLE001
        return None


def cpu_name() -> str:
    try:
        if sys.platform == "win32":
            out = subprocess.run(
                ["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return out.stdout.strip() or platform.processor()
        if sys.platform == "darwin":
            out = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=30
            )
            return out.stdout.strip() or platform.processor()
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor()


def environment() -> dict:
    try:
        import psutil

        ram = psutil.virtual_memory().total / 2**30
    except Exception:  # noqa: BLE001
        ram = None
    try:
        blas = np.__config__.CONFIG["Build Dependencies"]["blas"]
        blas = f"{blas.get('name')} {blas.get('version')}"
    except Exception:  # noqa: BLE001
        blas = None
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=Path(__file__).parent
        ).stdout.strip()
    except Exception:  # noqa: BLE001
        head = None
    return {
        "date": _dt.date.today().isoformat(),
        "machine": {
            "cpu": cpu_name(),
            "logical_cpus": os.cpu_count(),
            "ram_gib": None if ram is None else round(ram, 1),
            "os": platform.platform(),
        },
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "numpy": np.__version__,
        "numpy_blas": blas,
        "scipy": _dist_version("scipy"),
        "numba": _dist_version("numba"),
        "psutil": _dist_version("psutil"),
        "divsel": _dist_version("divsel"),
        "gist-select": _dist_version("gist-select"),
        "gist-sampling": _dist_version("gist-sampling"),
        "gist-sampling_commit": _git_commit_of("gist-sampling"),
        "divsel_git_head": head,
        "seed": SEED,
        "lam": LAM,
        "eps": EPS,
        "metric": METRIC,
        "mmr_lambda": MMR_LAMBDA,
    }


# --------------------------------------------------------------------------------------------------
# Worker: one (cell, method) in this process
# --------------------------------------------------------------------------------------------------


def _first_line(exc: BaseException) -> str:
    text = str(exc).strip().splitlines()
    return f"{type(exc).__name__}: {text[0] if text else ''}".strip()


def worker(args) -> dict:
    opts = {"diameter": args.diameter}
    fn = METHODS[args.method]
    result = {
        "n": args.n,
        "dim": args.dim,
        "k": args.k,
        "utility": args.utility,
        "method": args.method,
        "status": "ok",
        "python": sys.version.split()[0],
    }
    x, w = make_data(args.n, args.dim)
    result["baseline_rss_mib"] = peak_rss_mib()
    try:
        t0 = time.perf_counter()
        selected, reported = fn(x, w, args.k, args.utility, opts)
        warm = time.perf_counter() - t0
        result["warmup_s"] = warm
        if warm > args.timeout:
            result.update(status="timeout", reason=f"warm-up call took {warm:.1f} s > {args.timeout} s")
        else:
            times, selections = [], []
            for _ in range(args.runs):
                t0 = time.perf_counter()
                selected, reported = fn(x, w, args.k, args.utility, opts)
                times.append(time.perf_counter() - t0)
                selections.append(list(selected))
            result["wall_runs_s"] = times
            result["wall_median_s"] = float(np.median(times))
            result["selection_stable"] = all(s == selections[0] for s in selections)
    except Unavailable as exc:
        result.update(status="unavailable", reason=str(exc))
    except ImportError as exc:
        result.update(status="unavailable", reason=_first_line(exc))
    except Exception as exc:  # noqa: BLE001 -- the cell reports the error instead of dying
        result.update(status="error", reason=_first_line(exc))
    if result["status"] in ("ok", "timeout"):
        # A selection the evaluator rejects (duplicate indices, an index out of range) is the
        # library's defect and is reported as this cell's error, not as a driver crash.
        try:
            score = evaluate(x, w, selected, args.utility)
        except Exception as exc:  # noqa: BLE001
            result.update(status="error", reason=f"evaluate: {_first_line(exc)}")
            result["selected"] = [int(i) for i in selected]
            result["library_reported"] = reported
        else:
            result["selected"] = [int(i) for i in selected]
            result.update({f"eval_{key}": val for key, val in score.items()})
            result["library_reported"] = reported
            if args.method.startswith("divsel") and score["f"] is not None:
                result["self_check_abs_diff"] = abs(score["f"] - reported["f"])
    result["peak_rss_mib"] = peak_rss_mib()
    return result


# --------------------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------------------


# Slack added to a cell's own timeout before the driver stops waiting: the
# worker has to import numpy, build its fixture and print its record after the
# last measured run. A test overrides it to drive the timeout branch.
TIMEOUT_GRACE_S = 120


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill `proc` **and everything it spawned**.

    `subprocess.run(..., timeout=...)` kills and reaps only the direct child. A
    `gist-sampling` cell runs joblib/loky with `n_jobs=-1`, so a timeout there
    would leave its worker processes alive holding the multi-GiB working sets
    this harness reports -- and the next cell's wall clock and peak RSS would be
    measured against them. On POSIX the worker is started in its own session, so
    one `killpg` takes the group; on Windows `taskkill /T` walks the tree.
    """
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
            capture_output=True,
            check=False,
        )
    else:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    proc.kill()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:  # pragma: no cover - the OS refused a SIGKILL
        pass


# The two spellings of the same library. At n >= 1M they are the same cell.
DIVSEL_SPELLINGS = ("divsel", "divsel[diameter=approx]")


def resolve_methods(methods, n: int) -> list[tuple[str, str | None, str]]:
    """`(label, effective, reason)` for every requested method, in request order.

    `effective is None` means no cell is run and `reason` says why. Every label
    the caller asked for gets exactly one entry, which is the point: `cells_for`
    forces `diameter="approx"` at `n >= 1M` (R-G21), so plain `divsel` there
    names a configuration that did not run, and both spellings resolve to the
    one cell that did. An earlier version dropped the second spelling with
    `continue` and no record at all, so `--all --large` with the default methods
    wrote 6 rows where 7 were asked for and nothing said what happened to the
    seventh.

    Below 1M nothing is rewritten: the two divsel rows are genuinely different
    measurements there.
    """
    if n < 1_000_000:
        return [(m, m, "") for m in methods]
    out: list[tuple[str, str | None, str]] = []
    emitted = False
    for method in methods:
        if method not in DIVSEL_SPELLINGS:
            out.append(
                (method, None, "n = 1M is run for divsel[diameter=approx] only (R-G21)")
            )
            continue
        if emitted:
            out.append(
                (
                    method,
                    None,
                    "n = 1M forces diameter=approx (R-G21): this is the same cell as "
                    "divsel[diameter=approx], reported once under that label",
                )
            )
            continue
        emitted = True
        out.append((method, "divsel[diameter=approx]", ""))
    return out


def run_cell(method: str, n: int, dim: int, k: int, utility: str, args, diameter: str) -> dict:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--method",
        method,
        "--n",
        str(n),
        "--dim",
        str(dim),
        "--k",
        str(k),
        "--utility",
        utility,
        "--runs",
        str(args.runs),
        "--timeout",
        str(args.timeout),
        "--diameter",
        diameter,
    ]
    hard_limit = args.timeout * (args.runs + 1) + TIMEOUT_GRACE_S
    base = {"n": n, "dim": dim, "k": k, "utility": utility, "method": method}
    t0 = time.perf_counter()
    # Its own session (POSIX) so a timeout can take the whole process group; on
    # Windows `kill_tree` uses taskkill /T instead. See `kill_tree`.
    session = {} if os.name == "nt" else {"start_new_session": True}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        **session,
    )
    try:
        stdout, stderr = proc.communicate(timeout=hard_limit)
    except subprocess.TimeoutExpired:
        kill_tree(proc)
        # CPython documents the leak this closes, in `Popen._communicate`: "If
        # we time out, the threads remain reading and the fds left open in case
        # the user calls communicate again." Nothing here will, so the second
        # call is made now -- with the tree already dead it returns at once --
        # and the pipes are closed behind it. Without it every timed-out cell
        # leaves a reader thread blocked on a pipe whose last holder is a
        # descendant `kill_tree` could not reach, in the process that reports
        # peak RSS.
        try:
            proc.communicate(timeout=30)
        except (subprocess.TimeoutExpired, ValueError):  # pragma: no cover
            pass
        for pipe in (proc.stdout, proc.stderr):
            if pipe is not None and not pipe.closed:
                try:
                    pipe.close()
                except OSError:  # pragma: no cover
                    pass
        return {
            **base,
            "status": "timeout",
            "reason": f"worker and its descendants killed after {hard_limit} s",
            "worker_wall_s": time.perf_counter() - t0,
        }
    elapsed = time.perf_counter() - t0
    if proc.returncode != 0:
        err = [ln for ln in stderr.strip().splitlines() if ln.strip()]
        return {
            **base,
            "status": "error",
            "reason": f"worker exit {proc.returncode}: {err[-1] if err else '(no stderr)'}",
            "worker_wall_s": elapsed,
        }
    lines = [ln for ln in stdout.splitlines() if ln.startswith("{")]
    if not lines:
        return {
            **base,
            "status": "error",
            "reason": "worker exited 0 without printing a JSON result line",
            "worker_wall_s": elapsed,
        }
    out = json.loads(lines[-1])
    out["worker_wall_s"] = elapsed
    return out


def cells_for(args) -> list[tuple[int, int, int, str, str]]:
    """(n, dim, k, utility, diameter) cells, in run order."""
    cells = []
    if args.all:
        grid = [(n, d, k) for n in (10_000, 100_000) for d in (384, 768) for k in (10, 50, 100)]
        for n, d, k in grid:
            cells.append((n, d, k, "linear", "exact"))
            if n == 10_000:  # R-G21: facility location only at n = 10k
                cells.append((n, d, k, "facility_location", "exact"))
        if args.large:  # R-G21: n = 1M is divsel Linear with diameter="approx" only
            for d in (384, 768):
                for k in (10, 50, 100):
                    cells.append((1_000_000, d, k, "linear", "approx"))
        return cells
    utilities = UTILITIES if args.utility == "both" else (args.utility,)
    too_large = [n for n in args.n if n > FACILITY_LOCATION_MAX_N]
    if "facility_location" in utilities and too_large:
        raise SystemExit(
            f"facility_location is limited to n <= {FACILITY_LOCATION_MAX_N} (R-G21; --all applies the "
            f"same ceiling, and --large adds n = 1M for divsel Linear only), got --n {too_large}; "
            f"pass --utility linear for those n"
        )
    for n in args.n:
        for d in args.dim:
            for k in args.k:
                for u in utilities:
                    diameter = "approx" if n >= 1_000_000 else "exact"
                    cells.append((n, d, k, u, diameter))
    return cells


def fmt(v, nd=3):
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def markdown_table(results: list[dict]) -> str:
    head = (
        "| n | dim | k | utility | method | wall median s (min-max) | peak RSS MiB (baseline) "
        "| f(S) | g(S) | div(S) | \\|S\\| | library f | note |\n"
        "|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|"
    )
    rows = [head]
    for r in results:
        if r["status"] == "ok":
            wall = f"{r['wall_median_s']:.3f} ({min(r['wall_runs_s']):.3f}-{max(r['wall_runs_s']):.3f})"
        else:
            wall = r["status"]
        rss = f"{fmt(r.get('peak_rss_mib'), 0)} ({fmt(r.get('baseline_rss_mib'), 0)})"
        lib = r.get("library_reported") or {}
        note = r.get("reason") or ""
        extras = []
        if "self_check_abs_diff" in r:
            extras.append(f"self-check abs(f_eval - f_divsel) = {r['self_check_abs_diff']:.2e}")
        if lib.get("mode_used"):
            extras.append(f"mode_used={lib['mode_used']}")
        if lib.get("stage"):
            extras.append(f"stage={lib['stage']}, diameter={lib.get('diameter')}")
        if r.get("eval_div_note"):
            extras.append(r["eval_div_note"])
        if r.get("selection_stable") is False:
            extras.append("selection differs between runs")
        note = "; ".join([note, *extras]).strip("; ")
        rows.append(
            f"| {r['n']} | {r['dim']} | {r['k']} | {r['utility']} | {r['method']} | {wall} | {rss} "
            f"| {fmt(r.get('eval_f'), 4)} | {fmt(r.get('eval_g'), 4)} | {fmt(r.get('eval_div'), 4)} "
            f"| {fmt(r.get('eval_size'))} | {fmt(lib.get('f'), 4)} | {note} |"
        )
    return "\n".join(rows)


def merge_into(path: Path, meta: dict, results: list[dict]) -> dict:
    key = lambda r: (r["n"], r["dim"], r["k"], r["utility"], r["method"])  # noqa: E731
    doc = {"meta": meta, "results": []}
    if path.exists():
        doc = json.loads(path.read_text(encoding="utf-8"))
        doc.setdefault("meta", meta)
        doc.setdefault("results", [])
    existing = {key(r): i for i, r in enumerate(doc["results"])}
    for r in results:
        r = {**r}
        if "selected" in r and len(r["selected"]) > 100:
            r["selected"] = r["selected"][:100]
            r["selected_truncated"] = True
        if key(r) in existing:
            doc["results"][existing[key(r)]] = r
        else:
            doc["results"].append(r)
    doc["meta"] = meta
    doc["results"].sort(key=lambda r: (r["n"], r["dim"], r["k"], r["utility"], METHOD_NAMES.index(r["method"])))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return doc


def _int_list(text: str) -> list[int]:
    return [int(t.replace("_", "")) for t in text.split(",") if t.strip()]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=_int_list, default=[2000], help="comma-separated list")
    p.add_argument("--dim", type=_int_list, default=[64], help="comma-separated list")
    p.add_argument("--k", type=_int_list, default=[5], help="comma-separated list")
    p.add_argument("--utility", choices=(*UTILITIES, "both"), default="both")
    p.add_argument("--methods", default=",".join(METHOD_NAMES), help="comma-separated subset")
    p.add_argument("--all", action="store_true", help="the small/medium matrix (see cells_for)")
    p.add_argument("--large", action="store_true", help="with --all: add n = 1M, divsel Linear, diameter=approx")
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--timeout", type=int, default=600, help="seconds; a warm-up call longer than this is a timeout")
    p.add_argument("--out", type=Path, default=None, help="JSON file; existing cells with the same key are replaced")
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--method", choices=METHOD_NAMES, help=argparse.SUPPRESS)
    p.add_argument("--diameter", choices=("exact", "approx"), default="exact", help=argparse.SUPPRESS)
    args = p.parse_args(argv)
    if args.worker:
        args.n, args.dim, args.k = args.n[0], args.dim[0], args.k[0]
    return args


def main(argv=None) -> int:
    for stream in (sys.stdout, sys.stderr):  # Windows consoles default to cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    args = parse_args(argv)
    if args.worker:
        print(json.dumps(worker(args)))
        return 0
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    unknown = [m for m in methods if m not in METHODS]
    if unknown:
        print(f"unknown methods: {unknown}; known: {list(METHODS)}", file=sys.stderr)
        return 2
    meta = environment()
    meta.update(runs=args.runs, timeout_s=args.timeout)
    # The invocation is recorded on every cell, not in `meta`: `--out` merges cell by cell across
    # invocations, so a document-level argv could only ever describe the last one.
    argv = sys.argv[1:]
    print(json.dumps({**meta, "argv": argv}, indent=1), file=sys.stderr)
    results = []
    for n, dim, k, utility, diameter in cells_for(args):
        for method, effective, reason in resolve_methods(methods, n):
            if effective is None:
                results.append(
                    {"n": n, "dim": dim, "k": k, "utility": utility, "method": method,
                     "status": "not run", "reason": reason, "argv": argv}
                )
                continue
            print(f"[{_dt.datetime.now():%H:%M:%S}] n={n} dim={dim} k={k} {utility} {effective} ...",
                  file=sys.stderr, flush=True)
            r = run_cell(effective, n, dim, k, utility, args, diameter)
            r["argv"] = argv
            results.append(r)
            summary = (
                f"{r['wall_median_s']:.3f} s, f={r['eval_f']}" if r["status"] == "ok" else f"{r['status']}: {r.get('reason')}"
            )
            print(f"    -> {summary}", file=sys.stderr, flush=True)
    if args.out:
        merge_into(args.out, meta, results)
        print(f"wrote {args.out}", file=sys.stderr)
    print(markdown_table(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
