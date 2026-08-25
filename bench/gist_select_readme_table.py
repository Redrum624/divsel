#!/usr/bin/env python
"""Re-run the rows of gist-select 0.1.0's README "Performance" table, as the README states them.

The README (PyPI gist-select 0.1.0, section "Performance") says: "Benchmarked on Apple M-series,
single-threaded, eps=0.1" and lists 10K/64/k=50 0.3s, 100K/128/k=100 6s, 500K/128/k=100 ~30s,
2M/128/k=100 ~2 min. No script in the repository (kclaka/gist-select@f6281f3) produces those numbers --
the only scale test is tests/test_scale.py (1.5M points, 64-d, k=100, n_jobs=4, no timing assertion).

This script uses the README's own Quick Start recipe for the inputs -- ``rng = default_rng(42)``,
``rng.standard_normal((n, d)).astype(np.float32)``, ``weights = rng.random(n)``, ``LinearUtility``,
``EuclideanDistance()``, ``lam=1.0``, ``seed=42`` -- with ``eps=0.1`` and ``n_jobs=1`` as the table states,
and times one call per row with ``time.perf_counter``. Rows are run in the order given; pass ``--rows`` to
pick a subset (e.g. ``--rows 1,2,3``).

    python bench/gist_select_readme_table.py --rows 1,2,3,4
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np

ROWS = [  # (n, d, k, README time)
    (10_000, 64, 50, "0.3s"),
    (100_000, 128, 100, "6s"),
    (500_000, 128, 100, "~30s"),
    (2_000_000, 128, 100, "~2 min"),
]


def parse_rows(spec: str) -> list[int]:
    """1-based row numbers, validated.

    Unvalidated, ``--rows 0`` indexed ``ROWS[-1]`` and silently ran the
    2,000,000-point row (about 1 GB of float32 and 17 minutes) for a caller who
    asked for nothing, and a non-integer died with a bare ``ValueError``.
    """
    rows = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            raise SystemExit(f"--rows: empty entry in {spec!r}; rows are 1..{len(ROWS)}")
        try:
            idx = int(token)
        except ValueError:
            raise SystemExit(f"--rows: {token!r} is not an integer; rows are 1..{len(ROWS)}") from None
        if not 1 <= idx <= len(ROWS):
            raise SystemExit(f"--rows: row {idx} is out of range; rows are 1..{len(ROWS)}")
        rows.append(idx)
    if not rows:
        raise SystemExit(f"--rows: nothing selected; rows are 1..{len(ROWS)}")
    return rows


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", default="1,2,3,4",
                   help=f"comma-separated 1-based row numbers, 1..{len(ROWS)}")
    p.add_argument("--n-jobs", type=int, default=1)
    args = p.parse_args()
    rows = parse_rows(args.rows)
    from gist import EuclideanDistance, LinearUtility, gist

    import gist as gist_pkg
    import importlib.metadata as md

    print(f"python {sys.version.split()[0]}, numpy {np.__version__}, gist-select {md.version('gist-select')} "
          f"from {gist_pkg.__file__}", flush=True)
    for idx in rows:
        n, d, k, claimed = ROWS[idx - 1]
        rng = np.random.default_rng(42)
        t0 = time.perf_counter()
        points = rng.standard_normal((n, d)).astype(np.float32)
        weights = rng.random(n)
        gen = time.perf_counter() - t0
        print(f"row {idx}: n={n} d={d} k={k} README says {claimed}; data generated in {gen:.1f} s; "
              f"running gist(..., lam=1.0, eps=0.1, n_jobs={args.n_jobs}, seed=42) ...", flush=True)
        t0 = time.perf_counter()
        res = gist(points=points, utility=LinearUtility(weights), distance=EuclideanDistance(), k=k,
                   lam=1.0, eps=0.1, n_jobs=args.n_jobs, seed=42)
        wall = time.perf_counter() - t0
        print(f"row {idx}: measured {wall:.2f} s; |S|={len(res.indices)} objective={res.objective_value:.4f} "
              f"diversity={res.diversity:.4f}", flush=True)
        del points, weights, res
    return 0


if __name__ == "__main__":
    sys.exit(main())
