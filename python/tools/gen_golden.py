"""Generate ``test-assets/golden-selection.json`` — the cross-language golden fixtures.

divsel is the REFERENCE implementation of GIST (arXiv:2405.18754v3): the Aura
(Python) and limbic (TypeScript) ports prove conformance against the file this
script writes.  See ``docs/CONFORMANCE.md`` for the contract.

Usage::

    python python/tools/gen_golden.py           # (re)write test-assets/golden-selection.json
    python python/tools/gen_golden.py --check   # regenerate in memory, byte-compare with the file

Design rules (R-G14):

* All ``vectors`` and Linear ``utilities`` values are dyadic rationals —
  multiples of 1/64 in [-4, 4] — so every language prints and parses them
  exactly (they are exact in both f32 and f64).
* Expected values come from ``divsel.gist_select_full`` itself; floats are
  written with Python ``repr`` (shortest round-trip), which is what
  ``json.dump`` uses.
* **Robustness margin**: no fixture may sit on a knife edge that a 1-ulp
  platform difference could flip.  For every case (deliberate-tie cases
  exempted, their ties being exact dyadic arithmetic) a pure-Python float64
  brute force enumerates all subsets of size <= k and requires
    (a) the best and second-best *distinct* f values are >= 1e-4 relative apart,
    (b) ``expected_selected`` is the unique subset attaining its f value, and
    (c) that value is >= 1e-4 relative from every other distinct f value.
  Randomly drawn cases (13-15, all n <= 8) are re-drawn until the margin
  holds; hand-built cases fail hard (fix the construction, never the margin).
* Byte stability: the file is written with LF newlines and a fixed field
  order; ``.gitattributes`` marks it ``-text`` so no platform rewrites it.

If a hand computation below disagrees with the library, this script STOPS with
a non-zero exit: that is either a library bug or a wrong hand computation, and
neither may be papered over by adjusting the expectation.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from divsel import __version__, gist_select_full

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "test-assets" / "golden-selection.json"

SCHEMA = 1
F_REL = 1e-6  # conformance tolerance for float fields
MARGIN = 1e-4  # robustness margin between competing f values (relative)
SAME = 1e-9  # two brute-force f values closer than this (relative) are "the same"


def dy(v: float) -> float:
    """Assert v is a multiple of 1/64 in [-4, 4] and return it (guards typos)."""
    assert v * 64 == int(v * 64) and -4.0 <= v <= 4.0, f"not dyadic-in-range: {v}"
    return v


# ---------------------------------------------------------------------------
# Reference reimplementations (pure Python float64) used ONLY to gauge the
# robustness margin and to predict hand-case thresholds.  The expected values
# in the fixture always come from the library itself.
# ---------------------------------------------------------------------------


def thresholds_f32(d_max: float, eps: float, bound: float) -> list[float]:
    """Replicates gist.rs::thresholds_with_bound: repeated multiplication in
    f64 (eps and d_max widened from f32), each entry cast to f32, consecutive
    duplicates removed.  Never log+floor."""
    eps64 = float(np.float32(eps))
    d64 = float(np.float32(d_max))
    # The core's guard, replicated: gist.rs::thresholds_with_bound returns an
    # empty grid for anything below f32::EPSILON, where `p *= 1 + eps` can no
    # longer separate two consecutive f32 entries -- and below 2**-53 does not
    # advance `p` at all. Without it this loop diverges from the core (and, at a
    # small enough eps, does not terminate) exactly where the core reports
    # InvalidEps.
    if not (eps64 >= float(np.finfo(np.float32).eps) and math.isfinite(eps64)):
        return []
    out: list[float] = []
    p = 1.0
    while p <= bound:
        out.append(float(np.float32(p * eps64 * d64 / 2.0)))
        p *= 1.0 + eps64
    return [t for i, t in enumerate(out) if i == 0 or t != out[i - 1]]


def sweep_ceiling(diameter: str) -> float:
    """The numerator of the sweep's ceiling `bound = c / eps`, per diameter mode.

    The core widens it from the paper's `2/eps` to `4/eps` under
    `diameter="approx"` (gist.rs, DiameterMode::Approx), because it then only
    holds `d_hat >= d_max/2` and the grid still has to reach the true diameter.
    Predicting a hand case's threshold with `2/eps` under approx would miss
    every grid entry above about `d_hat`. No approx case carries a `Hand` today,
    which is exactly why this has to be right before one does.
    """
    if diameter not in {"exact", "approx"}:
        raise ValueError(f"unknown diameter mode {diameter!r}")
    return 4.0 if diameter == "approx" else 2.0


def _normalise(rows: list[list[float]]) -> list[list[float]]:
    out = []
    for row in rows:
        norm = math.sqrt(sum(x * x for x in row))
        if norm == 0.0:
            raise ValueError("zero-norm row under cosine")
        out.append([x / norm for x in row])
    return out


def _dist_matrix(vectors: list[list[float]], metric: str) -> list[list[float]]:
    n = len(vectors)
    rows = _normalise(vectors) if metric == "cosine" else vectors
    dist = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if metric == "cosine":
                d = 1.0 - sum(a * b for a, b in zip(rows[i], rows[j]))
                d = min(max(d, 0.0), 2.0)
            else:
                d = math.sqrt(sum((a - b) ** 2 for a, b in zip(rows[i], rows[j])))
            dist[i][j] = dist[j][i] = d
    return dist


def _approx_diameter(dist: list[list[float]], n: int, sweeps: int) -> float:
    """Replicates gist.rs::approx_diameter: repeated farthest-point double
    sweep; argmax ties to the lowest index; incumbent kept under the total
    order (larger distance, then smaller u, then smaller v)."""
    if n < 2:
        return 0.0

    def farthest(frm: int) -> int:
        best_d, best_j = -math.inf, None
        for j in range(n):
            if j == frm:
                continue
            if dist[frm][j] > best_d:
                best_d, best_j = dist[frm][j], j
        return best_j

    best = (-math.inf, n, n)
    current = 0
    for _ in range(max(sweeps, 1)):
        a = farthest(current)
        b = farthest(a)
        cand = (dist[a][b], min(a, b), max(a, b))
        if (cand[0] > best[0]) or (
            cand[0] == best[0] and (cand[1], cand[2]) < (best[1], best[2])
        ):
            best = cand
        current = b
    return best[0]


def _brute_f(case: dict) -> tuple[dict[frozenset, float], float]:
    """f(S) for every subset |S| <= k, in pure float64 with divsel's
    definitions.  Returns ({subset: f}, d_used) where d_used is d_max (exact
    mode) or d_hat (approx mode) — the |S| <= 1 div fallback."""
    vectors = case["vectors"]
    n = len(vectors)
    metric = case["metric"]
    lam = case["lam"]
    k = min(case["k"], n)
    dist = _dist_matrix(vectors, metric)

    d_exact = max((dist[i][j] for i in range(n) for j in range(i + 1, n)), default=0.0)
    if case["diameter"] == "approx":
        d_used = _approx_diameter(dist, n, case["diameter_sweeps"])
    else:
        d_used = d_exact

    utility = case["utility"]
    if utility == "linear":
        weights = case["utilities"] if case["utilities"] is not None else [1.0] * n

        def g(subset: tuple[int, ...]) -> float:
            return float(sum(weights[i] for i in subset))

    elif utility == "coverage":
        sets = [frozenset(s) for s in case["utilities"]]

        def g(subset: tuple[int, ...]) -> float:
            covered: set[int] = set()
            for i in subset:
                covered |= sets[i]
            return float(len(covered))

    elif utility == "facility_location":
        # The library's FacilityLocation::new always scales by the EXACT
        # diameter, whatever the diameter mode (CONFORMANCE.md rule 10).
        scale = 1.0 if metric == "cosine" else d_exact
        # d_max == 0 would make sim identically 1; no FL fixture is degenerate.
        assert scale > 0.0

        def sim(i: int, j: int) -> float:
            return max(0.0, 1.0 - dist[i][j] / scale)

        def g(subset: tuple[int, ...]) -> float:
            if not subset:
                return 0.0
            return sum(max(sim(i, j) for j in subset) for i in range(n))

    else:
        raise ValueError(f"unknown utility {utility}")

    def div(subset: tuple[int, ...]) -> float:
        if len(subset) <= 1:
            return d_used
        return min(dist[u][v] for u, v in itertools.combinations(subset, 2))

    values: dict[frozenset, float] = {}
    for size in range(0, k + 1):
        for subset in itertools.combinations(range(n), size):
            values[frozenset(subset)] = g(subset) + lam * div(subset)
    return values, d_used


def margin_check(case: dict, out: dict) -> tuple[bool, str, float]:
    """(ok, reason, min_gap_rel) per the robustness-margin rule."""
    values, _ = _brute_f(case)
    expected_set = frozenset(out["selected"])
    if expected_set not in values:
        return False, "expected_selected is not a subset of size <= k", 0.0
    f_expected = values[expected_set]

    # Reimplementation sanity: the pure-f64 brute force must agree with the
    # library's f32-kernel value to well under the margin scale.
    drift = abs(f_expected - out["f_value"]) / max(1.0, abs(f_expected))
    if drift > 2e-5:
        return False, f"brute-force f drifts {drift:.2e} from the library", 0.0

    # Distinct f values (values closer than SAME relative are the same value).
    ordered = sorted(values.values(), reverse=True)
    distinct: list[float] = []
    for v in ordered:
        if not distinct or abs(distinct[-1] - v) > SAME * max(1.0, abs(distinct[-1])):
            distinct.append(v)

    if len(distinct) < 2:
        return False, "fewer than two distinct f values", 0.0

    # (a) top-two distinct gap.
    gap_top = (distinct[0] - distinct[1]) / max(1.0, abs(distinct[0]))
    if gap_top < MARGIN:
        return False, f"top-two f gap {gap_top:.2e} < {MARGIN}", gap_top

    # (b) expected_selected is the unique subset at its own f value.
    same_value = [
        s for s, v in values.items() if abs(v - f_expected) <= SAME * max(1.0, abs(f_expected))
    ]
    if same_value != [expected_set]:
        others = [sorted(s) for s in same_value if s != expected_set]
        return False, f"f value shared with {others}", gap_top

    # (c) expected's value is MARGIN away from every other distinct value.
    gap_near = min(
        abs(f_expected - v) / max(1.0, abs(f_expected))
        for v in distinct
        if abs(v - f_expected) > SAME * max(1.0, abs(f_expected))
    )
    if gap_near < MARGIN:
        return False, f"nearest competing f gap {gap_near:.2e} < {MARGIN}", gap_near

    return True, "", min(gap_top, gap_near)


# ---------------------------------------------------------------------------
# Case construction
# ---------------------------------------------------------------------------


@dataclass
class Hand:
    """Hand-computed expectations, asserted EXACTLY against the library."""

    selected: list[int]
    f: float
    g: float
    div: float
    stage: str
    d_max: float
    # Either the exact winning threshold (0.0 / d_max), or the (lo, hi]
    # interval the winner must fall in: the predicted winner is then the
    # largest grid threshold in that interval (sweep folds ascending, >=).
    threshold: float | None = None
    interval: tuple[float, float] | None = None


def _case(
    name: str,
    note: str,
    *,
    metric: str = "euclidean",
    utility: str = "linear",
    vectors: list[list[float]],
    utilities=None,
    k: int,
    lam: float,
    eps: float = 0.1,
    exhaustive_thresholds: bool = False,
    diameter: str = "exact",
    diameter_sweeps: int = 3,
    hand: Hand | None = None,
    exempt: str | None = None,
) -> dict:
    for row in vectors:
        for v in row:
            dy(v)
    if utility == "linear" and utilities is not None:
        for w in utilities:
            dy(w)
            assert w >= 0.0
    return {
        "name": name,
        "note": note,
        "metric": metric,
        "utility": utility,
        "vectors": vectors,
        "utilities": utilities,
        "k": k,
        "lam": lam,
        "eps": eps,
        "exhaustive_thresholds": exhaustive_thresholds,
        "diameter": diameter,
        "diameter_sweeps": diameter_sweeps,
        "_hand": hand,
        "_exempt": exempt,
    }


def hand_cases() -> list[dict]:
    cases: list[dict] = []

    # -- 1. line_pick_widest_scaled -- 4 points on a line, uniform, k = 2 ----
    #
    #   x = [0, 0.5, 2.5, 3], w = 1, lam = 1, eps = 0.1.
    #   d01 = 0.5, d02 = 2.5, d03 = 3, d12 = 2, d13 = 2.5, d23 = 0.5; d_max = 3.
    #   Line 2, greedy d=0: ties -> [0, 1]; g = 2, div = 0.5, f = 2.5.
    #   Line 5, pair [0, 3]: g = 2, div = 3, f = 5 > 2.5 -> displaced.
    #   Sweep 0.15*1.1^i (0.15 .. 2.879):
    #     d <= 0.5: all admitted -> [0, 1], f = 2.5.
    #     d in (0.5, 2.5]: 0; then {2, 3} tie -> 2; 3 blocked (d23 = 0.5)
    #       -> [0, 2], f = 2 + 2.5 = 4.5 < 5.
    #     d in (2.5, 3]: 0; then only 3 -> [0, 3], f = 5 >= 5 -> sweep.
    #   Result: [0, 3], f = 5, g = 2, div = 3, stage sweep, thr in (2.5, 3].
    cases.append(
        _case(
            "line_pick_widest_scaled",
            "4 collinear points, uniform weights: the sweep's top threshold "
            "re-finds the diametrical pair and >= hands it the win",
            vectors=[[0.0], [0.5], [2.5], [3.0]],
            k=2,
            lam=1.0,
            hand=Hand([0, 3], 5.0, 2.0, 3.0, "sweep", 3.0, interval=(2.5, 3.0)),
        )
    )

    # -- 2. weighted_line_middle_threshold -- 6 points, weighted, k = 3 ------
    #
    #   x = [0, 0.5, 1, 1.5, 2, 4], w = [4, 1, 4, 1, 4, 3], lam = 1, eps = 0.1.
    #   Distances |xi - xj|; d_max = 4 (0, 5).
    #   Greedy d=0: weights 4 at 0, 2, 4 -> [0, 2, 4], g = 12,
    #     div = min(1, 2, 1) = 1, f = 13.
    #   Pair [0, 5]: g = 7, div = 4, f = 11 < 13 -> greedy survives line 5.
    #   Sweep 0.2*1.1^i (0.2 .. 3.839); greedy always opens with 0 (w4, the
    #   lowest index among the three weight-4 points):
    #     d <= 1: 2 admitted (d02 = 1 >= d) and wins on weight+index; then 4
    #       (d24 = 1 >= d) -> [0, 2, 4] again, f = 13 >= 13 -> sweep.
    #     d in (1, 2]: 2 blocked (d02 = 1 < d); 4 (d04 = 2 >= d) beats every
    #       other admitted point on weight; then only 5 clears d from both 0
    #       and 4 (d45 = 2 >= d) -> [0, 4, 5], g = 11, div = 2, f = 13 >= 13
    #       -> update.
    #     d in (2, 3.839]: only 5 clears d from 0 -> [0, 5], g = 7, div = 4,
    #       f = 11 < 13.
    #   Winner: largest thr <= 2 -> [0, 4, 5], f = 13, g = 11, div = 2.
    #   (This is Task 7's CASE_B at half scale with lam doubled: identical f.)
    cases.append(
        _case(
            "weighted_line_middle_threshold",
            "a middle threshold ties classic greedy on f and the >= rule "
            "hands the win to the larger threshold's selection",
            vectors=[[0.0], [0.5], [1.0], [1.5], [2.0], [4.0]],
            utilities=[4.0, 1.0, 4.0, 1.0, 4.0, 3.0],
            k=3,
            lam=1.0,
            hand=Hand([0, 4, 5], 13.0, 11.0, 2.0, "sweep", 4.0, interval=(1.5, 2.0)),
            exempt="f({0,2,4}) == f({0,4,5}) == 13 is an exact dyadic tie by "
            "construction — the >= rule choosing the later threshold IS the case",
        )
    )

    # -- 3. rectangle_short_return -- 3-4-5 rectangle, k = 3, |S| = 2 --------
    #
    #   p0=(0,0), p1=(3,0), p2=(0,4), p3=(3,4); w = 1; lam = 1.
    #   d01 = 3, d02 = 4, d03 = 5, d12 = 5, d13 = 4, d23 = 3.
    #   d_max = 5, realized by (0,3) AND (1,2): diameter tie -> smaller u -> (0,3).
    #   Greedy d=0: [0, 1, 2], g = 3, div = 3, f = 6.
    #   Pair [0, 3]: g = 2, div = 5, f = 7 > 6 -> displaced.
    #   Sweep 0.25*1.1^i (0.25 .. 4.799):
    #     d <= 3: [0, 1, 2], f = 6.   d in (3, 4]: [0, 2], f = 6.
    #     d in (4, 5]: [0, 3], f = 7 >= 7 -> sweep.
    #   Result: [0, 3] (2 < k = 3 points!), f = 7, thr in (4, 5].
    cases.append(
        _case(
            "rectangle_short_return",
            "diameter tie resolved to the smaller-u pair; the answer holds "
            "only 2 of the 3 allowed points",
            vectors=[[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]],
            k=3,
            lam=1.0,
            hand=Hand([0, 3], 7.0, 2.0, 5.0, "sweep", 5.0, interval=(4.0, 5.0)),
            exempt="f({0,3}) == f({1,2}) == 7 is an exact dyadic tie by "
            "construction — the diameter tie rule (smaller u) IS the case",
        )
    )

    # -- 4. pair_reached_by_sweep_tie -- n = 3, the sweep re-finds the pair --
    #
    #   x = [0, 2, 3], w = [2, 1, 1], lam = 1, eps = 0.1.
    #   d01 = 2, d02 = 3, d12 = 1; d_max = 3 (0, 2).
    #   Greedy d=0: 0 (w2); tie 1/2 -> 1 -> [0, 1], g = 3, div = 2, f = 5.
    #   Pair [0, 2]: g = 3, div = 3, f = 6 > 5 -> displaced.
    #   Sweep 0.15*1.1^i (0.15 .. 2.879):
    #     d <= 1: [0, 1], f = 5.  d in (1, 2]: 0; tie {1, 2} -> 1 -> [0, 1], f = 5.
    #     d in (2, 2.879]: 0; only 2 -> [0, 2], f = 6 >= 6 -> sweep steals the label.
    #   Result: [0, 2], f = 6, g = 3, div = 3, stage sweep (NOT diameter_pair!).
    cases.append(
        _case(
            "pair_reached_by_sweep_tie",
            "the pair displaces greedy but the sweep's top threshold "
            "re-produces it, so >= relabels the stage to sweep",
            vectors=[[0.0], [2.0], [3.0]],
            utilities=[2.0, 1.0, 1.0],
            k=2,
            lam=1.0,
            hand=Hand([0, 2], 6.0, 3.0, 3.0, "sweep", 3.0, interval=(2.0, 3.0)),
        )
    )

    # -- 5/6. near-duplicate cluster: 4 near-identical + 4 spread, k = 3 -----
    #
    #   x = [0, 1/64, 2/64, 3/64,  1, 2, 3, 4]
    #   w = [2.0625, 2, 1.9375, 1.875,  1, 1, 1, 1]; eps = 0.1; d_max = 4 (0, 7).
    #
    #   lam = 4 (case 5): greedy d=0 takes the cluster [0, 1, 2]:
    #     g = 2.0625 + 2 + 1.9375 = 6, div = 1/64, f = 6 + 4/64 = 6.0625.
    #   Pair [0, 7]: g = 3.0625, div = 4, f = 3.0625 + 16 = 19.0625 > 6.0625.
    #   Sweep 0.2*1.1^i (0.2 .. 3.839) -- every threshold exceeds the cluster
    #   spread 3/64, so at most ONE cluster member survives any sweep run:
    #     d in [0.2, 1]: [0, 4, 5], g = 4.0625, div = 1, f = 8.0625.
    #     d in (1, 2]:   [0, 5, 7], g = 4.0625, div = 2, f = 12.0625.
    #     d in (2, 3]:   [0, 6],    g = 3.0625, div = 3, f = 15.0625.
    #     d in (3, 3.839]: [0, 7],  g = 3.0625, div = 4, f = 19.0625 >= pair -> sweep.
    #   Result: [0, 7] -- exactly one cluster member -- f = 19.0625.
    cluster_vectors = [
        [0.0],
        [0.015625],
        [0.03125],
        [0.046875],
        [1.0],
        [2.0],
        [3.0],
        [4.0],
    ]
    cluster_weights = [2.0625, 2.0, 1.9375, 1.875, 1.0, 1.0, 1.0, 1.0]
    cases.append(
        _case(
            "near_duplicate_cluster_high_lambda",
            "4 near-identical high-weight points + 4 spread: at lam = 4 GIST "
            "keeps at most one cluster member",
            vectors=cluster_vectors,
            utilities=cluster_weights,
            k=3,
            lam=4.0,
            hand=Hand([0, 7], 19.0625, 3.0625, 4.0, "sweep", 4.0, interval=(3.0, 4.0)),
        )
    )

    #   lam = 0 (case 6): f = g alone; greedy takes the top-3 weights
    #   [0, 1, 2], g = 6, f = 6; the pair gives 3.0625 and every sweep
    #   threshold at most 4.0625 (one cluster member + two spread), so greedy
    #   is never displaced: stage greedy, threshold 0.
    cases.append(
        _case(
            "near_duplicate_cluster_lambda_zero",
            "same instance at lam = 0: pure weight maximisation keeps the "
            "whole cluster and the sweep never catches up",
            vectors=cluster_vectors,
            utilities=cluster_weights,
            k=3,
            lam=0.0,
            hand=Hand([0, 1, 2], 6.0, 6.0, 0.015625, "greedy", 4.0, threshold=0.0),
        )
    )

    # -- 7. k_one_div_equals_dmax -- the |S| <= 1 div edge case --------------
    #
    #   x = [0, 1, 2, 3, 4], w = [1, 4, 1, 1, 2], lam = 1, k = 1; d_max = 4.
    #   Every stage picks the single argmax-weight point 1; div(|S| <= 1) is
    #   d_max = 4 by definition (paper Sec. 2), so f = 4 + 4 = 8.  The pair
    #   step is skipped (k < 2); every sweep threshold reproduces [1] and the
    #   >= rule leaves the LAST threshold as the reported one.
    cases.append(
        _case(
            "k_one_div_equals_dmax",
            "k = 1: div(|S| <= 1) = d_max, the pair step is skipped, every "
            "threshold ties and the last one is reported",
            vectors=[[0.0], [1.0], [2.0], [3.0], [4.0]],
            utilities=[1.0, 4.0, 1.0, 1.0, 2.0],
            k=1,
            lam=1.0,
            hand=Hand([1], 8.0, 4.0, 4.0, "sweep", 4.0, interval=(0.0, 4.0)),
        )
    )

    # -- 8. k_exceeds_n_returns_all -- k = 10 > n = 4 clamps -----------------
    #
    #   x = [0, 1, 2, 4], w = 1, lam = 0.25, k = 10 -> clamped to 4.
    #   Greedy d=0: all four -> [0, 1, 2, 3], g = 4, div = 1, f = 4.25.
    #   Pair [0, 3]: g = 2, div = 4, f = 3 < 4.25.
    #   Sweep 0.2*1.1^i (0.2 .. 3.839):
    #     d <= 1: all four again, f = 4.25 >= 4.25 -> sweep.
    #     d in (1, 2]: [0, 2, 3], f = 3 + 0.5 = 3.5.  d in (2, 3.839]: [0, 3], f = 3.
    #   Winner: largest thr <= 1 -> all 4 points, stage sweep.
    cases.append(
        _case(
            "k_exceeds_n_returns_all",
            "k > n clamps: all n points are returned",
            vectors=[[0.0], [1.0], [2.0], [4.0]],
            k=10,
            lam=0.25,
            hand=Hand([0, 1, 2, 3], 4.25, 4.0, 1.0, "sweep", 4.0, interval=(0.0, 1.0)),
        )
    )

    # -- 9. sweep_tie_later_threshold_wins -- every threshold ties -----------
    #
    #   x = [0, 4], w = 1, lam = 1, k = 2; d_max = 4.
    #   Greedy: [0, 1], g = 2, div = 4, f = 6.  Pair [0, 1]: f = 6, NOT > (strict
    #   line 5) -> greedy keeps the slot.  Every sweep threshold (all <= 3.839
    #   < 4) admits both points -> [0, 1], f = 6 >= 6 at each of the 32
    #   thresholds: the ascending >= fold leaves the LAST one as the winner.
    cases.append(
        _case(
            "sweep_tie_later_threshold_wins",
            "all 32 thresholds yield the identical selection; the ascending "
            ">= fold reports the largest one",
            vectors=[[0.0], [4.0]],
            k=2,
            lam=1.0,
            hand=Hand([0, 1], 6.0, 2.0, 4.0, "sweep", 4.0, interval=(0.0, 4.0)),
        )
    )

    # -- 10. argmax_tie_lowest_index -- two identical rows ------------------
    #
    #   x = [0, 0, 4], w = 1, lam = 1, k = 2.  Rows 0 and 1 are identical, so
    #   f({0,2}) = f({1,2}) = 7 EXACTLY (dyadic); the argmax tie rule must
    #   pick index 0.  d_max = 4 realized by (0,2) and (1,2): pair tie -> (0,2).
    #   Greedy d=0: [0, 1], g = 2, div = 0, f = 2.  Pair [0, 2]: g = 2,
    #   div = 4, f = 6 > 2 -> displaced.  Every sweep threshold blocks
    #   row 1 (dist 0) and admits row 2 -> [0, 2], f = 6 >= 6 -> stage sweep,
    #   last threshold reported.  Exempt from the margin rule: the [1, 2] tie
    #   is exact by construction.
    cases.append(
        _case(
            "argmax_tie_lowest_index",
            "two identical rows: every argmax tie (greedy pick, diameter "
            "pair) resolves to the lowest index",
            vectors=[[0.0], [0.0], [4.0]],
            k=2,
            lam=1.0,
            hand=Hand([0, 2], 6.0, 2.0, 4.0, "sweep", 4.0, interval=(0.0, 4.0)),
            exempt="f({0,2}) == f({1,2}) is an exact dyadic tie by construction",
        )
    )

    # -- 11. diameter_pair_wins -- stage == diameter_pair --------------------
    #
    #   x = [-4, 0, 4], w = [1, 1.5, 1], lam = 1, k = 2; d_max = 8 (0, 2).
    #   Greedy d=0: 1 (w 1.5); tie {0, 2} -> 0 -> [1, 0], g = 2.5, div = 4,
    #     f = 6.5.
    #   Pair [0, 2]: g = 2, div = 8, f = 10 > 6.5 (STRICT line 5) -> displaced.
    #   Sweep 0.4*1.1^i (0.4 .. 7.678): greedy always starts at 1 (max weight):
    #     d <= 4: [1, 0], f = 6.5.  d in (4, 7.678]: [1] alone, g = 1.5,
    #     div = d_max = 8, f = 9.5.  Nothing reaches 10 -> the pair keeps the
    #     slot: stage diameter_pair, threshold = d_max = 8.
    cases.append(
        _case(
            "diameter_pair_wins",
            "the diametrical pair strictly beats greedy and no sweep "
            "threshold matches it: stage diameter_pair, threshold = d_max",
            vectors=[[-4.0], [0.0], [4.0]],
            utilities=[1.0, 1.5, 1.0],
            k=2,
            lam=1.0,
            hand=Hand([0, 2], 10.0, 2.0, 8.0, "diameter_pair", 8.0, threshold=8.0),
        )
    )

    # -- 12. greedy_wins_outright -- stage == greedy -------------------------
    #
    #   x = [0, 1/64, 4], w = [4, 4, 1], lam = 0.5, k = 2; d_max = 4.
    #   Greedy d=0: tie {0, 1} (w 4) -> 0; then 1 -> [0, 1], g = 8,
    #     div = 1/64, f = 8 + 0.5/64 = 8.0078125.
    #   Pair [0, 2]: g = 5, div = 4, f = 7 < 8.0078125.
    #   Sweep 0.2*1.1^i: every threshold >= 0.2 > 1/64 blocks point 1, so each
    #   run yields [0, 2], f = 7 < 8.0078125.  Greedy is never displaced:
    #   stage greedy, threshold 0.  (The geometric grid does NOT contain 0 --
    #   its smallest entry is eps*d_max/2 -- which is what makes this stage
    #   reachable; the exhaustive set always contains 0, see case 19.)
    cases.append(
        _case(
            "greedy_wins_outright",
            "a sub-threshold near-duplicate pair holds the best f: no sweep "
            "threshold can reproduce it, stage stays greedy",
            vectors=[[0.0], [0.015625], [4.0]],
            utilities=[4.0, 4.0, 1.0],
            k=2,
            lam=0.5,
            hand=Hand([0, 1], 8.0078125, 8.0, 0.015625, "greedy", 4.0, threshold=0.0),
        )
    )

    return cases


def coverage_cases() -> list[dict]:
    cases: list[dict] = []

    # -- 17. coverage_hand_counts -- hand-built newly-covered counts ---------
    #
    #   x = [0, 1, 2.5, 3.5]; sets s0={0,1,2}, s1={2,3}, s2={3,4,5}, s3={0,5};
    #   universe = 6; k = 2, lam = 0.5, eps = 0.1.
    #   d01 = 1, d02 = 2.5, d03 = 3.5, d12 = 1.5, d13 = 2.5, d23 = 1;
    #   d_max = 3.5 (0, 3).
    #   Greedy d=0: marginals |s0| = 3, |s1| = 2, |s2| = 3, |s3| = 2: tie 0/2
    #     -> 0; then newly-covered given {0,1,2}: s1 -> 1, s2 -> 3, s3 -> 1
    #     -> 2.  S = [0, 2], g = 6, div = 2.5, f = 6 + 1.25 = 7.25.
    #   Pair [0, 3]: g = |{0,1,2,5}| = 4, div = 3.5, f = 5.75 < 7.25.
    #   Sweep 0.175*1.1^i (0.175 .. 3.359):
    #     d <= 2.5: 0 first, then 2 by count -> [0, 2], f = 7.25 >= 7.25 -> sweep.
    #     d in (2.5, 3.359]: only 3 is far enough -> [0, 3], f = 5.75.
    #   Winner: largest thr <= 2.5 -> [0, 2], f = 7.25, g = 6, div = 2.5.
    cases.append(
        _case(
            "coverage_hand_counts",
            "coverage marginals are newly-covered counts: 3+3 covered items "
            "beat the wider pair",
            utility="coverage",
            vectors=[[0.0], [1.0], [2.5], [3.5]],
            utilities=[[0, 1, 2], [2, 3], [3, 4, 5], [0, 5]],
            k=2,
            lam=0.5,
            hand=Hand([0, 2], 7.25, 6.0, 2.5, "sweep", 3.5, interval=(0.0, 2.5)),
        )
    )

    # -- 18. coverage_exact_tie_lowest_index ---------------------------------
    #
    #   x = [0, 1, 3]; s0={0,1}, s1={2,3}, s2={4}; k = 2, lam = 0.5.
    #   d01 = 1, d02 = 3, d12 = 2; d_max = 3 (0, 2).
    #   Greedy d=0: marginals 2, 2, 1 -- an EXACT tie between 0 and 1 -> 0;
    #     then given {0,1}: s1 -> 2, s2 -> 1 -> 1.  S = [0, 1], g = 4,
    #     div = 1, f = 4.5.
    #   Pair [0, 2]: g = 3, div = 3, f = 4.5 -- NOT strictly greater -> greedy
    #     keeps the slot at line 5 (strict >).
    #   Sweep 0.15*1.1^i (0.15 .. 2.879):
    #     d <= 1: [0, 1], f = 4.5 >= 4.5 -> sweep.
    #     d in (1, 2.879]: 1 blocked, 2 admitted -> [0, 2], f = 3 + 1.5 = 4.5
    #       >= 4.5 -> the LATER thresholds hand the win to [0, 2].
    #   Result: [0, 2], f = 4.5, thr = the top of the grid.  Exempt from the
    #   margin rule: f({0,1}) == f({0,2}) == 4.5 exactly, by construction.
    cases.append(
        _case(
            "coverage_exact_tie_lowest_index",
            "an exact coverage-marginal tie resolves to the lowest index; an "
            "exact f tie then walks the >= fold to the top threshold",
            utility="coverage",
            vectors=[[0.0], [1.0], [3.0]],
            utilities=[[0, 1], [2, 3], [4]],
            k=2,
            lam=0.5,
            hand=Hand([0, 2], 4.5, 3.0, 3.0, "sweep", 3.0, interval=(0.0, 3.0)),
            exempt="f({0,1}) == f({0,2}) == pair f is an exact dyadic tie by construction",
        )
    )

    return cases


def degenerate_and_tie_cases() -> list[dict]:
    """Cases 21-22 (review fix round 1): the d_max == 0 pair check and the
    exact-diameter tie order, each pinned by a fixture of its own."""
    cases: list[dict] = []

    # -- 21. coincident_coverage_pair_check -- d_max == 0 still runs line 5 --
    #
    #   Three coincident points x = [0, 0, 0]; sets s0={0,1,2}, s1={3,4,5},
    #   s2={0,1,3,4}; universe = 6; k = 2, lam = 1.
    #   Every distance is 0, so d_max = 0 and the exact-diameter reduction
    #   returns the lexicographically smallest pair (0, 1).
    #   Greedy d=0: marginals 3, 3, 4 -> 2; then newly covered given s2:
    #     s0 -> {2} = 1, s1 -> {5} = 1, tie -> 0.  S = [2, 0], g = 5, div = 0,
    #     f = 5.
    #   Pair [0, 1]: g = |{0..5}| = 6, div = 0, f = 6 > 5 (strict line 5)
    #     -> displaced: stage diameter_pair, threshold = d_max = 0.
    #   Sweep: skipped (d_max == 0), so nothing can relabel the stage.
    #   Result: [0, 1], f = 6, g = 6, div = 0, stage diameter_pair, thr 0,
    #   d_max 0.  (Under a linear utility the same points would report stage
    #   greedy: greedy already holds the top-k weights, so a modular g can
    #   never let the pair win strictly.)
    cases.append(
        _case(
            "coincident_coverage_pair_check",
            "three coincident points (d_max == 0): the sweep is skipped but "
            "line 5 still runs, and the pair strictly beats greedy on g",
            utility="coverage",
            vectors=[[0.0], [0.0], [0.0]],
            utilities=[[0, 1, 2], [3, 4, 5], [0, 1, 3, 4]],
            k=2,
            lam=1.0,
            hand=Hand([0, 1], 6.0, 6.0, 0.0, "diameter_pair", 0.0, threshold=0.0),
        )
    )

    # -- 22. diameter_tie_smallest_pair -- R-G15 in the pair stage ------------
    #
    #   x = [-4, -4, 0, 4], w = [1, 1, 1.5, 1], lam = 1, k = 2, eps = 0.1.
    #   d01 = 0, d02 = 4, d03 = 8, d12 = 4, d13 = 8, d23 = 4; d_max = 8,
    #   realized by (0, 3) AND (1, 3): the total order (larger d, smaller u,
    #   smaller v) picks (0, 3).
    #   Greedy d=0: 2 (w 1.5); tie {0, 1, 3} at w = 1 -> 0 -> [2, 0],
    #     g = 2.5, div = 4, f = 6.5.
    #   Pair [0, 3]: g = 2, div = 8, f = 10 > 6.5 -> displaced.
    #   Sweep 0.4*1.1^i (0.4 .. 7.678): greedy always opens with 2 (max
    #   weight); every other point is exactly 4 from it:
    #     d <= 4: all three admitted (4 >= d), tie -> 0 -> [2, 0], f = 6.5.
    #     d in (4, 7.678]: nothing admitted -> [2] alone, g = 1.5,
    #       div = d_max = 8, f = 9.5 < 10.
    #   The pair keeps the slot: [0, 3], f = 10, stage diameter_pair, thr 8.
    #   Under the opposite tie order the pair would be (1, 3) and the reported
    #   selection [1, 3] with the same f -- a tie-direction case, exempt from
    #   the margin rule: f({0,3}) == f({1,3}) == 10 exactly by construction.
    cases.append(
        _case(
            "diameter_tie_smallest_pair",
            "d_max realised by (0,3) and (1,3): the exact-diameter tie resolves "
            "to the lexicographically smallest pair and the sweep never "
            "re-finds it, so the reported selection is the pair itself",
            vectors=[[-4.0], [-4.0], [0.0], [4.0]],
            utilities=[1.0, 1.0, 1.5, 1.0],
            k=2,
            lam=1.0,
            hand=Hand([0, 3], 10.0, 2.0, 8.0, "diameter_pair", 8.0, threshold=8.0),
            exempt="f({0,3}) == f({1,3}) == 10 is an exact dyadic tie by "
            "construction — the diameter tie order (smallest pair) IS the case",
        )
    )

    return cases


def structured_cases() -> list[dict]:
    """Cases 16, 19, 20: hand-listed dyadic constructions above n = 8 (kept
    deterministic rather than drawn; the margin check still applies)."""
    cases: list[dict] = []

    # -- 16. facility_location_euclidean_n12 -- three clusters of four -------
    # Asymmetric per-member offsets keep every swap of a cluster member a
    # >= 1e-4 relative change in f (verified by the margin check).
    c0, c1, c2 = (-3.0, -3.0), (0.0, 3.0), (3.0, -2.0)
    offsets = [
        [(0.0, 0.0), (0.25, 0.375), (-0.5, 0.125), (0.375, -0.4375)],
        [(0.0, 0.0), (-0.375, 0.25), (0.5, -0.125), (0.1875, 0.4375)],
        [(0.0, 0.0), (0.4375, 0.1875), (-0.25, -0.375), (-0.5, 0.375)],
    ]
    grid = [
        [center[0] + dx, center[1] + dy_]
        for center, offs in zip((c0, c1, c2), offsets)
        for dx, dy_ in offs
    ]
    cases.append(
        _case(
            "facility_location_euclidean_n12",
            "three tight clusters of four: facility location (scale = d_max) "
            "wants one representative per cluster",
            utility="facility_location",
            vectors=grid,
            k=3,
            lam=0.5,
        )
    )

    # -- 19. exhaustive_thresholds_linear_n10 --------------------------------
    # The exhaustive set {dist(u,v)/2} ALWAYS contains 0 (the u == v pairs),
    # so the d = 0 sweep run duplicates line 2 and the >= rule relabels the
    # stage: "greedy" is unreachable under exhaustive_thresholds (d_max > 0).
    cases.append(
        _case(
            "exhaustive_thresholds_linear_n10",
            "exhaustive threshold set {dist(u,v)/2}: contains 0, so stage "
            "'greedy' is unreachable here by construction",
            vectors=[[-4.0], [-3.25], [-2.5], [-1.5], [-0.75], [0.0], [1.0], [1.75], [2.5], [4.0]],
            utilities=[2.0, 0.25, 1.5, 0.5, 1.0, 0.75, 1.25, 0.25, 1.75, 2.25],
            k=3,
            lam=0.25,
            exhaustive_thresholds=True,
        )
    )

    # -- 20. approx_diameter_double_sweep ------------------------------------
    # diameter="approx", diameter_sweeps=2: d_hat comes from the farthest-point
    # double sweep (documented total order), the sweep bound widens to 4/eps,
    # and div(|S| <= 1) falls back to d_hat, not the exact d_max.  Ports that
    # skip DiameterMode::Approx skip this case and say so (CONFORMANCE.md).
    cases.append(
        _case(
            "approx_diameter_double_sweep",
            "diameter='approx' (2 double sweeps): d_max is the documented "
            "farthest-point estimate d_hat and the sweep bound widens to 4/eps",
            vectors=[
                [-4.0, 0.0],
                [4.0, 0.25],
                [-3.0, 3.0],
                [2.0, -3.5],
                [0.0, 4.0],
                [3.5, 2.0],
                [-2.0, -3.0],
                [1.0, 1.0],
                [-1.0, 2.5],
                [2.5, -1.0],
                [-3.5, -2.0],
                [0.5, -4.0],
            ],
            utilities=[1.5, 1.25, 1.0, 1.0, 1.25, 0.75, 1.0, 0.5, 0.75, 1.0, 1.25, 1.0],
            k=3,
            lam=0.5,
            diameter="approx",
            diameter_sweeps=2,
        )
    )

    return cases


def random_cases() -> list[dict]:
    """Cases 13-15: seeded random draws at n <= 8, re-drawn until the
    robustness margin holds.  The seed is fixed, so regeneration is
    deterministic on this machine."""
    cases: list[dict] = []
    rng = random.Random(0x51D5E1)

    def draw_rows(n: int, dim: int, cosine: bool) -> list[list[float]]:
        while True:
            rows = [[rng.randint(-256, 256) / 64.0 for _ in range(dim)] for _ in range(n)]
            if not cosine or all(any(x != 0.0 for x in row) for row in rows):
                return rows

    def accept(name: str, build) -> dict:
        for _ in range(500):
            case = build()
            out = run_library(case)
            ok, _reason, _gap = margin_check(case, out)
            if ok:
                return case
        raise SystemExit(f"{name}: no draw passed the robustness margin in 500 attempts")

    cases.append(
        accept(
            "cosine_linear_random",
            lambda: _case(
                "cosine_linear_random",
                "cosine metric (rows L2-normalised, dist = clamp(1 - a.b, 0, 2)) "
                "with dyadic linear weights",
                metric="cosine",
                vectors=draw_rows(6, 3, cosine=True),
                utilities=[rng.randint(16, 256) / 64.0 for _ in range(6)],
                k=3,
                lam=1.0,
            ),
        )
    )

    cases.append(
        accept(
            "cosine_facility_location_random",
            lambda: _case(
                "cosine_facility_location_random",
                "facility location under cosine: sim = max(0, 1 - dist), "
                "scale = 1.0 (the paper's own s(i,j))",
                metric="cosine",
                utility="facility_location",
                vectors=draw_rows(7, 3, cosine=True),
                k=3,
                lam=0.5,
            ),
        )
    )

    cases.append(
        accept(
            "facility_location_euclidean_n8",
            lambda: _case(
                "facility_location_euclidean_n8",
                "facility location under euclidean: sim = max(0, 1 - dist/d_max), "
                "scale = the exact diameter",
                utility="facility_location",
                vectors=draw_rows(8, 2, cosine=False),
                k=3,
                lam=1.0,
            ),
        )
    )

    return cases


# ---------------------------------------------------------------------------
# Running, hand-verification, serialisation
# ---------------------------------------------------------------------------


def run_library(case: dict) -> dict:
    vectors = np.ascontiguousarray(np.array(case["vectors"], dtype=np.float32))
    utilities = case["utilities"]
    if case["utility"] == "linear" and utilities is not None:
        utilities = np.array(utilities, dtype=np.float64)
    return gist_select_full(
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


def verify_hand(case: dict, out: dict) -> None:
    """The library must reproduce the hand arithmetic EXACTLY (all values in
    the hand cases are dyadic, so every platform computes them exactly).  A
    mismatch is a library bug or a wrong hand computation: STOP either way."""
    hand: Hand | None = case["_hand"]
    if hand is None:
        return
    name = case["name"]
    problems = []
    if out["selected"] != hand.selected:
        problems.append(f"selected {out['selected']} != hand {hand.selected}")
    if out["stage"] != hand.stage:
        problems.append(f"stage {out['stage']} != hand {hand.stage}")
    for field, expected in (
        ("f_value", hand.f),
        ("g_value", hand.g),
        ("div", hand.div),
        ("d_max", hand.d_max),
    ):
        if out[field] != expected:
            problems.append(f"{field} {out[field]!r} != hand {expected!r}")
    if hand.threshold is not None:
        if out["threshold"] != hand.threshold:
            problems.append(f"threshold {out['threshold']!r} != hand {hand.threshold!r}")
    if hand.interval is not None:
        lo, hi = hand.interval
        grid = thresholds_f32(
            hand.d_max,
            case["eps"],
            sweep_ceiling(case["diameter"]) / float(np.float32(case["eps"])),
        )
        winners = [t for t in grid if lo < t <= hi]
        if not winners:
            problems.append(f"no grid threshold in ({lo}, {hi}]")
        elif out["threshold"] != winners[-1]:
            problems.append(
                f"threshold {out['threshold']!r} != predicted {winners[-1]!r} "
                f"(largest grid entry in ({lo}, {hi}])"
            )
    if problems:
        raise SystemExit(
            f"HAND-COMPUTATION MISMATCH in case {name!r} — library bug or wrong "
            f"hand arithmetic; do NOT adjust the expectation blindly:\n  "
            + "\n  ".join(problems)
        )


def build() -> dict:
    specs = []
    hand = hand_cases()
    specs.extend(hand[:12])  # cases 1-12
    specs.extend(random_cases())  # cases 13-15
    structured = structured_cases()
    specs.append(structured[0])  # case 16
    specs.extend(coverage_cases())  # cases 17-18
    specs.append(structured[1])  # case 19
    specs.append(structured[2])  # case 20
    specs.extend(degenerate_and_tie_cases())  # cases 21-22

    cases_json = []
    for case in specs:
        out = run_library(case)
        verify_hand(case, out)
        if case["_exempt"] is not None:
            note = f"{case['note']} [margin: exempt — {case['_exempt']}]"
        else:
            ok, reason, gap = margin_check(case, out)
            if not ok:
                raise SystemExit(
                    f"case {case['name']!r} fails the robustness margin: {reason}"
                )
            note = f"{case['note']} [margin: {gap:.3e} rel]"
        print(
            f"  {case['name']:<38} n={len(case['vectors']):>2} k={case['k']:>2} "
            f"{case['metric']:<9} {case['utility']:<17} stage={out['stage']}"
        )
        cases_json.append(
            {
                "name": case["name"],
                "note": note,
                "metric": case["metric"],
                "utility": case["utility"],
                "vectors": case["vectors"],
                "utilities": case["utilities"],
                "k": case["k"],
                "lam": case["lam"],
                "eps": case["eps"],
                "exhaustive_thresholds": case["exhaustive_thresholds"],
                "diameter": case["diameter"],
                "diameter_sweeps": case["diameter_sweeps"],
                "expected_selected": out["selected"],
                "expected_f": out["f_value"],
                "expected_g": out["g_value"],
                "expected_div": out["div"],
                "expected_threshold": out["threshold"],
                "expected_stage": out["stage"],
                "expected_d_max": out["d_max"],
            }
        )

    assert len(cases_json) == 22, f"expected 22 cases, built {len(cases_json)}"
    return {
        "generator": f"divsel {__version__}",
        "paper": "arXiv:2405.18754v3",
        "schema": SCHEMA,
        "tolerance": {"f_rel": F_REL, "selected": "exact"},
        "cases": cases_json,
    }


def serialise(data: dict) -> bytes:
    # indent=1 + LF: a fixed, diffable layout.  json.dump prints floats with
    # repr (shortest round-trip) and the field order above is fixed, so the
    # bytes are a pure function of the expected values.
    return (json.dumps(data, indent=1) + "\n").encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="regenerate and byte-compare against the committed file; exit 1 on drift",
    )
    args = parser.parse_args()

    print(f"generating golden fixtures with divsel {__version__} ...")
    payload = serialise(build())

    if args.check:
        if not OUT_PATH.exists():
            print(f"--check: {OUT_PATH} does not exist", file=sys.stderr)
            return 1
        committed = OUT_PATH.read_bytes()
        if committed != payload:
            print(
                f"--check: DRIFT — regenerated bytes differ from {OUT_PATH} "
                f"({len(payload)} vs {len(committed)} bytes)",
                file=sys.stderr,
            )
            return 1
        print(f"--check: OK — {OUT_PATH} is byte-identical ({len(payload)} bytes)")
        return 0

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "wb") as fh:
        fh.write(payload)
    print(f"wrote {OUT_PATH} ({len(payload)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
