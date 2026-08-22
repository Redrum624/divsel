"""Hand-computed GIST fixtures shared with the Rust core (R-G25).

Every case here is small enough to run Algorithm 1 (arXiv:2405.18754v3) on paper.
The same three cases, with the same expected values, live in
``crates/divsel/tests/shared_fixture.rs``; both sides must agree with the
arithmetic below, which is what "the Python result equals the Rust result"
means for this task. Task 11's golden file supersedes this.

Conventions that the arithmetic relies on (all ``[divsel choice]`` rules that the
core documents):

* Euclidean metric, so distances are exact for these dyadic coordinates.
* ``GreedyIndependentSet`` breaks ``argmax`` ties towards the **lowest index**.
* Line 5 (diametrical pair) compares **strictly** (``>``); line 10 (sweep)
  compares **non-strictly** (``>=``) and the sweep is folded in ascending
  threshold order, so the **largest** threshold attaining the best ``f`` wins.
* ``div(S)`` is the minimum pairwise distance for ``|S| >= 2`` and ``d_max``
  for ``|S| <= 1``.
* The threshold set for ``eps = 0.1`` is ``{0.05 * d_max * 1.1**i : i = 0..31}``
  (``1.1**31 = 19.19 <= 20 < 1.1**32 = 21.1``). Only which *interval between
  consecutive pairwise distances* each threshold lands in matters, so the
  decisive powers are quoted below; ``1.1**i`` for ``i = 10, 13, 17, 22, 25,
  28, 30, 31`` is ``2.594, 3.452, 5.054, 8.140, 10.83, 14.42, 17.45, 19.19``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Case:
    """One shared fixture: inputs, the exact expected outputs, and a threshold range."""

    name: str
    vectors: np.ndarray
    k: int
    lam: float
    utilities: np.ndarray | None
    selected: list[int]
    f_value: float
    g_value: float
    div: float
    stage: str
    # The winning threshold is ``0.05 * d_max * 1.1**i`` for some ``i`` and is not
    # hand-computable to the bit, but the interval it must land in is.
    threshold_range: tuple[float, float] = field(default=(0.0, float("inf")))


def _f32(rows: list[list[float]]) -> np.ndarray:
    return np.ascontiguousarray(np.array(rows, dtype=np.float32))


# ---------------------------------------------------------------------------
# Case A: "line_pick_widest" -- four points on a line, uniform utility, k = 2.
#
#   x = [0, 1, 5, 6], weights = [1, 1, 1, 1], lambda = 1, eps = 0.1.
#   Pairwise: d01 = 1, d02 = 5, d03 = 6, d12 = 4, d13 = 5, d23 = 1.
#   d_max = 6, realized by (0, 3).
#
#   Line 2, greedy at d = 0: all marginals are 1, lowest index wins twice
#     -> S = [0, 1], g = 2, div = d01 = 1, f = 2 + 1*1 = 3.
#   Lines 4-5, diametrical pair T = [0, 3]: g = 2, div = 6, f = 8 > 3
#     -> S = [0, 3], f = 8 (stage would be diameter_pair if nothing else won).
#   Lines 7-11, thresholds 0.3 * 1.1**i, i = 0..31 (0.3 .. 5.758):
#     d in (0, 1]  (i <= 12, 0.3*3.138 = 0.94): greedy admits all -> [0, 1], f = 3.
#     d in (1, 5]  (i = 13..29):            0 first; then v with dist(v,0) >= d:
#                                            {2, 3} -> lowest index 2; then nothing
#                                            is >= d from both 0 and 2 (d13 = 5 but
#                                            d23 = 1) -> [0, 2], f = 2 + 5 = 7 < 8.
#     d in (5, 6]  (i = 30, 31: 5.235, 5.758): 0 first; only 3 has dist >= d
#                                            -> [0, 3], f = 8 >= 8 -> stage = sweep.
#   Result: selected = [0, 3], f = 8.0, g = 2.0, div = 6.0, stage = "sweep",
#           threshold = 0.3 * 1.1**31 = 5.758, in (5, 6].
# ---------------------------------------------------------------------------
CASE_A = Case(
    name="line_pick_widest",
    vectors=_f32([[0.0], [1.0], [5.0], [6.0]]),
    k=2,
    lam=1.0,
    utilities=None,
    selected=[0, 3],
    f_value=8.0,
    g_value=2.0,
    div=6.0,
    stage="sweep",
    threshold_range=(5.0, 6.0),
)

# ---------------------------------------------------------------------------
# Case B: "weighted_line_middle_threshold" -- six points, weighted, k = 3,
# lambda = 0.5. Neither classic greedy nor the diametrical pair is the answer; a
# middle threshold ties greedy on f and the >= rule hands it the win.
#
#   x = [0, 1, 2, 3, 4, 8], weights = [4, 1, 4, 1, 4, 3], lambda = 0.5.
#   Pairwise distances are |xi - xj|; d_max = 8, realized by (0, 5).
#
#   Line 2, greedy at d = 0: weights 4 at 0, 2, 4, lowest index first
#     -> S = [0, 2, 4], g = 12, div = min(d02, d04, d24) = min(2, 4, 2) = 2,
#        f = 12 + 0.5*2 = 13.
#   Lines 4-5, pair T = [0, 5]: g = 4 + 3 = 7, div = 8, f = 7 + 0.5*8 = 11 < 13
#     -> greedy survives.
#   Lines 7-11, thresholds 0.4 * 1.1**i, i = 0..31 (0.4 .. 7.678):
#     d in (0, 1]  (i <= 9):   all admitted -> [0, 2, 4], f = 13 >= 13 -> sweep.
#     d in (1, 2]  (i = 10..16, 1.037 .. 1.838):
#        0; then dist(v,0) >= d: {2, 3, 4, 5}, argmax weight -> 2 (4);
#        then dist(v,{0,2}) >= d: 3 (d23 = 1) no, 4 (d24 = 2) yes, 5 yes
#        -> argmax weight 4 (4) over 5 (3) -> [0, 2, 4], f = 13 >= 13.
#     d in (2, 3]  (i = 17..21, 2.022 .. 2.960):
#        0; then dist(v,0) >= d: {3, 4, 5}, argmax weight -> 4 (4);
#        then dist(v,{0,4}) >= d: 1 (d01 = 1) no, 2 (d02 = 2) no, 3 (d34 = 1) no,
#        5 (d05 = 8, d45 = 4) yes -> [0, 4, 5],
#        g = 4 + 4 + 3 = 11, div = min(4, 8, 4) = 4, f = 11 + 0.5*4 = 13 >= 13
#        -> selected becomes [0, 4, 5].
#     d in (3, 4]  (i = 22..24, 3.256 .. 3.940):
#        0; then {4, 5} -> 4; then 5 (d45 = 4 >= d) -> [0, 4, 5], f = 13 >= 13.
#     d in (4, 8]  (i = 25..31, 4.334 .. 7.678):
#        0; then only 5 (d05 = 8); then nothing (every other point is within 4
#        of 0) -> [0, 5], f = 11 < 13.
#   Result: selected = [0, 4, 5], f = 13.0, g = 11.0, div = 4.0, stage = "sweep",
#           threshold = 0.4 * 1.1**24 = 3.940, in (3, 4].
# ---------------------------------------------------------------------------
CASE_B = Case(
    name="weighted_line_middle_threshold",
    vectors=_f32([[0.0], [1.0], [2.0], [3.0], [4.0], [8.0]]),
    k=3,
    lam=0.5,
    utilities=np.array([4.0, 1.0, 4.0, 1.0, 4.0, 3.0], dtype=np.float64),
    selected=[0, 4, 5],
    f_value=13.0,
    g_value=11.0,
    div=4.0,
    stage="sweep",
    threshold_range=(3.0, 4.0),
)

# ---------------------------------------------------------------------------
# Case C: "rectangle_short_return" -- a 3-4-5 rectangle, uniform, k = 3. The
# diameter is realized by two pairs (a tie the core resolves to the smaller u),
# and the answer holds only 2 of the 3 allowed points.
#
#   p0 = (0,0), p1 = (3,0), p2 = (0,4), p3 = (3,4); weights = 1; lambda = 1.
#   d01 = 3, d02 = 4, d03 = 5, d12 = 5, d13 = 4, d23 = 3.
#   d_max = 5, realized by both (0, 3) and (1, 2); the tie rule picks u = 0,
#   so T = [0, 3].
#
#   Line 2, greedy at d = 0: [0, 1, 2], g = 3, div = min(3, 4, 5) = 3, f = 6.
#   Lines 4-5, pair T = [0, 3]: g = 2, div = 5, f = 7 > 6 -> S = [0, 3].
#   Lines 7-11, thresholds 0.25 * 1.1**i, i = 0..31 (0.25 .. 4.799):
#     d in (0, 3]  (i <= 26, 0.25*11.92 = 2.98): all admitted -> [0, 1, 2], f = 6.
#     d in (3, 4]  (i = 27..29, 3.277 .. 3.966):
#        0; then dist(v,0) >= d: {2, 3} -> 2; then 1 (d01 = 3) no, 3 (d23 = 3) no
#        -> [0, 2], g = 2, div = 4, f = 6 < 7.
#     d in (4, 5]  (i = 30, 31: 4.362, 4.799):
#        0; then only 3 (d03 = 5); then nothing -> [0, 3], f = 7 >= 7 -> sweep.
#   Result: selected = [0, 3], f = 7.0, g = 2.0, div = 5.0, stage = "sweep",
#           threshold = 0.25 * 1.1**31 = 4.799, in (4, 5]; |selected| = 2 < k.
# ---------------------------------------------------------------------------
CASE_C = Case(
    name="rectangle_short_return",
    vectors=_f32([[0.0, 0.0], [3.0, 0.0], [0.0, 4.0], [3.0, 4.0]]),
    k=3,
    lam=1.0,
    utilities=None,
    selected=[0, 3],
    f_value=7.0,
    g_value=2.0,
    div=5.0,
    stage="sweep",
    threshold_range=(4.0, 5.0),
)

CASES: tuple[Case, ...] = (CASE_A, CASE_B, CASE_C)
