"""Type stubs for ``divsel._divsel``, the native (Rust, PyO3) extension module.

The wheel is ``abi3`` (``cp311-abi3``) and installs on CPython 3.11 through 3.14
with the GIL. Free-threaded CPython (3.14t) is supported too, but ``abi3`` does
not cover it: that build is a version-specific ``cp314t`` wheel.
"""

from collections.abc import Sequence
from typing import Literal, TypedDict

import numpy as np
from numpy.typing import NDArray

__version__: str

class GistResult(TypedDict):
    """The dict returned by :func:`gist_select_full`."""

    selected: list[int]
    """The selected row indices, in selection order; at most ``k`` of them."""
    f_value: float
    """``g(S) + lam * div(S)``, the value that was maximized."""
    g_value: float
    """``g(S)`` alone."""
    div: float
    """The minimum pairwise distance inside ``S``, or ``d_max`` when ``|S| <= 1``."""
    threshold: float
    """The distance threshold that produced ``selected``: ``0.0`` for
    ``"greedy"``, ``d_max`` for ``"diameter_pair"``, the winning ``d`` for
    ``"sweep"``."""
    stage: Literal["greedy", "diameter_pair", "sweep"]
    """Which branch of the paper's Algorithm 1 won."""
    d_max: float
    """The diameter the threshold sweep was built from: exact under
    ``diameter="exact"``, the estimate ``d_hat`` in ``[d_max/2, d_max]`` under
    ``diameter="approx"``."""

def gist_select(
    vectors: NDArray[np.float32],
    utilities: NDArray[np.float64] | Sequence[Sequence[int]] | None = None,
    *,
    k: int,
    lam: float = 1.0,
    eps: float = 0.1,
    metric: Literal["cosine", "euclidean"] = "cosine",
    utility: Literal["linear", "coverage", "facility_location"] = "linear",
    exhaustive_thresholds: bool = False,
    diameter: Literal["exact", "approx"] = "exact",
    diameter_sweeps: int = 3,
) -> list[int]:
    """Select up to ``k`` rows of ``vectors`` with GIST (arXiv:2405.18754v3, NeurIPS 2025).

    GIST maximizes ``f(S) = g(S) + lam * div(S)`` over ``|S| <= k``, where ``g``
    is a monotone submodular utility (``"linear"``: a weighted count,
    ``"coverage"``: the number of distinct items covered, ``"facility_location"``:
    the paper's own choice) and ``div(S)`` is the minimum pairwise distance inside
    ``S``, and it guarantees ``f(S) >= (1/2 - eps) * OPT`` for any such ``g``
    (Theorem 3.1) and ``(2/3 - eps) * OPT`` for a linear ``g`` (Theorem 3.3).
    Those bounds are proven for a true metric, so they hold for
    ``metric="euclidean"``; raw cosine distance ``1 - cos`` -- the paper's own
    experimental setting and the default here -- is used as a well-behaved
    heuristic. ``vectors`` is consumed zero-copy for ``metric="euclidean"`` (the
    numpy buffer is borrowed in place, under a read lock held for the whole
    call); ``metric="cosine"`` makes exactly one L2-normalised copy. The caller's
    array is never modified, and the solve runs with the interpreter lock
    released, so other Python threads keep running while Rust parallelises the
    threshold sweep.

    Args:
        vectors: A C-contiguous ``float32`` array of shape ``(n, d)``. Anything
            else -- another dtype, a 1-D array, a strided view, a Fortran-ordered
            array, a list -- raises ``TypeError``; ``np.ascontiguousarray(x,
            dtype=np.float32)`` is the fix.
        utilities: Depends on ``utility``. For ``"linear"``, a C-contiguous
            ``float64`` array of shape ``(n,)`` of finite, non-negative weights,
            or ``None`` for uniform weights (``g(S) = |S|``). For ``"coverage"``,
            a sequence of ``n`` sequences of non-negative int item ids; the
            universe is inferred as the largest id plus one, and a bitmap of that
            size is allocated, so ids must be dense (a huge sparse id allocates a
            bitmap of that size). For ``"facility_location"``, must be ``None``.
        k: The budget; ``|S| <= k``. Must be greater than zero: ``k <= 0``
            raises ``ValueError``. ``k > n`` is clamped to ``n``. The result can
            be shorter than ``k`` when no remaining point is at least the winning
            threshold away from every selected one.
        lam: The weight of the diversity term, finite and ``>= 0``.
        eps: The sweep accuracy in ``(0, 1]``; the threshold set has
            ``1 + floor(log(2/eps) / log(1+eps))`` entries (32 at the default).
        metric: ``"cosine"`` (rows are L2-normalised into a copy, distance is
            ``1 - a.b``) or ``"euclidean"`` (zero-copy).
        utility: ``"linear"``, ``"coverage"`` or ``"facility_location"``.
        exhaustive_thresholds: Replace the geometric threshold set with every
            ``dist(u, v) / 2`` -- ``O(n^2)`` greedy runs, intended for ``n`` of
            roughly 2000 or below; turns ``2/3 - eps`` into an exact ``2/3`` for
            a linear ``g``.
        diameter: ``"exact"`` for the ``O(n^2)`` diameter scan the paper uses,
            ``"approx"`` for a farthest-point double sweep whose estimate lies in
            ``[d_max/2, d_max]``.
        diameter_sweeps: Number of double sweeps under ``diameter="approx"``.
            Must be ``>= 0`` (a negative value raises ``ValueError``); ``0`` is
            treated as ``1``; ignored under ``"exact"``.

    Returns:
        The selected row indices, in selection order, at most ``k`` of them.

    Raises:
        TypeError: ``vectors`` or a linear ``utilities`` is not the required
            C-contiguous array, or a coverage ``utilities`` is not a sequence of
            int sequences.
        ValueError: An unknown ``metric``/``utility``/``diameter`` string,
            ``k <= 0``, a negative ``diameter_sweeps``, ``eps`` outside
            ``(0, 1]``, a negative or non-finite ``lam``, ``utilities`` whose
            length is not ``n``, a negative weight,
            a negative coverage id, a ``utilities`` array given with
            ``"facility_location"``, no ``utilities`` with ``"coverage"``, an
            empty or zero-dimensional ``vectors``, a NaN or infinite coordinate,
            or (cosine only) a row that cannot be normalised.
    """

def gist_select_full(
    vectors: NDArray[np.float32],
    utilities: NDArray[np.float64] | Sequence[Sequence[int]] | None = None,
    *,
    k: int,
    lam: float = 1.0,
    eps: float = 0.1,
    metric: Literal["cosine", "euclidean"] = "cosine",
    utility: Literal["linear", "coverage", "facility_location"] = "linear",
    exhaustive_thresholds: bool = False,
    diameter: Literal["exact", "approx"] = "exact",
    diameter_sweeps: int = 3,
) -> GistResult:
    """:func:`gist_select` with the full result.

    Same arguments, objective, guarantees (Theorems 3.1 and 3.3 of
    arXiv:2405.18754v3), metric caveat and zero-copy behaviour as
    :func:`gist_select`; instead of the bare index list it returns a
    :class:`GistResult` dict with ``selected`` (the same list), ``f_value``,
    ``g_value``, ``div``, ``threshold``, ``stage`` and ``d_max``, so the split of
    the objective and which branch of Algorithm 1 produced the answer are
    observable. ``f_value == g_value + lam * div`` holds exactly (the same
    double-precision arithmetic the Rust core performs).

    Returns:
        A :class:`GistResult` dict.

    Raises:
        TypeError, ValueError: As for :func:`gist_select`.
    """
