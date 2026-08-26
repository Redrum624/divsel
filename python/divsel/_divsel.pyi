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
    diameter_sweeps: int | None = None,
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
    threshold sweep. That read lock only excludes other *native* mutable borrows
    of the buffer: it does not stop another Python thread from writing into
    ``vectors`` while a call is in flight, so do not mutate the array
    concurrently (under ``"euclidean"`` the solve reads it in place).

    Args:
        vectors: A C-contiguous ``float32`` array of shape ``(n, d)``. Anything
            else -- another dtype, a 1-D array, a strided view, a Fortran-ordered
            array, a list -- raises ``TypeError``; ``np.ascontiguousarray(x,
            dtype=np.float32)`` is the fix.
        utilities: Depends on ``utility``. For ``"linear"``, a C-contiguous
            ``float64`` array of shape ``(n,)`` of finite, non-negative weights,
            or ``None`` for uniform weights (``g(S) = |S|``). For ``"coverage"``,
            a sequence of ``n`` sequences of non-negative int item ids; the
            universe is inferred as the largest id plus one and a coverage flag
            is allocated per item id -- one **byte**, not one bit, and the
            parallel sweep clones that array once per worker job, measured at
            about 17.5 bytes per item id of peak working set. Ids must therefore
            be dense: one stray sparse id sets the footprint for the whole call
            (``2**24`` costs roughly 280 MB, ``2**31`` roughly 36 GB, which will
            abort the process). Ids above ``2**32 - 1`` -- of any magnitude,
            including ones too large for a signed 64-bit integer -- or a
            universe that does not fit the platform's ``usize``, raise
            ``ValueError`` naming the row, never ``OverflowError`` and never
            ``TypeError``. For
            ``"facility_location"``, must be ``None``.
        k: The budget; ``|S| <= k``. Must be an ``int`` greater than zero:
            ``k <= 0`` raises ``ValueError``, a ``bool`` raises ``TypeError``
            (``True`` is not read as ``1``), and a ``k`` too large for a signed
            64-bit integer raises ``ValueError`` (never ``OverflowError``).
            ``k > n`` is clamped to ``n``. The result can be shorter than ``k``
            when no remaining point is at least the winning threshold away from
            every selected one.
        lam: The weight of the diversity term, finite and ``>= 0``.
        eps: The sweep accuracy, from ``np.finfo(np.float32).eps``
            (``1.1920929e-07``) up to ``1.0`` inclusive; anything outside that
            range raises ``ValueError``. The threshold set has
            ``1 + floor(log(2/eps) / log(1+eps))`` entries (32 at the default),
            or ``1 + floor(log(4/eps) / log(1+eps))`` under ``diameter="approx"``
            (39 at the default), so the cost grows like ``1/eps``: ``1e-4``
            already means 99 040 thresholds, and one greedy run each. The lower
            end of the range is where the ``float32`` grid stops being able to
            separate two consecutive entries -- below it the set is unbounded,
            not precise. That floor is a representability bound, not a resource
            one: ``eps = 1.1920929e-07`` itself builds 139 548 968 thresholds
            (about 560 MB) and runs greedy that many times. Nothing caps the
            threshold count.
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
            ``None``, the default, means **3** sweeps -- the default object
            really is ``None``, in the signature, in the stub and at runtime, so
            passing it explicitly is the same as omitting it. Otherwise it must
            be an ``int`` ``>= 0`` (a negative value raises ``ValueError``, and
            so does one too large for a signed 64-bit integer; a ``bool`` raises
            ``TypeError``, exactly as for ``k``); ``0`` is treated as ``1``;
            ignored under ``"exact"``. No upper bound is enforced: each sweep
            costs ``O(n * d)``, so a very large value simply runs that long.

    Returns:
        The selected row indices, in selection order, at most ``k`` of them.

    Raises:
        TypeError: ``vectors`` or a linear ``utilities`` is not the required
            C-contiguous array, a coverage ``utilities`` is not a sequence of
            int sequences, or ``k``/``diameter_sweeps`` is a ``bool`` (or any
            other non-``int``).
        ValueError: An unknown ``metric``/``utility``/``diameter`` string,
            ``k <= 0`` or a ``k``/``diameter_sweeps`` outside the signed 64-bit
            range, a negative ``diameter_sweeps``, ``eps`` outside
            ``[np.finfo(np.float32).eps, 1]``, a negative or non-finite ``lam``,
            ``utilities`` whose length is not ``n``, a negative weight,
            a negative coverage id, a ``utilities`` array given with
            ``"facility_location"``, no ``utilities`` with ``"coverage"``, an
            empty or zero-dimensional ``vectors``, a NaN or infinite coordinate,
            or (cosine only) a row that cannot be normalised.

    Note:
        The solve runs on rayon's process-global thread pool, created on the
        first call and never shut down. Its size comes from ``RAYON_NUM_THREADS``
        (read once, at first use) or from the available parallelism; there is no
        per-call thread count. On Linux, forking after a first call --
        ``os.fork``, or ``multiprocessing`` with the default ``fork`` start
        method -- gives the child a pool whose worker threads do not exist, and
        the child's next call deadlocks. Use the ``spawn`` or ``forkserver``
        start method, or make the first ``divsel`` call inside each child.
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
    diameter_sweeps: int | None = None,
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
