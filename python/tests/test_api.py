"""API tests for ``divsel.gist_select`` / ``divsel.gist_select_full`` (Task 7)."""

from __future__ import annotations

import numpy as np
import pytest

import divsel
from divsel import gist_select, gist_select_full
from fixtures import CASES

STAGES = {"greedy", "diameter_pair", "sweep"}
UTILITY_KINDS = ("linear", "coverage", "facility_location")


def _random_vectors(n: int = 200, d: int = 32, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.ascontiguousarray(rng.standard_normal((n, d)), dtype=np.float32)


def _random_sets(n: int, universe: int, seed: int = 1) -> list[list[int]]:
    rng = np.random.default_rng(seed)
    return [
        sorted(set(rng.integers(0, universe, size=rng.integers(0, 6)).tolist()))
        for _ in range(n)
    ]


def _utilities_for(kind: str, x: np.ndarray):
    if kind == "linear":
        return np.random.default_rng(2).random(x.shape[0])
    if kind == "coverage":
        return _random_sets(x.shape[0], universe=50)
    return None


def _f32_column(values):
    return np.ascontiguousarray(np.array(values, dtype=np.float32).reshape(-1, 1))


# ---- package surface --------------------------------------------------------


def test_package_exports_both_entry_points_and_version():
    assert set(divsel.__all__) == {"__version__", "gist_select", "gist_select_full"}
    assert callable(divsel.gist_select)
    assert callable(divsel.gist_select_full)
    assert gist_select.__doc__ and "GIST" in gist_select.__doc__
    assert gist_select_full.__doc__ and "GIST" in gist_select_full.__doc__


# ---- shape / dtype / argument errors ----------------------------------------


def test_float64_vectors_raise_typeerror_naming_the_fix():
    x = np.zeros((4, 3), dtype=np.float64)
    with pytest.raises(TypeError, match="ascontiguousarray"):
        gist_select(x, k=2)


def test_one_dimensional_vectors_raise_typeerror():
    with pytest.raises(TypeError, match="ascontiguousarray"):
        gist_select(np.zeros(8, dtype=np.float32), k=2)


def test_non_contiguous_view_raises_typeerror():
    x = _random_vectors(8, 8)
    view = x[:, ::2]
    assert not view.flags.c_contiguous
    with pytest.raises(TypeError, match="ascontiguousarray"):
        gist_select(view, k=2)


def test_fortran_ordered_vectors_raise_typeerror():
    x = np.asfortranarray(_random_vectors(8, 4))
    assert x.flags.f_contiguous and not x.flags.c_contiguous
    with pytest.raises(TypeError, match="ascontiguousarray"):
        gist_select(x, k=2)


def test_non_array_vectors_raise_typeerror():
    with pytest.raises(TypeError, match="ascontiguousarray"):
        gist_select([[0.0, 1.0], [1.0, 0.0]], k=1)


def test_utilities_of_wrong_length_raise_valueerror():
    x = _random_vectors(6, 4)
    with pytest.raises(ValueError, match="6"):
        gist_select(x, np.ones(5, dtype=np.float64), k=2)


def test_utilities_of_wrong_dtype_raise_typeerror_naming_the_fix():
    x = _random_vectors(6, 4)
    with pytest.raises(TypeError, match=r"ascontiguousarray\(u, dtype=np.float64\)"):
        gist_select(x, np.ones(6, dtype=np.float32), k=2)


def test_negative_utility_raises_valueerror():
    x = _random_vectors(4, 2)
    with pytest.raises(ValueError, match="non-negative"):
        gist_select(x, np.array([1.0, -1.0, 1.0, 1.0]), k=2)


def test_unknown_metric_raises_valueerror_listing_the_choices():
    with pytest.raises(ValueError, match="cosine.*euclidean"):
        gist_select(_random_vectors(4, 2), k=2, metric="manhattan")


def test_unknown_utility_raises_valueerror_listing_the_choices():
    with pytest.raises(ValueError, match="linear.*coverage.*facility_location"):
        gist_select(_random_vectors(4, 2), k=2, utility="saturated")


def test_unknown_diameter_mode_raises_valueerror_listing_the_choices():
    with pytest.raises(ValueError, match="exact.*approx"):
        gist_select(_random_vectors(4, 2), k=2, diameter="guess")


def test_k_zero_raises_valueerror():
    with pytest.raises(ValueError, match="greater than zero"):
        gist_select(_random_vectors(4, 2), k=0)


@pytest.mark.parametrize("k", [-1, -(2**63)], ids=["minus_one", "i64_min"])
def test_negative_k_raises_valueerror(k):
    # Not pyo3's OverflowError from a usize extraction: a bad budget is a
    # ValueError like every other bad argument, and the message names no Rust type.
    x = _random_vectors(4, 2)
    with pytest.raises(ValueError, match="greater than zero") as info:
        gist_select(x, k=k)
    assert "unsigned" not in str(info.value)
    with pytest.raises(ValueError, match="greater than zero"):
        gist_select_full(x, k=k)


@pytest.mark.parametrize("k", [True, False], ids=["true", "false"])
def test_bool_k_raises_typeerror(k):
    # bool is an int subclass, so an integer extraction would read True as 1.
    x = _random_vectors(4, 2)
    with pytest.raises(TypeError, match="k must be an int, not bool"):
        gist_select(x, k=k)
    with pytest.raises(TypeError, match="k must be an int, not bool"):
        gist_select_full(x, k=k)


@pytest.mark.parametrize("k", [2**63, 2**200, -(2**64)], ids=["i64_max_plus", "huge", "very_negative"])
def test_k_outside_the_i64_range_raises_valueerror_not_overflowerror(k):
    # OverflowError is not in the documented Raises block, and a budget out of
    # range is a ValueError like every other bad argument.
    x = _random_vectors(4, 2)
    for fn in (gist_select, gist_select_full):
        with pytest.raises(ValueError) as info:
            fn(x, k=k)
        assert not isinstance(info.value, OverflowError)


@pytest.mark.parametrize("sweeps", [True, False], ids=["true", "false"])
def test_bool_diameter_sweeps_raises_typeerror(sweeps):
    # Same rule as `k`: bool is an int subclass, and True must not mean 1 sweep.
    x = _random_vectors(4, 2)
    with pytest.raises(TypeError, match="diameter_sweeps must be an int, not bool"):
        gist_select(x, k=2, diameter="approx", diameter_sweeps=sweeps)
    with pytest.raises(TypeError, match="diameter_sweeps must be an int, not bool"):
        gist_select_full(x, k=2, diameter="approx", diameter_sweeps=sweeps)


def test_diameter_sweeps_outside_the_i64_range_raises_valueerror():
    x = _random_vectors(4, 2)
    with pytest.raises(ValueError, match="diameter_sweeps") as info:
        gist_select(x, k=2, diameter="approx", diameter_sweeps=2**200)
    assert not isinstance(info.value, OverflowError)


def test_the_declared_default_sweeps_is_the_object_the_binding_uses():
    # Three sources used to give three answers: the pyo3 default (`None`), the
    # hand-written text signature (`3`) and the stub (`3`). They agree on `None`
    # now, and the docstring is where "None means 3 sweeps" is written.
    # The *behaviour* of that default is pinned below, on a fixture where the
    # sweep count decides the answer.
    import inspect

    assert (
        inspect.signature(gist_select).parameters["diameter_sweeps"].default is None
    )
    assert (
        inspect.signature(gist_select_full).parameters["diameter_sweeps"].default
        is None
    )
    assert "diameter_sweeps=None" in str(
        getattr(gist_select, "__text_signature__", "")
    )


def test_negative_diameter_sweeps_raises_valueerror():
    x = _random_vectors(4, 2)
    with pytest.raises(ValueError, match="diameter_sweeps must be non-negative"):
        gist_select(x, k=2, diameter="approx", diameter_sweeps=-1)
    with pytest.raises(ValueError, match="diameter_sweeps must be non-negative"):
        gist_select_full(x, k=2, diameter="approx", diameter_sweeps=-1)
    # Rejected even under diameter="exact", where the value is otherwise unused.
    with pytest.raises(ValueError, match="diameter_sweeps must be non-negative"):
        gist_select(x, k=2, diameter_sweeps=-1)


def test_facility_location_with_utilities_array_raises_valueerror():
    x = _random_vectors(4, 2)
    with pytest.raises(ValueError, match="facility_location takes no utilities array"):
        gist_select(x, np.ones(4), k=2, utility="facility_location")


def test_coverage_without_utilities_raises_valueerror():
    with pytest.raises(ValueError, match="coverage"):
        gist_select(_random_vectors(4, 2), None, k=2, utility="coverage")


def test_coverage_with_negative_item_id_raises_valueerror():
    x = _random_vectors(3, 2)
    with pytest.raises(ValueError, match="non-negative"):
        gist_select(x, [[0, 1], [2, -3], [4]], k=2, utility="coverage")


def test_coverage_with_non_sequence_utilities_raises_typeerror():
    x = _random_vectors(3, 2)
    with pytest.raises(TypeError, match="coverage"):
        gist_select(x, np.ones(3), k=2, utility="coverage")


def test_coverage_with_wrong_number_of_sets_raises_valueerror():
    x = _random_vectors(3, 2)
    with pytest.raises(ValueError, match="3"):
        gist_select(x, [[0], [1]], k=2, utility="coverage")


def test_invalid_eps_and_lambda_raise_valueerror():
    x = _random_vectors(4, 2)
    with pytest.raises(ValueError, match=r"epsilon 0e0 must be in the range"):
        gist_select(x, k=2, eps=0.0)
    with pytest.raises(ValueError, match="epsilon"):
        gist_select(x, k=2, eps=1.0000001)
    with pytest.raises(ValueError, match="lambda"):
        gist_select(x, k=2, lam=-1.0)


def test_eps_of_exactly_one_is_accepted():
    # The range is closed at the top; the message says so.
    x = _random_vectors(6, 3)
    assert 0 < len(gist_select(x, k=2, eps=1.0)) <= 2


@pytest.mark.parametrize(
    "eps",
    [1e-30, 1e-16, 1e-8, float(np.finfo(np.float32).eps) / 2],
    ids=["1e-30", "1e-16", "1e-8", "half_f32_eps"],
)
def test_eps_below_the_float32_epsilon_raises_instead_of_killing_the_process(eps):
    """An eps too small for the f32 grid is rejected, not attempted.

    The grid is built by repeated ``p *= 1 + eps`` in f64 and cast to f32: below
    ``f32::EPSILON`` two consecutive entries cannot differ, and below ``2**-53``
    the multiplication does not advance at all -- an unbounded push into a Vec
    that used to end with ``memory allocation of 137438953472 bytes failed`` and
    a dead interpreter (exit 127), with no Python exception to catch.
    """
    x = _random_vectors(4, 2)
    with pytest.raises(ValueError, match="epsilon"):
        gist_select(x, k=2, eps=eps)
    with pytest.raises(ValueError, match="epsilon"):
        gist_select_full(x, k=2, eps=eps)


def test_eps_of_exactly_the_float32_epsilon_is_accepted():
    # The bottom of the range is inclusive. One point means d_max == 0, so the
    # sweep is skipped and this does not build the ~1.4e8 thresholds that eps
    # would otherwise ask for.
    x = _random_vectors(1, 2)
    assert gist_select(x, k=1, eps=float(np.finfo(np.float32).eps)) == [0]


def test_empty_and_zero_dim_inputs_raise_valueerror():
    with pytest.raises(ValueError):
        gist_select(np.zeros((0, 4), dtype=np.float32), k=1)
    with pytest.raises(ValueError):
        gist_select(np.zeros((4, 0), dtype=np.float32), k=1)


def test_nan_coordinate_raises_valueerror():
    x = _random_vectors(4, 2)
    x[2, 1] = np.nan
    with pytest.raises(ValueError, match="row 2, column 1"):
        gist_select(x, k=2, metric="euclidean")


def test_zero_row_raises_valueerror_under_cosine_only():
    x = _random_vectors(4, 2)
    x[1] = 0.0
    with pytest.raises(ValueError, match="row 1"):
        gist_select(x, k=2, metric="cosine")
    assert len(gist_select(x, k=2, metric="euclidean")) == 2


# ---- budget and results -----------------------------------------------------


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_k_larger_than_n_returns_every_index_as_a_permutation(metric):
    x = _random_vectors(7, 5)
    out = gist_select(x, k=50, metric=metric)
    assert sorted(out) == list(range(7))
    assert len(set(out)) == 7


def test_gist_select_full_keys_and_types():
    x = _random_vectors(30, 6)
    r = gist_select_full(x, k=5)
    assert set(r) == {"selected", "f_value", "g_value", "div", "threshold", "stage", "d_max"}
    assert isinstance(r["selected"], list) and all(isinstance(i, int) for i in r["selected"])
    for key in ("f_value", "g_value", "div", "threshold", "d_max"):
        assert isinstance(r[key], float), key
    assert isinstance(r["stage"], str) and r["stage"] in STAGES
    assert 0 < len(r["selected"]) <= 5
    assert len(set(r["selected"])) == len(r["selected"])
    assert r["div"] >= 0.0
    assert r["d_max"] >= r["div"] >= 0.0
    assert r["f_value"] == r["g_value"] + 1.0 * r["div"]


def test_gist_select_matches_gist_select_full_selected():
    x = _random_vectors(40, 8)
    u = np.random.default_rng(3).random(40)
    assert gist_select(x, u, k=6) == gist_select_full(x, u, k=6)["selected"]


@pytest.mark.parametrize("kind", UTILITY_KINDS)
def test_determinism_across_ten_runs(kind):
    x = _random_vectors()
    u = _utilities_for(kind, x)
    first = gist_select_full(x, u, k=10, utility=kind)
    assert 0 < len(first["selected"]) <= 10
    for _ in range(9):
        again = gist_select_full(x, u, k=10, utility=kind)
        assert again["selected"] == first["selected"]
        assert again["f_value"].hex() == first["f_value"].hex()
        assert again["stage"] == first["stage"]


def test_approx_diameter_mode_runs_and_bounds_d_max():
    x = _random_vectors(100, 16)
    exact = gist_select_full(x, k=5, metric="euclidean", diameter="exact")
    approx = gist_select_full(x, k=5, metric="euclidean", diameter="approx", diameter_sweeps=2)
    assert exact["d_max"] / 2.0 <= approx["d_max"] <= exact["d_max"]
    assert 0 < len(approx["selected"]) <= 5


def test_exhaustive_thresholds_runs_on_a_small_instance():
    x = _random_vectors(25, 4)
    r = gist_select_full(x, k=4, metric="euclidean", exhaustive_thresholds=True)
    assert 0 < len(r["selected"]) <= 4
    assert r["stage"] in STAGES


# ---- zero-copy observability ------------------------------------------------


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_input_array_is_never_mutated(metric):
    x = _random_vectors(50, 8)
    before = x.copy()
    gist_select(x, k=5, metric=metric)
    assert np.array_equal(x, before)
    assert x.flags.writeable


@pytest.mark.parametrize("metric", ["cosine", "euclidean"])
def test_read_only_array_is_accepted(metric):
    x = _random_vectors(50, 8)
    expected = gist_select(x, k=5, metric=metric)
    x.setflags(write=False)
    assert not x.flags.writeable
    assert gist_select(x, k=5, metric=metric) == expected


# ---- shared hand-computed fixture (R-G25) -----------------------------------


@pytest.mark.parametrize("case", CASES, ids=[c.name for c in CASES])
def test_shared_fixture_matches_hand_computation(case):
    r = gist_select_full(
        case.vectors, case.utilities, k=case.k, lam=case.lam, metric="euclidean"
    )
    assert r["selected"] == case.selected
    assert r["f_value"].hex() == float(case.f_value).hex()
    assert r["g_value"].hex() == float(case.g_value).hex()
    assert r["div"].hex() == float(case.div).hex()
    assert r["stage"] == case.stage
    lo, hi = case.threshold_range
    assert lo < r["threshold"] <= hi
    assert gist_select(
        case.vectors, case.utilities, k=case.k, lam=case.lam, metric="euclidean"
    ) == case.selected


# ---- coverage ---------------------------------------------------------------


def test_coverage_counts_newly_covered_items():
    # Four points on a line at x = 0, 1, 2, 10 with lambda = 0, so f is pure
    # coverage. Sets cover a five-item universe {0..4}:
    #   marginals from empty: |{0,1,2}| = 3, |{2,3}| = 2, |{3,4}| = 2, |{0,1}| = 2
    #   -> pick 0 (gain 3); then newly covered: set1 -> {3} = 1, set2 -> {3,4} = 2,
    #      set3 -> {} = 0 -> pick 2 (gain 2). g = 5 = the whole universe.
    # Thresholds d <= 2 keep point 2 admissible after 0, so the sweep reproduces
    # [0, 2] with f = 5; larger d leaves only point 3 (gain 0), f = 3 < 5.
    x = _f32_column([0.0, 1.0, 2.0, 10.0])
    sets = [[0, 1, 2], [2, 3], [3, 4], [0, 1]]
    r = gist_select_full(x, sets, k=2, lam=0.0, metric="euclidean", utility="coverage")
    assert r["selected"] == [0, 2]
    assert r["g_value"] == 5.0
    assert r["f_value"] == 5.0
    # k = 1 is just the largest set.
    one = gist_select_full(x, sets, k=1, lam=0.0, metric="euclidean", utility="coverage")
    assert one["selected"] == [0]
    assert one["g_value"] == 3.0


def test_coverage_universe_is_inferred_and_duplicates_count_once():
    x = _f32_column([0.0, 1.0, 2.0, 3.0])
    # Universe is max id + 1 = 8; ids 7,7 count once; an empty set is legal.
    r = gist_select_full(
        x, [[7, 7], [], [0, 7], [5]], k=4, lam=0.0, metric="euclidean", utility="coverage"
    )
    assert r["g_value"] == 3.0  # items {0, 5, 7}
    assert sorted(r["selected"]) == [0, 1, 2, 3]


def test_coverage_with_all_empty_sets_runs():
    x = _f32_column([0.0, 1.0, 2.0])
    r = gist_select_full(x, [[], [], []], k=2, lam=1.0, metric="euclidean", utility="coverage")
    assert r["g_value"] == 0.0
    assert r["selected"] == [0, 2]  # only diversity is left: the widest pair


def test_coverage_item_id_above_u32_max_raises_valueerror():
    # The binding stores ids as u32; the universe (largest id + 1) is checked
    # against usize separately, which only a 32-bit build can trip.
    x = _random_vectors(2, 2)
    with pytest.raises(ValueError, match=r"row 1 must be a non-negative int no larger than 4294967295"):
        gist_select(x, [[0], [2**32]], k=1, utility="coverage")


def test_coverage_item_id_above_i64_max_is_the_same_valueerror():
    """An id too large for `i64` is a range error, not a type error.

    `_divsel.pyi` promises `ValueError` for "Ids above 2**32 - 1" and reserves
    `TypeError` for "a coverage `utilities` is not a sequence of int sequences".
    A blanket `map_err` turned the extraction's `OverflowError` into that
    `TypeError`, whose message ("must be a sequence of sequences of
    non-negative int item ids") is also false for `[[0], [2**200]]` -- which is
    exactly such a sequence. Same rule as `k`/`diameter_sweeps`: the error keys
    on the range, whatever int-like the id arrived as.
    """
    x = _random_vectors(2, 2)
    for oversized in (2**200, 2**64, -(2**200)):
        with pytest.raises(ValueError, match=r"row 1 must be a non-negative int"):
            gist_select(x, [[0], [oversized]], k=1, utility="coverage")

    # A genuinely wrong element type is still a TypeError.
    with pytest.raises(TypeError, match="sequence of sequences"):
        gist_select(x, [[0], ["nope"]], k=1, utility="coverage")
    # numpy integer scalars are ints for this purpose, in range and out of it.
    assert gist_select(x, [[np.uint32(0)], [np.int64(3)]], k=1, utility="coverage")


def test_a_coverage_id_whose_index_raises_propagates_that_exception():
    """"The error keys on the exception, not on the argument's type" -- both ways.

    `coverage_sets` maps `OverflowError` to the range `ValueError` above and
    used to map *everything else* to the shape `TypeError`, so a
    `KeyboardInterrupt` raised inside a custom `__index__` was discarded and
    reported as "utilities ... must be a sequence of sequences of non-negative
    int item ids". `budget` gets the same shape of call right (`Err(err) =>
    Err(err)`), and `k=<that object>` propagates the Ctrl-C, so the two halves
    of the same documented rule disagreed.
    """
    x = _random_vectors(2, 2)

    class Raises:
        def __init__(self, exc):
            self._exc = exc

        def __index__(self):
            raise self._exc

    for exc in (KeyboardInterrupt("ctrl-c in an id"), ValueError("a picky __index__")):
        with pytest.raises(type(exc), match=str(exc)):
            gist_select(x, [[0], [Raises(exc)]], k=1, utility="coverage")
        # ... exactly as the same object raises through `k`.
        with pytest.raises(type(exc), match=str(exc)):
            gist_select(x, None, k=Raises(exc))

    # And a plain non-int is still the shape TypeError, not a leaked one.
    with pytest.raises(TypeError, match="sequence of sequences"):
        gist_select(x, [[0], [object()]], k=1, utility="coverage")


def test_an_unprintable_out_of_range_coverage_id_still_names_its_row():
    """The `"<unprintable>"` arm of `out_of_range`, which nothing drove.

    The range message interpolates `item.str()`, and `__str__` is caller code
    that can fail. It must still be a `ValueError` naming the row, not a
    secondary exception from inside the error path.
    """
    x = _random_vectors(2, 2)

    class Unprintable:
        def __index__(self):
            return 2**200

        def __str__(self):
            raise RuntimeError("no str for you")

    with pytest.raises(ValueError, match=r"coverage item id <unprintable> at row 1"):
        gist_select(x, [[0], [Unprintable()]], k=1, utility="coverage")


# --- round-2 gaps ------------------------------------------------------------

# A 12x3 Gaussian set on which the farthest-point walk needs four double sweeps
# to converge, so d_hat is strictly increasing for sweeps = 1, 2, 3, 4:
# 3.8591561, 4.2479930, 4.4063721, 4.4678040. Every other fixture in this file
# converges on the first sweep, which is why the default-sweeps tests below can
# say something the sweep count actually decides.
_SWEEP_SENSITIVE = [
    0.42473456, 0.53754765, -0.65292674, -0.495182, 0.76948655, -0.13159879,
    0.39689574, -0.192833, 1.8935074, -1.3601785, -0.45732597, 0.49488983,
    -0.23497039, 0.33717105, -1.7059377, 1.992929, -0.9904514, 0.55411506,
    -0.29105875, 0.18395717, 0.650892, -0.45368975, 2.3688433, -0.32602257,
    -0.5333204, -1.0139397, -1.6846485, -0.49256995, -1.9239715, -0.8081258,
    1.9454916, 0.95573187, 1.4676777, -0.5040848, 1.3585472, -1.523295,
]


def _sweep_sensitive_vectors() -> np.ndarray:
    return np.ascontiguousarray(
        np.array(_SWEEP_SENSITIVE, dtype=np.float32).reshape(12, 3)
    )


def _d_hat(sweeps) -> float:
    kw = {} if sweeps is None else {"diameter_sweeps": sweeps}
    return gist_select_full(
        _sweep_sensitive_vectors(), k=1, metric="euclidean", diameter="approx", **kw
    )["d_max"]


def test_the_sweep_sensitive_fixture_separates_every_sweep_count():
    # Guards the guard: without this, the two tests below would be assertions
    # about a value no sweep count can change.
    assert _d_hat(1) < _d_hat(2) < _d_hat(3) < _d_hat(4)


def test_diameter_sweeps_defaults_to_exactly_three_when_omitted():
    # The pyo3 default is None (the argument is taken as an object so a bool can
    # be rejected); omitting it must mean the documented 3 sweeps, and this
    # fixture can tell 3 apart from both 2 and 4.
    assert _d_hat(None) == _d_hat(3)
    assert _d_hat(None) != _d_hat(2)
    assert _d_hat(None) != _d_hat(4)


def test_passing_none_for_diameter_sweeps_is_the_default():
    # `None` is what the signature itself says, so it has to be accepted.
    assert _d_hat(None) == gist_select_full(
        _sweep_sensitive_vectors(),
        k=1,
        metric="euclidean",
        diameter="approx",
        diameter_sweeps=None,
    )["d_max"]


def test_zero_diameter_sweeps_is_one_sweep_not_zero_sweeps():
    assert _d_hat(0) == _d_hat(1)
    assert _d_hat(0) != _d_hat(2)


@pytest.mark.parametrize("name", ["k", "diameter_sweeps"])
def test_a_non_int_budget_raises_typeerror(name):
    # `budget`'s passthrough arm: not an int, not a bool, so the extraction's own
    # TypeError stands. Documented in _divsel.pyi as "or any other non-``int``".
    x = _random_vectors(4, 2)
    kwargs = {"k": 2, "diameter": "approx"}
    kwargs[name] = "3"
    with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
        gist_select(x, **kwargs)
    with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
        gist_select_full(x, **kwargs)


@pytest.mark.parametrize(
    "value",
    [np.uint64(2**63), np.uint64(2**64 - 1)],
    ids=["uint64_i64_max_plus_one", "uint64_max"],
)
def test_a_numpy_integer_outside_the_i64_range_raises_valueerror(value):
    # The stub promises ValueError, "never OverflowError", without restricting
    # that to `int` subclasses: a numpy integer scalar is an int-like the binding
    # accepts everywhere else, so the range error has to be a ValueError too.
    x = _random_vectors(4, 2)
    for fn in (gist_select, gist_select_full):
        with pytest.raises(ValueError) as info:
            fn(x, k=value)
        assert not isinstance(info.value, OverflowError)
        with pytest.raises(ValueError) as info:
            fn(x, k=2, diameter="approx", diameter_sweeps=value)
        assert not isinstance(info.value, OverflowError)


def test_an_index_object_outside_the_i64_range_raises_valueerror():
    class Huge:
        def __index__(self):
            return 2**200

    x = _random_vectors(4, 2)
    with pytest.raises(ValueError) as info:
        gist_select(x, k=Huge())
    assert not isinstance(info.value, OverflowError)


def test_a_numpy_integer_budget_inside_the_range_is_accepted():
    # The other side of the rule above: an int-like that fits is still a budget.
    x = _random_vectors(6, 2)
    assert gist_select(x, k=np.int64(2)) == gist_select(x, k=2)


def test_a_strided_linear_utilities_vector_raises_typeerror_naming_the_fix():
    # `is_c_contiguous` on the 1-D `utilities`: a slice with a stride has the
    # right dtype and the right length, and `as_slice` would hand back the
    # wrong values.
    x = _random_vectors(4, 2)
    strided = np.ones(8, dtype=np.float64)[::2]
    assert strided.shape == (4,) and not strided.flags["C_CONTIGUOUS"]
    with pytest.raises(TypeError, match=r"ascontiguousarray\(u, dtype=np.float64\)"):
        gist_select(x, strided, k=2)


def test_identical_rows_give_a_zero_diameter_through_the_binding():
    # d_max == 0 with n > 1: the sweep is skipped entirely, so the stage stays
    # "greedy" and div is 0. Pinned in Rust, never through the binding.
    x = np.ascontiguousarray(np.tile(np.array([1.0, 2.0, 3.0]), (4, 1)), dtype=np.float32)
    r = gist_select_full(x, k=2, lam=1.0, metric="euclidean")
    assert r["d_max"] == 0.0
    assert r["div"] == 0.0
    assert r["threshold"] == 0.0
    assert r["stage"] == "greedy"
    assert r["selected"] == [0, 1]
    assert r["f_value"] == r["g_value"]


# --- round-4 gaps ------------------------------------------------------------


def test_lambda_zero_with_an_infinite_div_is_g_not_nan():
    """``lam == 0`` contributes a literal ``0.0``, never ``0.0 * div``.

    ``Points::new`` validates coordinates, not distances, so a point set of
    finite float32 coordinates far enough apart overflows the squared distance
    and ``div`` is ``+inf``. ``0.0 * inf`` is ``nan``, and the stub used to
    promise ``f_value == g_value + lam * div`` "exactly" -- which is ``nan``
    here, so the promise was false on the one input the short circuit exists
    for. ``docs/CONFORMANCE.md`` rule 18 is the port-facing statement, and no
    fixture can pin it: every fixture input is a dyadic rational in [-4, 4].
    """
    x = np.ascontiguousarray(np.array([[-3.0e38], [3.0e38]], dtype=np.float32))
    out = gist_select_full(x, None, k=2, lam=0.0, metric="euclidean")
    assert out["selected"] == [0, 1]
    assert np.isinf(out["div"]) and out["div"] > 0
    assert np.isinf(out["d_max"])
    assert out["f_value"] == out["g_value"] == 2.0
    assert out["stage"] == "sweep"
    # The literal formula, for the record: what a port transcribing it gets.
    assert np.isnan(out["g_value"] + 0.0 * out["div"])

    # A non-zero lam still forms the product, so f is legitimately infinite.
    positive = gist_select_full(x, None, k=2, lam=1.0, metric="euclidean")
    assert np.isinf(positive["f_value"])


@pytest.mark.parametrize("metric", ["euclidean", "cosine"])
@pytest.mark.parametrize("utility", ["linear", "facility_location"])
def test_approx_diameter_works_for_every_metric_and_utility(metric, utility):
    """``diameter="approx"`` outside euclidean + linear, which nothing covered.

    Every ``diameter="approx"`` call in this file passed ``metric="euclidean"``
    with the default utility, the one approx fixture is euclidean/linear, and
    every Rust approx unit test was linear/euclidean -- so the whole cosine
    approx path, and ``FacilityLocation`` built while the driver sweeps on
    ``d_hat`` (CONFORMANCE rule 10's exception), were asserted by nothing.
    """
    x = _sweep_sensitive_vectors()
    approx = gist_select_full(
        x, None, k=3, metric=metric, utility=utility, diameter="approx",
        diameter_sweeps=1,
    )
    exact = gist_select_full(x, None, k=3, metric=metric, utility=utility)

    assert len(approx["selected"]) <= 3
    assert len(set(approx["selected"])) == len(approx["selected"])
    # d_hat is the reported diameter, and it lies in [d_max/2, d_max].
    assert exact["d_max"] / 2.0 <= approx["d_max"] <= exact["d_max"]
    if utility == "facility_location":
        # Rule 10's exception, pinned: the two runs disagree about `d_max` (the
        # sweep genuinely misses here) and agree bit-for-bit about `g`, because
        # the similarity scale stays the EXACT diameter whatever the mode says.
        assert approx["d_max"] < exact["d_max"], "the sweep must miss, or this pins nothing"
        assert approx["selected"] == exact["selected"]
        assert approx["g_value"] == exact["g_value"]


def test_a_huge_diameter_sweeps_is_clamped_to_n_instead_of_hanging():
    """``diameter_sweeps`` is unvalidated, and each sweep is ``O(n * d)``.

    Measured on a five-point set in release: 1 sweep 8.7 us, 20e6 sweeps 3.46 s
    -- strictly linear -- so ``diameter_sweeps=2**62`` was about 1e12 seconds
    and hung the interpreter uninterruptibly. The core clamps to ``n``, which is
    result-preserving: each sweep starts where the last ended, so the sequence
    of starting points repeats within ``n`` steps.
    """
    x = _sweep_sensitive_vectors()
    at_n = gist_select_full(
        x, None, k=3, metric="euclidean", diameter="approx", diameter_sweeps=len(x)
    )
    for sweeps in (len(x) + 1, 2**62, 2**63 - 1):
        out = gist_select_full(
            x, None, k=3, metric="euclidean", diameter="approx",
            diameter_sweeps=sweeps,
        )
        assert out == at_n, f"diameter_sweeps={sweeps}"
