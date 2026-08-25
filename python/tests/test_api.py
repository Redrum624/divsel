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
    with pytest.raises(ValueError, match=r"epsilon 0 must be in the range 0 < eps <= 1"):
        gist_select(x, k=2, eps=0.0)
    with pytest.raises(ValueError, match="epsilon"):
        gist_select(x, k=2, eps=1.0000001)
    with pytest.raises(ValueError, match="lambda"):
        gist_select(x, k=2, lam=-1.0)


def test_eps_of_exactly_one_is_accepted():
    # The range is 0 < eps <= 1, closed at the top; the message says so.
    x = _random_vectors(6, 3)
    assert 0 < len(gist_select(x, k=2, eps=1.0)) <= 2


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
