//! Python bindings for `divsel`: the `divsel._divsel` extension module.
//!
//! Two entry points, `gist_select` and `gist_select_full`, run [`divsel::gist()`]
//! over a numpy `float32` matrix. Under `metric="euclidean"` the numpy buffer is
//! borrowed in place through [`Points::borrowed`] -- zero-copy; under
//! `metric="cosine"` that same constructor makes one L2-normalised copy. The
//! caller's array is never written to in either case. The solve runs with the
//! interpreter detached ([`Python::detach`]), so rayon's threshold sweep
//! parallelises and other Python threads keep running; the numpy read guard is
//! held for the whole call.
//!
//! Free-threaded CPython (3.14t) is supported by default -- PyO3 0.28+ makes
//! that opt-out and this module does not opt out -- but such a wheel is version
//! specific (`cp314t`), not `abi3`.

#![deny(missing_docs)]
#![warn(clippy::all)]

use divsel::{
    gist, Coverage, DiameterMode, DivselError, FacilityLocation, GistConfig, GistResult, Linear,
    Metric, Points, Stage, Utility,
};
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyUntypedArrayMethods};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Raised for any `vectors` that is not exactly a C-contiguous `float32` matrix.
const VECTORS_TYPE_ERROR: &str = "vectors must be a C-contiguous float32 array of shape (n, d); \
     use np.ascontiguousarray(x, dtype=np.float32)";

/// Raised for a linear `utilities` that is not a C-contiguous `float64` vector.
const UTILITIES_TYPE_ERROR: &str = "utilities must be a C-contiguous float64 array of shape (n,); \
     use np.ascontiguousarray(u, dtype=np.float64)";

/// Raised for a coverage `utilities` that is not a sequence of int sequences.
const COVERAGE_TYPE_ERROR: &str = "utilities for utility='coverage' must be a sequence of \
     sequences of non-negative int item ids, one per vector";

/// The `utility=` choices, parsed from the keyword string.
#[derive(Clone, Copy)]
enum UtilityKind {
    Linear,
    Coverage,
    FacilityLocation,
}

/// Which `g` to build. Decided while attached to the interpreter -- it needs
/// the Python objects -- then carried into the detached solve as plain `Send`
/// data, since no `Bound` may cross into [`Python::detach`].
enum UtilitySpec {
    /// [`Linear`] with explicit weights, or [`Linear::uniform`] for `None`.
    Linear(Option<Vec<f64>>),
    /// [`Coverage`] over these item lists; the universe is inferred as the
    /// largest id plus one (`0` when every list is empty).
    Coverage(Vec<Vec<u32>>),
    /// [`FacilityLocation`], which needs the points themselves (and, under
    /// Euclidean distance, their `O(n^2)` diameter -- hence built detached).
    FacilityLocation,
}

impl UtilitySpec {
    /// Materialises the utility against the point set.
    fn build(self, pts: &Points<'_>) -> Result<Box<dyn Utility>, DivselError> {
        Ok(match self {
            Self::Linear(None) => Box::new(Linear::uniform(pts.n())),
            Self::Linear(Some(weights)) => Box::new(Linear::new(weights)),
            Self::Coverage(sets) => {
                let universe = sets
                    .iter()
                    .flatten()
                    .max()
                    .map_or(0, |&largest| largest as usize + 1);
                Box::new(Coverage::new(sets, universe)?)
            }
            Self::FacilityLocation => Box::new(FacilityLocation::new(pts)),
        })
    }
}

fn parse_metric(name: &str) -> PyResult<Metric> {
    match name {
        "cosine" => Ok(Metric::Cosine),
        "euclidean" => Ok(Metric::Euclidean),
        other => Err(PyValueError::new_err(format!(
            "metric must be 'cosine' or 'euclidean', got '{other}'"
        ))),
    }
}

fn parse_utility_kind(name: &str) -> PyResult<UtilityKind> {
    match name {
        "linear" => Ok(UtilityKind::Linear),
        "coverage" => Ok(UtilityKind::Coverage),
        "facility_location" => Ok(UtilityKind::FacilityLocation),
        other => Err(PyValueError::new_err(format!(
            "utility must be one of 'linear', 'coverage', 'facility_location', got '{other}'"
        ))),
    }
}

fn parse_diameter(name: &str, sweeps: usize) -> PyResult<DiameterMode> {
    match name {
        "exact" => Ok(DiameterMode::Exact),
        "approx" => Ok(DiameterMode::Approx { sweeps }),
        other => Err(PyValueError::new_err(format!(
            "diameter must be 'exact' or 'approx', got '{other}'"
        ))),
    }
}

fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::Greedy => "greedy",
        Stage::DiameterPair => "diameter_pair",
        Stage::Sweep => "sweep",
    }
}

/// The `utilities` argument for `utility="linear"`: `None`, or a C-contiguous
/// `float64` vector whose values are copied out (one `(n,)` copy; the length is
/// checked by [`gist`] against the point count).
fn linear_weights(utilities: Option<&Bound<'_, PyAny>>) -> PyResult<Option<Vec<f64>>> {
    let Some(obj) = utilities else {
        return Ok(None);
    };
    let array = obj
        .cast::<PyArray1<f64>>()
        .map_err(|_| PyTypeError::new_err(UTILITIES_TYPE_ERROR))?;
    if !array.is_c_contiguous() {
        return Err(PyTypeError::new_err(UTILITIES_TYPE_ERROR));
    }
    let guard = array.readonly();
    let weights = guard
        .as_slice()
        .map_err(|_| PyTypeError::new_err(UTILITIES_TYPE_ERROR))?;
    Ok(Some(weights.to_vec()))
}

/// The `utilities` argument for `utility="coverage"`: a sequence of sequences
/// of non-negative ints. Negative or oversized ids are a `ValueError` naming the
/// row; anything that is not such a nested sequence is a `TypeError`.
fn coverage_sets(utilities: Option<&Bound<'_, PyAny>>) -> PyResult<Vec<Vec<u32>>> {
    let Some(obj) = utilities else {
        return Err(PyValueError::new_err(
            "utility='coverage' requires utilities: a sequence of sequences of item ids, \
             one per vector",
        ));
    };
    let rows: Vec<Vec<i64>> = obj
        .extract()
        .map_err(|_| PyTypeError::new_err(COVERAGE_TYPE_ERROR))?;
    let mut sets = Vec::with_capacity(rows.len());
    for (row, items) in rows.into_iter().enumerate() {
        let mut set = Vec::with_capacity(items.len());
        for item in items {
            let id = u32::try_from(item).map_err(|_| {
                PyValueError::new_err(format!(
                    "coverage item id {item} at row {row} must be a non-negative int no larger \
                     than {}",
                    u32::MAX
                ))
            })?;
            set.push(id);
        }
        sets.push(set);
    }
    Ok(sets)
}

/// The shared body of both entry points: validate and cast the Python
/// arguments while attached, then run [`gist`] detached from the interpreter.
#[allow(clippy::too_many_arguments)]
fn solve(
    py: Python<'_>,
    vectors: &Bound<'_, PyAny>,
    utilities: Option<&Bound<'_, PyAny>>,
    k: i64,
    lam: f64,
    eps: f32,
    metric: &str,
    utility: &str,
    exhaustive_thresholds: bool,
    diameter: &str,
    diameter_sweeps: i64,
) -> PyResult<GistResult> {
    // Both budgets arrive as `i64` so that a negative Python int is a
    // `ValueError` here rather than pyo3's `OverflowError` from a `usize`
    // extraction. The core rejects `k == 0` too; folding it in keeps one message.
    let k = usize::try_from(k)
        .ok()
        .filter(|&k| k > 0)
        .ok_or_else(|| PyValueError::new_err("k must be greater than zero"))?;
    let diameter_sweeps = usize::try_from(diameter_sweeps)
        .map_err(|_| PyValueError::new_err("diameter_sweeps must be non-negative"))?;
    let metric = parse_metric(metric)?;
    let kind = parse_utility_kind(utility)?;
    let diameter = parse_diameter(diameter, diameter_sweeps)?;
    let cfg = GistConfig {
        k,
        lambda: lam,
        eps,
        exhaustive_thresholds,
        diameter,
    };

    // `cast` already rejects a wrong dtype or ndim; the C-order check is extra
    // because `as_slice` would happily hand back a Fortran-ordered buffer.
    let array = vectors
        .cast::<PyArray2<f32>>()
        .map_err(|_| PyTypeError::new_err(VECTORS_TYPE_ERROR))?;
    if !array.is_c_contiguous() {
        return Err(PyTypeError::new_err(VECTORS_TYPE_ERROR));
    }
    // A borrow-checked read lock on the numpy buffer. It lives to the end of
    // this function, i.e. across the detached solve that borrows `data`.
    let guard = array.readonly();
    let data: &[f32] = guard
        .as_slice()
        .map_err(|_| PyTypeError::new_err(VECTORS_TYPE_ERROR))?;
    let dim = guard.shape()[1];

    let spec = match kind {
        UtilityKind::Linear => UtilitySpec::Linear(linear_weights(utilities)?),
        UtilityKind::Coverage => UtilitySpec::Coverage(coverage_sets(utilities)?),
        UtilityKind::FacilityLocation => {
            if utilities.is_some() {
                return Err(PyValueError::new_err(
                    "facility_location takes no utilities array",
                ));
            }
            UtilitySpec::FacilityLocation
        }
    };

    // Only plain `Send` data crosses into the closure: the borrowed `&[f32]`,
    // the config and the utility spec. `Points::borrowed` keeps the Euclidean
    // buffer in place and copies once for cosine.
    py.detach(move || {
        let pts = Points::borrowed(data, dim, metric)?;
        let mut util = spec.build(&pts)?;
        gist(&pts, util.as_mut(), &cfg)
    })
    .map_err(|err| PyValueError::new_err(err.to_string()))
}

/// Select up to `k` rows of `vectors` with GIST (arXiv:2405.18754v3, NeurIPS 2025).
///
/// GIST maximizes `f(S) = g(S) + lam * div(S)` over `|S| <= k`, where `g` is a
/// monotone submodular utility (`"linear"`: a weighted count, `"coverage"`: the
/// number of distinct items covered, `"facility_location"`: the paper's own
/// choice) and `div(S)` is the minimum pairwise distance inside `S`, and it
/// guarantees `f(S) >= (1/2 - eps) * OPT` for any such `g` (Theorem 3.1) and
/// `(2/3 - eps) * OPT` for a linear `g` (Theorem 3.3). Those bounds are proven
/// for a true metric, so they hold for `metric="euclidean"`; raw cosine
/// distance `1 - cos` -- the paper's own experimental setting and the default
/// here -- is used as a well-behaved heuristic. `vectors` is consumed zero-copy
/// for `metric="euclidean"` (the numpy buffer is borrowed in place under a read
/// lock held for the whole call); `metric="cosine"` makes exactly one
/// L2-normalised copy. The caller's array is never modified, and the solve runs
/// with the interpreter lock released.
///
/// Returns the selected row indices, in selection order, at most `k` of them.
/// See `gist_select_full` for the objective value and the other diagnostics.
#[pyfunction]
#[pyo3(signature = (vectors, utilities=None, *, k, lam=1.0, eps=0.1, metric="cosine",
    utility="linear", exhaustive_thresholds=false, diameter="exact", diameter_sweeps=3))]
#[allow(clippy::too_many_arguments)]
fn gist_select(
    py: Python<'_>,
    vectors: &Bound<'_, PyAny>,
    utilities: Option<&Bound<'_, PyAny>>,
    k: i64,
    lam: f64,
    eps: f32,
    metric: &str,
    utility: &str,
    exhaustive_thresholds: bool,
    diameter: &str,
    diameter_sweeps: i64,
) -> PyResult<Vec<usize>> {
    Ok(solve(
        py,
        vectors,
        utilities,
        k,
        lam,
        eps,
        metric,
        utility,
        exhaustive_thresholds,
        diameter,
        diameter_sweeps,
    )?
    .selected)
}

/// `gist_select` with the full GIST result: a dict with keys `selected` (the
/// same list `gist_select` returns), `f_value` (`g(S) + lam * div(S)`, the value
/// maximized), `g_value`, `div` (the minimum pairwise distance in `S`, or
/// `d_max` when `|S| <= 1`), `threshold` (the distance threshold that produced
/// `selected`: `0.0` for `"greedy"`, `d_max` for `"diameter_pair"`, the winning
/// `d` for `"sweep"`), `stage` (which branch of Algorithm 1 won: `"greedy"`,
/// `"diameter_pair"` or `"sweep"`) and `d_max` (the diameter the sweep was built
/// from -- exact under `diameter="exact"`, the double-sweep estimate `d_hat` in
/// `[d_max/2, d_max]` under `diameter="approx"`). Same objective, guarantees,
/// metric caveat and zero-copy behaviour as `gist_select`.
#[pyfunction]
#[pyo3(signature = (vectors, utilities=None, *, k, lam=1.0, eps=0.1, metric="cosine",
    utility="linear", exhaustive_thresholds=false, diameter="exact", diameter_sweeps=3))]
#[allow(clippy::too_many_arguments)]
fn gist_select_full<'py>(
    py: Python<'py>,
    vectors: &Bound<'py, PyAny>,
    utilities: Option<&Bound<'py, PyAny>>,
    k: i64,
    lam: f64,
    eps: f32,
    metric: &str,
    utility: &str,
    exhaustive_thresholds: bool,
    diameter: &str,
    diameter_sweeps: i64,
) -> PyResult<Bound<'py, PyDict>> {
    let result = solve(
        py,
        vectors,
        utilities,
        k,
        lam,
        eps,
        metric,
        utility,
        exhaustive_thresholds,
        diameter,
        diameter_sweeps,
    )?;
    let out = PyDict::new(py);
    out.set_item("selected", result.selected)?;
    out.set_item("f_value", result.f_value)?;
    out.set_item("g_value", result.g_value)?;
    out.set_item("div", result.div)?;
    out.set_item("threshold", result.threshold)?;
    out.set_item("stage", stage_name(result.stage))?;
    out.set_item("d_max", result.d_max)?;
    Ok(out)
}

/// The native module `divsel._divsel`: `gist_select`, `gist_select_full` and
/// `__version__`. `python/divsel/__init__.py` re-exports all three.
#[pymodule]
fn _divsel(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(gist_select, m)?)?;
    m.add_function(wrap_pyfunction!(gist_select_full, m)?)?;
    Ok(())
}
