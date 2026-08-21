//! Row-major storage for the point set, plus its distance and diameter queries.

use std::borrow::Cow;
use std::cmp::Ordering;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::error::DivselError;
use crate::metric::{dot, sq_euclid, Metric};

/// The identity element of the diameter reduction: it loses against every real
/// candidate under [`better_pair`].
const NO_PAIR: (f32, usize, usize) = (f32::NEG_INFINITY, usize::MAX, usize::MAX);

/// Picks the better of two `(distance, i, j)` candidates under a total order:
/// larger distance wins; on a tie the smaller `i` wins; on a further tie the
/// smaller `j` wins.
///
/// Being a total order, the winner is the same however `rayon` splits the work.
fn better_pair(a: (f32, usize, usize), b: (f32, usize, usize)) -> (f32, usize, usize) {
    match a.0.total_cmp(&b.0) {
        Ordering::Greater => a,
        Ordering::Less => b,
        Ordering::Equal => match a.1.cmp(&b.1) {
            Ordering::Less => a,
            Ordering::Greater => b,
            Ordering::Equal => {
                if a.2 <= b.2 {
                    a
                } else {
                    b
                }
            }
        },
    }
}

/// Shared shape and finiteness validation for both constructors, in the order
/// `ZeroDim`, `EmptyInput`, `LengthNotMultipleOfDim`, `NonFinite`. Returns `n`.
fn validate(data: &[f32], dim: usize) -> Result<usize, DivselError> {
    if dim == 0 {
        return Err(DivselError::ZeroDim);
    }
    if data.is_empty() {
        return Err(DivselError::EmptyInput);
    }
    if data.len() % dim != 0 {
        return Err(DivselError::LengthNotMultipleOfDim {
            len: data.len(),
            dim,
        });
    }
    for (index, value) in data.iter().enumerate() {
        if !value.is_finite() {
            return Err(DivselError::NonFinite {
                row: index / dim,
                col: index % dim,
            });
        }
    }
    Ok(data.len() / dim)
}

/// L2-normalizes each row in place, dividing by the norm rather than multiplying
/// by its reciprocal so that each coordinate stays correctly rounded.
///
/// The norm is derived from a sum of squares accumulated in `f32`, so it is
/// rejected as un-normalizable when that sum overflows to infinity (a row of very
/// large coordinates) as well as when it is zero -- either because the row really
/// is all zeros or because every square underflowed. Dividing by an infinite norm
/// would otherwise fail open, silently yielding an all-zero, non-unit row.
fn normalize_rows(data: &mut [f32], dim: usize) -> Result<(), DivselError> {
    for (row, values) in data.chunks_exact_mut(dim).enumerate() {
        let norm = dot(values, values).sqrt();
        if norm == 0.0 || !norm.is_finite() {
            return Err(DivselError::ZeroNormRow { row });
        }
        for value in values.iter_mut() {
            *value /= norm;
        }
    }
    Ok(())
}

/// An immutable, row-major matrix of `n` points in `dim` dimensions, tagged with
/// the [`Metric`] used to compare them.
///
/// `Points` either owns its buffer or borrows one from the caller. Construct it
/// with [`Points::new`] to hand over a `Vec<f32>`, or [`Points::borrowed`] to
/// point at an existing slice. Borrowing is zero-copy for [`Metric::Euclidean`];
/// [`Metric::Cosine`] always L2-normalizes into an owned copy, so that
/// [`Points::dist`] is a plain `1 - dot`.
#[derive(Clone, Debug)]
pub struct Points<'a> {
    data: Cow<'a, [f32]>,
    n: usize,
    dim: usize,
    metric: Metric,
}

impl<'a> Points<'a> {
    /// Takes ownership of a row-major `data` buffer of `n * dim` values.
    ///
    /// Under [`Metric::Cosine`] every row is L2-normalized in place.
    ///
    /// # Normalizable range
    ///
    /// Row norms are derived from a sum of squares accumulated in `f32`, so a row
    /// is normalized exactly when its L2 norm lands in roughly
    /// `[1.0842022e-19, 1.8446743e19]` -- that is, `sqrt(f32::MIN_POSITIVE)` up to
    /// `sqrt(f32::MAX)`. Above the top of that band the sum of squares overflows,
    /// and far below the bottom every square underflows to zero; both are rejected
    /// with [`DivselError::ZeroNormRow`] rather than silently producing a non-unit
    /// row. Note that the second case rejects a row that is not literally zero.
    ///
    /// One narrow gap is *not* rejected: when the sum of squares is subnormal but
    /// still nonzero (norms just under `1.0842022e-19`), it keeps too few
    /// significant bits and the row normalizes to a length near, but not equal to,
    /// `1.0`. For example `[3e-23, 4e-23]` becomes `[0.566684, 0.75557864]`, of
    /// length `0.9444733` rather than `[0.6, 0.8]`. Rescale such embeddings before
    /// handing them over.
    ///
    /// # Errors
    ///
    /// Returns [`DivselError::ZeroDim`] if `dim == 0`, [`DivselError::EmptyInput`]
    /// if `data` is empty, [`DivselError::LengthNotMultipleOfDim`] if `data.len()`
    /// is not a multiple of `dim`, [`DivselError::NonFinite`] for the first `NaN`
    /// or infinite coordinate in row-major order, and, under [`Metric::Cosine`],
    /// [`DivselError::ZeroNormRow`] for the first row that cannot be scaled to unit
    /// length, as described above.
    pub fn new(data: Vec<f32>, dim: usize, metric: Metric) -> Result<Points<'static>, DivselError> {
        let n = validate(&data, dim)?;
        let mut data = data;
        if metric == Metric::Cosine {
            normalize_rows(&mut data, dim)?;
        }
        Ok(Points {
            data: Cow::Owned(data),
            n,
            dim,
            metric,
        })
    }

    /// Borrows a row-major `data` slice of `n * dim` values.
    ///
    /// Under [`Metric::Euclidean`] the slice is used in place with no copy. Under
    /// [`Metric::Cosine`] an L2-normalized copy is made, since normalization
    /// requires mutation. The same normalizable range documented on
    /// [`Points::new`] applies.
    ///
    /// # Errors
    ///
    /// Identical to [`Points::new`].
    pub fn borrowed(
        data: &'a [f32],
        dim: usize,
        metric: Metric,
    ) -> Result<Points<'a>, DivselError> {
        let n = validate(data, dim)?;
        let data = match metric {
            Metric::Cosine => {
                let mut owned = data.to_vec();
                normalize_rows(&mut owned, dim)?;
                Cow::Owned(owned)
            }
            Metric::Euclidean => Cow::Borrowed(data),
        };
        Ok(Points {
            data,
            n,
            dim,
            metric,
        })
    }

    /// Number of points.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Dimensionality of each point.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// The metric these points are compared with.
    pub fn metric(&self) -> Metric {
        self.metric
    }

    /// The `i`-th row, as a slice of `dim` values.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.n()`.
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    /// Distance between points `i` and `j` under this set's [`Metric`].
    ///
    /// `dist(i, i)` is exactly `0.0`, short-circuited before any arithmetic.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.n()` or `j >= self.n()`.
    pub fn dist(&self, i: usize, j: usize) -> f32 {
        if i == j {
            return 0.0;
        }
        let a = self.row(i);
        let b = self.row(j);
        match self.metric {
            Metric::Cosine => (1.0 - dot(a, b)).clamp(0.0, 2.0),
            Metric::Euclidean => sq_euclid(a, b).sqrt(),
        }
    }

    /// The exact diameter of the point set: `(d_max, u, v)`, where `u < v` are the
    /// two points that realize it. A single-point set yields `(0.0, 0, 0)`.
    ///
    /// Runs the full `O(n^2)` pair scan in parallel over rows. The winning pair is
    /// chosen by a total order -- larger distance wins; on a tie, the smaller `u`
    /// wins; on a further tie, the smaller `v` -- so the result does not depend on
    /// how `rayon` happens to split the work.
    pub fn diameter(&self) -> (f32, usize, usize) {
        if self.n < 2 {
            return (0.0, 0, 0);
        }
        (0..self.n)
            .into_par_iter()
            .map(|i| {
                let mut best = NO_PAIR;
                for j in (i + 1)..self.n {
                    best = better_pair(best, (self.dist(i, j), i, j));
                }
                best
            })
            .reduce(|| NO_PAIR, better_pair)
    }

    /// Whether the backing buffer is borrowed rather than owned.
    #[cfg(test)]
    pub(crate) fn is_borrowed(&self) -> bool {
        matches!(self.data, Cow::Borrowed(_))
    }
}

#[cfg(test)]
mod tests {
    use super::Points;
    use crate::error::DivselError;
    use crate::metric::Metric;

    const TOL: f32 = 1e-6;

    #[track_caller]
    fn assert_close(got: f32, want: f32) {
        assert!(
            (got - want).abs() < TOL,
            "got {got}, want {want} (tolerance {TOL})"
        );
    }

    fn sample(n: usize, dim: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        let mut out = Vec::with_capacity(n * dim);
        for _ in 0..n * dim {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = f32::from(((state >> 16) & 0xffff) as u16) / 65_536.0;
            out.push(unit - 0.5);
        }
        out
    }

    // ---- validation -------------------------------------------------------

    #[test]
    fn zero_dim_is_rejected_before_anything_else() {
        assert_eq!(
            Points::new(Vec::new(), 0, Metric::Euclidean).unwrap_err(),
            DivselError::ZeroDim
        );
        assert_eq!(
            Points::borrowed(&[], 0, Metric::Euclidean).unwrap_err(),
            DivselError::ZeroDim
        );
    }

    #[test]
    fn empty_input_is_rejected() {
        assert_eq!(
            Points::new(Vec::new(), 3, Metric::Euclidean).unwrap_err(),
            DivselError::EmptyInput
        );
        assert_eq!(
            Points::borrowed(&[], 3, Metric::Cosine).unwrap_err(),
            DivselError::EmptyInput
        );
    }

    #[test]
    fn length_not_a_multiple_of_dim_is_rejected() {
        assert_eq!(
            Points::new(vec![1.0; 7], 3, Metric::Euclidean).unwrap_err(),
            DivselError::LengthNotMultipleOfDim { len: 7, dim: 3 }
        );
    }

    #[test]
    fn non_finite_reports_the_first_offender_in_row_major_order() {
        let mut data = vec![1.0f32; 12];
        data[7] = f32::NAN;
        data[9] = f32::INFINITY;
        assert_eq!(
            Points::new(data, 4, Metric::Euclidean).unwrap_err(),
            DivselError::NonFinite { row: 1, col: 3 }
        );

        let mut data = vec![1.0f32; 12];
        data[6] = f32::NEG_INFINITY;
        assert_eq!(
            Points::borrowed(&data, 3, Metric::Cosine).unwrap_err(),
            DivselError::NonFinite { row: 2, col: 0 }
        );
    }

    #[test]
    fn a_zero_norm_row_is_rejected_under_cosine_only() {
        let data = vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0];
        assert_eq!(
            Points::new(data.clone(), 2, Metric::Cosine).unwrap_err(),
            DivselError::ZeroNormRow { row: 1 }
        );
        let pts = Points::new(data, 2, Metric::Euclidean).expect("euclidean allows zero rows");
        assert_eq!(pts.n(), 3);
    }

    #[test]
    fn a_row_whose_sum_of_squares_overflows_is_rejected_under_cosine() {
        // Every coordinate is finite, so validation passes, but row 1's sum of
        // squares overflows f32 to infinity. Dividing by an infinite norm used to
        // fail open and silently yield an all-zero, non-unit row.
        let data = vec![1.0, 0.0, 1e20, 2e20];
        assert_eq!(
            Points::new(data.clone(), 2, Metric::Cosine).unwrap_err(),
            DivselError::ZeroNormRow { row: 1 }
        );
        assert_eq!(
            Points::borrowed(&data, 2, Metric::Cosine).unwrap_err(),
            DivselError::ZeroNormRow { row: 1 }
        );
        // Euclidean never normalizes, so it accepts the very same buffer.
        assert!(Points::new(data, 2, Metric::Euclidean).is_ok());
    }

    #[test]
    fn a_row_whose_sum_of_squares_underflows_is_rejected_under_cosine() {
        // Pins the documented behaviour at the other end of the band: this row is
        // not literally zero, but every square underflows to zero in f32, so it
        // cannot be scaled to unit length and is reported as a zero-norm row.
        let data = vec![1.0, 0.0, 1e-25, 1e-25];
        assert_eq!(
            Points::new(data, 2, Metric::Cosine).unwrap_err(),
            DivselError::ZeroNormRow { row: 1 }
        );
    }

    // ---- storage ----------------------------------------------------------

    #[test]
    fn shape_accessors_report_the_constructor_arguments() {
        let pts = Points::new(vec![0.0; 12], 4, Metric::Euclidean).unwrap();
        assert_eq!(pts.n(), 3);
        assert_eq!(pts.dim(), 4);
        assert_eq!(pts.metric(), Metric::Euclidean);
    }

    #[test]
    fn row_returns_the_requested_slice() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let pts = Points::new(data, 3, Metric::Euclidean).unwrap();
        assert_eq!(pts.row(0), &[1.0, 2.0, 3.0]);
        assert_eq!(pts.row(1), &[4.0, 5.0, 6.0]);
    }

    #[test]
    fn borrowed_euclidean_is_zero_copy() {
        let data = [1.0f32, 2.0, 3.0, 4.0];
        let pts = Points::borrowed(&data, 2, Metric::Euclidean).unwrap();
        assert!(pts.is_borrowed());
        assert_eq!(pts.row(1), &[3.0, 4.0]);
    }

    #[test]
    fn borrowed_cosine_owns_a_normalized_copy() {
        let data = [3.0f32, 4.0];
        let pts = Points::borrowed(&data, 2, Metric::Cosine).unwrap();
        assert!(!pts.is_borrowed());
        assert_close(pts.row(0)[0], 0.6);
        assert_close(pts.row(0)[1], 0.8);
        // The caller's own buffer is left untouched.
        assert_eq!(data, [3.0, 4.0]);
    }

    #[test]
    fn new_cosine_normalizes_every_row() {
        let pts = Points::new(vec![3.0, 4.0, 0.0, -2.0], 2, Metric::Cosine).unwrap();
        for i in 0..pts.n() {
            let norm: f32 = pts.row(i).iter().map(|x| x * x).sum::<f32>().sqrt();
            assert_close(norm, 1.0);
        }
        assert_close(pts.row(1)[1], -1.0);
    }

    #[test]
    fn new_euclidean_leaves_the_data_untouched() {
        let pts = Points::new(vec![3.0, 4.0], 2, Metric::Euclidean).unwrap();
        assert!(!pts.is_borrowed());
        assert_eq!(pts.row(0), &[3.0, 4.0]);
    }

    // ---- distances --------------------------------------------------------

    #[test]
    fn dist_to_self_is_exactly_zero() {
        let euclid = Points::new(vec![3.0, 4.0, 1.0, 1.0], 2, Metric::Euclidean).unwrap();
        assert_eq!(euclid.dist(1, 1), 0.0);
        let cosine = Points::new(vec![3.0, 4.0, 1.0, 1.0], 2, Metric::Cosine).unwrap();
        assert_eq!(cosine.dist(0, 0), 0.0);
    }

    #[test]
    fn cosine_distance_between_orthogonal_rows_is_one() {
        let pts = Points::new(vec![5.0, 0.0, 0.0, 7.0], 2, Metric::Cosine).unwrap();
        assert_close(pts.dist(0, 1), 1.0);
        assert_close(pts.dist(1, 0), 1.0);
    }

    #[test]
    fn cosine_distance_between_identical_directions_is_zero() {
        let pts = Points::new(vec![1.0, 2.0, 3.0, 2.0, 4.0, 6.0], 3, Metric::Cosine).unwrap();
        assert_close(pts.dist(0, 1), 0.0);
    }

    #[test]
    fn cosine_distance_between_opposite_directions_is_two() {
        let pts = Points::new(vec![1.0, 0.0, -4.0, 0.0], 2, Metric::Cosine).unwrap();
        assert_close(pts.dist(0, 1), 2.0);
    }

    #[test]
    fn euclidean_distance_over_a_three_four_five_triangle() {
        let pts = Points::new(vec![0.0, 0.0, 3.0, 0.0, 0.0, 4.0], 2, Metric::Euclidean).unwrap();
        assert_close(pts.dist(0, 1), 3.0);
        assert_close(pts.dist(0, 2), 4.0);
        assert_close(pts.dist(1, 2), 5.0);
        assert_close(pts.dist(2, 1), 5.0);
    }

    // ---- diameter ---------------------------------------------------------

    #[test]
    fn diameter_of_a_single_point_is_zero() {
        let pts = Points::new(vec![1.0, 2.0, 3.0], 3, Metric::Euclidean).unwrap();
        assert_eq!(pts.n(), 1);
        assert_eq!(pts.diameter(), (0.0, 0, 0));
    }

    #[test]
    fn diameter_of_a_known_four_point_set() {
        // (0,0) (3,4) (-6,-8) (1,1): the unique widest pair is 1..2, at distance 15.
        let pts = Points::new(
            vec![0.0, 0.0, 3.0, 4.0, -6.0, -8.0, 1.0, 1.0],
            2,
            Metric::Euclidean,
        )
        .unwrap();
        let (d, u, v) = pts.diameter();
        assert_close(d, 15.0);
        assert_eq!((u, v), (1, 2));
    }

    #[test]
    fn diameter_tie_returns_the_lexicographically_smallest_pair() {
        // Unit square: the two diagonals (0,3) and (1,2) tie at sqrt(2).
        let pts = Points::new(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
            Metric::Euclidean,
        )
        .unwrap();
        assert_eq!(pts.dist(0, 3).to_bits(), pts.dist(1, 2).to_bits());
        let (d, u, v) = pts.diameter();
        assert_close(d, std::f32::consts::SQRT_2);
        assert_eq!((u, v), (0, 3));
        // And the same answer however often it is asked.
        for _ in 0..8 {
            assert_eq!(pts.diameter(), (d, u, v));
        }
    }

    #[test]
    fn diameter_is_independent_of_the_rayon_split() {
        let pts = Points::new(sample(257, 8, 0x5eed_1234), 8, Metric::Euclidean).unwrap();
        let baseline = pts.diameter();
        for threads in [1usize, 2, 3, 7] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .expect("thread pool");
            let got = pool.install(|| pts.diameter());
            assert_eq!(
                (got.0.to_bits(), got.1, got.2),
                (baseline.0.to_bits(), baseline.1, baseline.2),
                "diameter differed with {threads} threads"
            );
        }
    }

    #[test]
    fn diameter_agrees_with_a_serial_scan() {
        let pts = Points::new(sample(64, 5, 0x0f0f_0f0f), 5, Metric::Cosine).unwrap();
        let mut want = (f32::NEG_INFINITY, usize::MAX, usize::MAX);
        for i in 0..pts.n() {
            for j in (i + 1)..pts.n() {
                let d = pts.dist(i, j);
                if d > want.0 {
                    want = (d, i, j);
                }
            }
        }
        let got = pts.diameter();
        assert_eq!(got.0.to_bits(), want.0.to_bits());
        assert_eq!((got.1, got.2), (want.1, want.2));
    }
}
