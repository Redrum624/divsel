//! The error type returned by fallible `divsel` operations.

use std::fmt;

/// Errors produced when an input violates a `divsel` precondition.
///
/// The enum is `#[non_exhaustive]`: matching on it must include a wildcard arm so
/// that future variants do not break downstream code.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DivselError {
    /// The supplied point matrix contained no values at all.
    EmptyInput,
    /// The supplied dimensionality was zero.
    ZeroDim,
    /// The flat, row-major buffer length is not an exact multiple of `dim`.
    LengthNotMultipleOfDim {
        /// Length of the flat buffer.
        len: usize,
        /// Requested row dimensionality.
        dim: usize,
    },
    /// A coordinate was `NaN` or infinite.
    NonFinite {
        /// Zero-based row index of the offending coordinate.
        row: usize,
        /// Zero-based column index of the offending coordinate.
        col: usize,
    },
    /// A row had a zero or un-normalizable L2 norm, so it cannot be scaled to
    /// unit length for cosine distance.
    ///
    /// Covers an all-zero row, a row whose sum of squares overflows `f32` to
    /// infinity, and a row so small that every square underflows to zero.
    ZeroNormRow {
        /// Zero-based index of the offending row.
        row: usize,
    },
    /// A supplied weight vector did not have one entry per point.
    WeightsLength {
        /// Number of points, i.e. the required length.
        expected: usize,
        /// Length actually supplied.
        got: usize,
    },
    /// A supplied weight was negative, `NaN`, or infinite.
    ///
    /// GIST's utility `g` must be monotone, which for a linear `g` means every
    /// weight is finite and non-negative.
    InvalidWeight {
        /// Zero-based index of the offending weight.
        index: usize,
        /// The offending weight.
        value: f64,
    },
    /// A supplied coverage list did not have one entry per point.
    CoverageLength {
        /// Number of points, i.e. the required length.
        expected: usize,
        /// Length actually supplied.
        got: usize,
    },
    /// A coverage entry referenced an item outside the coverage universe.
    CoverageItemOutOfRange {
        /// Zero-based row index holding the offending item.
        row: usize,
        /// The offending item identifier.
        item: u32,
        /// Number of items in the coverage universe.
        universe: usize,
    },
    /// The requested subset size `k` was zero. (A `k` above the number of
    /// points is clamped, not rejected.)
    InvalidK,
    /// The requested epsilon was outside `f32::EPSILON <= eps <= 1` (which also
    /// rejects `NaN` and the infinities). The lower end is not cosmetic: the
    /// threshold grid is built by repeated multiplication by `1 + eps` and its
    /// entries are `f32`, so below `f32::EPSILON` consecutive entries cannot
    /// differ and `|D|` runs away toward the count of representable `f32`s in
    /// the range -- for an `eps` below `2^-53` the multiplication does not
    /// advance at all.
    InvalidEps(f32),
    /// The requested lambda was out of range.
    InvalidLambda(f64),
}

impl fmt::Display for DivselError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyInput => write!(f, "the point matrix is empty"),
            Self::ZeroDim => write!(f, "the dimensionality must be greater than zero"),
            Self::LengthNotMultipleOfDim { len, dim } => write!(
                f,
                "the data buffer holds {len} values, which is not a multiple of the dimensionality {dim}"
            ),
            Self::NonFinite { row, col } => write!(
                f,
                "the coordinate at row {row}, column {col} is NaN or infinite"
            ),
            Self::ZeroNormRow { row } => write!(
                f,
                "row {row} has a zero or un-normalizable L2 norm and cannot be scaled to unit length for cosine distance"
            ),
            Self::WeightsLength { expected, got } => write!(
                f,
                "expected {expected} weights, one per point, but got {got}"
            ),
            Self::InvalidWeight { index, value } => write!(
                f,
                "the weight {value} at index {index} must be finite and non-negative"
            ),
            Self::CoverageLength { expected, got } => write!(
                f,
                "expected {expected} coverage lists, one per point, but got {got}"
            ),
            Self::CoverageItemOutOfRange {
                row,
                item,
                universe,
            } => write!(
                f,
                "coverage item {item} at row {row} is outside the universe of {universe} items"
            ),
            Self::InvalidK => write!(f, "k must be greater than zero"),
            Self::InvalidEps(eps) => write!(
                f,
                "epsilon {eps:e} must be in the range f32::EPSILON (1.1920929e-7) <= eps <= 1"
            ),
            Self::InvalidLambda(lambda) => {
                write!(f, "lambda {lambda} must be finite and non-negative")
            }
        }
    }
}

impl std::error::Error for DivselError {}

#[cfg(test)]
mod tests {
    use super::DivselError;
    use std::error::Error;

    #[test]
    fn display_names_the_offending_values() {
        let cases: [(DivselError, &[&str]); 12] = [
            (DivselError::EmptyInput, &["empty"]),
            (DivselError::ZeroDim, &["dimensionality"]),
            (
                DivselError::LengthNotMultipleOfDim { len: 7, dim: 3 },
                &["7", "3"],
            ),
            (DivselError::NonFinite { row: 2, col: 5 }, &["2", "5"]),
            (DivselError::ZeroNormRow { row: 4 }, &["4"]),
            (
                DivselError::WeightsLength {
                    expected: 8,
                    got: 3,
                },
                &["8", "3"],
            ),
            (
                DivselError::InvalidWeight {
                    index: 4,
                    value: -1.5,
                },
                &["4", "-1.5"],
            ),
            (
                DivselError::CoverageLength {
                    expected: 8,
                    got: 3,
                },
                &["8", "3"],
            ),
            (
                DivselError::CoverageItemOutOfRange {
                    row: 1,
                    item: 99,
                    universe: 10,
                },
                &["1", "99", "10"],
            ),
            (DivselError::InvalidK, &["k"]),
            (DivselError::InvalidEps(1.5), &["1.5"]),
            (DivselError::InvalidLambda(-2.0), &["-2"]),
        ];
        for (err, needles) in cases {
            let rendered = err.to_string();
            assert!(!rendered.is_empty(), "{err:?} rendered empty");
            for needle in needles {
                assert!(
                    rendered.contains(needle),
                    "{err:?} rendered as {rendered:?}, missing {needle:?}"
                );
            }
        }
    }

    #[test]
    fn is_a_std_error() {
        let err = DivselError::ZeroDim;
        let dynamic: &dyn Error = &err;
        assert!(dynamic.source().is_none());
        assert_eq!(err.clone(), err);
    }
}
