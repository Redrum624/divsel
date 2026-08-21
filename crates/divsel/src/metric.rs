//! The distance metric selector and the fixed-order scalar kernels behind it.

/// The distance function used to compare two points.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Metric {
    /// Cosine distance `1 - a.b`, evaluated on L2-normalized rows.
    ///
    /// This is the metric the GIST paper's ImageNet experiment uses, and
    /// [`Points`](crate::Points) L2-normalizes every row on construction so that
    /// the dot product is a true cosine.
    ///
    /// # Triangle inequality
    ///
    /// On unit vectors, `1 - a.b` satisfies the triangle inequality only in its
    /// square-root (angular) form, `sqrt(2 * (1 - a.b))`. GIST's approximation
    /// guarantee has a metric precondition, so that guarantee holds strictly for
    /// [`Metric::Euclidean`] and for angular distance; for raw cosine distance
    /// `divsel` uses it as a well-behaved heuristic, which is exactly what the
    /// paper's own experiments do.
    Cosine,
    /// Standard Euclidean (L2) distance, `sqrt(sum((a_i - b_i)^2))`.
    ///
    /// A true metric, so GIST's approximation guarantee applies without caveat.
    Euclidean,
}

/// Number of independent accumulator lanes used by the scalar kernels.
///
/// The lane count is part of the kernels' numeric contract: a future SIMD
/// implementation must reduce in this same order to stay bit-identical.
pub(crate) const LANES: usize = 16;

/// Dot product of two equal-length slices.
///
/// The summation order is a hard contract, not an implementation detail:
/// [`LANES`] independent accumulators are advanced in lockstep over
/// `chunks_exact(LANES)`, the tail folds into lane `index % LANES`, and the
/// accumulators are summed in lane order at the end. A SIMD rewrite that reduces
/// this way is bit-identical to this scalar version. Do not replace the explicit
/// loops with `sum()` or `fold`, and do not fuse the multiply and the add: an
/// `mul_add` would change the rounding.
///
/// # Panics
///
/// Panics in debug builds if `a` and `b` have different lengths.
// The two lints below would rewrite these loops into iterator adaptors, which
// changes the reduction order that a SIMD implementation has to match exactly.
#[allow(clippy::assign_op_pattern, clippy::needless_range_loop)]
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot operands must have equal lengths");

    let mut acc = [0.0f32; LANES];
    let mut chunks_a = a.chunks_exact(LANES);
    let mut chunks_b = b.chunks_exact(LANES);
    for (ca, cb) in (&mut chunks_a).zip(&mut chunks_b) {
        for l in 0..LANES {
            let p = ca[l] * cb[l];
            acc[l] = acc[l] + p;
        }
    }
    for (l, (x, y)) in chunks_a
        .remainder()
        .iter()
        .zip(chunks_b.remainder())
        .enumerate()
    {
        let p = x * y;
        acc[l] = acc[l] + p;
    }
    let mut total = 0.0f32;
    for l in 0..LANES {
        total = total + acc[l];
    }
    total
}

/// Squared Euclidean distance between two equal-length slices.
///
/// Shares the fixed [`LANES`]-accumulator reduction order documented on [`dot`],
/// with `(a[i] - b[i])^2` in place of `a[i] * b[i]`.
///
/// # Panics
///
/// Panics in debug builds if `a` and `b` have different lengths.
// The two lints below would rewrite these loops into iterator adaptors, which
// changes the reduction order that a SIMD implementation has to match exactly.
#[allow(clippy::assign_op_pattern, clippy::needless_range_loop)]
pub(crate) fn sq_euclid(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "sq_euclid operands must have equal lengths"
    );

    let mut acc = [0.0f32; LANES];
    let mut chunks_a = a.chunks_exact(LANES);
    let mut chunks_b = b.chunks_exact(LANES);
    for (ca, cb) in (&mut chunks_a).zip(&mut chunks_b) {
        for l in 0..LANES {
            let d = ca[l] - cb[l];
            let p = d * d;
            acc[l] = acc[l] + p;
        }
    }
    for (l, (x, y)) in chunks_a
        .remainder()
        .iter()
        .zip(chunks_b.remainder())
        .enumerate()
    {
        let d = x - y;
        let p = d * d;
        acc[l] = acc[l] + p;
    }
    let mut total = 0.0f32;
    for l in 0..LANES {
        total = total + acc[l];
    }
    total
}

#[cfg(test)]
mod tests {
    use super::{dot, sq_euclid, LANES};

    /// Deterministic sample data. Signs depend only on the index, so `dot` of two
    /// such vectors is a sum of strictly positive products and stays far from zero,
    /// which keeps the relative-error assertion meaningful.
    fn sample(len: usize, seed: u32) -> Vec<f32> {
        let mut state = seed;
        let mut out = Vec::with_capacity(len);
        for i in 0..len {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = f32::from(((state >> 16) & 0xffff) as u16) / 65_536.0;
            let magnitude = 0.25 + unit;
            out.push(if i % 3 == 0 { -magnitude } else { magnitude });
        }
        out
    }

    /// Seed pairs swept by the structure guards below.
    ///
    /// One pair is nowhere near enough. A fused-multiply-add rewrite of either
    /// kernel is bit-identical to the correct reduction on most inputs -- measured
    /// here, only 5 of these 16 pairs separate a fused `dot` and only 4 separate a
    /// fused `sq_euclid` -- so a single sample would wave an FMA straight through.
    /// `the_structure_guard_can_detect_a_fused_multiply_add` holds this sweep to
    /// that job, and fails if a change to `sample` ever makes it blind.
    const STRUCTURE_SEEDS: [(u32, u32); 16] = [
        (0x0000_0001, 0x0000_0002),
        (0x0000_0003, 0x0000_0004),
        (0x1234_5678, 0x9e37_79b9),
        (0x0bad_c0de, 0xdead_beef),
        (0x00c0_ffee, 0x00fe_ed01),
        (0x5eed_1234, 0x0f0f_0f0f),
        (0xa5a5_a5a5, 0x5a5a_5a5a),
        (0x0000_beef, 0xcafe_0000),
        (0x2718_2818, 0x3141_5926),
        (0x1618_0339, 0x6180_3398),
        (0x0102_0304, 0x0a0b_0c0d),
        (0xfeed_face, 0xabcd_ef01),
        (0x7fff_ffff, 0x8000_0001),
        (0x1357_9bdf, 0x2468_ace0),
        (0xdead_c0de, 0xb16b_00b5),
        (0x0000_0005, 0x0000_0006),
    ];

    /// Hand-written 16-accumulator reference, written with explicit indices rather
    /// than `chunks_exact`, so it guards the kernel's reduction *structure*.
    ///
    /// The lane count is spelled as a literal `16` on purpose, not as
    /// [`super::LANES`]: the reference has to be an independent statement of the
    /// contract, so that redefining the constant is caught here instead of being
    /// silently mirrored.
    // Allowed for the same reason as the kernels: the rewrites these lints
    // suggest would not preserve the order this reference exists to pin.
    #[allow(clippy::assign_op_pattern, clippy::needless_range_loop)]
    fn reference(a: &[f32], b: &[f32], square_difference: bool) -> f32 {
        let mut acc = [0.0f32; 16];
        let full = a.len() / 16;
        let term = |idx: usize| {
            if square_difference {
                let d = a[idx] - b[idx];
                d * d
            } else {
                a[idx] * b[idx]
            }
        };
        for chunk in 0..full {
            for l in 0..16 {
                let idx = chunk * 16 + l;
                acc[l] = acc[l] + term(idx);
            }
        }
        for idx in (full * 16)..a.len() {
            let l = idx - full * 16;
            acc[l] = acc[l] + term(idx);
        }
        let mut total = 0.0f32;
        for l in 0..16 {
            total = total + acc[l];
        }
        total
    }

    fn reference_dot(a: &[f32], b: &[f32]) -> f32 {
        reference(a, b, false)
    }

    fn reference_sq_euclid(a: &[f32], b: &[f32]) -> f32 {
        reference(a, b, true)
    }

    /// The same 16-lane shape as [`reference`], but fusing each multiply into its
    /// add. This is the tempting "optimisation" the R-G22 contract forbids, since
    /// it skips the intermediate rounding; it exists here only so the structure
    /// guards can be shown to detect it.
    #[allow(clippy::assign_op_pattern, clippy::needless_range_loop)]
    fn fused_reference(a: &[f32], b: &[f32], square_difference: bool) -> f32 {
        let mut acc = [0.0f32; 16];
        let full = a.len() / 16;
        let fuse = |idx: usize, carry: f32| {
            if square_difference {
                let d = a[idx] - b[idx];
                d.mul_add(d, carry)
            } else {
                a[idx].mul_add(b[idx], carry)
            }
        };
        for chunk in 0..full {
            for l in 0..16 {
                let idx = chunk * 16 + l;
                acc[l] = fuse(idx, acc[l]);
            }
        }
        for idx in (full * 16)..a.len() {
            let l = idx - full * 16;
            acc[l] = fuse(idx, acc[l]);
        }
        let mut total = 0.0f32;
        for l in 0..16 {
            total = total + acc[l];
        }
        total
    }

    #[test]
    fn dot_matches_an_f64_reference() {
        let a = sample(1001, 0x1234_5678);
        let b = sample(1001, 0x9e37_79b9);
        let want: f64 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| f64::from(*x) * f64::from(*y))
            .sum();
        assert!(want.abs() > 1.0, "degenerate reference value {want}");
        let got = f64::from(dot(&a, &b));
        let relative = (got - want).abs() / want.abs();
        assert!(relative < 1e-3, "dot = {got}, reference = {want}");
    }

    #[test]
    fn sq_euclid_matches_an_f64_reference() {
        let a = sample(1001, 0x0bad_c0de);
        let b = sample(1001, 0xdead_beef);
        let want: f64 = a
            .iter()
            .zip(&b)
            .map(|(x, y)| {
                let d = f64::from(*x) - f64::from(*y);
                d * d
            })
            .sum();
        assert!(want.abs() > 1.0, "degenerate reference value {want}");
        let got = f64::from(sq_euclid(&a, &b));
        let relative = (got - want).abs() / want.abs();
        assert!(relative < 1e-3, "sq_euclid = {got}, reference = {want}");
    }

    #[test]
    fn dot_reduces_in_the_fixed_sixteen_accumulator_order() {
        // 1001 = 62 * 16 + 9, so the tail path is exercised too.
        for (seed_a, seed_b) in STRUCTURE_SEEDS {
            let a = sample(1001, seed_a);
            let b = sample(1001, seed_b);
            assert_eq!(
                dot(&a, &b).to_bits(),
                reference_dot(&a, &b).to_bits(),
                "dot left the fixed reduction order for seeds ({seed_a:#010x}, {seed_b:#010x})"
            );
        }
    }

    #[test]
    fn sq_euclid_reduces_in_the_fixed_sixteen_accumulator_order() {
        for (seed_a, seed_b) in STRUCTURE_SEEDS {
            let a = sample(1001, seed_a);
            let b = sample(1001, seed_b);
            assert_eq!(
                sq_euclid(&a, &b).to_bits(),
                reference_sq_euclid(&a, &b).to_bits(),
                "sq_euclid left the fixed reduction order for seeds ({seed_a:#010x}, {seed_b:#010x})"
            );
        }
    }

    /// Guards the guard. The two tests above are the automated backstop for the
    /// "no `mul_add`/FMA" half of the kernel contract, and they are only worth
    /// anything if their seed sweep can actually tell a fused kernel apart from a
    /// correct one. On any single input it often cannot.
    #[test]
    fn the_structure_guard_can_detect_a_fused_multiply_add() {
        let mut differing_dot = 0usize;
        let mut differing_sq_euclid = 0usize;
        for (seed_a, seed_b) in STRUCTURE_SEEDS {
            let a = sample(1001, seed_a);
            let b = sample(1001, seed_b);
            if fused_reference(&a, &b, false).to_bits() != reference_dot(&a, &b).to_bits() {
                differing_dot += 1;
            }
            if fused_reference(&a, &b, true).to_bits() != reference_sq_euclid(&a, &b).to_bits() {
                differing_sq_euclid += 1;
            }
        }
        assert!(
            differing_dot > 0,
            "no seed pair separates a fused dot from the reference, so              dot_reduces_in_the_fixed_sixteen_accumulator_order could not see an FMA rewrite"
        );
        assert!(
            differing_sq_euclid > 0,
            "no seed pair separates a fused sq_euclid from the reference, so              sq_euclid_reduces_in_the_fixed_sixteen_accumulator_order could not see an FMA rewrite"
        );
    }

    #[test]
    fn kernels_handle_lengths_around_the_lane_boundary() {
        for len in [0, 1, LANES - 1, LANES, LANES + 1, 2 * LANES] {
            let a = sample(len, 0x00c0_ffee);
            let b = sample(len, 0x00fe_ed01);
            assert_eq!(
                dot(&a, &b).to_bits(),
                reference_dot(&a, &b).to_bits(),
                "dot mismatch at len {len}"
            );
            assert_eq!(
                sq_euclid(&a, &b).to_bits(),
                reference_sq_euclid(&a, &b).to_bits(),
                "sq_euclid mismatch at len {len}"
            );
        }
    }
}
