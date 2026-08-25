//! The distance metric selector and the fixed-order distance kernels behind it.
//!
//! Two kernels do all of `divsel`'s floating-point work: `dot` and
//! `sq_euclid`. Each has a scalar reference body (`dot_scalar`,
//! `sq_euclid_scalar`) and a `pulp` runtime-dispatched SIMD body, and the two
//! are **bit-identical** -- not "equal to a tolerance", identical under
//! `f32::to_bits` -- on every input and on every instruction set `pulp` can
//! select. That is contract R-G22, and
//! `tests::the_dispatched_kernels_are_bit_identical_to_the_scalar_ones` is its
//! automated half.
//!
//! # How bit-identity is achieved
//!
//! Floating-point addition is not associative, so the only way two reductions
//! agree to the last bit is to perform the *same* additions in the *same* order.
//! Both bodies therefore implement one fixed shape:
//!
//! * `LANES` = 16 logical accumulators; element `i` always lands in
//!   accumulator `i % 16`.
//! * The multiply and the add stay separate. A fused multiply-add keeps the
//!   product at full precision and rounds once instead of twice, so `mul_add`
//!   and `pulp`'s `mul_add_f32s` are banned from both paths.
//! * The final reduction is `acc[0] + acc[1] + ... + acc[15]`, left to right.
//!   `pulp`'s `reduce_sum_f32s` is *not* used: its internal tree order is the
//!   instruction set's business, not ours.
//!
//! The SIMD body expresses those 16 accumulators as `16 / W` vector registers of
//! `W` lanes each, so register `r` lane `l` *is* logical accumulator `r * W + l`.
//! `W` is `S::F32_LANES`: 16 on AVX-512, 8 on AVX2, 4 on SSE/NEON, 1 on the
//! scalar fallback -- all divisors of 16, all giving a whole number of registers.
//! A hypothetical `W` that does not divide 16 falls back to the scalar body
//! rather than reassociating.

use std::sync::OnceLock;

use pulp::{Arch, Simd, WithSimd};

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

/// Number of independent accumulator lanes both kernel bodies use.
///
/// The lane count is part of the kernels' numeric contract, not a tuning knob:
/// the SIMD bodies lay `LANES` logical accumulators across `LANES / W` vector
/// registers and reduce in exactly this order, which is what keeps them
/// bit-identical to the scalar bodies. Sixteen is also the widest f32 register
/// any supported instruction set has, so every `W` divides it.
pub(crate) const LANES: usize = 16;

/// The instruction set `pulp` selected for this process.
///
/// [`Arch::new`] runs CPU feature detection, which is cheap but not free, and
/// the kernels are called once per point pair -- hundreds of millions of times
/// in an `O(n^2)` diameter scan. Detecting once and copying the result out of a
/// [`OnceLock`] keeps that off the hot path; the selection cannot change during
/// a process's lifetime anyway.
fn arch() -> Arch {
    static ARCH: OnceLock<Arch> = OnceLock::new();
    *ARCH.get_or_init(Arch::new)
}

/// Dot product of two equal-length slices, the scalar reference body.
///
/// The summation order is a hard contract, not an implementation detail:
/// [`LANES`] independent accumulators are advanced in lockstep over
/// `chunks_exact(LANES)`, the tail folds into lane `index % LANES`, and the
/// accumulators are summed in lane order at the end. [`dot`] reduces this way on
/// every instruction set and is bit-identical to this body. Do not replace the
/// explicit loops with `sum()` or `fold`, and do not fuse the multiply and the
/// add: an `mul_add` would change the rounding.
///
/// # Panics
///
/// Panics in debug builds if `a` and `b` have different lengths.
// The two lints below would rewrite these loops into iterator adaptors, which
// changes the reduction order that the SIMD bodies have to match exactly.
#[allow(clippy::assign_op_pattern, clippy::needless_range_loop)]
// `pub` + `#[doc(hidden)]`, re-exported from `testutil`, so `benches/gist.rs`
// can measure the real body instead of a copy of it; see that module's doc.
// `#[inline]` so the cross-crate call from the bench costs what the in-crate
// call costs -- inlining cannot reassociate the sum, because Rust emits no
// fast-math flags and this loop's order is written out explicitly.
#[doc(hidden)]
#[inline]
pub fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
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

/// Squared Euclidean distance between two equal-length slices, the scalar
/// reference body.
///
/// Shares the fixed [`LANES`]-accumulator reduction order documented on
/// [`dot_scalar`], with `(a[i] - b[i])^2` in place of `a[i] * b[i]`.
///
/// # Panics
///
/// Panics in debug builds if `a` and `b` have different lengths.
// The two lints below would rewrite these loops into iterator adaptors, which
// changes the reduction order that the SIMD bodies have to match exactly.
#[allow(clippy::assign_op_pattern, clippy::needless_range_loop)]
// See the note on `dot_scalar`: hidden-public and `#[inline]` for the bench.
#[doc(hidden)]
#[inline]
pub fn sq_euclid_scalar(a: &[f32], b: &[f32]) -> f32 {
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

/// The SIMD body both kernels share, parameterised by the number of vector
/// registers the 16 logical accumulators occupy and by which element-wise term
/// is accumulated.
///
/// `REGISTERS` must be `LANES / S::F32_LANES`; [`Kernel::with_simd`] is the only
/// caller and picks it from the lane count. `SQUARED_DIFFERENCE` selects
/// `(a-b)^2` over `a*b`; it is a const parameter rather than a runtime flag so
/// the branch disappears at monomorphisation.
///
/// The layout, and why it is bit-identical to the scalar bodies:
///
/// * The slices are cut at `full * LANES`, a whole number of 16-element groups,
///   so `S::as_simd_f32s` yields exactly `full * REGISTERS` vectors and an empty
///   remainder. Group `g` occupies vectors `g * REGISTERS .. (g+1) * REGISTERS`.
/// * Register `r` of the accumulator holds logical accumulators
///   `r * W .. (r+1) * W`, so element `g * 16 + r * W + l` lands in logical
///   accumulator `r * W + l` -- that is, in `index % 16`, exactly as the scalar
///   body puts it.
/// * `mul` and `add` are separate vector ops; nothing here is fused.
/// * The registers are stored back into a plain `[f32; LANES]` through
///   `S::as_mut_simd_f32s`, which writes register `r` at offset `r * W`, so the
///   array is the scalar body's `acc` verbatim. The tail then folds into it by
///   index, and the sum runs left to right over the array.
///
/// Behaviour on a length mismatch is **unspecified**, and deliberately not the
/// scalar bodies' behaviour. Both entry points, [`dot`] and [`sq_euclid`],
/// `debug_assert` equal lengths and every caller inside this crate passes equal
/// lengths, so the case is a precondition violation rather than an input. This
/// body truncates to `a.len().min(b.len())` only so that a release build does
/// not index out of bounds; the result then differs from the scalar body's,
/// which zips two independent `chunks_exact(16)` and so pairs elements by their
/// position *within each operand*. For `a.len() == 33, b.len() == 20` the scalar
/// body pairs `a[32]` with `b[16]` in lane 0, while this body folds `a[16..20]`
/// into lanes 0..4 and drops the rest. Neither is meaningful; the parity tests
/// only ever compare equal-length operands.
// Same two lints, same reason: the order these loops fix is the contract.
#[allow(clippy::assign_op_pattern, clippy::needless_range_loop)]
#[inline(always)]
fn accumulate<S: Simd, const REGISTERS: usize, const SQUARED_DIFFERENCE: bool>(
    simd: S,
    a: &[f32],
    b: &[f32],
) -> f32 {
    debug_assert_eq!(REGISTERS * S::F32_LANES, LANES);

    let len = a.len().min(b.len());
    let grouped = (len / LANES) * LANES;
    let (a_groups, a_tail) = (&a[..grouped], &a[grouped..len]);
    let (b_groups, b_tail) = (&b[..grouped], &b[grouped..len]);

    let (a_vectors, a_extra) = S::as_simd_f32s(a_groups);
    let (b_vectors, b_extra) = S::as_simd_f32s(b_groups);
    debug_assert!(a_extra.is_empty() && b_extra.is_empty());

    let mut acc = [simd.splat_f32s(0.0); REGISTERS];
    for (group_a, group_b) in a_vectors
        .chunks_exact(REGISTERS)
        .zip(b_vectors.chunks_exact(REGISTERS))
    {
        for r in 0..REGISTERS {
            let term = if SQUARED_DIFFERENCE {
                let d = simd.sub_f32s(group_a[r], group_b[r]);
                simd.mul_f32s(d, d)
            } else {
                simd.mul_f32s(group_a[r], group_b[r])
            };
            acc[r] = simd.add_f32s(acc[r], term);
        }
    }

    let mut lanes = [0.0f32; LANES];
    {
        let (registers, extra) = S::as_mut_simd_f32s(&mut lanes);
        debug_assert!(extra.is_empty());
        // `registers.len()` is `LANES / W`, which is `REGISTERS`; the length
        // check inside `copy_from_slice` is a free assertion of exactly that.
        registers.copy_from_slice(&acc);
    }
    for (l, (x, y)) in a_tail.iter().zip(b_tail).enumerate() {
        let p = if SQUARED_DIFFERENCE {
            let d = x - y;
            d * d
        } else {
            x * y
        };
        lanes[l] = lanes[l] + p;
    }

    let mut total = 0.0f32;
    for l in 0..LANES {
        total = total + lanes[l];
    }
    total
}

/// The `pulp` dispatch payload: a pair of operands plus the choice of term.
///
/// `pulp` calls [`WithSimd::with_simd`] with the concrete `Simd` type for the
/// instruction set it selected, which is where the register count becomes known.
struct Kernel<'a, const SQUARED_DIFFERENCE: bool> {
    a: &'a [f32],
    b: &'a [f32],
}

impl<const SQUARED_DIFFERENCE: bool> WithSimd for Kernel<'_, SQUARED_DIFFERENCE> {
    type Output = f32;

    /// Turns the runtime lane count into the const register count
    /// [`accumulate`] needs.
    ///
    /// `S::F32_LANES` is an associated constant, so this match folds away at
    /// monomorphisation and only the selected arm is generated. The arms cover
    /// every lane count a supported instruction set has -- 16 (AVX-512), 8
    /// (AVX2), 4 (SSE, NEON, WASM SIMD128), 2 and 1 (the scalar fallback). A
    /// lane count that does not divide [`LANES`] cannot host the fixed
    /// accumulator layout, so it takes the scalar body rather than a reduction
    /// order of its own.
    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> f32 {
        match S::F32_LANES {
            16 => accumulate::<S, 1, SQUARED_DIFFERENCE>(simd, self.a, self.b),
            8 => accumulate::<S, 2, SQUARED_DIFFERENCE>(simd, self.a, self.b),
            4 => accumulate::<S, 4, SQUARED_DIFFERENCE>(simd, self.a, self.b),
            2 => accumulate::<S, 8, SQUARED_DIFFERENCE>(simd, self.a, self.b),
            1 => accumulate::<S, 16, SQUARED_DIFFERENCE>(simd, self.a, self.b),
            _ if SQUARED_DIFFERENCE => sq_euclid_scalar(self.a, self.b),
            _ => dot_scalar(self.a, self.b),
        }
    }
}

/// Dot product of two equal-length slices, dispatched to the widest instruction
/// set this CPU supports.
///
/// Bit-identical to [`dot_scalar`] on every input and every instruction set; see
/// the module documentation for how, and R-G22 for why that is a contract rather
/// than a courtesy.
///
/// # Panics
///
/// Panics in debug builds if `a` and `b` have different lengths.
pub(crate) fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "dot operands must have equal lengths");
    arch().dispatch(Kernel::<false> { a, b })
}

/// Squared Euclidean distance between two equal-length slices, dispatched to the
/// widest instruction set this CPU supports.
///
/// Bit-identical to [`sq_euclid_scalar`]; see [`dot`].
///
/// # Panics
///
/// Panics in debug builds if `a` and `b` have different lengths.
pub(crate) fn sq_euclid(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(
        a.len(),
        b.len(),
        "sq_euclid operands must have equal lengths"
    );
    arch().dispatch(Kernel::<true> { a, b })
}

#[cfg(test)]
mod tests {
    use pulp::{Simd, WithSimd};

    use super::{arch, dot, dot_scalar, sq_euclid, sq_euclid_scalar, LANES};
    use crate::metric::Metric;
    use crate::points::Points;
    use crate::testutil::SplitMix64;

    // ----------------------------------------------------------------------
    // R-G22: the dispatched kernels are bit-identical to the scalar ones.
    // ----------------------------------------------------------------------

    /// Reports the SIMD lane count of the instruction set `pulp` selected, so the
    /// parity test can name the register layout it actually exercised.
    struct LaneCount;

    impl WithSimd for LaneCount {
        type Output = usize;

        #[inline(always)]
        fn with_simd<S: Simd>(self, _simd: S) -> usize {
            S::F32_LANES
        }
    }

    /// The dimensions the parity sweep covers: below, at and above the 16-element
    /// group, an odd multiple-plus-remainder, and the three embedding widths the
    /// benches use.
    const PARITY_DIMS: [usize; 10] = [1, 7, 15, 16, 17, 31, 64, 384, 768, 1001];

    /// Seed pairs per dimension in the parity sweep, per the R-G22 contract.
    const PARITY_PAIRS: u64 = 20;

    /// Coordinates whose magnitudes span seven decades, `1e-3` to `1e3`.
    ///
    /// A well-scaled Gaussian keeps every partial sum inside one exponent range,
    /// where almost any summation order rounds the same way. Mixing decades makes
    /// the accumulators absorb small terms into large ones, which is exactly the
    /// regime where a reduction that reassociates diverges from this one.
    fn mixed_magnitude(len: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64(seed);
        (0..len)
            .map(|_| {
                // `below(7) - 3` is the closed range -3..=3.
                let exponent = rng.below(7) as i32 - 3;
                rng.normal() * 10f32.powi(exponent)
            })
            .collect()
    }

    /// The raw bits of every value in a slice, for an order-sensitive comparison
    /// that `==` on `f32` would not give (`-0.0 == 0.0`).
    fn bits(values: &[f32]) -> Vec<u32> {
        values.iter().map(|x| x.to_bits()).collect()
    }

    /// L2-normalizes a row through the **scalar** kernel, mirroring
    /// `points::normalize_rows` exactly.
    fn scalar_normalized(row: &[f32]) -> Vec<f32> {
        let norm = dot_scalar(row, row).sqrt();
        assert!(norm.is_finite() && norm != 0.0, "row is not normalizable");
        row.iter().map(|x| x / norm).collect()
    }

    /// The local half of the cross-ISA promise.
    ///
    /// `pulp` picks the instruction set at runtime, so this test only proves
    /// bit-identity for whatever the machine running it selected -- printed below
    /// so a failure report names the ISA. The other half is CI: the same test on
    /// an AVX-512 host (16 lanes, one register) and on an `aarch64` NEON host
    /// (4 lanes, four registers) covers the remaining register layouts, and the
    /// scalar fallback (1 lane, sixteen registers) is reachable on any host by
    /// building `pulp` without its `x86-v3` feature.
    ///
    /// On the two architectures this crate claims SIMD dispatch for the lane
    /// count is **asserted**, not merely printed. At one lane the dispatched
    /// kernel *is* the scalar body, so every equality below compares a value to
    /// itself and the test proves nothing — which is precisely how a silent pulp
    /// fallback on a CI runner would look. Elsewhere a one-lane arch is a
    /// legitimate outcome and only the bit-identity claim is checked.
    #[test]
    fn the_dispatched_kernels_are_bit_identical_to_the_scalar_ones() {
        let lanes = arch().dispatch(LaneCount);
        println!(
            "pulp selected {:?}: {lanes} f32 lane(s) per register, {} register(s) per {LANES}-element group",
            arch(),
            LANES / lanes.max(1)
        );
        #[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
        assert!(
            lanes > 1,
            "pulp selected {:?}, a {lanes}-lane arch, on a target that has SIMD: the comparisons \
             below would then be the scalar body against itself and prove nothing",
            arch()
        );

        for &dim in &PARITY_DIMS {
            for pair in 0..PARITY_PAIRS {
                let seed = ((dim as u64) << 32) | pair;
                let a = mixed_magnitude(dim, seed ^ 0xa5a5_a5a5_0000_0001);
                let b = mixed_magnitude(dim, seed ^ 0x5a5a_5a5a_0000_0002);

                assert_eq!(
                    dot(&a, &b).to_bits(),
                    dot_scalar(&a, &b).to_bits(),
                    "dot diverged at dim {dim}, pair {pair}: dispatched {:e} vs scalar {:e}",
                    dot(&a, &b),
                    dot_scalar(&a, &b)
                );
                assert_eq!(
                    sq_euclid(&a, &b).to_bits(),
                    sq_euclid_scalar(&a, &b).to_bits(),
                    "sq_euclid diverged at dim {dim}, pair {pair}: dispatched {:e} vs scalar {:e}",
                    sq_euclid(&a, &b),
                    sq_euclid_scalar(&a, &b)
                );
            }
        }
    }

    /// The same contract one level up: [`Points::dist`] under both metrics, and
    /// the row normalization [`Metric::Cosine`] performs on construction, are bit
    /// for bit what the scalar kernels would have produced.
    #[test]
    fn points_dist_is_bit_identical_to_the_scalar_path() {
        for &dim in &PARITY_DIMS {
            for pair in 0..PARITY_PAIRS {
                let seed = ((dim as u64) << 32) | pair;
                let a = mixed_magnitude(dim, seed ^ 0x1234_5678_0000_0003);
                let b = mixed_magnitude(dim, seed ^ 0x9e37_79b9_0000_0004);
                let mut data = a.clone();
                data.extend_from_slice(&b);

                let euclidean = Points::new(data.clone(), dim, Metric::Euclidean)
                    .expect("two finite rows are a valid point set");
                assert_eq!(
                    euclidean.dist(0, 1).to_bits(),
                    sq_euclid_scalar(&a, &b).sqrt().to_bits(),
                    "Euclidean dist diverged at dim {dim}, pair {pair}"
                );

                let cosine = Points::new(data, dim, Metric::Cosine)
                    .expect("no row of this generator has a degenerate norm");
                let normalized_a = scalar_normalized(&a);
                let normalized_b = scalar_normalized(&b);
                assert_eq!(
                    bits(cosine.row(0)),
                    bits(&normalized_a),
                    "cosine normalization diverged on row 0 at dim {dim}, pair {pair}"
                );
                assert_eq!(
                    bits(cosine.row(1)),
                    bits(&normalized_b),
                    "cosine normalization diverged on row 1 at dim {dim}, pair {pair}"
                );
                assert_eq!(
                    cosine.dist(0, 1).to_bits(),
                    (1.0 - dot_scalar(&normalized_a, &normalized_b))
                        .clamp(0.0, 2.0)
                        .to_bits(),
                    "cosine dist diverged at dim {dim}, pair {pair}"
                );
            }
        }
    }

    /// Every length from `0` to `2 * LANES + 1`, so the empty group loop, a lone
    /// tail element, an exact group boundary and a group-plus-tail all agree.
    #[test]
    fn the_dispatched_kernels_agree_on_every_short_length() {
        for len in 0..=(2 * LANES + 1) {
            let a = mixed_magnitude(len, 0x0bad_c0de_dead_beef ^ len as u64);
            let b = mixed_magnitude(len, 0x00c0_ffee_00fe_ed01 ^ len as u64);
            assert_eq!(
                dot(&a, &b).to_bits(),
                dot_scalar(&a, &b).to_bits(),
                "dot diverged at len {len}"
            );
            assert_eq!(
                sq_euclid(&a, &b).to_bits(),
                sq_euclid_scalar(&a, &b).to_bits(),
                "sq_euclid diverged at len {len}"
            );
        }
    }

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
            "no seed pair separates a fused dot from the reference, so \
             dot_reduces_in_the_fixed_sixteen_accumulator_order could not see an FMA rewrite"
        );
        assert!(
            differing_sq_euclid > 0,
            "no seed pair separates a fused sq_euclid from the reference, so \
             sq_euclid_reduces_in_the_fixed_sixteen_accumulator_order could not see an FMA rewrite"
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
