//! Deterministic, dependency-free random data for `divsel`'s own tests, doc
//! examples and benches.
//!
//! This module is `#[doc(hidden)] pub`: it has to be reachable from integration
//! tests and benches, which compile against the crate as an external library, but
//! it is not part of the supported API and carries no stability promise.
//!
//! # Platform determinism
//!
//! [`SplitMix64::next_u64`], [`SplitMix64::next_f64`], [`SplitMix64::next_f32`],
//! [`SplitMix64::below`] and [`SplitMix64::uniform_weights`] are integer
//! arithmetic plus exact power-of-two divisions. A given seed yields the same
//! bits from them on every platform, every Rust version and every run, so a
//! fixture may pin their exact values.
//!
//! [`SplitMix64::normal`] and [`SplitMix64::gaussian_points`] are **not** in
//! that set. Box-Muller needs `f64::ln` and `f64::cos`, and the standard
//! library promises nothing about either: both are documented under
//! "Unspecified precision -- the precision of this function is
//! non-deterministic. This means it varies by platform, Rust version, and can
//! even differ within the same execution from one invocation to the next." Of
//! the operations used here only `sqrt` is correctly rounded ("guaranteed to be
//! the rounded infinite-precision result ... specified by IEEE 754 as
//! squareRoot"). A Gaussian coordinate can therefore differ in its last ulp
//! between two machines, and a selection derived from one can differ by a whole
//! index once two marginals land close enough together.
//!
//! So: **never back an exact-value fixture with Gaussian draws.** Use them for
//! statistical or structural assertions that carry headroom -- an independence
//! property, an agreement between two code paths, a count with slack -- and
//! reach for the integer draws when a test needs to pin bytes.

/// [SplitMix64], the reference finalizer used to seed `xoshiro`, as a small
/// standalone generator.
///
/// The state is public so a test can construct one with `SplitMix64(seed)` and
/// read or rewind it. Any `u64` is a valid seed, including `0`.
///
/// [SplitMix64]: https://dl.acm.org/doi/10.1145/2714064.2660195
///
/// # Examples
///
/// ```
/// # use divsel::testutil::SplitMix64;
/// let mut rng = SplitMix64(0x5eed);
/// let a = rng.next_u64();
/// let b = SplitMix64(0x5eed).next_u64();
/// assert_eq!(a, b);
/// ```
pub struct SplitMix64(pub u64);

impl SplitMix64 {
    /// The next 64 raw bits, advancing the state by the SplitMix64 gamma.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A `f64` in `[0, 1)`, built from the top 53 bits so every value is a
    /// multiple of `2^-53` and the conversion is exact.
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0
    }

    /// A `f32` in `[0, 1)`, built from the top 24 bits so every value is a
    /// multiple of `2^-24` and the conversion is exact.
    ///
    /// Note this consumes a whole `u64` draw, exactly like [`Self::next_f64`], so
    /// mixing the two does not perturb the stream in surprising ways.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / 16_777_216.0
    }

    /// An index in `0..bound`, by modulo.
    ///
    /// The modulo bias is at most one part in `2^64 / bound`, which is far below
    /// anything a test can observe; this is deliberately not a rejection sampler,
    /// so the number of draws per call is fixed at one.
    ///
    /// # Panics
    ///
    /// Panics if `bound` is `0`.
    pub fn below(&mut self, bound: usize) -> usize {
        assert!(bound > 0, "below(0) has no valid output");
        (self.next_u64() % bound as u64) as usize
    }

    /// A standard normal deviate, `N(0, 1)`.
    ///
    /// Box-Muller in `f64`, cast down to `f32` at the end. Only the cosine half of
    /// the pair is kept -- caching the sine half would save a call but would make
    /// the stream depend on how many deviates a caller has drawn so far, which is
    /// exactly the kind of hidden state that makes seeded tests hard to reason
    /// about.
    ///
    /// # Platform determinism
    ///
    /// Unlike the integer draws, this is **not** bit-stable across platforms:
    /// `f64::ln` and `f64::cos` carry no precision guarantee in the standard
    /// library, so the last ulp is the platform's business, not this crate's. Do
    /// not pin exact values derived from it; see the module documentation.
    pub fn normal(&mut self) -> f32 {
        // `next_f64` yields [0, 1); the reflection moves it to (0, 1] so the
        // logarithm is always finite.
        let u1 = 1.0 - self.next_f64();
        let u2 = self.next_f64();
        let radius = (-2.0 * u1.ln()).sqrt();
        (radius * (std::f64::consts::TAU * u2).cos()) as f32
    }

    /// A row-major `n * dim` buffer of independent `N(0, 1)` coordinates, ready
    /// for [`Points::new`](crate::Points::new).
    ///
    /// Gaussian coordinates give a point cloud with no preferred direction and no
    /// zero rows, so it is usable under both [`Metric`](crate::Metric) variants.
    ///
    /// # Platform determinism
    ///
    /// Inherits the caveat on [`SplitMix64::normal`]: the coordinates are
    /// reproducible for a given seed on a given machine, but not bit-stable
    /// across platforms or Rust versions. Assertions over these points must
    /// carry headroom -- a golden fixture that pins a selection computed from
    /// them would be fragile, and its failures would be misread as bugs in the
    /// code under test.
    pub fn gaussian_points(&mut self, n: usize, dim: usize) -> Vec<f32> {
        (0..n * dim).map(|_| self.normal()).collect()
    }

    /// `n` weights drawn uniformly from `[0, 1)`.
    ///
    /// Finite and non-negative, so the result always passes
    /// [`Linear::validate`](crate::Utility::validate).
    pub fn uniform_weights(&mut self, n: usize) -> Vec<f64> {
        (0..n).map(|_| self.next_f64()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::SplitMix64;

    #[test]
    fn the_stream_is_reproducible_and_seed_dependent() {
        let first: Vec<u64> = (0..8).map(|_| SplitMix64(7).next_u64()).collect();
        assert!(first.iter().all(|&x| x == first[0]), "a fresh seed rewinds");

        let mut a = SplitMix64(7);
        let mut b = SplitMix64(7);
        let mut c = SplitMix64(8);
        let stream_a: Vec<u64> = (0..8).map(|_| a.next_u64()).collect();
        let stream_b: Vec<u64> = (0..8).map(|_| b.next_u64()).collect();
        let stream_c: Vec<u64> = (0..8).map(|_| c.next_u64()).collect();
        assert_eq!(stream_a, stream_b);
        assert_ne!(stream_a, stream_c);
    }

    #[test]
    fn splitmix64_matches_the_reference_vector() {
        // The published SplitMix64 test vector for seed 0.
        let mut rng = SplitMix64(0);
        assert_eq!(rng.next_u64(), 0xe220_a839_7b1d_cdaf);
        assert_eq!(rng.next_u64(), 0x6e78_9e6a_a1b9_65f4);
        assert_eq!(rng.next_u64(), 0x06c4_5d18_8009_454f);
    }

    #[test]
    fn floats_stay_inside_the_unit_interval() {
        let mut rng = SplitMix64(0xabcd_ef01);
        for _ in 0..2_000 {
            let x = rng.next_f64();
            assert!((0.0..1.0).contains(&x), "next_f64 out of range: {x}");
            let y = rng.next_f32();
            assert!((0.0..1.0).contains(&y), "next_f32 out of range: {y}");
        }
    }

    #[test]
    fn below_stays_in_range_and_covers_it() {
        let mut rng = SplitMix64(0x1234_5678);
        let mut seen = [false; 5];
        for _ in 0..500 {
            let i = rng.below(5);
            assert!(i < 5);
            seen[i] = true;
        }
        assert!(seen.iter().all(|&s| s), "below(5) never reached some index");
    }

    #[test]
    fn normal_deviates_look_standard() {
        let mut rng = SplitMix64(0x5eed_5eed);
        const N: usize = 20_000;
        let mut sum = 0.0f64;
        let mut sum_sq = 0.0f64;
        for _ in 0..N {
            let x = f64::from(rng.normal());
            assert!(x.is_finite(), "Box-Muller produced {x}");
            sum += x;
            sum_sq += x * x;
        }
        let mean = sum / N as f64;
        let variance = sum_sq / N as f64 - mean * mean;
        assert!(mean.abs() < 0.05, "mean {mean} is not near 0");
        assert!(
            (variance - 1.0).abs() < 0.05,
            "variance {variance} is not near 1"
        );
    }

    #[test]
    fn generators_produce_the_requested_shapes() {
        let mut rng = SplitMix64(99);
        let data = rng.gaussian_points(7, 3);
        assert_eq!(data.len(), 21);
        assert!(data.iter().all(|x| x.is_finite()));

        let weights = rng.uniform_weights(11);
        assert_eq!(weights.len(), 11);
        assert!(weights.iter().all(|&w| (0.0..1.0).contains(&w)));
    }
}
