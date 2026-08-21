//! The monotone submodular utility `g`, and the three implementations `divsel`
//! ships: [`Linear`], [`Coverage`] and [`FacilityLocation`].
//!
//! GIST maximizes `f(S) = g(S) + lambda * min-pairwise-distance(S)`. This module
//! owns the `g` half. Every implementation is expected to be **monotone**
//! (`g(v | S) >= 0` for all `v` and `S`) and **submodular**
//! (`g(v | S) >= g(v | T)` whenever `S` is a subset of `T`); `divsel` cannot
//! verify either property and does not try to, so a custom [`Utility`] that
//! breaks them silently voids GIST's approximation guarantee.
//!
//! Implementations keep an internal cache of the current selection, so the
//! sequence [`Utility::marginal`] then [`Utility::commit`] is the contract:
//! `marginal` reads the cache, `commit` advances it, and [`Utility::reset`]
//! rewinds it to the empty selection.

use crate::error::DivselError;
use crate::metric::Metric;
use crate::points::Points;

/// Monotone submodular `g`. Implementations must be monotone and submodular;
/// `divsel` cannot verify this and says so here.
///
/// `g` is never evaluated on a whole set: GIST only ever needs marginal gains, so
/// the trait exposes `g(v | S)` and a `commit` hook that lets an implementation
/// keep whatever incremental state makes that marginal cheap. `g(empty set)` is
/// `0` by convention, which is the state [`Utility::reset`] restores.
pub trait Utility: Send + Sync {
    /// `g(v | S)` — the marginal gain of adding `v` to the current selection
    /// `selected`.
    ///
    /// `selected` is the selection built by the [`Utility::commit`] calls made so
    /// far, in selection order. Implementations that keep their own cache may
    /// ignore it; it is passed so that a stateless implementation is possible.
    ///
    /// # Panics
    ///
    /// All three built-ins index a per-point table by `v`, so they panic when `v`
    /// is out of range for the point set. Call [`Utility::validate`] once up
    /// front to rule that out.
    fn marginal(&self, v: usize, selected: &[usize], pts: &Points<'_>) -> f64;

    /// Called after `v` is committed so caches can update.
    fn commit(&mut self, v: usize, pts: &Points<'_>);

    /// Return to the empty-selection state.
    fn reset(&mut self);

    /// True when marginals never change (enables the CELF fast path).
    fn is_linear(&self) -> bool {
        false
    }

    /// Check this utility is compatible with `pts` (lengths etc.).
    ///
    /// # Errors
    ///
    /// Implementation-defined; see each built-in. The default accepts everything.
    fn validate(&self, _pts: &Points<'_>) -> Result<(), DivselError> {
        Ok(())
    }

    /// Clone into a box so callers can run independent copies on parallel threads.
    fn boxed_clone(&self) -> Box<dyn Utility>;
}

impl Clone for Box<dyn Utility> {
    fn clone(&self) -> Self {
        self.boxed_clone()
    }
}

/// A modular (linear) utility: `g(S) = sum over v in S of weights[v]`.
///
/// Modular functions are the boundary case of submodularity — marginals do not
/// depend on the selection at all — so [`Utility::is_linear`] is `true` and the
/// greedy loop can use its CELF fast path.
#[derive(Clone, Debug)]
pub struct Linear {
    /// One weight per point, in point order. Must be finite and non-negative for
    /// `g` to be monotone; [`Utility::validate`] enforces that.
    pub weights: Vec<f64>,
}

impl Linear {
    /// Builds a linear utility from one weight per point.
    ///
    /// Infallible: the weights are only checked against a point set, by
    /// [`Utility::validate`].
    pub fn new(weights: Vec<f64>) -> Self {
        Self { weights }
    }

    /// Builds a linear utility that weights all `n` points equally, at `1.0` each,
    /// so that `g(S) == |S|`.
    pub fn uniform(n: usize) -> Self {
        Self {
            weights: vec![1.0; n],
        }
    }
}

impl Utility for Linear {
    fn marginal(&self, v: usize, _selected: &[usize], _pts: &Points<'_>) -> f64 {
        self.weights[v]
    }

    fn commit(&mut self, _v: usize, _pts: &Points<'_>) {}

    fn reset(&mut self) {}

    fn is_linear(&self) -> bool {
        true
    }

    fn validate(&self, pts: &Points<'_>) -> Result<(), DivselError> {
        if self.weights.len() != pts.n() {
            return Err(DivselError::WeightsLength {
                expected: pts.n(),
                got: self.weights.len(),
            });
        }
        for (index, &value) in self.weights.iter().enumerate() {
            if !value.is_finite() || value < 0.0 {
                return Err(DivselError::InvalidWeight { index, value });
            }
        }
        Ok(())
    }

    fn boxed_clone(&self) -> Box<dyn Utility> {
        Box::new(self.clone())
    }
}

/// Unweighted set coverage: `g(S)` is the number of distinct items covered by the
/// sets attached to the points in `S`.
///
/// Monotone and submodular by construction — an item can only be covered once, so
/// a larger selection can only shrink what is left to cover.
#[derive(Clone, Debug)]
pub struct Coverage {
    /// The item ids each point covers, deduplicated and ascending.
    sets: Vec<Vec<u32>>,
    /// Number of items in the coverage universe; every id is `< universe`.
    universe: usize,
    /// Which items the current selection already covers, indexed by item id.
    covered: Vec<bool>,
}

impl Coverage {
    /// Builds a coverage utility from one item list per point, over a universe of
    /// `universe` items with ids `0..universe`.
    ///
    /// Each list is sorted and deduplicated, so an item repeated inside a single
    /// list is counted once.
    ///
    /// # Errors
    ///
    /// Returns [`DivselError::CoverageItemOutOfRange`] for the first item id that
    /// is not less than `universe`, scanning rows in order.
    pub fn new(mut sets: Vec<Vec<u32>>, universe: usize) -> Result<Self, DivselError> {
        for (row, items) in sets.iter_mut().enumerate() {
            for &item in items.iter() {
                if item as usize >= universe {
                    return Err(DivselError::CoverageItemOutOfRange {
                        row,
                        item,
                        universe,
                    });
                }
            }
            items.sort_unstable();
            items.dedup();
        }
        Ok(Self {
            sets,
            universe,
            covered: vec![false; universe],
        })
    }

    /// Number of items in the coverage universe.
    pub fn universe(&self) -> usize {
        self.universe
    }
}

impl Utility for Coverage {
    fn marginal(&self, v: usize, _selected: &[usize], _pts: &Points<'_>) -> f64 {
        self.sets[v]
            .iter()
            .filter(|&&item| !self.covered[item as usize])
            .count() as f64
    }

    fn commit(&mut self, v: usize, _pts: &Points<'_>) {
        let items = &self.sets[v];
        let covered = &mut self.covered;
        for &item in items {
            covered[item as usize] = true;
        }
    }

    fn reset(&mut self) {
        self.covered.fill(false);
    }

    fn validate(&self, pts: &Points<'_>) -> Result<(), DivselError> {
        if self.sets.len() != pts.n() {
            return Err(DivselError::CoverageLength {
                expected: pts.n(),
                got: self.sets.len(),
            });
        }
        Ok(())
    }

    fn boxed_clone(&self) -> Box<dyn Utility> {
        Box::new(self.clone())
    }
}

/// The facility-location utility the GIST paper's own experiments use:
/// `g(S) = sum over i in V of max over j in S of sim(i, j)`, with `g(empty) = 0`.
///
/// Every point in the whole set is represented by its most similar selected
/// point, so `g` rewards a selection that covers the data rather than one that
/// merely scores well on its own members.
///
/// # Similarity
///
/// ```text
/// sim(i, j) = max(0, 1 - dist(i, j) / scale)
/// scale     = 1.0    for Metric::Cosine
/// scale     = d_max  for Metric::Euclidean
/// ```
///
/// For [`Metric::Cosine`] this is the paper's own `s(i, j) = 1 - dist(x_i, x_j)`
/// (Sec. 5.2); clamping the result at `0`, which matters because cosine distance
/// reaches `2` for opposed vectors, is a **`[divsel choice]`**. Euclidean distance
/// has no natural upper bound, so dividing by the set's diameter `d_max` is also a
/// **`[divsel choice]`**; when `d_max` is `0` — every point identical — the scale
/// falls back to `1.0`, which makes `sim` identically `1` instead of `0/0`.
///
/// # Monotonicity and submodularity
///
/// Monotone: each term of a marginal is `max(0, sim(i, v) - best[i])`, so a
/// marginal is never negative. Submodular: `best` is a running maximum over
/// non-negative similarities, so it only grows as the selection grows, and every
/// term of the marginal only shrinks. Note `sim(v, v) == 1`, so the first point
/// selected always has a marginal of at least `1`.
///
/// # Cost
///
/// The running maximum is cached, one `f64` per point, so both `marginal` and
/// `commit` are `O(n)` distance evaluations, i.e. `O(n * dim)` work.
#[derive(Clone, Debug)]
pub struct FacilityLocation {
    /// Distance at which similarity reaches `0`; always finite and `> 0`.
    scale: f32,
    /// `best[i]` is `max over j in S of sim(i, j)`, `0` for the empty selection.
    best: Vec<f64>,
}

/// Forces `scale` into the finite, strictly positive range `sim` needs, mapping
/// anything else — `0`, negatives, infinities, `NaN` — to `1.0`.
fn usable_scale(scale: f32) -> f32 {
    if scale.is_finite() && scale > 0.0 {
        scale
    } else {
        1.0
    }
}

impl FacilityLocation {
    /// Builds a facility-location utility for `pts`, deriving the similarity scale
    /// from the metric: `1.0` for [`Metric::Cosine`], the exact diameter for
    /// [`Metric::Euclidean`].
    ///
    /// Computing the Euclidean diameter is an `O(n^2)` parallel scan. A caller
    /// that already holds `d_max` should use [`FacilityLocation::with_scale`]
    /// instead.
    ///
    /// A diameter that is not finite and strictly positive — `0` when every point
    /// is identical, or an infinity from coordinates large enough to overflow the
    /// squared distance — falls back to a scale of `1.0` rather than producing a
    /// `NaN` similarity.
    pub fn new(pts: &Points<'_>) -> Self {
        let scale = match pts.metric() {
            Metric::Cosine => 1.0,
            Metric::Euclidean => usable_scale(pts.diameter().0),
        };
        Self {
            scale,
            best: vec![0.0; pts.n()],
        }
    }

    /// Builds a facility-location utility over `n` points with an explicit
    /// similarity scale, for callers that already know the diameter.
    ///
    /// Panic-free: a `scale` that is not finite and strictly positive is replaced
    /// by `1.0`.
    pub fn with_scale(n: usize, scale: f32) -> Self {
        Self {
            scale: usable_scale(scale),
            best: vec![0.0; n],
        }
    }

    /// `sim(i, j) = max(0, 1 - dist(i, j) / scale)`, evaluated in `f64`.
    fn sim(&self, i: usize, j: usize, pts: &Points<'_>) -> f64 {
        let distance = f64::from(pts.dist(i, j));
        (1.0 - distance / f64::from(self.scale)).max(0.0)
    }
}

impl Utility for FacilityLocation {
    fn marginal(&self, v: usize, _selected: &[usize], pts: &Points<'_>) -> f64 {
        let mut total = 0.0;
        for (i, &best) in self.best.iter().enumerate() {
            total += (self.sim(i, v, pts) - best).max(0.0);
        }
        total
    }

    fn commit(&mut self, v: usize, pts: &Points<'_>) {
        for i in 0..self.best.len() {
            let similarity = self.sim(i, v, pts);
            if similarity > self.best[i] {
                self.best[i] = similarity;
            }
        }
    }

    fn reset(&mut self) {
        self.best.fill(0.0);
    }

    /// # Errors
    ///
    /// Returns [`DivselError::WeightsLength`] — reused here for the per-point
    /// cache — when the cache was built for a different number of points.
    fn validate(&self, pts: &Points<'_>) -> Result<(), DivselError> {
        if self.best.len() != pts.n() {
            return Err(DivselError::WeightsLength {
                expected: pts.n(),
                got: self.best.len(),
            });
        }
        Ok(())
    }

    fn boxed_clone(&self) -> Box<dyn Utility> {
        Box::new(self.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::{usable_scale, Coverage, FacilityLocation, Linear, Utility};
    use crate::error::DivselError;
    use crate::metric::Metric;
    use crate::points::Points;

    const TOL: f64 = 1e-6;

    #[track_caller]
    fn assert_close(got: f64, want: f64) {
        assert!(
            (got - want).abs() < TOL,
            "got {got}, want {want} (tolerance {TOL})"
        );
    }

    /// SplitMix64: a deterministic, dependency-free generator for the randomized
    /// spot-checks. Task 3 promotes this to a shared test-support module.
    struct SplitMix64(u64);

    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self(seed)
        }

        fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            z ^ (z >> 31)
        }

        /// A float in `[-1, 1)`.
        fn next_coord(&mut self) -> f32 {
            let unit = (self.next_u64() >> 40) as f32 / 16_777_216.0;
            unit * 2.0 - 1.0
        }

        /// An index in `0..bound`.
        fn below(&mut self, bound: usize) -> usize {
            (self.next_u64() % bound as u64) as usize
        }
    }

    fn sample(n: usize, dim: usize, seed: u64) -> Vec<f32> {
        let mut rng = SplitMix64::new(seed);
        (0..n * dim).map(|_| rng.next_coord()).collect()
    }

    /// Four points, used wherever a `Points` is needed only to satisfy a
    /// signature.
    fn four_points() -> Points<'static> {
        Points::new(
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0],
            2,
            Metric::Euclidean,
        )
        .expect("four points")
    }

    /// The hand-built triangle used for the facility-location arithmetic:
    /// `(0,0)`, `(3,0)`, `(0,4)`, so `d(0,1) = 3`, `d(0,2) = 4`, `d(1,2) = 5` and
    /// the diameter is exactly `5`.
    fn triangle() -> Points<'static> {
        Points::new(vec![0.0, 0.0, 3.0, 0.0, 0.0, 4.0], 2, Metric::Euclidean)
            .expect("triangle points")
    }

    // ---- Linear -----------------------------------------------------------

    #[test]
    fn linear_marginals_ignore_the_selection() {
        let pts = four_points();
        let mut linear = Linear::new(vec![0.5, 2.0, 3.0, 0.0]);
        for v in 0..pts.n() {
            let empty = linear.marginal(v, &[], &pts);
            assert_close(empty, linear.weights[v]);
            assert_close(linear.marginal(v, &[0, 2, 3], &pts), empty);
        }
        // Committing changes nothing, which is exactly what is_linear() promises.
        linear.commit(0, &pts);
        linear.commit(2, &pts);
        for v in 0..pts.n() {
            assert_close(linear.marginal(v, &[0, 2], &pts), linear.weights[v]);
        }
        linear.reset();
        assert_close(linear.marginal(1, &[], &pts), 2.0);
    }

    #[test]
    fn linear_is_linear() {
        assert!(Linear::uniform(4).is_linear());
    }

    #[test]
    fn linear_uniform_weights_every_point_at_one() {
        let pts = four_points();
        let linear = Linear::uniform(pts.n());
        assert_eq!(linear.weights, vec![1.0; 4]);
        assert_eq!(linear.validate(&pts), Ok(()));
        for v in 0..pts.n() {
            assert_close(linear.marginal(v, &[], &pts), 1.0);
        }
    }

    #[test]
    fn linear_validate_catches_a_length_mismatch() {
        let pts = four_points();
        assert_eq!(
            Linear::new(vec![1.0, 2.0]).validate(&pts).unwrap_err(),
            DivselError::WeightsLength {
                expected: 4,
                got: 2
            }
        );
    }

    #[test]
    fn linear_validate_rejects_a_negative_or_non_finite_weight() {
        let pts = four_points();
        assert_eq!(
            Linear::new(vec![1.0, 2.0, -0.25, 4.0])
                .validate(&pts)
                .unwrap_err(),
            DivselError::InvalidWeight {
                index: 2,
                value: -0.25
            }
        );
        assert_eq!(
            Linear::new(vec![1.0, f64::INFINITY, 3.0, 4.0])
                .validate(&pts)
                .unwrap_err(),
            DivselError::InvalidWeight {
                index: 1,
                value: f64::INFINITY
            }
        );
        // NaN is rejected too, but does not compare equal, so match on the shape.
        match Linear::new(vec![f64::NAN, 2.0, 3.0, 4.0])
            .validate(&pts)
            .unwrap_err()
        {
            DivselError::InvalidWeight { index, value } => {
                assert_eq!(index, 0);
                assert!(value.is_nan());
            }
            other => panic!("expected InvalidWeight, got {other:?}"),
        }
        // Zero is a legal weight.
        assert_eq!(Linear::new(vec![0.0; 4]).validate(&pts), Ok(()));
    }

    #[test]
    fn linear_boxed_clone_yields_equal_marginals() {
        let pts = four_points();
        let linear = Linear::new(vec![0.5, 2.0, 3.0, 7.25]);
        let cloned: Box<dyn Utility> = linear.boxed_clone();
        assert!(cloned.is_linear());
        for v in 0..pts.n() {
            assert_eq!(
                cloned.marginal(v, &[], &pts).to_bits(),
                linear.marginal(v, &[], &pts).to_bits()
            );
        }
    }

    // ---- Coverage ---------------------------------------------------------

    /// Four points over a five-item universe. Initial marginals are 3, 2, 2, 5.
    fn coverage_fixture() -> Coverage {
        Coverage::new(
            vec![vec![0, 1, 2], vec![2, 3], vec![3, 4], vec![0, 1, 2, 3, 4]],
            5,
        )
        .expect("coverage fixture")
    }

    #[test]
    fn coverage_marginal_counts_newly_covered_items() {
        let pts = four_points();
        let mut coverage = coverage_fixture();
        assert_eq!(coverage.universe(), 5);
        let marginals = |c: &Coverage| {
            (0..4)
                .map(|v| c.marginal(v, &[], &pts))
                .collect::<Vec<f64>>()
        };
        assert_eq!(marginals(&coverage), vec![3.0, 2.0, 2.0, 5.0]);

        // {0,1,2} covered: set 1 keeps only item 3, set 3 keeps items 3 and 4.
        coverage.commit(0, &pts);
        assert_eq!(marginals(&coverage), vec![0.0, 1.0, 2.0, 2.0]);

        // Adding {3,4} covers the universe, so nothing is left to gain.
        coverage.commit(2, &pts);
        assert_eq!(marginals(&coverage), vec![0.0, 0.0, 0.0, 0.0]);

        coverage.reset();
        assert_eq!(marginals(&coverage), vec![3.0, 2.0, 2.0, 5.0]);
    }

    #[test]
    fn coverage_counts_a_repeated_item_once() {
        let pts = four_points();
        let coverage = Coverage::new(
            vec![vec![1, 1, 1, 2], vec![2, 2], vec![0], vec![2, 1, 2, 0]],
            3,
        )
        .expect("duplicates are deduplicated, not rejected");
        assert_eq!(coverage.marginal(0, &[], &pts), 2.0);
        assert_eq!(coverage.marginal(1, &[], &pts), 1.0);
        assert_eq!(coverage.marginal(3, &[], &pts), 3.0);
    }

    #[test]
    fn coverage_rejects_an_item_outside_the_universe() {
        assert_eq!(
            Coverage::new(vec![vec![0, 1], vec![2, 9]], 5).unwrap_err(),
            DivselError::CoverageItemOutOfRange {
                row: 1,
                item: 9,
                universe: 5
            }
        );
        // The boundary itself is out of range: ids are 0..universe.
        assert_eq!(
            Coverage::new(vec![vec![5]], 5).unwrap_err(),
            DivselError::CoverageItemOutOfRange {
                row: 0,
                item: 5,
                universe: 5
            }
        );
    }

    #[test]
    fn coverage_validate_catches_a_length_mismatch() {
        let pts = four_points();
        assert_eq!(coverage_fixture().validate(&pts), Ok(()));
        let short = Coverage::new(vec![vec![0], vec![1]], 5).expect("short coverage");
        assert_eq!(
            short.validate(&pts).unwrap_err(),
            DivselError::CoverageLength {
                expected: 4,
                got: 2
            }
        );
    }

    // ---- FacilityLocation -------------------------------------------------

    #[test]
    fn facility_location_marginals_on_a_hand_built_triangle() {
        let pts = triangle();
        let mut fl = FacilityLocation::new(&pts);
        // Diameter is 5, so sim = 1 - d/5:
        //   sim(0,1) = 0.4, sim(0,2) = 0.2, sim(1,2) = 0.0, sim(i,i) = 1.
        assert_eq!(fl.scale, 5.0);

        // g({0}) = 1 + 0.4 + 0.2
        assert_close(fl.marginal(0, &[], &pts), 1.6);
        fl.commit(0, &pts);

        // g({0,1}) = max(1,0.4) + max(0.4,1) + max(0.2,0) = 2.2, so the gain is 0.6.
        assert_close(fl.marginal(1, &[0], &pts), 0.6);
        fl.commit(1, &pts);

        // g({0,1,2}) = 3, so the gain is 0.8.
        assert_close(fl.marginal(2, &[0, 1], &pts), 0.8);

        // Re-adding a committed point gains exactly nothing.
        assert_eq!(fl.marginal(0, &[0, 1], &pts), 0.0);
        assert_eq!(fl.marginal(1, &[0, 1], &pts), 0.0);

        // reset rewinds to the empty selection, where g(empty) = 0.
        fl.reset();
        assert!(fl.best.iter().all(|b| *b == 0.0));
        assert_close(fl.marginal(1, &[], &pts), 1.4);
    }

    #[test]
    fn facility_location_first_marginal_of_any_point_is_at_least_one() {
        for metric in [Metric::Cosine, Metric::Euclidean] {
            let pts = Points::new(sample(30, 6, 0x5eed_0001), 6, metric).expect("sample points");
            let fl = FacilityLocation::new(&pts);
            for v in 0..pts.n() {
                let first = fl.marginal(v, &[], &pts);
                assert!(
                    first >= 1.0,
                    "sim(v,v) = 1, so the first marginal of {v} should be >= 1, got {first}"
                );
            }
        }
    }

    #[test]
    fn facility_location_marginals_stay_non_negative_after_commits() {
        for metric in [Metric::Cosine, Metric::Euclidean] {
            let pts = Points::new(sample(30, 6, 0x5eed_0002), 6, metric).expect("sample points");
            let mut fl = FacilityLocation::new(&pts);
            for &chosen in &[7usize, 2, 19, 11, 25] {
                fl.commit(chosen, &pts);
                for v in 0..pts.n() {
                    let gain = fl.marginal(v, &[], &pts);
                    assert!(gain >= 0.0, "marginal of {v} went negative: {gain}");
                }
                // A committed point is fully represented by itself.
                assert_eq!(fl.marginal(chosen, &[], &pts), 0.0);
            }
        }
    }

    #[test]
    fn facility_location_marginals_telescope_to_the_direct_definition() {
        for metric in [Metric::Cosine, Metric::Euclidean] {
            let pts = Points::new(sample(30, 6, 0x5eed_0003), 6, metric).expect("sample points");
            let mut fl = FacilityLocation::new(&pts);
            let mut running = 0.0f64;
            let mut selected: Vec<usize> = Vec::new();
            for &chosen in &[3usize, 17, 8, 21, 0, 29] {
                running += fl.marginal(chosen, &selected, &pts);
                fl.commit(chosen, &pts);
                selected.push(chosen);

                // g(S) straight from the definition, independent of the cache.
                let direct: f64 = (0..pts.n())
                    .map(|i| {
                        selected
                            .iter()
                            .map(|&j| fl.sim(i, j, &pts))
                            .fold(0.0f64, f64::max)
                    })
                    .sum();
                assert!(
                    (running - direct).abs() < 1e-9,
                    "accumulated marginals {running} != g(S) {direct} for {selected:?}"
                );
            }
        }
    }

    #[test]
    fn facility_location_is_submodular_on_a_random_thirty_point_set() {
        const N: usize = 30;
        for metric in [Metric::Cosine, Metric::Euclidean] {
            let pts = Points::new(sample(N, 6, 0x5eed_0004), 6, metric).expect("sample points");
            let mut rng = SplitMix64::new(0xabcd_ef01_2345_6789);
            let mut subset = FacilityLocation::new(&pts);
            let mut superset = FacilityLocation::new(&pts);
            let mut pool: Vec<usize> = (0..N).collect();
            let mut strict = 0usize;
            for trial in 0..200 {
                // Partial Fisher-Yates gives distinct indices, so v is outside T.
                for i in 0..N {
                    let j = i + rng.below(N - i);
                    pool.swap(i, j);
                }
                let t_len = 1 + rng.below(8);
                let s_len = rng.below(t_len); // strict subset: |S| < |T|
                let t = &pool[..t_len];
                let s = &pool[..s_len];
                let v = pool[t_len];

                subset.reset();
                for &x in s {
                    subset.commit(x, &pts);
                }
                superset.reset();
                for &x in t {
                    superset.commit(x, &pts);
                }
                let on_subset = subset.marginal(v, s, &pts);
                let on_superset = superset.marginal(v, t, &pts);
                if on_subset > on_superset + 1e-12 {
                    strict += 1;
                }
                assert!(
                    on_subset >= on_superset - 1e-9,
                    "trial {trial} ({metric:?}): g({v}|S) = {on_subset} < g({v}|T) = {on_superset} \
                     for S = {s:?}, T = {t:?}"
                );
            }
            // Guards the guard: an implementation whose `commit` never updated
            // `best` would make every marginal selection-independent and sail
            // through the inequality above. Measured here, 193 of the 200 cosine
            // trials and 186 of the euclidean ones are strict.
            assert!(
                strict >= 150,
                "only {strict} of 200 {metric:?} trials were strict; the check has gone blind"
            );
        }
    }

    #[test]
    fn facility_location_scale_matches_the_metric() {
        let data = sample(30, 6, 0x5eed_0005);
        let cosine = Points::new(data.clone(), 6, Metric::Cosine).expect("cosine points");
        assert_eq!(FacilityLocation::new(&cosine).scale, 1.0);

        let euclidean = Points::new(data, 6, Metric::Euclidean).expect("euclidean points");
        let d_max = euclidean.diameter().0;
        assert!(d_max > 0.0, "degenerate fixture");
        assert_eq!(
            FacilityLocation::new(&euclidean).scale.to_bits(),
            d_max.to_bits()
        );
    }

    #[test]
    fn facility_location_scale_falls_back_to_one_when_it_cannot_be_used() {
        for bad in [0.0f32, -2.5, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert_eq!(usable_scale(bad), 1.0, "scale {bad} should fall back to 1");
            assert_eq!(FacilityLocation::with_scale(3, bad).scale, 1.0);
        }
        let explicit = FacilityLocation::with_scale(3, 2.0);
        assert_eq!(explicit.scale, 2.0);
        assert_eq!(explicit.best.len(), 3);

        // Identical points have a zero diameter, so sim is identically 1.
        let identical =
            Points::new(vec![1.0, 2.0, 1.0, 2.0], 2, Metric::Euclidean).expect("identical points");
        let fl = FacilityLocation::new(&identical);
        assert_eq!(identical.diameter().0, 0.0);
        assert_eq!(fl.scale, 1.0);
        assert_eq!(fl.sim(0, 1, &identical), 1.0);
        assert_close(fl.marginal(0, &[], &identical), 2.0);
    }

    #[test]
    fn facility_location_validate_catches_a_size_mismatch() {
        let pts = triangle();
        assert_eq!(FacilityLocation::new(&pts).validate(&pts), Ok(()));
        assert_eq!(
            FacilityLocation::with_scale(5, 1.0)
                .validate(&pts)
                .unwrap_err(),
            DivselError::WeightsLength {
                expected: 3,
                got: 5
            }
        );
    }

    #[test]
    fn facility_location_is_not_linear() {
        let pts = triangle();
        assert!(!FacilityLocation::new(&pts).is_linear());
        assert!(!coverage_fixture().is_linear());
    }

    // ---- the boxed trait object -------------------------------------------

    #[test]
    fn boxed_utility_is_send_and_clones_independently() {
        fn assert_send<T: Send>() {}
        assert_send::<Box<dyn Utility>>();

        let pts = triangle();
        let boxed: Vec<Box<dyn Utility>> = vec![
            Box::new(Linear::uniform(pts.n())),
            Box::new(FacilityLocation::new(&pts)),
        ];
        let mut copies = boxed.clone();
        copies[1].commit(0, &pts);

        // Committing to the copy left the original at the empty selection.
        assert_close(boxed[1].marginal(1, &[], &pts), 1.4);
        assert_close(copies[1].marginal(1, &[0], &pts), 0.6);
    }
}
