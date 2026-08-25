//! `GreedyIndependentSet`, the subroutine GIST calls once per distance
//! threshold, with CELF lazy evaluation of the marginal gains.
//!
//! The paper (arXiv:2405.18754v3) writes it as, counting the `function` header
//! as line 1:
//!
//! ```text
//! 1: function GreedyIndependentSet(V, g, d, k)
//! 2:   Initialize S <- {}
//! 3:   for i = 1 to k do
//! 4:     Let C <- {v in V \ S : dist(v, S) >= d}
//! 5:     if C = {} then
//! 6:       return S                    > S is a maximal independent set of G_d(V)
//! 7:     Find t <- argmax_{v in C} g(v | S)
//! 8:     Update S <- S u {t}
//! 9:   return S
//! ```
//!
//! Two things about it are worth stating plainly, because they are easy to get
//! subtly wrong:
//!
//! * The selection rule maximizes the marginal gain of `g` **alone**. The
//!   diversity term never enters it; diversity is only the hard feasibility
//!   filter on line 4.
//! * With `d = 0` the filter admits everything, so the subroutine degenerates to
//!   classic Nemhauser-Wolsey-Fisher greedy -- which is exactly how GIST obtains
//!   its initial solution.

use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

use crate::points::Points;
use crate::utility::Utility;

/// Runs the paper's `GreedyIndependentSet(V, g, d, k)` over `pts`.
///
/// Returns the selected indices **in selection order**, at most `k` of them,
/// with every pair at distance `>= d`. The return is short only when the
/// candidate set runs dry (line 5), in which case the result is a maximal
/// independent set of `G_d(V)`.
///
/// # Tie-breaking
///
/// The paper leaves the `argmax` on line 7 unqualified. `divsel` breaks ties
/// towards the **lowest index**, always, on both evaluation paths -- a
/// `[divsel choice]` that makes every run reproducible. Gains are compared with
/// [`f64::total_cmp`], so a `NaN` from a misbehaving [`Utility`] cannot make the
/// comparison intransitive.
///
/// # Utility contract
///
/// `util` is expected to be in its reset state, but a dirty one is tolerated:
/// this function calls [`Utility::reset`] before it starts. On return, `util`
/// holds the state for the returned selection, so the caller can read `g(S)` out
/// of it (GIST needs exactly that) and is responsible for resetting it before
/// the next threshold.
///
/// [`Utility::validate`] is *not* called here; the GIST driver validates once,
/// up front. Passing a utility built for a different number of points will
/// therefore panic rather than return an error.
///
/// # Budget
///
/// `k == 0` returns empty. `k > pts.n()` is clamped to `pts.n()`.
///
/// # Lazy evaluation
///
/// When `util` reports [`Utility::is_linear`] the marginals are constants, so
/// CELF cannot skip a single evaluation -- every cached gain is already exact
/// and the heap only adds bookkeeping. This function therefore takes a plain
/// scan for linear utilities and builds the heap only when there is something to
/// be lazy about. For a submodular `util` both paths return the same vector (see
/// [`greedy_independent_set_naive`], the oracle that equality is tested against).
/// A utility that is **not** submodular voids that equality: a cached gain may
/// then under-estimate the true one and the lazy path can hand back a point the
/// plain scan would not have picked -- silently, since nothing here can detect
/// it. The [`Utility`] contract requires submodularity for exactly this reason.
///
/// # Panics
///
/// Panics if `util` is not sized for `pts` — a per-point table shorter than
/// `pts.n()`, or a [`crate::FacilityLocation`] cache built for another point
/// count. Both panic in release as well as in debug; see [`Utility::marginal`].
pub fn greedy_independent_set(
    pts: &Points<'_>,
    util: &mut dyn Utility,
    d: f32,
    k: usize,
) -> Vec<usize> {
    if util.is_linear() {
        run_scan(pts, util, d, k)
    } else {
        run_celf(pts, util, d, k)
    }
}

/// The unconditional-re-evaluation reference: identical semantics to
/// [`greedy_independent_set`], but never lazy, even for a submodular utility.
///
/// Kept as the oracle the CELF path is tested against. It is `pub` so the
/// crate's own integration tests can reach it, and `#[doc(hidden)]` because it
/// is not part of the supported API.
#[doc(hidden)]
pub fn greedy_independent_set_naive(
    pts: &Points<'_>,
    util: &mut dyn Utility,
    d: f32,
    k: usize,
) -> Vec<usize> {
    run_scan(pts, util, d, k)
}

/// The state both paths share: the running selection and the incremental form of
/// the paper's candidate set `C` (line 4).
struct Frontier {
    /// `nearest[v]` is `dist(v, S)`, so `f32::INFINITY` while `S` is empty --
    /// the paper's `dist(u, {}) = infinity` convention -- and
    /// `f32::NEG_INFINITY` once `v` itself has been selected.
    nearest: Vec<f32>,
    /// `chosen[v]` is whether `v` is in `S`.
    ///
    /// Redundant with the sentinel in `nearest` on purpose: once `v` is taken,
    /// `nearest[v]` is `NEG_INFINITY`, which fails `>= d` for every threshold
    /// the driver passes (`d >= 0`), so either guard alone enforces the
    /// membership half of line 4 (`v in V \ S`). Both are kept because that
    /// half *must* hold and neither guard is where a reader expects it: at
    /// `d == 0` the distance half of line 4 would admit a selected point right
    /// back in (`dist(v, S) == 0 >= 0`), and with a linear utility the argmax
    /// would then return the same point every round.
    chosen: Vec<bool>,
    /// The selection, in selection order.
    selected: Vec<usize>,
}

impl Frontier {
    /// An empty selection over `n` points.
    fn new(n: usize, k: usize) -> Self {
        Self {
            nearest: vec![f32::INFINITY; n],
            chosen: vec![false; n],
            selected: Vec::with_capacity(k),
        }
    }

    /// The paper's lines 8 and 4: commit `t`, then refresh `dist(v, S)` for every
    /// point still outside `S` in one `O(n)` pass, since
    /// `dist(v, S u {t}) = min(dist(v, S), dist(v, t))`.
    fn take(&mut self, pts: &Points<'_>, util: &mut dyn Utility, t: usize) {
        self.selected.push(t);
        util.commit(t, pts);
        self.chosen[t] = true;
        self.nearest[t] = f32::NEG_INFINITY;
        for (v, near) in self.nearest.iter_mut().enumerate() {
            if !self.chosen[v] {
                *near = near.min(pts.dist(v, t));
            }
        }
    }
}

/// `argmax_{v in C} g(v | S)` by a full scan of the candidate set (line 7).
///
/// Indices are visited in ascending order and the incumbent is replaced only on
/// a **strictly** greater gain, so the lowest index wins every tie.
fn scan_argmax(pts: &Points<'_>, util: &dyn Utility, d: f32, frontier: &Frontier) -> Option<usize> {
    let mut best: Option<(f64, usize)> = None;
    for (v, &near) in frontier.nearest.iter().enumerate() {
        // Line 4: v is in V \ S, and dist(v, S) >= d -- `>=`, not `>`, so two
        // points exactly d apart are feasible.
        if frontier.chosen[v] || near < d {
            continue;
        }
        let gain = util.marginal(v, &frontier.selected, pts);
        let better = match best {
            None => true,
            Some((incumbent, _)) => gain.total_cmp(&incumbent) == Ordering::Greater,
        };
        if better {
            best = Some((gain, v));
        }
    }
    best.map(|(_, v)| v)
}

/// The paper's loop with every candidate re-evaluated every round.
fn run_scan(pts: &Points<'_>, util: &mut dyn Utility, d: f32, k: usize) -> Vec<usize> {
    util.reset();
    let k = k.min(pts.n());
    let mut frontier = Frontier::new(pts.n(), k);
    while frontier.selected.len() < k {
        // Line 5: an empty candidate set ends the round early.
        let Some(t) = scan_argmax(pts, util, d, &frontier) else {
            break;
        };
        frontier.take(pts, util, t);
    }
    frontier.selected
}

/// One cached marginal gain in the CELF heap.
///
/// Ordered by `gain` first (via [`f64::total_cmp`], since `f64` is not [`Ord`]),
/// then by `Reverse(index)`, so that the entry that pops first is the one with
/// the largest gain and, among equal gains, the **lowest** index -- the same tie
/// rule the plain scan applies.
///
/// `last_eval_round` is deliberately outside the ordering: it is bookkeeping, not
/// priority. [`PartialEq`] is derived from [`Ord`] to keep the two consistent.
struct Entry {
    gain: f64,
    index: Reverse<usize>,
    last_eval_round: usize,
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.gain
            .total_cmp(&other.gain)
            .then_with(|| self.index.cmp(&other.index))
    }
}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Entry {}

/// The same loop, with Leskovec's CELF lazy evaluation of line 7.
///
/// # Why this is exact
///
/// Submodularity says a cached gain can only be an over-estimate: once `S` has
/// grown, `g(v | S)` is no larger than the value cached for a smaller `S`. So if
/// the top of the heap was evaluated against the *current* `S`, every entry
/// below it has a true gain no larger than its own cached gain, which is no
/// larger than the top's -- the top is the argmax, and no other candidate needs
/// touching.
///
/// The tie half of that argument is why a re-evaluated entry is always pushed
/// back rather than accepted in place: an entry sitting below the top may hold
/// an equal gain with a lower index, and only re-heapifying lets it surface. The
/// ordering on [`Entry`] then hands ties to the lowest index, exactly as
/// [`scan_argmax`] does.
///
/// A [`Utility`] that is not submodular voids this equivalence -- the
/// re-evaluated gain could exceed the cached one and the heap would hand back a
/// point the plain scan would not have picked. The trait's own documentation
/// already requires submodularity; this is one of the places where breaking it
/// is silently wrong rather than loudly wrong.
fn run_celf(pts: &Points<'_>, util: &mut dyn Utility, d: f32, k: usize) -> Vec<usize> {
    util.reset();
    let k = k.min(pts.n());
    let mut frontier = Frontier::new(pts.n(), k);
    if k == 0 {
        // Nothing to select, and the heap would cost n evaluations to build.
        return frontier.selected;
    }

    // Round 0's candidate set is everything (dist(v, {}) = infinity), evaluated
    // once. Entries whose `nearest` later falls below `d` are not removed here;
    // they are discarded lazily when they reach the top.
    let mut heap: BinaryHeap<Entry> = (0..pts.n())
        .map(|v| Entry {
            gain: util.marginal(v, &frontier.selected, pts),
            index: Reverse(v),
            last_eval_round: 0,
        })
        .collect();

    'rounds: while frontier.selected.len() < k {
        let round = frontier.selected.len();
        let t = loop {
            // An empty heap means an empty candidate set: the paper's line 5.
            let Some(mut entry) = heap.pop() else {
                break 'rounds;
            };
            let index = entry.index.0;
            // `nearest` only ever decreases and `chosen` never reverts, so an
            // entry that fails the filter now can never pass it again. The
            // `chosen` half is unreachable for any `d >= 0` -- a taken point's
            // `nearest` is `NEG_INFINITY` -- and kept as the second, independent
            // guard described on `Frontier::chosen`.
            if frontier.chosen[index] || frontier.nearest[index] < d {
                continue;
            }
            if entry.last_eval_round < round {
                entry.gain = util.marginal(index, &frontier.selected, pts);
                entry.last_eval_round = round;
                heap.push(entry);
                continue;
            }
            break index;
        };
        frontier.take(pts, util, t);
    }
    frontier.selected
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::{greedy_independent_set, greedy_independent_set_naive};
    use crate::metric::Metric;
    use crate::points::Points;
    use crate::testutil::SplitMix64;
    use crate::utility::{Coverage, FacilityLocation, Linear, Utility};

    /// Six points on a line at `x = 0, 1, 2, 3, 4, 8`.
    ///
    /// The diameter is exactly `8`, so `FacilityLocation::new` uses a scale of
    /// `8` and every similarity `1 - |dx| / 8` is an exact multiple of `1/8`.
    /// Every marginal is therefore a sum of exact eighths, which makes the ties
    /// below exact ties rather than near-misses -- the tie-breaking rule is
    /// tested, not floating-point luck.
    fn line_six() -> Points<'static> {
        Points::new(vec![0.0, 1.0, 2.0, 3.0, 4.0, 8.0], 1, Metric::Euclidean).expect("line fixture")
    }

    /// Weights over [`line_six`] whose greedy order is unambiguous apart from one
    /// exact tie: `5.0` at index 1 and index 2.
    fn line_six_weights() -> Vec<f64> {
        vec![3.0, 5.0, 5.0, 1.0, 4.0, 2.0]
    }

    /// `n` random sparse subsets of `0..universe`, 1 to 4 items each.
    ///
    /// A small universe makes collisions -- and therefore exactly equal marginals
    /// -- common, which is what stresses the tie rule.
    fn random_coverage(rng: &mut SplitMix64, n: usize, universe: usize) -> Coverage {
        let mut sets = Vec::with_capacity(n);
        for _ in 0..n {
            let size = 1 + rng.below(4);
            let mut items = Vec::with_capacity(size);
            for _ in 0..size {
                items.push(rng.below(universe) as u32);
            }
            sets.push(items);
        }
        Coverage::new(sets, universe).expect("every item id is below the universe")
    }

    /// A [`Utility`] that counts `marginal` calls and can misreport
    /// [`Utility::is_linear`], so one instance can be driven down either path.
    struct Counting {
        inner: Linear,
        calls: AtomicUsize,
        claims_linear: bool,
    }

    impl Counting {
        fn new(weights: Vec<f64>, claims_linear: bool) -> Self {
            Self {
                inner: Linear::new(weights),
                calls: AtomicUsize::new(0),
                claims_linear,
            }
        }

        fn calls(&self) -> usize {
            self.calls.load(Ordering::Relaxed)
        }
    }

    impl Utility for Counting {
        fn marginal(&self, v: usize, selected: &[usize], pts: &Points<'_>) -> f64 {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.inner.marginal(v, selected, pts)
        }

        fn commit(&mut self, v: usize, pts: &Points<'_>) {
            self.inner.commit(v, pts);
        }

        fn reset(&mut self) {
            self.inner.reset();
        }

        fn is_linear(&self) -> bool {
            self.claims_linear
        }

        fn boxed_clone(&self) -> Box<dyn Utility> {
            Box::new(Self {
                inner: self.inner.clone(),
                calls: AtomicUsize::new(self.calls()),
                claims_linear: self.claims_linear,
            })
        }
    }

    // ---- (a) d = 0 is classic greedy --------------------------------------

    #[test]
    fn d_zero_reproduces_classic_greedy_on_a_linear_instance() {
        let pts = line_six();
        let mut util = Linear::new(line_six_weights());

        // Weights 3, 5, 5, 1, 4, 2: descending order is 5 (indices 1 and 2, an
        // exact tie broken towards the lower index), then 4, 3, 2, 1.
        let want = vec![1, 2, 4, 0, 5, 3];
        assert_eq!(greedy_independent_set(&pts, &mut util, 0.0, 6), want);
        assert_eq!(greedy_independent_set(&pts, &mut util, 0.0, 3), &want[..3]);

        // For a linear utility `greedy_independent_set_naive` *is* the plain
        // scan the call above took, so this line only pins that the two entry
        // points agree on the linear fast path -- it is not a cross-path check.
        assert_eq!(greedy_independent_set_naive(&pts, &mut util, 0.0, 6), want);
        // The real cross-path check: the same weights hidden behind a utility
        // that does not advertise linearity, which forces the CELF heap.
        let mut lying = Counting::new(line_six_weights(), false);
        assert!(!lying.is_linear());
        assert_eq!(greedy_independent_set(&pts, &mut lying, 0.0, 6), want);
    }

    #[test]
    fn d_zero_facility_location_picks_are_hand_checked() {
        let pts = line_six();
        assert_eq!(pts.diameter().0, 8.0);

        // Round 0, g({v}) = sum_i (1 - |x_i - x_v| / 8):
        //   v=0: 6 - 18/8 = 3.75    v=3: 6 - 12/8 = 4.5
        //   v=1: 6 - 14/8 = 4.25    v=4: 6 - 14/8 = 4.25
        //   v=2: 6 - 12/8 = 4.5     v=5: 6 - 30/8 = 2.25
        let cold = FacilityLocation::new(&pts);
        let round0: Vec<f64> = (0..6).map(|v| cold.marginal(v, &[], &pts)).collect();
        assert_eq!(round0, vec![3.75, 4.25, 4.5, 4.5, 4.25, 2.25]);
        // 2 and 3 tie exactly, so the lower index is taken.

        // Round 1, against best = sim(., 2) = [.75, .875, 1, .875, .75, .25]:
        //   v=0: .25   v=1: .25   v=3: .375   v=4: .5   v=5: .75  -> 5 wins outright.
        // Round 2, against best = [.75, .875, 1, .875, .75, 1]:
        //   v=0: .25   v=1: .25   v=3: .25    v=4: .25            -> a four-way
        //   exact tie, so index 0 wins.
        let mut fl = FacilityLocation::new(&pts);
        let s = greedy_independent_set(&pts, &mut fl, 0.0, 3);
        assert_eq!(s, vec![2, 5, 0]);
        assert_eq!(greedy_independent_set_naive(&pts, &mut fl, 0.0, 3), s);

        // On return the utility holds the state for S, ready for the driver:
        // the round-3 marginals against best = [1, .875, 1, .875, .75, 1].
        let after: Vec<f64> = (0..6).map(|v| fl.marginal(v, &s, &pts)).collect();
        assert_eq!(after, vec![0.0, 0.125, 0.0, 0.25, 0.25, 0.0]);
    }

    // ---- (b) a threshold above the diameter -------------------------------

    #[test]
    fn a_threshold_above_the_diameter_returns_exactly_one_point() {
        let pts = line_six();
        let d = pts.diameter().0 + 1.0;

        let mut linear = Linear::new(line_six_weights());
        assert_eq!(greedy_independent_set(&pts, &mut linear, d, 3), vec![1]);
        let mut fl = FacilityLocation::new(&pts);
        assert_eq!(greedy_independent_set(&pts, &mut fl, d, 3), vec![2]);

        // Same on a random cloud, where the first pick is not hand-checked but
        // the cardinality still is.
        let mut rng = SplitMix64(0x0001_5e70_0000_000b);
        for metric in [Metric::Cosine, Metric::Euclidean] {
            let data = rng.gaussian_points(30, 5);
            let pts = Points::new(data, 5, metric).expect("random points");
            let d = pts.diameter().0 + 1.0;
            let mut fl = FacilityLocation::new(&pts);
            assert_eq!(greedy_independent_set(&pts, &mut fl, d, 3).len(), 1);
            let mut linear = Linear::new(rng.uniform_weights(30));
            assert_eq!(greedy_independent_set(&pts, &mut linear, d, 3).len(), 1);
        }
    }

    // ---- (c) the result is always an independent set -----------------------

    #[test]
    fn every_selection_is_an_independent_set() {
        const N: usize = 40;
        const DIM: usize = 4;
        const K: usize = 6;

        let mut rng = SplitMix64(0x0001_5e70_0000_000c);
        let mut nontrivial = 0usize;
        for trial in 0..50 {
            let data = rng.gaussian_points(N, DIM);
            let weights = rng.uniform_weights(N);
            let pts = Points::new(data, DIM, Metric::Euclidean).expect("random points");
            let d = rng.next_f32() * pts.diameter().0;

            let mut fl = FacilityLocation::new(&pts);
            let mut linear = Linear::new(weights);
            let utils: [(&str, &mut dyn Utility); 2] =
                [("facility-location", &mut fl), ("linear", &mut linear)];
            for (name, util) in utils {
                let s = greedy_independent_set(&pts, util, d, K);
                assert!(
                    s.len() <= K,
                    "trial {trial} ({name}): |S| = {} exceeds k = {K}",
                    s.len()
                );
                for (pos, &a) in s.iter().enumerate() {
                    for &b in &s[pos + 1..] {
                        assert_ne!(a, b, "trial {trial} ({name}): {a} selected twice");
                        assert!(
                            pts.dist(a, b) >= d,
                            "trial {trial} ({name}): dist({a}, {b}) = {} < d = {d}",
                            pts.dist(a, b)
                        );
                    }
                }
                if s.len() >= 2 {
                    nontrivial += 1;
                }
            }
        }
        // Guards the guard: a function that always returned a single point would
        // satisfy every assertion above. Measured here: 67 of the 100 runs.
        assert!(
            nontrivial >= 50,
            "only {nontrivial} of 100 runs selected more than one point"
        );
    }

    // ---- (d) CELF is the naive scan, exactly -------------------------------

    #[test]
    fn celf_and_the_naive_scan_agree_on_facility_location() {
        const N: usize = 60;
        const DIM: usize = 8;
        const K: usize = 8;

        let mut rng = SplitMix64(0x0001_5e70_0000_000d);
        let mut selected = 0usize;
        for trial in 0..100 {
            let metric = if trial % 2 == 0 {
                Metric::Cosine
            } else {
                Metric::Euclidean
            };
            let data = rng.gaussian_points(N, DIM);
            let pts = Points::new(data, DIM, metric).expect("random points");
            let d = rng.next_f32() * pts.diameter().0;

            // One utility for both calls: each entry point resets it first.
            let mut util = FacilityLocation::new(&pts);
            assert!(!util.is_linear(), "this instance must take the CELF path");
            let lazy = greedy_independent_set(&pts, &mut util, d, K);
            let naive = greedy_independent_set_naive(&pts, &mut util, d, K);
            assert_eq!(lazy, naive, "trial {trial} ({metric:?}), d = {d}");
            selected += lazy.len();
        }
        // Out of a possible 800; measured here: 494. A threshold that always
        // emptied C after the first pick would make the comparison above
        // trivially true.
        assert!(
            selected >= 400,
            "only {selected} points selected over 100 instances; the comparison \
             has gone vacuous"
        );
    }

    #[test]
    fn celf_and_the_naive_scan_agree_on_coverage() {
        const N: usize = 60;
        const DIM: usize = 8;
        const K: usize = 8;
        const UNIVERSE: usize = 20;

        let mut rng = SplitMix64(0x0001_5e70_0000_000e);
        let mut selected = 0usize;
        for trial in 0..100 {
            let metric = if trial % 2 == 0 {
                Metric::Cosine
            } else {
                Metric::Euclidean
            };
            let data = rng.gaussian_points(N, DIM);
            let pts = Points::new(data, DIM, metric).expect("random points");
            let d = rng.next_f32() * pts.diameter().0;

            let mut util = random_coverage(&mut rng, N, UNIVERSE);
            assert!(!util.is_linear(), "this instance must take the CELF path");
            let lazy = greedy_independent_set(&pts, &mut util, d, K);
            let naive = greedy_independent_set_naive(&pts, &mut util, d, K);
            assert_eq!(lazy, naive, "trial {trial} ({metric:?}), d = {d}");
            selected += lazy.len();
        }
        // Out of a possible 800; measured here: 524. A threshold that always
        // emptied C after the first pick would make the comparison above
        // trivially true.
        assert!(
            selected >= 400,
            "only {selected} points selected over 100 instances; the comparison \
             has gone vacuous"
        );
    }

    // ---- (e) the budget ----------------------------------------------------

    #[test]
    fn k_zero_is_empty_and_k_above_n_is_clamped() {
        let pts = line_six();

        let mut linear = Linear::new(line_six_weights());
        assert!(greedy_independent_set(&pts, &mut linear, 0.0, 0).is_empty());
        assert!(greedy_independent_set_naive(&pts, &mut linear, 0.0, 0).is_empty());
        assert_eq!(
            greedy_independent_set(&pts, &mut linear, 0.0, 99),
            vec![1, 2, 4, 0, 5, 3]
        );

        // The facility-location order over the whole line, hand-checked the same
        // way as the first three picks.
        let mut fl = FacilityLocation::new(&pts);
        assert!(greedy_independent_set(&pts, &mut fl, 0.0, 0).is_empty());
        assert_eq!(
            greedy_independent_set(&pts, &mut fl, 0.0, 99),
            vec![2, 5, 0, 3, 1, 4]
        );
    }

    // ---- (f) a linear utility never builds the heap ------------------------

    #[test]
    fn a_linear_utility_skips_the_celf_heap() {
        const N: usize = 10;
        // Distinct weights, so there is nothing for the tie rule to do here.
        let weights = vec![3.0, 9.0, 1.0, 7.0, 5.0, 2.0, 8.0, 4.0, 6.0, 10.0];
        let mut rng = SplitMix64(0x0001_5e70_0000_000f);
        let pts =
            Points::new(rng.gaussian_points(N, 3), 3, Metric::Euclidean).expect("random points");

        let mut scan = Counting::new(weights.clone(), true);
        let by_scan = greedy_independent_set(&pts, &mut scan, 0.0, 3);
        assert_eq!(by_scan, vec![9, 1, 6]);
        assert_eq!(
            scan.calls(),
            27,
            "the linear path evaluates every feasible candidate once per round: \
             10 + 9 + 8"
        );

        // Same weights, same points, but now the utility hides its linearity, so
        // the heap is built. CELF must agree, and must not cost more.
        let mut lazy = Counting::new(weights, false);
        let by_celf = greedy_independent_set(&pts, &mut lazy, 0.0, 3);
        assert_eq!(by_celf, by_scan);
        assert!(
            lazy.calls() >= N,
            "the initial heap alone is n evaluations, got {}",
            lazy.calls()
        );
        assert!(
            lazy.calls() <= 27,
            "CELF evaluated {} marginals, more than the plain scan's 27",
            lazy.calls()
        );
    }

    /// Line 4's filter is **non-strict**, on both evaluation paths: a candidate
    /// exactly `d` away from the selection is feasible.
    ///
    /// `x = 0, 1, 5` with coverage sets `{0,1,2}`, `{3,4}`, `{5}`. The first pick
    /// is point 0 (marginal 3); point 1 then sits at exactly `d = 1` and carries
    /// the larger marginal, so it is taken if and only if `dist(v, S) >= d`
    /// admits it -- a strict `>` reports `[0, 2]` instead. The CELF branch is the
    /// one every submodular utility takes, and the golden fixture's only rule-15
    /// case is linear, so it never reaches this filter.
    #[test]
    fn line_four_admits_a_candidate_exactly_d_away_on_both_paths() {
        let pts = Points::new(vec![0.0, 1.0, 5.0], 1, Metric::Euclidean).expect("line fixture");
        assert_eq!(pts.dist(0, 1), 1.0);
        let sets = vec![vec![0, 1, 2], vec![3, 4], vec![5]];
        let mut lazy = Coverage::new(sets.clone(), 6).expect("coverage sets");
        let mut plain = Coverage::new(sets, 6).expect("coverage sets");
        assert!(
            !lazy.is_linear(),
            "Coverage has to take the CELF path for this to test it"
        );

        assert_eq!(greedy_independent_set(&pts, &mut lazy, 1.0, 2), vec![0, 1]);
        assert_eq!(
            greedy_independent_set_naive(&pts, &mut plain, 1.0, 2),
            vec![0, 1]
        );

        // One ulp above the boundary the same instance drops the near point,
        // which is what makes the two assertions above a boundary test rather
        // than a restatement of the `d = 0` run.
        let above = 1.0 + f32::EPSILON;
        assert_eq!(
            greedy_independent_set(&pts, &mut lazy, above, 2),
            vec![0, 2]
        );
        assert_eq!(
            greedy_independent_set_naive(&pts, &mut plain, above, 2),
            vec![0, 2]
        );
    }

    /// The CELF path opens with [`Utility::reset`], so a dirty utility is
    /// tolerated -- the documented contract, and the one the sweep relies on when
    /// a worker reuses its clone across thresholds.
    #[test]
    fn the_celf_path_resets_a_dirty_utility_on_entry() {
        let pts = Points::new(vec![0.0, 1.0, 5.0], 1, Metric::Euclidean).expect("line fixture");
        let sets = vec![vec![0, 1, 2], vec![3, 4], vec![5]];
        let mut clean = Coverage::new(sets.clone(), 6).expect("coverage sets");
        let expected = greedy_independent_set(&pts, &mut clean, 0.0, 2);
        assert_eq!(expected, vec![0, 1]);

        let mut dirty = Coverage::new(sets, 6).expect("coverage sets");
        dirty.commit(0, &pts);
        // Without the reset point 0's marginal is 0 and the run reports [1, 2].
        assert_eq!(dirty.marginal(0, &[], &pts), 0.0);
        assert_eq!(greedy_independent_set(&pts, &mut dirty, 0.0, 2), expected);
    }
}
