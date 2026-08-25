//! Brute-force oracle: does GIST actually meet its approximation guarantee?
//!
//! GIST's selling point is not that it returns a good set — every heuristic
//! claims that — it is that the set it returns is provably within a constant
//! factor of the exhaustive optimum. This file is the evidence. For 500 seeded
//! random instances small enough to solve exactly (`n <= 16`, `2 <= k <= 4`) it
//! computes `OPT = max_{|S| <= k} f(S)` by enumerating **every** subset, runs
//! [`gist`] on the same instance, and checks the ratio against the paper's three
//! bounds (arXiv:2405.18754v3):
//!
//! * Theorem 3.1, monotone submodular `g`: `f(GIST) >= (1/2 - eps) * OPT`.
//! * Theorem 3.3, linear `g`: `f(GIST) >= (2/3 - eps) * OPT`, and exactly
//!   `(2/3) * OPT` with the exhaustive threshold set.
//! * Section 2.1's warm-up baseline, on every instance regardless of utility or
//!   threshold set: `f(GIST) >= (e - 1)/(2e - 1) * OPT ~= 0.387 * OPT`.
//!
//! The oracle evaluates `f` through the library's own [`eval_g`] and [`div`], so
//! the oracle and the driver cannot drift apart about *what* is being maximized;
//! the sanity group then closes the loop by checking that `GistResult::f_value`
//! is reproducible from `GistResult::selected`.
//!
//! The metric is [`Metric::Euclidean`] throughout the three theorem groups: the
//! guarantee has a metric precondition and Euclidean distance satisfies it. Raw
//! cosine distance does not (only its angular form does), so the 20 Cosine
//! instances live under the warm-up regression group alone and are labelled as
//! such — they are a "did it get worse" tripwire, not a claim about the theorem.
//!
//! **If one of these assertions ever fails, the implementation is wrong.** The
//! bound is the contract: record the instance, do not relax the constant.

use std::sync::OnceLock;

use divsel::testutil::SplitMix64;
use divsel::{
    div, eval_g, gist, DiameterMode, FacilityLocation, GistConfig, GistResult, Linear, Metric,
    Points, Utility,
};

// ---- instance distribution ------------------------------------------------

/// Number of seeded Euclidean instances the theorem groups run over.
const INSTANCES: usize = 500;
/// Instance `i` of that set is drawn from `SplitMix64(EUCLIDEAN_SEED_BASE + i)`.
const EUCLIDEAN_SEED_BASE: u64 = 1000;
/// The extra Cosine instances (warm-up regression group only) start here.
const COSINE_SEED_BASE: u64 = 2000;
/// How many of those there are.
const COSINE_INSTANCES: usize = 20;
/// The sweep accuracy every instance is run with.
const EPS: f32 = 0.1;
/// The three diversity weights, cycled across instances.
const LAMBDAS: [f64; 3] = [0.5, 1.0, 5.0];

// ---- the bounds under test ------------------------------------------------

/// Theorem 3.1, monotone submodular `g`, at `eps = 0.1`.
const SUBMODULAR_BOUND: f64 = 0.5 - 0.1;
/// Theorem 3.3, linear `g`, geometric threshold set, at `eps = 0.1`.
const LINEAR_BOUND: f64 = 2.0 / 3.0 - 0.1;
/// Theorem 3.3, linear `g`, exhaustive threshold set: the exact `2/3`, no `eps`.
const LINEAR_EXACT_BOUND: f64 = 2.0 / 3.0;
/// Section 2.1's warm-up baseline, `(e - 1)/(2e - 1) ~= 0.387`.
const WARMUP_BOUND: f64 = (std::f64::consts::E - 1.0) / (2.0 * std::f64::consts::E - 1.0);

// ---- instances ------------------------------------------------------------

/// Which utility an instance carries. `Linear` is modular, so Theorem 3.3
/// applies to it; `FacilityLocation` is strictly submodular, so only Theorem 3.1
/// does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum UtilKind {
    Linear,
    FacilityLocation,
}

impl UtilKind {
    fn label(self) -> &'static str {
        match self {
            UtilKind::Linear => "Linear",
            UtilKind::FacilityLocation => "FacilityLocation",
        }
    }
}

/// One instance plus its exhaustively computed optimum.
struct Case {
    seed: u64,
    n: usize,
    k: usize,
    dim: usize,
    lambda: f64,
    metric: Metric,
    kind: UtilKind,
    pts: Points<'static>,
    /// Kept so the `Linear` utility can be rebuilt identically for every run.
    weights: Vec<f64>,
    /// `max_{|S| <= min(k, n)} f(S)`, from [`brute_force_opt`].
    opt: f64,
    /// One maximizer of `opt`. Ties are not resolved — only the value is
    /// asserted — but the set makes a failure reproducible by hand.
    opt_set: Vec<usize>,
}

impl Case {
    /// A fresh utility in the empty-selection state, identical on every call.
    fn utility(&self) -> Box<dyn Utility> {
        match self.kind {
            UtilKind::Linear => Box::new(Linear::new(self.weights.clone())),
            UtilKind::FacilityLocation => Box::new(FacilityLocation::new(&self.pts)),
        }
    }

    /// Runs GIST on this instance with the geometric (`false`) or exhaustive
    /// (`true`) threshold set.
    fn run(&self, exhaustive: bool) -> GistResult {
        let mut util = self.utility();
        let cfg = GistConfig {
            k: self.k,
            lambda: self.lambda,
            eps: EPS,
            exhaustive_thresholds: exhaustive,
            diameter: DiameterMode::Exact,
        };
        gist(&self.pts, util.as_mut(), &cfg).expect("gist rejected a valid instance")
    }

    /// Every parameter needed to reproduce a failure, on one line.
    fn describe(&self, exhaustive: bool, out: &GistResult) -> String {
        format!(
            "seed={} n={} k={} dim={} lambda={} metric={:?} utility={} thresholds={} \
             f_gist={:.12} opt={:.12} ratio={:.12} gist_set={:?} opt_set={:?} stage={:?}",
            self.seed,
            self.n,
            self.k,
            self.dim,
            self.lambda,
            self.metric,
            self.kind.label(),
            threshold_label(exhaustive),
            out.f_value,
            self.opt,
            ratio(out.f_value, self.opt),
            out.selected,
            self.opt_set,
            out.stage,
        )
    }
}

fn threshold_label(exhaustive: bool) -> &'static str {
    if exhaustive {
        "exhaustive"
    } else {
        "geometric"
    }
}

/// Draws instance `index` from `SplitMix64(seed)` and solves it exactly.
///
/// The draw order is fixed — `n`, `k`, `dim`, coordinates, weights — so a given
/// seed always yields the same instance. `index` (not the seed) drives the two
/// deterministic cycles: the utility alternates `Linear` / `FacilityLocation`,
/// and `lambda` cycles through `LAMBDAS`.
///
/// `k = 1` is left out of the draw on purpose: the diametrical pair and the
/// sweep are both no-ops there, so such an instance would test the bound
/// against nothing but line 2.
fn build_case(index: usize, seed: u64, metric: Metric) -> Case {
    let mut rng = SplitMix64(seed);
    let n = 4 + rng.below(13); // n in [4, 16]
    let k = (2 + rng.below(3)).min(n); // k in [2, 4]; n >= 4, so `min` never binds
    let dim = if rng.below(2) == 0 { 2 } else { 8 };
    let data = rng.gaussian_points(n, dim);
    let weights = rng.uniform_weights(n);
    let pts = Points::new(data, dim, metric).expect("gaussian coordinates are valid points");

    let kind = if index % 2 == 0 {
        UtilKind::Linear
    } else {
        UtilKind::FacilityLocation
    };
    let lambda = LAMBDAS[index % LAMBDAS.len()];

    let mut case = Case {
        seed,
        n,
        k,
        dim,
        lambda,
        metric,
        kind,
        pts,
        weights,
        opt: 0.0,
        opt_set: Vec::new(),
    };
    let mut util = case.utility();
    let (opt, opt_set) = brute_force_opt(&case.pts, util.as_mut(), lambda, k);
    case.opt = opt;
    case.opt_set = opt_set;
    case
}

// ---- the oracle -----------------------------------------------------------

/// `f(S) = g(S) + lambda * div(S)`, the objective GIST maximizes.
///
/// Evaluated through the library's own [`eval_g`] and [`div`] on purpose: an
/// oracle carrying its own reimplementation of `f` would be testing agreement
/// between two transcriptions of the paper, not the algorithm. `div` of a set of
/// size `<= 1` is the diameter of the whole point set, per the paper's Sec. 2
/// convention, which is what makes `div` monotone decreasing in `S`.
fn f_of(pts: &Points<'_>, util: &mut dyn Utility, lambda: f64, s: &[usize]) -> f64 {
    eval_g(util, s, pts) + lambda * f64::from(div(pts, s))
}

/// `OPT = max_{|S| <= min(k, n)} f(S)`, by enumerating every such subset.
///
/// The cardinality constraint is `|S| <= k`, **not** `|S| = k`, so sizes `0` and
/// `1` are enumerated too. They are dominated on essentially every instance once
/// `n >= 2`, but leaving them out would be quietly solving a different problem
/// than the one GIST is constrained by. For `n <= 16, k <= 4` the count is at
/// most `sum_{j<=4} C(16,j) = 2517`, which the enumeration asserts it hit.
///
/// Ties are not broken; only the returned value is asserted anywhere.
fn brute_force_opt(
    pts: &Points<'_>,
    util: &mut dyn Utility,
    lambda: f64,
    k: usize,
) -> (f64, Vec<usize>) {
    let n = pts.n();
    assert!(
        n <= 16,
        "the oracle is only affordable for n <= 16, got {n}"
    );
    let budget = k.min(n);

    let mut best_value = f64::NEG_INFINITY;
    let mut best_set: Vec<usize> = Vec::new();
    let mut subset: Vec<usize> = Vec::with_capacity(budget);
    let mut enumerated = 0usize;

    for mask in 0u32..(1u32 << n) {
        if mask.count_ones() as usize > budget {
            continue;
        }
        subset.clear();
        subset.extend((0..n).filter(|&v| (mask >> v) & 1 == 1));
        enumerated += 1;

        let value = f_of(pts, util, lambda, &subset);
        if value > best_value {
            best_value = value;
            best_set.clone_from(&subset);
        }
    }

    assert_eq!(
        enumerated,
        subsets_up_to(n, budget),
        "the enumeration missed subsets for n={n}, k={budget}"
    );
    (best_value, best_set)
}

/// `sum_{j=0..=k} C(n, j)`, the number of subsets the oracle must visit.
fn subsets_up_to(n: usize, k: usize) -> usize {
    (0..=k).map(|j| binomial(n, j)).sum()
}

/// `C(n, k)`, multiplied and divided in an order that stays exact in integers:
/// after step `i` the accumulator holds `C(n, i+1)`, and `C(n, i) * (n - i)` is
/// always divisible by `i + 1`.
fn binomial(n: usize, k: usize) -> usize {
    if k > n {
        return 0;
    }
    let mut out = 1usize;
    for i in 0..k {
        out = out * (n - i) / (i + 1);
    }
    out
}

// ---- shared case sets -----------------------------------------------------

static EUCLIDEAN_CASES: OnceLock<Vec<Case>> = OnceLock::new();
static COSINE_CASES: OnceLock<Vec<Case>> = OnceLock::new();

/// The 500 seeded Euclidean instances with their exhaustive optima. Built once
/// per test binary and shared by all five groups — the enumeration, not GIST, is
/// what costs anything here.
fn euclidean_cases() -> &'static [Case] {
    EUCLIDEAN_CASES.get_or_init(|| {
        build_cases(
            (0..INSTANCES)
                .map(|i| (i, EUCLIDEAN_SEED_BASE + i as u64, Metric::Euclidean))
                .collect(),
        )
    })
}

/// The 20 Cosine instances used by the warm-up regression group only.
fn cosine_cases() -> &'static [Case] {
    COSINE_CASES.get_or_init(|| {
        build_cases(
            (0..COSINE_INSTANCES)
                .map(|i| (i, COSINE_SEED_BASE + i as u64, Metric::Cosine))
                .collect(),
        )
    })
}

/// Builds and solves a batch of cases across a handful of OS threads.
///
/// The work is embarrassingly parallel and each case depends only on its seed,
/// so the batch is split into contiguous chunks and reassembled in order: the
/// resulting `Vec` is identical to the sequential one. `rayon` is a dependency
/// of the library, not of this test target, so this uses `std::thread` only.
fn build_cases(specs: Vec<(usize, u64, Metric)>) -> Vec<Case> {
    let workers = std::thread::available_parallelism()
        .map(std::num::NonZeroUsize::get)
        .unwrap_or(1)
        .min(specs.len().max(1));
    let chunk = specs.len().div_ceil(workers).max(1);

    std::thread::scope(|scope| {
        let handles: Vec<_> = specs
            .chunks(chunk)
            .map(|slice| {
                scope.spawn(move || {
                    slice
                        .iter()
                        .map(|&(index, seed, metric)| build_case(index, seed, metric))
                        .collect::<Vec<_>>()
                })
            })
            .collect();
        handles
            .into_iter()
            .flat_map(|handle| handle.join().expect("case builder panicked"))
            .collect()
    })
}

// ---- assertion plumbing ---------------------------------------------------

/// `f_gist / OPT`. `OPT` is a maximum over a family containing the empty set,
/// whose `f` is `lambda * d_max >= 0`, so it is never negative; the `opt == 0`
/// guard is for the degenerate all-points-coincide case, where `f_gist == 0`
/// too and the ratio is vacuously perfect.
fn ratio(f_gist: f64, opt: f64) -> f64 {
    if opt > 0.0 {
        f_gist / opt
    } else {
        1.0
    }
}

/// `f_gist >= bound * OPT`, with `1e-9 * max(1, OPT)` of slack.
///
/// That slack is **floating-point tolerance, not a weakened bound**: distances
/// are `f32` and `g(S)` is a running `f64` sum of marginals, so the oracle's
/// `f(S)` and the driver's `f(S)` for the very same `S` may differ in the last
/// ulps. At ~1e-9 relative it is eight orders of magnitude below the margin any
/// of these theorems claims, so no real violation can hide inside it.
fn meets_bound(f_gist: f64, bound: f64, opt: f64) -> bool {
    f_gist >= bound * opt - 1e-9 * opt.max(1.0)
}

/// Running tally of the tightest ratio a group has seen, for the summary line.
struct Tally {
    ratio: f64,
    worst: String,
    checked: usize,
}

impl Tally {
    fn new() -> Self {
        Self {
            ratio: f64::INFINITY,
            worst: String::from("(none)"),
            checked: 0,
        }
    }

    fn record(&mut self, case: &Case, exhaustive: bool, out: &GistResult) {
        self.checked += 1;
        let observed = ratio(out.f_value, case.opt);
        if observed < self.ratio {
            self.ratio = observed;
            self.worst = case.describe(exhaustive, out);
        }
    }

    fn report(&self, group: &str, bound: f64) {
        println!(
            "{group}: {} runs, bound {bound:.9}, min f(GIST)/OPT = {:.9}\n    worst: {}",
            self.checked, self.ratio, self.worst
        );
    }
}

// ---- 1. Theorem 3.1 -------------------------------------------------------

/// Theorem 3.1: for a monotone submodular `g`, `f(GIST) >= (1/2 - eps) * OPT`.
#[test]
fn theorem_3_1_submodular_half_minus_eps() {
    let mut tally = Tally::new();

    for case in euclidean_cases() {
        if case.kind != UtilKind::FacilityLocation {
            continue;
        }
        let out = case.run(false);
        // The `1e-9 * max(1, OPT)` inside `meets_bound` is floating-point
        // tolerance, not a weakened bound.
        assert!(
            meets_bound(out.f_value, SUBMODULAR_BOUND, case.opt),
            "Theorem 3.1 violated: f(GIST) < (1/2 - eps) * OPT\n    {}",
            case.describe(false, &out)
        );
        tally.record(case, false, &out);
    }

    // The utility alternates by index, so exactly half the instances are here.
    assert_eq!(tally.checked, INSTANCES / 2);
    tally.report("Theorem 3.1 (submodular, geometric)", SUBMODULAR_BOUND);
}

// ---- 2. Theorem 3.3, geometric thresholds ---------------------------------

/// Theorem 3.3: for a linear `g`, `f(GIST) >= (2/3 - eps) * OPT`.
#[test]
fn theorem_3_3_linear_two_thirds_minus_eps() {
    let mut tally = Tally::new();

    for case in euclidean_cases() {
        if case.kind != UtilKind::Linear {
            continue;
        }
        let out = case.run(false);
        // The `1e-9 * max(1, OPT)` inside `meets_bound` is floating-point
        // tolerance, not a weakened bound.
        assert!(
            meets_bound(out.f_value, LINEAR_BOUND, case.opt),
            "Theorem 3.3 violated: f(GIST) < (2/3 - eps) * OPT\n    {}",
            case.describe(false, &out)
        );
        tally.record(case, false, &out);
    }

    assert_eq!(tally.checked, INSTANCES / 2);
    tally.report("Theorem 3.3 (linear, geometric)", LINEAR_BOUND);
}

// ---- 3. Theorem 3.3, exhaustive thresholds --------------------------------

/// Theorem 3.3's exact variant: a linear `g` with the exhaustive threshold set
/// `{dist(u,v)/2}` drops the `eps` entirely — `f(GIST) >= (2/3) * OPT`.
#[test]
fn theorem_3_3_linear_exact_two_thirds_with_exhaustive_thresholds() {
    let mut tally = Tally::new();

    for case in euclidean_cases() {
        if case.kind != UtilKind::Linear {
            continue;
        }
        let out = case.run(true);
        // The `1e-9 * max(1, OPT)` inside `meets_bound` is floating-point
        // tolerance, not a weakened bound — the constant here is an exact 2/3.
        assert!(
            meets_bound(out.f_value, LINEAR_EXACT_BOUND, case.opt),
            "Theorem 3.3 (exact) violated: f(GIST) < (2/3) * OPT\n    {}",
            case.describe(true, &out)
        );
        tally.record(case, true, &out);
    }

    assert_eq!(tally.checked, INSTANCES / 2);
    tally.report("Theorem 3.3 (linear, exhaustive)", LINEAR_EXACT_BOUND);
}

// ---- 4. Section 2.1 warm-up regression ------------------------------------

/// Regression floor: every instance, both utilities and both threshold sets,
/// clears the warm-up algorithm of Sec. 2.1 at `(e - 1)/(2e - 1) ~= 0.387`.
///
/// The 20 Cosine instances are folded in here and **only** here. Raw cosine
/// distance is not a metric, so the theorems' precondition does not hold for
/// them; this group is a tripwire against a regression, not a proof.
#[test]
fn warm_up_baseline_holds_on_every_instance() {
    let mut euclidean = Tally::new();

    for case in euclidean_cases() {
        for exhaustive in [false, true] {
            let out = case.run(exhaustive);
            // The `1e-9 * max(1, OPT)` inside `meets_bound` is floating-point
            // tolerance, not a weakened bound.
            assert!(
                meets_bound(out.f_value, WARMUP_BOUND, case.opt),
                "Sec. 2.1 warm-up baseline violated: f(GIST) < (e-1)/(2e-1) * OPT\n    {}",
                case.describe(exhaustive, &out)
            );
            euclidean.record(case, exhaustive, &out);
        }
    }

    let mut cosine = Tally::new();
    for case in cosine_cases() {
        for exhaustive in [false, true] {
            let out = case.run(exhaustive);
            // The `1e-9 * max(1, OPT)` inside `meets_bound` is floating-point
            // tolerance, not a weakened bound.
            assert!(
                meets_bound(out.f_value, WARMUP_BOUND, case.opt),
                "Sec. 2.1 warm-up baseline violated on a COSINE regression instance \
                 (outside the theorems' metric precondition)\n    {}",
                case.describe(exhaustive, &out)
            );
            cosine.record(case, exhaustive, &out);
        }
    }

    assert_eq!(euclidean.checked, 2 * INSTANCES);
    assert_eq!(cosine.checked, 2 * COSINE_INSTANCES);
    euclidean.report("Sec. 2.1 warm-up (Euclidean, both sets)", WARMUP_BOUND);
    cosine.report(
        "Sec. 2.1 warm-up (Cosine regression only, not a theorem instance)",
        WARMUP_BOUND,
    );
}

// ---- 5. sanity ------------------------------------------------------------

/// GIST maximizes `f` over a subfamily of what the oracle enumerates, so it can
/// never beat `OPT`. If it does, the oracle and the driver disagree about `f`
/// itself and every ratio above is meaningless — which is why this group also
/// recomputes `f_value` from `selected` through the public `eval_g` + `div`.
#[test]
fn gist_never_beats_the_oracle_and_reports_a_reproducible_f() {
    let mut runs = 0usize;
    let mut strict_geometric = 0usize;
    let mut strict_exhaustive = 0usize;

    for case in euclidean_cases() {
        for exhaustive in [false, true] {
            let out = case.run(exhaustive);
            let label = || case.describe(exhaustive, &out);

            // Same `1e-9 * max(1, OPT)` floating-point tolerance as `meets_bound`.
            assert!(
                out.f_value <= case.opt + 1e-9 * case.opt.max(1.0),
                "GIST beat the exhaustive optimum — eval_g/div disagree between \
                 the oracle and the driver\n    {}",
                label()
            );

            let mut sorted = out.selected.clone();
            sorted.sort_unstable();
            let before = sorted.len();
            sorted.dedup();
            assert_eq!(
                before,
                sorted.len(),
                "duplicate index in selected\n    {}",
                label()
            );
            assert!(
                out.selected.len() <= case.k,
                "selected exceeds the budget k\n    {}",
                label()
            );

            let mut util = case.utility();
            let recomputed = f_of(&case.pts, util.as_mut(), case.lambda, &out.selected);
            assert!(
                (recomputed - out.f_value).abs() <= 1e-9,
                "f_value {} is not reproducible from selected: recomputed {}\n    {}",
                out.f_value,
                recomputed,
                label()
            );

            // Non-vacuity bookkeeping: how often the oracle actually finds
            // something GIST does not. A set of instances GIST solves exactly
            // every time would say nothing about the bound.
            if out.f_value < case.opt - 1e-9 * case.opt.max(1.0) {
                if exhaustive {
                    strict_exhaustive += 1;
                } else {
                    strict_geometric += 1;
                }
            }
            runs += 1;
        }
    }

    assert_eq!(runs, 2 * INSTANCES);

    // Non-vacuity is asserted, not merely printed: on a set of instances GIST
    // solves exactly every time every ratio above is `1.0`, and the three
    // theorem groups then prove nothing. Replacing `gist` with a brute-force
    // optimum drives both counts to `0` and fails here -- the only assertion in
    // this file that would notice.
    //
    // Measured on this tree: 114/500 geometric and 270/500 exhaustive. The
    // floors sit far below that because the instances are Gaussian and so not
    // bit-stable across platforms; they guard non-vacuity, they do not pin the
    // exact counts.
    assert!(
        strict_geometric >= 40,
        "the geometric half has gone vacuous: OPT is strictly above f(GIST) on only \
         {strict_geometric}/{INSTANCES} runs, so the Theorem 3.1 and 3.3 groups are \
         asserting a bound no instance approaches"
    );
    assert!(
        strict_exhaustive >= 100,
        "the exhaustive half has gone vacuous: OPT is strictly above f(GIST) on only \
         {strict_exhaustive}/{INSTANCES} runs"
    );
    println!(
        "sanity: {runs} runs over {INSTANCES} instances; OPT strictly above f(GIST) on \
         {strict_geometric}/{INSTANCES} geometric and \
         {strict_exhaustive}/{INSTANCES} exhaustive runs"
    );
}
