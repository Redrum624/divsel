//! GIST itself: the driver that turns [`greedy_independent_set`] into an
//! approximation algorithm for `max_{|S| <= k} f(S) = g(S) + lambda * div(S)`.
//!
//! The paper (arXiv:2405.18754v3) writes Algorithm 1 as, counting the `function`
//! header as line 1:
//!
//! ```text
//!  1: function GIST(V, g, k, eps)
//!  2:   Initialize S <- GreedyIndependentSet(V, g, 0, k)   > classic greedy
//!  3:   Let d_max = max_{u,v in V} dist(u,v)               > the diameter of V
//!  4:   Let T <- {u,v} such that dist(u,v) = d_max
//!  5:   if f(T) > f(S) and k >= 2 then
//!  6:     Update S <- T
//!  7:   Let D <- {(1+eps)^i * eps*d_max/2 : (1+eps)^i <= 2/eps, i in Z>=0}
//!  8:   for d in D do
//!  9:     Set T <- GreedyIndependentSet(V, g, d, k)
//! 10:     if f(T) >= f(S) then
//! 11:       Update S <- T
//! 12:   return S
//! ```
//!
//! Three details of that transcription are load-bearing, and `divsel` treats them
//! as the contract every port conforms to:
//!
//! * Line 5 compares **strictly** (`>`), so the diametrical pair only displaces
//!   the greedy solution when it is genuinely better.
//! * Line 10 compares **non-strictly** (`>=`), so a later threshold displaces an
//!   equally good earlier one. Line 8 iterates a *set*, which fixes no order, so
//!   which of several equally good thresholds survives is left open: `divsel`
//!   folds `D` in **ascending order** of `d` — a **`[divsel choice]`** — making
//!   "the largest threshold attaining the best `f`" the answer, deterministically.
//!   A parallel sweep must reproduce that fold order to reproduce the answer.
//! * `D` is a geometric sequence of ratio `1 + eps` starting at `eps*d_max/2`.
//!   It is not a quantile grid and not an absolute grid, and only `d_max` is
//!   needed to build it.
//!
//! `f` is not submodular (Remark 2.1) and plain greedy on `f` has no
//! constant-factor guarantee (Appendix B); the threshold sweep is what buys the
//! `(1/2 - eps)` guarantee of Theorem 3.1, and `(2/3 - eps)` for a linear `g`
//! (Theorem 3.3).

use std::cmp::Ordering;

use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

use crate::error::DivselError;
use crate::greedy::greedy_independent_set;
use crate::points::Points;
use crate::utility::Utility;

/// How [`gist`] obtains the diameter `d_max` of the point set (paper line 3).
///
/// The exact diameter is an `O(n^2 * dim)` scan, which for a large `n` can cost
/// more than the whole threshold sweep. [`DiameterMode::Approx`] is the escape
/// hatch.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum DiameterMode {
    /// The exact diameter, via [`Points::diameter`]. The default, and the only
    /// mode the paper describes.
    #[default]
    Exact,
    /// A farthest-point double sweep, `O(sweeps * n * dim)`, yielding a `d_hat`
    /// in `[d_max/2, d_max]`. A **`[divsel choice]`**; see [`approx_diameter`].
    Approx {
        /// Number of double sweeps to run. `0` is treated as `1`.
        sweeps: usize,
    },
}

/// The knobs of [`gist`].
///
/// [`GistConfig::default`] is `k = 10`, `lambda = 1.0`, `eps = 0.1`,
/// `exhaustive_thresholds = false`, `diameter = DiameterMode::Exact`, so
/// `GistConfig { k: 3, ..Default::default() }` is the idiomatic way to change one
/// field.
#[derive(Clone, Debug, PartialEq)]
pub struct GistConfig {
    /// Budget: the returned selection holds at most `k` points. Must be nonzero;
    /// a `k` larger than the number of points is silently clamped.
    pub k: usize,
    /// Weight `lambda >= 0` of the diversity term in `f(S) = g(S) + lambda *
    /// div(S)`. Must be finite.
    pub lambda: f64,
    /// Accuracy `eps` of the threshold sweep, in `(0, 1]`. Smaller means more
    /// thresholds: `|D| = 1 + floor(log_{1+eps}(2/eps))`.
    pub eps: f32,
    /// Replace the geometric threshold set with the exhaustive one,
    /// `{dist(u,v)/2 : u,v in V}` — the paper's exact-`2/3` variant for a linear
    /// `g`. Costs `O(n^2)` thresholds; see [`gist`].
    pub exhaustive_thresholds: bool,
    /// How `d_max` is obtained.
    pub diameter: DiameterMode,
}

impl Default for GistConfig {
    fn default() -> Self {
        Self {
            k: 10,
            lambda: 1.0,
            eps: 0.1,
            exhaustive_thresholds: false,
            diameter: DiameterMode::Exact,
        }
    }
}

/// Which of Algorithm 1's three candidate solutions [`gist`] returned.
///
/// Not part of the paper: it is `divsel` reporting which branch won, which makes
/// the tie rules on lines 5 and 10 observable from the outside.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    /// The classic greedy solution of line 2 was never displaced.
    Greedy,
    /// The diametrical pair of line 4 strictly beat greedy (line 5) and nothing
    /// in the sweep matched it.
    DiameterPair,
    /// A threshold from `D` won (line 10). With the non-strict `>=` there, this
    /// is the **largest** `d` attaining the best `f`.
    Sweep,
}

/// What [`gist`] found.
#[derive(Clone, Debug, PartialEq)]
pub struct GistResult {
    /// The selected point indices, in the order the winning stage produced them
    /// (selection order for [`Stage::Greedy`] and [`Stage::Sweep`], ascending for
    /// [`Stage::DiameterPair`]). At most `k` of them.
    pub selected: Vec<usize>,
    /// `f(S) = g(S) + lambda * div(S)`, the value that was maximized.
    pub f_value: f64,
    /// `g(S)` alone, so a caller can see how the objective splits.
    pub g_value: f64,
    /// `div(S)`: the minimum pairwise distance in `S`, or `d_max` when
    /// `|S| <= 1`.
    pub div: f32,
    /// The distance threshold that produced `selected`: `0.0` for
    /// [`Stage::Greedy`], `d_max` for [`Stage::DiameterPair`], and the winning
    /// `d` for [`Stage::Sweep`].
    pub threshold: f32,
    /// Which branch of Algorithm 1 won.
    pub stage: Stage,
    /// The diameter used to build the sweep — exact under
    /// [`DiameterMode::Exact`], the estimate `d_hat` under
    /// [`DiameterMode::Approx`].
    pub d_max: f32,
}

/// The paper's threshold set `D` of line 7:
/// `{(1+eps)^i * eps*d_max/2 : (1+eps)^i <= 2/eps, i in Z>=0}`.
///
/// Returned in ascending order, with `1 + floor(log_{1+eps}(2/eps))` entries. The
/// powers of `1 + eps` are built by repeated multiplication in `f64` rather than
/// by `powf` or by flooring a logarithm: the standard library promises nothing
/// about the precision of `ln` or `powf`, so a log-and-floor count can differ by
/// one between platforms exactly at the boundary, and the count is part of this
/// crate's cross-language contract.
///
/// A `d_max` of `0` collapses every entry to `0.0`; the duplicates are removed,
/// so the result is `[0.0]` rather than a run of zeros. An `eps` outside
/// `(0, +inf)` — which [`gist`] rejects before ever calling this — yields an
/// empty set rather than looping forever.
///
/// # Examples
///
/// ```
/// # use divsel::thresholds;
/// let d = thresholds(1.0, 1.0);
/// assert_eq!(d, vec![0.5, 1.0]);
/// assert_eq!(thresholds(1.0, 0.1).len(), 32);
/// ```
pub fn thresholds(d_max: f32, eps: f32) -> Vec<f32> {
    thresholds_with_bound(d_max, eps, 2.0 / f64::from(eps))
}

/// [`thresholds`] with the ceiling on `(1+eps)^i` supplied by the caller.
///
/// The paper's bound is `2/eps`, which is what [`thresholds`] passes. Under
/// [`DiameterMode::Approx`] the driver only holds `d_hat >= d_max/2`, so it
/// passes `4/eps` instead: the top of the sweep then reaches `~2*d_hat >= d_max`
/// and the grid still contains a threshold within a factor `1 + eps` of the
/// target `div(S*)/2`. That widening is a **`[divsel choice]`** — the paper only
/// describes the exact-diameter case.
fn thresholds_with_bound(d_max: f32, eps: f32, bound: f64) -> Vec<f32> {
    let eps = f64::from(eps);
    let d_max = f64::from(d_max);
    // `p *= 1 + eps` only grows -- and so only terminates -- for a finite,
    // strictly positive eps. The driver has already rejected everything else;
    // this keeps the public entry point from hanging on a hand-written call.
    if !(eps > 0.0 && eps.is_finite()) {
        return Vec::new();
    }

    let mut out = Vec::new();
    let mut p = 1.0f64;
    while p <= bound {
        out.push((p * eps * d_max / 2.0) as f32);
        p *= 1.0 + eps;
    }
    // The entries are strictly increasing whenever `d_max > 0`, so this only
    // ever collapses the all-zero set of a degenerate point cloud to `[0.0]`.
    out.dedup();
    out
}

/// The exhaustive threshold set `{dist(u,v)/2 : u,v in V}` (paper Sec. 5.1),
/// ascending and deduplicated. Includes `0.0`, from the `u == v` pairs.
///
/// Note the halving is the paper's, and it puts the ceiling of this set at
/// `d_max/2` — half of where the geometric set of [`thresholds`] ends. The two
/// sets are therefore *not* nested and neither dominates the other on the raw
/// objective: the exhaustive one resolves every breakpoint below `d_max/2`, the
/// geometric one alone can reach the high-`div` thresholds above it. What the
/// exhaustive set buys is the worst-case ratio — an exact `2/3` for a linear `g`
/// instead of `2/3 - eps`.
fn exhaustive_threshold_set(pts: &Points<'_>) -> Vec<f32> {
    let n = pts.n();
    let mut out = Vec::with_capacity(n * (n - 1) / 2 + 1);
    for u in 0..n {
        // `u == v` contributes the paper's `dist(u,u)/2 = 0`, which reproduces
        // the line-2 greedy call inside the sweep.
        for v in u..n {
            out.push(pts.dist(u, v) / 2.0);
        }
    }
    out.sort_by(f32::total_cmp);
    out.dedup();
    out
}

/// `div(S)`: the minimum pairwise distance inside `S`, or the diameter of the
/// whole point set when `|S| <= 1`.
///
/// The `|S| <= 1` case is `d_max`, **not** `0` — that is the paper's definition
/// (Sec. 2), and the reason for it is that it makes `div` monotone decreasing in
/// `S`, which the analysis relies on.
///
/// Computing that fallback costs a full `O(n^2)` diameter scan, so this function
/// is `O(n^2 * dim)` in the worst case even for a two-element `s`. [`gist`]
/// computes `d_max` once and never pays it again.
///
/// # Panics
///
/// Panics if any index in `s` is out of range for `pts`.
pub fn div(pts: &Points<'_>, s: &[usize]) -> f32 {
    if s.len() <= 1 {
        return pts.diameter().0;
    }
    // The diameter is already handled above, so the value passed here is never
    // read: two or more indices always take the pairwise-minimum path.
    div_with_dmax(pts, s, 0.0)
}

/// [`div`] with `d_max` supplied, so the sweep never recomputes the diameter.
fn div_with_dmax(pts: &Points<'_>, s: &[usize], d_max: f32) -> f32 {
    if s.len() <= 1 {
        return d_max;
    }
    let mut best = f32::INFINITY;
    for (position, &u) in s.iter().enumerate() {
        for &v in &s[position + 1..] {
            best = best.min(pts.dist(u, v));
        }
    }
    best
}

/// `g(S)`, by replaying `S` through `util` in order and summing the marginals.
///
/// [`Utility`] exposes marginals only, so `g(S)` is `sum_i g(s_i | {s_0..s_i-1})`.
/// `util` is [`Utility::reset`] both **before** the replay — so a dirty utility is
/// tolerated — and **after** it, so the caller gets it back in the empty-selection
/// state. That matters for the diametrical pair, whose `g` cannot be read off the
/// greedy loop's leftover state.
///
/// # Panics
///
/// Panics if any index in `s` is out of range for `util`'s per-point tables.
pub fn eval_g(util: &mut dyn Utility, s: &[usize], pts: &Points<'_>) -> f64 {
    util.reset();
    let mut total = 0.0f64;
    for (position, &v) in s.iter().enumerate() {
        total += util.marginal(v, &s[..position], pts);
        util.commit(v, pts);
    }
    util.reset();
    total
}

/// A farthest-point double sweep, the cheap stand-in for [`Points::diameter`].
///
/// Starting from index `0`, each sweep takes `a = argmax_j dist(cur, j)` and then
/// `b = argmax_j dist(a, j)`, keeps `(dist(a,b), min(a,b), max(a,b))` if it beats
/// the incumbent, and starts the next sweep from `b`. Ties in either `argmax` go
/// to the lowest index, and the incumbent is chosen by the same total order
/// [`Points::diameter`] uses — larger distance, then smaller `u`, then smaller
/// `v` — so the result is deterministic and directly comparable to the exact one.
///
/// # Guarantee
///
/// `d_max/2 <= d_hat <= d_max`. The upper bound is trivial. For the lower one,
/// let `c` be a sweep's starting point and `a` the farthest point from it: for any
/// `x, y`, the triangle inequality gives
/// `dist(x,y) <= dist(x,c) + dist(c,y) <= 2*dist(c,a)`, so `dist(c,a) >= d_max/2`;
/// and `b` is the farthest point from `a`, so `dist(a,b) >= dist(a,c) >= d_max/2`.
///
/// This is a **`[divsel choice]`**: the paper always uses the exact diameter, and
/// its guarantees are stated in terms of that `d_max`. Under this mode they hold
/// relative to the returned `d_hat` instead, which is why [`gist`] widens the
/// sweep's ceiling to `4/eps` — see [`thresholds_with_bound`].
///
/// `sweeps == 0` is treated as `1`. A set of fewer than two points yields
/// `(0.0, 0, 0)`, matching [`Points::diameter`].
///
/// Not part of the supported API: it is `pub` only so this crate's benches can
/// reach it.
#[doc(hidden)]
pub fn approx_diameter(pts: &Points<'_>, sweeps: usize) -> (f32, usize, usize) {
    if pts.n() < 2 {
        return (0.0, 0, 0);
    }
    let mut best = (f32::NEG_INFINITY, usize::MAX, usize::MAX);
    let mut current = 0usize;
    for _ in 0..sweeps.max(1) {
        let a = farthest_from(pts, current);
        let b = farthest_from(pts, a);
        best = better_pair(best, (pts.dist(a, b), a.min(b), a.max(b)));
        current = b;
    }
    best
}

/// `argmax_{j != from} dist(from, j)`, ties to the lowest index.
///
/// Excluding `from` itself keeps the returned pair distinct even when every
/// distance is `0`.
fn farthest_from(pts: &Points<'_>, from: usize) -> usize {
    let mut best = (f32::NEG_INFINITY, usize::MAX);
    for j in 0..pts.n() {
        if j == from {
            continue;
        }
        let distance = pts.dist(from, j);
        if distance.total_cmp(&best.0) == Ordering::Greater {
            best = (distance, j);
        }
    }
    best.1
}

/// The total order [`Points::diameter`] reduces with: larger distance wins, then
/// the smaller `u`, then the smaller `v`.
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

/// Runs GIST over `pts`, maximizing `f(S) = g(S) + lambda * div(S)` subject to
/// `|S| <= k`.
///
/// This is Algorithm 1 verbatim: classic greedy, then the diametrical pair, then
/// the threshold sweep, keeping the best `f` under the paper's comparison
/// directions (strict on line 5, non-strict on line 10). The sweep is folded in
/// ascending order of `d`, so the largest threshold attaining the best `f` is the
/// one reported.
///
/// `util` may be in any state on entry and is returned [`Utility::reset`].
///
/// The elided lifetime on `pts` accepts either flavour of point set: the
/// `Points<'static>` that [`Points::new`] hands back for an owned buffer, and the
/// `Points<'a>` that [`Points::borrowed`] hands back for a borrowed one.
///
/// # Cost
///
/// `1 + |D|` calls to [`greedy_independent_set`], plus one diameter scan. The
/// `|D|` sweep calls are independent, so they run on the `rayon` thread pool,
/// each worker holding its own [`Utility::boxed_clone`] of `util`; only the fold
/// that picks the winner is sequential, because its tie rule is order dependent.
/// The sweep therefore materialises one candidate selection per threshold --
/// `O(|D| * k)` indices -- before folding, which is nothing for the geometric
/// set and worth knowing about for the exhaustive one.
/// With the geometric set that is `O(n * k * log_{1+eps}(1/eps))` marginal
/// evaluations, the bound of Theorem 3.1. With
/// [`GistConfig::exhaustive_thresholds`] the set has up to `n*(n-1)/2 + 1`
/// distinct entries, so the sweep costs `O(n^2)` greedy runs -- `O(n^3 * k)`
/// marginal evaluations. That is the paper's own choice for its `n = 1000`
/// synthetic experiment with a linear utility, and it is what turns Theorem 3.3's
/// `2/3 - eps` into an exact `2/3`. No cap is enforced; it is intended for `n` of
/// roughly 2000 or below. It buys a worst-case ratio, not a better answer on any
/// given instance: neither threshold set dominates the other empirically,
/// because the exhaustive set's largest entry is `d_max/2`, half the ceiling of
/// the geometric one, so only the geometric set reaches the thresholds that
/// force a high `div`.
///
/// # Degenerate point sets
///
/// When `d_max == 0` — every point coincides, which includes `n == 1` — every
/// threshold in either set is `0`, so each sweep iteration would re-run the
/// line-2 call and return the same selection. `divsel` skips the sweep entirely
/// in that case, a **`[divsel choice]`** that cannot change `selected`,
/// `f_value`, `g_value` or `div`, and only stops [`GistResult::stage`] from being
/// relabelled [`Stage::Sweep`] by a threshold that did no work.
///
/// # Errors
///
/// In this order: [`DivselError::InvalidK`] if `cfg.k == 0`;
/// [`DivselError::InvalidEps`] unless `0 < eps <= 1`;
/// [`DivselError::InvalidLambda`] unless `lambda` is finite and non-negative;
/// then whatever [`Utility::validate`] returns for `pts` — typically
/// [`DivselError::WeightsLength`].
///
/// # Panics
///
/// Panics if `util` indexes a per-point table shorter than `pts.n()`. A
/// [`Utility`] whose [`Utility::validate`] checks its own length, as all three
/// built-ins do, cannot reach that.
///
/// # Examples
///
/// ```
/// # use divsel::{gist, GistConfig, Linear, Metric, Points, Stage};
/// let pts = Points::new(vec![0.0, 1.0, 5.0, 6.0], 1, Metric::Euclidean)?;
/// let mut util = Linear::uniform(pts.n());
/// let out = gist(&pts, &mut util, &GistConfig { k: 2, ..Default::default() })?;
/// // Two points, both weight 1, so f is decided entirely by div: the widest pair.
/// assert_eq!(out.selected, vec![0, 3]);
/// assert_eq!(out.div, 6.0);
/// assert_eq!(out.stage, Stage::Sweep);
/// # Ok::<(), divsel::DivselError>(())
/// ```
pub fn gist(
    pts: &Points<'_>,
    util: &mut dyn Utility,
    cfg: &GistConfig,
) -> Result<GistResult, DivselError> {
    // Validation first, in a fixed order, so a caller who got two things wrong
    // always hears about the same one.
    if cfg.k == 0 {
        return Err(DivselError::InvalidK);
    }
    if !(cfg.eps > 0.0 && cfg.eps <= 1.0) {
        return Err(DivselError::InvalidEps(cfg.eps));
    }
    if !(cfg.lambda >= 0.0 && cfg.lambda.is_finite()) {
        return Err(DivselError::InvalidLambda(cfg.lambda));
    }
    util.validate(pts)?;

    let n = pts.n();
    // The paper constrains |S| <= k, so a budget past the ground set is not an
    // error; it just cannot bind.
    let k = cfg.k.min(n);
    let lambda = cfg.lambda;

    // Paper lines 3-4.
    let (d_max, u, v) = match cfg.diameter {
        DiameterMode::Exact => pts.diameter(),
        DiameterMode::Approx { sweeps } => approx_diameter(pts, sweeps),
    };

    // `f(S)` for a selection, with `d_max` already in hand. `eval_g` resets `util`
    // on the way in and on the way out, so every evaluation starts from
    // `g(empty) = 0` -- the greedy loop's leftover state describes its own
    // selection, which is no help for the diametrical pair.
    let evaluate = |selected: &[usize], util: &mut dyn Utility| -> (f64, f64, f32) {
        let g_value = eval_g(util, selected, pts);
        let div_value = div_with_dmax(pts, selected, d_max);
        (g_value + lambda * f64::from(div_value), g_value, div_value)
    };

    // Paper line 2: the classic greedy solution, i.e. the sweep at d = 0.
    let mut selected = greedy_independent_set(pts, util, 0.0, k);
    let (mut f_value, mut g_value, mut div_value) = evaluate(&selected, util);
    let mut stage = Stage::Greedy;
    let mut threshold = 0.0f32;

    // Paper lines 5-6: the diametrical pair, compared **strictly**, and only when
    // the budget can hold two points. `n >= 2` is implied by `k >= 2` here, since
    // k is clamped to n, but stating it keeps the guard honest if that changes.
    if k >= 2 && n >= 2 {
        let pair = vec![u.min(v), u.max(v)];
        let (f_pair, g_pair, div_pair) = evaluate(&pair, util);
        if f_pair > f_value {
            selected = pair;
            f_value = f_pair;
            g_value = g_pair;
            div_value = div_pair;
            stage = Stage::DiameterPair;
            threshold = d_max;
        }
    }

    // Paper lines 7-11. Every threshold of a zero-diameter point set is zero, so
    // the sweep would only repeat the line-2 call; see the note on this function.
    if d_max > 0.0 {
        let set = if cfg.exhaustive_thresholds {
            exhaustive_threshold_set(pts)
        } else {
            // Approx mode only knows d_max <= 2 * d_hat, so the sweep's ceiling
            // doubles to keep the true diameter inside it.
            let bound = match cfg.diameter {
                DiameterMode::Exact => 2.0 / f64::from(cfg.eps),
                DiameterMode::Approx { .. } => 4.0 / f64::from(cfg.eps),
            };
            thresholds_with_bound(d_max, cfg.eps, bound)
        };

        // The thresholds are independent of one another -- the paper's own
        // observation, Sec. 5: "By using parallelism for different d values, we
        // can keep the GIST-submod subset selection runtime the same as the
        // submod algorithm" -- so the greedy runs go on the `rayon` pool.
        //
        // `util` is reset first and then shared immutably as the prototype every
        // worker clones from, so no worker ever sees another's selection state.
        // `map_init` clones once per worker rather than once per threshold;
        // `greedy_independent_set` resets what it is handed, so reusing a clone
        // across the thresholds a worker happens to receive is safe, and the
        // determinism test pins that.
        util.reset();
        let prototype: &dyn Utility = &*util;
        let scanned: Vec<(f32, Vec<usize>, f64, f64, f32)> = set
            .par_iter()
            .map_init(
                || prototype.boxed_clone(),
                |worker, &d| {
                    let candidate = greedy_independent_set(pts, worker.as_mut(), d, k);
                    let (f_candidate, g_candidate, div_candidate) =
                        evaluate(&candidate, worker.as_mut());
                    (d, candidate, f_candidate, g_candidate, div_candidate)
                },
            )
            .collect();

        // `collect` restores the order of `set`, so this fold runs ascending and
        // compares **non-strictly**, exactly as a sequential sweep would: the
        // largest threshold attaining the best f is the one that survives. The
        // fold stays sequential precisely because that tie rule is order
        // dependent -- a parallel reduction would have to re-derive it.
        for (d, candidate, f_candidate, g_candidate, div_candidate) in scanned {
            if f_candidate >= f_value {
                selected = candidate;
                f_value = f_candidate;
                g_value = g_candidate;
                div_value = div_candidate;
                stage = Stage::Sweep;
                threshold = d;
            }
        }
    }

    util.reset();
    Ok(GistResult {
        selected,
        f_value,
        g_value,
        div: div_value,
        threshold,
        stage,
        d_max,
    })
}

#[cfg(test)]
mod tests {
    use std::cmp::Ordering;
    use std::sync::atomic::{self, AtomicBool};
    use std::sync::Arc;

    use super::{
        approx_diameter, div, div_with_dmax, eval_g, exhaustive_threshold_set, gist, thresholds,
        thresholds_with_bound, DiameterMode, GistConfig, GistResult, Stage,
    };
    use crate::error::DivselError;
    use crate::greedy::greedy_independent_set;
    use crate::metric::Metric;
    use crate::points::Points;
    use crate::testutil::SplitMix64;
    use crate::utility::{FacilityLocation, Linear, Utility};

    /// Five points on a line at `x = 0, 1, 3, 7, 12`; the diameter is exactly 12.
    fn line_five() -> Points<'static> {
        Points::new(vec![0.0, 1.0, 3.0, 7.0, 12.0], 1, Metric::Euclidean).expect("line fixture")
    }

    /// `f(S) = g(S) + lambda * div(S)`, evaluated the same way the driver does.
    fn f_of(pts: &Points<'_>, util: &mut dyn Utility, s: &[usize], lambda: f64, d_max: f32) -> f64 {
        eval_g(util, s, pts) + lambda * f64::from(div_with_dmax(pts, s, d_max))
    }

    // ---- (a) the threshold set --------------------------------------------

    #[test]
    fn thresholds_match_the_paper_definition() {
        let d = thresholds(1.0, 0.1);

        // |D| = 1 + floor(log_{1.1}(20)) = 1 + 31 = 32.
        let counted = 1 + (20.0f64.ln() / 1.1f64.ln()).floor() as usize;
        assert_eq!(counted, 32, "the reference count itself moved");
        assert_eq!(d.len(), 32, "got {d:?}");

        assert!(
            (d[0] - 0.05).abs() < 1e-7,
            "first entry is eps*d_max/2: {}",
            d[0]
        );
        let last = *d.last().expect("non-empty");
        assert!(last <= 1.0, "last entry {last} exceeds d_max");
        for pair in d.windows(2) {
            assert!(pair[1] > pair[0], "not strictly increasing: {pair:?}");
            // Consecutive entries are in ratio 1 + eps.
            let ratio = f64::from(pair[1]) / f64::from(pair[0]);
            assert!((ratio - 1.1).abs() < 1e-5, "ratio {ratio} is not 1 + eps");
        }

        // A zero diameter collapses the whole set onto one entry.
        assert_eq!(thresholds(0.0, 0.1), vec![0.0]);

        // eps = 1: (1+eps)^i <= 2 admits i = 0 and i = 1 only.
        let one = thresholds(1.0, 1.0);
        assert_eq!(one.len(), 2, "got {one:?}");
        assert!((one[0] - 0.5).abs() < 1e-7);
        assert!((one[1] - 1.0).abs() < 1e-7);

        // An eps the driver would have rejected must still terminate.
        assert!(thresholds(1.0, 0.0).is_empty());
        assert!(thresholds(1.0, -0.5).is_empty());
        assert!(thresholds(1.0, f32::NAN).is_empty());

        // The Approx-mode set extends the exact one rather than replacing it, and
        // its top reaches ~2*d_hat, so a true d_max of up to 2*d_hat is covered.
        let wide = thresholds_with_bound(1.0, 0.1, 40.0);
        assert_eq!(
            wide.len(),
            1 + (40.0f64.ln() / 1.1f64.ln()).floor() as usize
        );
        assert_eq!(&wide[..d.len()], &d[..]);
        assert!(*wide.last().expect("non-empty") >= 2.0 / 1.1);
    }

    // ---- (b) div ----------------------------------------------------------

    #[test]
    fn div_is_the_diameter_for_small_sets_and_the_min_pair_otherwise() {
        let pts = line_five();
        let d_max = pts.diameter().0;
        assert_eq!(d_max, 12.0);

        // |S| <= 1 is the diameter, not zero: that is what makes div monotone
        // decreasing.
        assert_eq!(div(&pts, &[]), 12.0);
        assert_eq!(div(&pts, &[3]), 12.0);
        assert_eq!(div(&pts, &[4]), 12.0);

        // Hand-checked over x = 0, 1, 3, 7, 12.
        assert_eq!(div(&pts, &[0, 1]), 1.0);
        assert_eq!(div(&pts, &[0, 1, 2]), 1.0); // min(1, 3, 2)
        assert_eq!(div(&pts, &[0, 2, 4]), 3.0); // min(3, 12, 9)
        assert_eq!(div(&pts, &[2, 3, 4]), 4.0); // min(4, 9, 5)
        assert_eq!(div(&pts, &[0, 4]), 12.0);

        // The private helper agrees, which is what the driver relies on.
        for s in [&[][..], &[3][..], &[0, 2, 4][..], &[2, 3, 4][..]] {
            assert_eq!(div_with_dmax(&pts, s, d_max), div(&pts, s), "for {s:?}");
        }
    }

    // ---- (c) the scaled-down paper synthetic setup -------------------------

    /// The paper's synthetic utility (Sec. 5.1):
    /// `g(S) = alpha * min{(1/k) * sum_{i in S} w_i, beta}`.
    ///
    /// Monotone and submodular: a concave, non-decreasing function of a modular
    /// function. Marginals shrink once the budget `beta` binds, so this exercises
    /// the CELF path rather than the linear fast path.
    #[derive(Clone)]
    struct BudgetAdditive {
        weights: Vec<f64>,
        alpha: f64,
        beta: f64,
        k: usize,
        running_sum: f64,
    }

    impl BudgetAdditive {
        fn new(weights: Vec<f64>, alpha: f64, beta: f64, k: usize) -> Self {
            Self {
                weights,
                alpha,
                beta,
                k,
                running_sum: 0.0,
            }
        }

        fn value(&self, sum: f64) -> f64 {
            self.alpha * (sum / self.k as f64).min(self.beta)
        }
    }

    impl Utility for BudgetAdditive {
        fn marginal(&self, v: usize, _selected: &[usize], _pts: &Points<'_>) -> f64 {
            self.value(self.running_sum + self.weights[v]) - self.value(self.running_sum)
        }

        fn commit(&mut self, v: usize, _pts: &Points<'_>) {
            self.running_sum += self.weights[v];
        }

        fn reset(&mut self) {
            self.running_sum = 0.0;
        }

        fn boxed_clone(&self) -> Box<dyn Utility> {
            Box::new(self.clone())
        }
    }

    /// Plain greedy on `f` itself — the baseline the paper's Appendix B shows has
    /// no constant-factor guarantee.
    ///
    /// Each round adds the point with the largest `f(S u {v}) - f(S)`, diversity
    /// term included, until `k` points are held -- the textbook loop, which is
    /// what Appendix B analyses. `div(S u {v}) = min(div(S), dist(v, S))` is
    /// maintained incrementally, exactly as the greedy subroutine maintains its
    /// frontier, so a round costs the same `O(n)` marginals plus `O(n * dim)`
    /// distance work that one round of `GreedyIndependentSet` costs.
    ///
    /// It is a strong heuristic on an i.i.d. Gaussian cloud, and on this instance
    /// family it is not one GIST dominates; see the measurements on the test
    /// below and Appendix B for why no constant-factor guarantee backs it.
    fn greedy_on_f(
        pts: &Points<'_>,
        util: &mut dyn Utility,
        k: usize,
        lambda: f64,
        d_max: f32,
    ) -> Vec<usize> {
        util.reset();
        let n = pts.n();
        let mut nearest = vec![f32::INFINITY; n];
        let mut chosen = vec![false; n];
        let mut selected: Vec<usize> = Vec::new();
        let mut current_div = d_max;
        let mut current_g = 0.0f64;
        let mut current_f = lambda * f64::from(d_max);

        while selected.len() < k.min(n) {
            let mut best: Option<(f64, usize, f64, f32)> = None;
            for v in 0..n {
                if chosen[v] {
                    continue;
                }
                let g_new = current_g + util.marginal(v, &selected, pts);
                let div_new = current_div.min(nearest[v]);
                let gain = g_new + lambda * f64::from(div_new) - current_f;
                let better = match best {
                    None => true,
                    Some((incumbent, _, _, _)) => gain.total_cmp(&incumbent) == Ordering::Greater,
                };
                if better {
                    best = Some((gain, v, g_new, div_new));
                }
            }
            let Some((_, t, g_new, div_new)) = best else {
                break;
            };
            selected.push(t);
            util.commit(t, pts);
            chosen[t] = true;
            current_g = g_new;
            current_div = div_new;
            current_f = g_new + lambda * f64::from(div_new);
            for (v, near) in nearest.iter_mut().enumerate() {
                if !chosen[v] {
                    *near = near.min(pts.dist(v, t));
                }
            }
        }
        util.reset();
        selected
    }

    /// `k` distinct indices drawn uniformly, by partial Fisher-Yates.
    fn random_subset(rng: &mut SplitMix64, n: usize, k: usize) -> Vec<usize> {
        let mut pool: Vec<usize> = (0..n).collect();
        let k = k.min(n);
        for i in 0..k {
            let j = i + rng.below(n - i);
            pool.swap(i, j);
        }
        pool.truncate(k);
        pool
    }

    /// GIST against the paper's own synthetic instance family (Sec. 5.1), scaled
    /// down from `n = 1000` to `n = 200`.
    ///
    /// # What is asserted, and why it is not "GIST wins every comparison"
    ///
    /// Per instance, `f(GIST) >= f(greedy-on-g)` holds **exactly**: Algorithm 1
    /// line 2 seeds `S` with that solution and every later update only raises `f`.
    /// That is a theorem, and it is asserted as an exact inequality on all 80
    /// instances.
    ///
    /// `f(GIST) >= f(greedy-on-f)` is **not** a theorem, per instance or on the
    /// mean. Measured over these 20 seeds at the default `eps = 0.1`:
    ///
    /// ```text
    ///  k | GIST     | greedy-on-g | greedy-on-f | random
    ///  5 | 1.308235 |    1.213987 |    1.323919 | 0.943228
    /// 10 | 1.273587 |    1.176339 |    1.289976 | 0.919074
    /// 25 | 1.223206 |    1.144013 |    1.247859 | 0.884398
    /// 50 | 1.184705 |    1.122270 |    1.167358 | 0.877457
    /// ```
    ///
    /// GIST wins at `k = 50` and trails by 1.2%, 1.3% and 2.0% at `k = 5, 10, 25`.
    /// That gap is the `eps` quantization of the geometric grid, not a defect: a
    /// 401-point uniform sweep of `[0, d_max]`, which upper-bounds what *any*
    /// threshold can achieve, scores 1.331096 / 1.297391 / 1.251197 / 1.208640,
    /// and GIST climbs monotonically towards those as `eps` falls -- at
    /// `eps = 0.01` it reaches 1.329765 / 1.296795 / 1.249884 / 1.207464 and beats
    /// greedy-on-f at all four `k`. On an i.i.d. Gaussian cloud in 64 dimensions
    /// every pairwise distance concentrates, so the best achievable `div` sits in
    /// a narrow band that a 10%-spaced grid cannot resolve; greedy-on-f optimizes
    /// `div` directly and lands inside it. Appendix B's point is that greedy-on-f
    /// has no *worst-case* guarantee, not that it is weak on nice instances.
    ///
    /// So this test asserts the mean relations that hold: strictly above random
    /// and above greedy-on-g, and within 5% of greedy-on-f (measured worst case
    /// 2.0%, at `k = 25`). Sharpening the last one to a strict win needs
    /// `eps = 0.01`, which is 533 thresholds -- 17x the runtime -- for a 0.16%
    /// margin at `k = 25`, far too thin to survive the platform variance that
    /// `gaussian_points` carries.
    ///
    /// # Runtime
    ///
    /// About 19 s in a debug build and 0.5 s in a release build.
    #[test]
    fn gist_beats_the_baselines_on_the_scaled_down_paper_setup() {
        // Paper Sec. 5.1, scaled down from n = 1000 to n = 200 for test runtime:
        // x_i ~ N(0, I_64), w_i ~ U[0,1), and the budget-additive utility with the
        // paper's headline alpha = 0.95, beta = 0.75, lambda = 1 - alpha.
        const N: usize = 200;
        const DIM: usize = 64;
        const ALPHA: f64 = 0.95;
        const BETA: f64 = 0.75;
        const LAMBDA: f64 = 1.0 - ALPHA;
        const SEEDS: usize = 20;

        for k in [5usize, 10, 25, 50] {
            let mut sum_gist = 0.0f64;
            let mut sum_greedy_f = 0.0f64;
            let mut sum_random = 0.0f64;
            let mut sum_greedy_g = 0.0f64;

            for seed in 0..SEEDS {
                let mut rng = SplitMix64(0x0001_5e70_0000_0000 + seed as u64);
                let data = rng.gaussian_points(N, DIM);
                let weights = rng.uniform_weights(N);
                let pts = Points::new(data, DIM, Metric::Euclidean).expect("random points");
                let d_max = pts.diameter().0;
                let mut util = BudgetAdditive::new(weights, ALPHA, BETA, k);

                let cfg = GistConfig {
                    k,
                    lambda: LAMBDA,
                    ..Default::default()
                };
                let out = gist(&pts, &mut util, &cfg).expect("valid configuration");
                assert!(out.selected.len() <= k);
                assert_eq!(out.d_max, d_max);

                // Line 2 seeds S with the classic greedy solution and every later
                // update only ever raises f, so this holds per instance, exactly.
                let on_g = greedy_independent_set(&pts, &mut util, 0.0, k);
                let f_on_g = f_of(&pts, &mut util, &on_g, LAMBDA, d_max);
                assert!(
                    out.f_value >= f_on_g,
                    "k = {k}, seed {seed}: GIST f = {} below greedy-on-g f = {f_on_g}",
                    out.f_value
                );
                // The reported value really is f of the reported set.
                let recomputed = f_of(&pts, &mut util, &out.selected, LAMBDA, d_max);
                assert!(
                    (out.f_value - recomputed).abs() < 1e-12,
                    "k = {k}, seed {seed}: reported {} but f(S) = {recomputed}",
                    out.f_value
                );

                let on_f = greedy_on_f(&pts, &mut util, k, LAMBDA, d_max);
                let picked = random_subset(&mut rng, N, k);

                sum_gist += out.f_value;
                sum_greedy_g += f_on_g;
                sum_greedy_f += f_of(&pts, &mut util, &on_f, LAMBDA, d_max);
                sum_random += f_of(&pts, &mut util, &picked, LAMBDA, d_max);
            }

            let seeds = SEEDS as f64;
            let mean_gist = sum_gist / seeds;
            let mean_greedy_g = sum_greedy_g / seeds;
            let mean_greedy_f = sum_greedy_f / seeds;
            let mean_random = sum_random / seeds;
            assert!(
                mean_gist >= mean_random,
                "k = {k}: mean f(GIST) = {mean_gist} < mean f(random) = {mean_random}"
            );
            assert!(
                mean_gist >= mean_greedy_g,
                "k = {k}: mean f(GIST) = {mean_gist} < mean f(greedy-on-g) = {mean_greedy_g}"
            );
            assert!(
                mean_gist >= 0.95 * mean_greedy_f,
                "k = {k}: mean f(GIST) = {mean_gist} is more than 5% below mean                  f(greedy-on-f) = {mean_greedy_f}"
            );
        }
    }

    // ---- (d) the two comparison directions --------------------------------

    /// Three points at `x = 0, 1, 10` with weights `1, 5, 1`: every threshold in
    /// `D` yields the same selection, so the sweep's `>=` is what decides.
    fn tie_instance() -> (Points<'static>, Linear) {
        (
            Points::new(vec![0.0, 1.0, 10.0], 1, Metric::Euclidean).expect("tie fixture"),
            Linear::new(vec![1.0, 5.0, 1.0]),
        )
    }

    #[test]
    fn the_sweep_keeps_the_largest_threshold_that_ties() {
        let (pts, mut util) = tie_instance();
        let cfg = GistConfig {
            k: 2,
            lambda: 0.2,
            eps: 0.5,
            ..Default::default()
        };
        let d_max = pts.diameter().0;
        assert_eq!(d_max, 10.0);

        // Evaluate the sweep by hand: which thresholds attain the best f?
        let set = thresholds(d_max, cfg.eps);
        assert_eq!(set.len(), 4, "got {set:?}");
        let scored: Vec<(f32, f64)> = set
            .iter()
            .map(|&d| {
                let s = greedy_independent_set(&pts, &mut util, d, cfg.k);
                (d, f_of(&pts, &mut util, &s, cfg.lambda, d_max))
            })
            .collect();
        let best = scored
            .iter()
            .map(|&(_, f)| f)
            .fold(f64::NEG_INFINITY, f64::max);
        let winners: Vec<f32> = scored
            .iter()
            .filter(|&&(_, f)| f == best)
            .map(|&(d, _)| d)
            .collect();
        assert!(
            winners.len() >= 2,
            "the fixture no longer produces a tie: {scored:?}"
        );

        let out = gist(&pts, &mut util, &cfg).expect("valid configuration");
        assert_eq!(out.stage, Stage::Sweep);
        assert_eq!(out.selected, vec![1, 2]);
        assert_eq!(
            out.threshold,
            *winners.last().expect("at least two winners"),
            "the sweep must keep the largest tying threshold, not the first"
        );
        assert!((out.f_value - best).abs() < 1e-12);
    }

    /// Two heavy points almost on top of each other, two lighter ones far apart:
    /// `x = 0, 0.01, -50, 50` with weights `10, 10, 6, 6`.
    ///
    /// The pair `{2, 3}` has weight 12, more than any single point, so it beats
    /// both the greedy pair (weight 20 but no diversity) and every singleton the
    /// sweep can produce.
    fn pair_instance() -> (Points<'static>, Linear) {
        (
            Points::new(vec![0.0, 0.01, -50.0, 50.0], 1, Metric::Euclidean).expect("pair fixture"),
            Linear::new(vec![10.0, 10.0, 6.0, 6.0]),
        )
    }

    #[test]
    fn the_diametrical_pair_wins_only_when_it_strictly_beats_greedy() {
        let (pts, mut util) = pair_instance();
        let d_max = pts.diameter().0;
        assert_eq!(d_max, 100.0);

        let cfg = GistConfig {
            k: 2,
            lambda: 0.2,
            eps: 0.5,
            ..Default::default()
        };
        let out = gist(&pts, &mut util, &cfg).expect("valid configuration");
        // f(greedy) = 20 + 0.2*0.01, f(pair) = 12 + 20 = 32, best sweep = 30.
        assert_eq!(out.stage, Stage::DiameterPair);
        assert_eq!(out.selected, vec![2, 3]);
        assert_eq!(out.threshold, d_max);
        assert_eq!(out.d_max, d_max);
        assert_eq!(out.div, 100.0);
        assert!((out.f_value - 32.0).abs() < 1e-9, "got {}", out.f_value);
        assert!((out.g_value - 12.0).abs() < 1e-9, "got {}", out.g_value);

        // Same instance, lambda = 0: diversity stops paying, so the greedy pair
        // wins outright. Nothing in the sweep matches its f, but the assertion is
        // on the value and the selection rather than on the stage, because a
        // threshold that reproduced greedy's selection would legitimately take
        // the stage under the non-strict `>=`.
        let flat = GistConfig {
            lambda: 0.0,
            ..cfg.clone()
        };
        let out = gist(&pts, &mut util, &flat).expect("valid configuration");
        let on_g = greedy_independent_set(&pts, &mut util, 0.0, flat.k);
        assert_eq!(on_g, vec![0, 1]);
        assert_eq!(out.selected, on_g);
        assert!((out.f_value - 20.0).abs() < 1e-9, "got {}", out.f_value);
        assert_eq!(out.div, 0.01);
    }

    // ---- (e) validation ----------------------------------------------------

    #[test]
    fn the_configuration_is_validated_before_anything_runs() {
        let pts = line_five();
        let mut util = Linear::uniform(pts.n());

        let bad_k = GistConfig {
            k: 0,
            ..Default::default()
        };
        assert_eq!(gist(&pts, &mut util, &bad_k), Err(DivselError::InvalidK));

        for eps in [0.0f32, -0.1, 1.5] {
            let cfg = GistConfig {
                eps,
                ..Default::default()
            };
            assert_eq!(
                gist(&pts, &mut util, &cfg),
                Err(DivselError::InvalidEps(eps)),
                "eps = {eps}"
            );
        }
        // NaN does not compare equal, so match on the shape.
        let nan_eps = GistConfig {
            eps: f32::NAN,
            ..Default::default()
        };
        match gist(&pts, &mut util, &nan_eps).unwrap_err() {
            DivselError::InvalidEps(eps) => assert!(eps.is_nan()),
            other => panic!("expected InvalidEps, got {other:?}"),
        }

        for lambda in [-1.0f64, f64::INFINITY] {
            let cfg = GistConfig {
                lambda,
                ..Default::default()
            };
            assert_eq!(
                gist(&pts, &mut util, &cfg),
                Err(DivselError::InvalidLambda(lambda)),
                "lambda = {lambda}"
            );
        }

        // The utility gets the last word.
        let mut short = Linear::new(vec![1.0, 2.0]);
        assert_eq!(
            gist(&pts, &mut short, &GistConfig::default()),
            Err(DivselError::WeightsLength {
                expected: 5,
                got: 2
            })
        );
    }

    // ---- (f) the budget and a one-point set --------------------------------

    #[test]
    fn k_above_n_selects_everything_and_a_single_point_stays_at_greedy() {
        let pts = line_five();
        let mut util = Linear::new(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let cfg = GistConfig {
            k: 99,
            lambda: 0.0,
            ..Default::default()
        };
        let out = gist(&pts, &mut util, &cfg).expect("valid configuration");
        let mut got = out.selected.clone();
        got.sort_unstable();
        assert_eq!(got, vec![0, 1, 2, 3, 4]);
        assert!((out.f_value - 15.0).abs() < 1e-9, "got {}", out.f_value);
        assert!((out.g_value - 15.0).abs() < 1e-9);
        assert_eq!(out.div, 1.0);

        // A single point: the diameter is 0, so div(S) is 0 as well, the pair
        // branch is skipped for want of two points, and every threshold in D is
        // zero -- the sweep would only re-run line 2, so it does not run at all.
        let one = Points::new(vec![0.5, -1.5], 2, Metric::Euclidean).expect("one point");
        let mut util = Linear::uniform(1);
        let out = gist(&one, &mut util, &GistConfig::default()).expect("valid configuration");
        assert_eq!(out.selected, vec![0]);
        assert_eq!(out.div, 0.0);
        assert_eq!(out.d_max, 0.0);
        assert_eq!(out.threshold, 0.0);
        assert_eq!(out.stage, Stage::Greedy);
        assert!((out.f_value - 1.0).abs() < 1e-12);
    }

    // ---- (g) the exhaustive threshold set ----------------------------------

    /// Twelve points in 3-D with bit-stable coordinates and weights: `next_f32`
    /// and `uniform_weights` are integer draws scaled by powers of two, so this
    /// fixture is identical on every platform (unlike `gaussian_points`).
    fn twelve_points() -> (Points<'static>, Linear) {
        let mut rng = SplitMix64(0x0001_5e70_0000_0107);
        let data: Vec<f32> = (0..12 * 3).map(|_| rng.next_f32() * 4.0 - 2.0).collect();
        let pts = Points::new(data, 3, Metric::Euclidean).expect("twelve points");
        let weights = rng.uniform_weights(12);
        (pts, Linear::new(weights))
    }

    #[test]
    fn the_exhaustive_threshold_set_is_at_least_as_good_as_the_geometric_one() {
        let (pts, mut util) = twelve_points();
        let base = GistConfig {
            k: 4,
            lambda: 0.5,
            ..Default::default()
        };

        let set = exhaustive_threshold_set(&pts);
        // n*(n-1)/2 distinct pairs plus the zero from the u == v pairs, minus any
        // exact duplicates.
        assert!(set.len() > 1 && set.len() <= 12 * 11 / 2 + 1, "got {set:?}");
        assert_eq!(set[0], 0.0, "the paper's set includes dist(u,u)/2 = 0");
        for pair in set.windows(2) {
            assert!(pair[1] > pair[0], "not ascending or not deduplicated");
        }
        assert_eq!(*set.last().expect("non-empty"), pts.diameter().0 / 2.0);

        let geometric = gist(&pts, &mut util, &base).expect("valid configuration");
        let exhaustive = gist(
            &pts,
            &mut util,
            &GistConfig {
                exhaustive_thresholds: true,
                ..base.clone()
            },
        )
        .expect("valid configuration");
        assert!(
            exhaustive.f_value >= geometric.f_value - 1e-9,
            "exhaustive f = {} below geometric f = {}",
            exhaustive.f_value,
            geometric.f_value
        );
        // That inequality holds on this fixture, and it is deterministic here --
        // the coordinates come from the bit-stable integer draws. It is NOT a
        // theorem, and a future reader should not generalize it: over 300 random
        // 12-point instances crossed with k in {2,3,4,6} and lambda in
        // {0.1,0.5,2.0}, the exhaustive sweep scored *below* the geometric one in
        // 778 of 3600 runs and strictly above it in 8. The reason is the ceiling
        // noted on `exhaustive_threshold_set`: the exhaustive set stops at
        // d_max/2, so it never tries the high thresholds that drive div up.
        // What does hold in both modes is line-2 dominance, asserted next.
        for out in [&geometric, &exhaustive] {
            assert_eq!(out.selected.len(), base.k);
            assert!(out.f_value > 0.0);
            assert!(div_with_dmax(&pts, &out.selected, out.d_max) >= out.threshold);
            let on_g = greedy_independent_set(&pts, &mut util, 0.0, base.k);
            let f_on_g = f_of(&pts, &mut util, &on_g, base.lambda, out.d_max);
            assert!(
                out.f_value >= f_on_g,
                "f = {} fell below the classic greedy solution's {f_on_g}",
                out.f_value
            );
        }
    }

    // ---- (h) determinism ---------------------------------------------------

    #[test]
    fn two_runs_with_the_same_inputs_agree_exactly() {
        let (pts, weights) = twelve_points();
        let cfg = GistConfig {
            k: 5,
            lambda: 0.75,
            eps: 0.25,
            ..Default::default()
        };

        let run = |cfg: &GistConfig| -> GistResult {
            let mut util = weights.clone();
            gist(&pts, &mut util, cfg).expect("valid configuration")
        };
        assert_eq!(run(&cfg), run(&cfg));

        // A single utility reused across calls must give the same answer as a
        // fresh one: gist resets it on entry and on exit.
        let mut util = weights.clone();
        let first = gist(&pts, &mut util, &cfg).expect("valid configuration");
        let second = gist(&pts, &mut util, &cfg).expect("valid configuration");
        assert_eq!(first, second);
        assert_eq!(first, run(&cfg));

        // Same for the exhaustive set and for the approximate diameter.
        for cfg in [
            GistConfig {
                exhaustive_thresholds: true,
                ..cfg.clone()
            },
            GistConfig {
                diameter: DiameterMode::Approx { sweeps: 3 },
                ..cfg.clone()
            },
        ] {
            assert_eq!(run(&cfg), run(&cfg));
        }
    }

    // ---- (i) the approximate diameter --------------------------------------

    #[test]
    fn the_approximate_diameter_is_within_a_factor_of_two() {
        const N: usize = 100;
        const DIM: usize = 8;

        let mut rng = SplitMix64(0x0001_5e70_0000_0109);
        let mut exact_hits = 0usize;
        let mut ratio_sum = 0.0f64;
        for trial in 0..30 {
            let pts = Points::new(rng.gaussian_points(N, DIM), DIM, Metric::Euclidean)
                .expect("random points");
            let (d_max, _, _) = pts.diameter();
            let (d_hat, u, v) = approx_diameter(&pts, 2);

            assert!(
                d_hat >= d_max / 2.0 && d_hat <= d_max,
                "trial {trial}: d_hat = {d_hat} outside [{}, {d_max}]",
                d_max / 2.0
            );
            assert!(u < v, "trial {trial}: pair ({u}, {v}) is not ordered");
            assert_eq!(
                pts.dist(u, v),
                d_hat,
                "trial {trial}: the reported pair does not realize d_hat"
            );
            if d_hat == d_max {
                exact_hits += 1;
            }
            ratio_sum += f64::from(d_hat) / f64::from(d_max);
        }
        // Guards the guard: the factor-of-two bound above is loose enough that a
        // sloppy sweep would still clear it -- the worst ratio measured here is
        // 0.7998, comfortably above 0.5. These two pin the quality instead.
        // Measured here: 14 of 30 double sweeps land on the exact diameter, mean
        // ratio 0.9631. Gaussian coordinates are not bit-stable across platforms,
        // so both thresholds carry generous headroom.
        assert!(
            exact_hits >= 8,
            "only {exact_hits} of 30 double sweeps found the exact diameter"
        );
        assert!(
            ratio_sum / 30.0 >= 0.9,
            "mean d_hat / d_max is only {}",
            ratio_sum / 30.0
        );

        // Degenerate inputs, and the sweeps = 0 convention.
        let one = Points::new(vec![1.0, 2.0], 2, Metric::Euclidean).expect("one point");
        assert_eq!(approx_diameter(&one, 4), (0.0, 0, 0));
        let pts = line_five();
        assert_eq!(approx_diameter(&pts, 0), approx_diameter(&pts, 1));
        assert_eq!(approx_diameter(&pts, 1), (12.0, 0, 4));

        // The driver uses d_hat as its d_max, and stays at least as good as the
        // classic greedy solution it starts from.
        let (pts, mut util) = twelve_points();
        let cfg = GistConfig {
            k: 4,
            lambda: 0.5,
            diameter: DiameterMode::Approx { sweeps: 2 },
            ..Default::default()
        };
        let out = gist(&pts, &mut util, &cfg).expect("valid configuration");
        let (d_hat, _, _) = approx_diameter(&pts, 2);
        assert_eq!(out.d_max, d_hat);
        let on_g = greedy_independent_set(&pts, &mut util, 0.0, cfg.k);
        let f_on_g = f_of(&pts, &mut util, &on_g, cfg.lambda, d_hat);
        assert!(out.f_value >= f_on_g);
    }

    // ---- (g) the parallel threshold sweep ---------------------------------

    /// A [`Linear`] utility that records whether any of its marginals was
    /// evaluated on a `rayon` worker thread.
    ///
    /// [`Utility::boxed_clone`] shares the flag, so every clone the sweep hands
    /// to a worker reports into the same place.
    #[derive(Clone)]
    struct PoolWitness {
        weights: Vec<f64>,
        on_pool: Arc<AtomicBool>,
    }

    impl Utility for PoolWitness {
        fn marginal(&self, v: usize, _selected: &[usize], _pts: &Points<'_>) -> f64 {
            if rayon::current_thread_index().is_some() {
                self.on_pool.store(true, atomic::Ordering::Relaxed);
            }
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
            Ok(())
        }

        fn boxed_clone(&self) -> Box<dyn Utility> {
            Box::new(self.clone())
        }
    }

    /// The sweep must actually run on the thread pool, not merely produce the
    /// same answer as a sequential one.
    ///
    /// `gist` is called from the test's own thread, which is not a `rayon`
    /// worker, so `rayon::current_thread_index()` is `None` everywhere a
    /// sequential sweep could evaluate a marginal. Observing `Some(_)` therefore
    /// proves a marginal ran inside the pool. This is the assertion a sequential
    /// implementation fails; the determinism test below cannot fail for it,
    /// since a sequential sweep satisfies determinism trivially.
    #[test]
    fn the_threshold_sweep_runs_on_the_rayon_pool() {
        assert!(
            rayon::current_thread_index().is_none(),
            "this test has to run outside a rayon worker for the probe to mean anything"
        );

        let mut rng = SplitMix64(0x9e37_79b9_5eed_0001);
        let pts = Points::new(rng.gaussian_points(200, 8), 8, Metric::Euclidean)
            .expect("gaussian point set");
        let flag = Arc::new(AtomicBool::new(false));
        let mut util = PoolWitness {
            weights: rng.uniform_weights(pts.n()),
            on_pool: Arc::clone(&flag),
        };

        let out = gist(&pts, &mut util, &GistConfig::default()).expect("valid configuration");
        assert!(!out.selected.is_empty());
        assert!(
            flag.load(atomic::Ordering::Relaxed),
            "no marginal was evaluated on a rayon worker: the threshold sweep is still sequential"
        );
    }

    /// Twenty random instances, each run twice: once on a one-thread pool and
    /// once on the default pool. The two [`GistResult`]s must be identical, which
    /// is what pins the sweep's answer to the ascending, `>=` fold rather than to
    /// however `rayon` split the thresholds.
    #[test]
    fn the_sweep_is_independent_of_the_rayon_split() {
        let single = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .expect("a one-thread pool");

        for instance in 0..20u64 {
            let mut rng = SplitMix64(0x5eed_0000_0000_0000 ^ instance);
            let dim = 1 + (instance % 5) as usize;
            let n = 20 + (instance % 11) as usize;
            let pts = Points::new(rng.gaussian_points(n, dim), dim, Metric::Euclidean)
                .expect("gaussian point set");
            let cfg = GistConfig {
                k: 2 + (instance % 4) as usize,
                lambda: 0.25 + f64::from(instance as u32) / 10.0,
                eps: 0.3,
                exhaustive_thresholds: instance % 2 == 0,
                diameter: DiameterMode::Exact,
            };

            let linear = Linear::new(rng.uniform_weights(n));
            let mut a = linear.clone();
            let mut b = linear;
            let sequential = single.install(|| gist(&pts, &mut a, &cfg).expect("valid"));
            let parallel = gist(&pts, &mut b, &cfg).expect("valid");
            assert_eq!(
                sequential, parallel,
                "Linear instance {instance} depends on the rayon split"
            );

            let facility = FacilityLocation::new(&pts);
            let mut a = facility.clone();
            let mut b = facility;
            let sequential = single.install(|| gist(&pts, &mut a, &cfg).expect("valid"));
            let parallel = gist(&pts, &mut b, &cfg).expect("valid");
            assert_eq!(
                sequential, parallel,
                "FacilityLocation instance {instance} depends on the rayon split"
            );
        }
    }
}
