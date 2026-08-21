//! `divsel`'s benchmark harness.
//!
//! Two things are measured here:
//!
//! 1. **The distance kernels.** `Points::dist` goes through the `pulp`
//!    runtime-dispatched kernel; the `scalar` arm of the same group runs the
//!    identical loop over the identical rows through a copy of `metric.rs`'s
//!    scalar body. The ratio is the number that justifies taking a SIMD
//!    dependency at all. The two are bit-identical by contract (R-G22), so this
//!    is a pure speed comparison -- see
//!    `metric::tests::the_dispatched_kernels_are_bit_identical_to_the_scalar_ones`.
//! 2. **GIST end to end**, over the paper's shape of problem: `n` points in
//!    `dim` dimensions, budget `k`, `eps = 0.1`, Euclidean, with the
//!    `Linear` and `FacilityLocation` utilities.
//!
//! Everything is generated from `divsel::testutil::SplitMix64`, so a run is
//! reproducible from this repository alone -- no downloads, no fixtures, no
//! private dataset.
//!
//! # Running
//!
//! ```text
//! cargo bench -p divsel                              # kernels + n = 10_000
//! cargo bench -p divsel -- --quick kernel            # just the kernels, seconds
//! cargo bench -p divsel -- --quick 'gist/linear'     # the modular utility, minutes
//! cargo bench -p divsel --features bench-large       # adds n = 100_000 and 1_000_000
//! ```
//!
//! **Budget the time.** The argument after `--quick` is a name filter, and you
//! usually want one. On the reference machine below, a single iteration takes
//! roughly 0.4 s (`linear`, `dim = 64`, `k = 10`) to 7.6 s (`linear`,
//! `dim = 768`, `k = 100`), and 84 s to 122 s for `facility_location` at
//! `dim = 64`. Criterion takes at least ten samples per benchmark, so the
//! unfiltered default run is measured in hours, essentially all of it
//! `facility_location`: its marginal is `O(n * dim)`, which makes CELF's first
//! greedy round `O(n^2 * dim)` -- a full diameter scan's worth of work, once per
//! threshold. That group is kept at the brief's matrix rather than trimmed;
//! filter it out (or pass `--quick`) when you only want the kernel numbers.
//!
//! The `bench-large` groups are opt-in: `n = 1_000_000` at `dim = 768` alone
//! allocates about 3 GiB of `f32` and takes minutes to generate before any
//! measurement begins. They also use `DiameterMode::Approx { sweeps: 3 }`,
//! because an exact `O(n^2 * dim)` diameter at that size dwarfs the sweep it is
//! supposed to feed.
//!
//! # Reference machine
//!
//! The numbers recorded in `task-6-report.md` were taken on:
//!
//! ```text
//! CPU:    Intel(R) Core(TM) i7-10875H @ 2.30GHz (8 cores / 16 threads, AVX2, no AVX-512)
//! ISA:    pulp selected V3 -> 8 f32 lanes per register, 2 registers per 16-element group
//! OS:     Windows 11 Pro 26200
//! Rust:   rustc 1.92.0 (ded5c06cf 2025-12-08), release profile
//! Date:   2026-08-21
//! ```

use std::hint::black_box;
use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

use divsel::testutil::SplitMix64;
use divsel::{gist, DiameterMode, FacilityLocation, GistConfig, Linear, Metric, Points, Utility};

/// Embedding widths: a small projection, a sentence-transformer width, and an
/// OpenAI-style width.
const DIMS: [usize; 3] = [64, 384, 768];

/// Budgets from the brief's matrix.
const BUDGETS: [usize; 2] = [10, 100];

/// Rows in the kernel fixtures. 64 rows is 2016 unordered pairs per iteration,
/// which keeps criterion's per-iteration overhead far below the measurement even
/// at `dim = 64`.
const KERNEL_ROWS: usize = 64;

/// The `n` the default (non-`bench-large`) GIST groups use.
const N_DEFAULT: usize = 10_000;

// ---------------------------------------------------------------------------
// Scalar baselines
//
// `metric::dot_scalar` and `metric::sq_euclid_scalar` are `pub(crate)`, and this
// task deliberately changed no public API, so the baseline is mirrored here.
// These two functions are a verbatim copy of the scalar bodies in
// `crates/divsel/src/metric.rs`; the parity test in that module is what pins the
// dispatched kernels to them. If you edit one, edit both.
// ---------------------------------------------------------------------------

/// Accumulator lanes; mirrors `metric::LANES`.
const LANES: usize = 16;

/// Mirror of `metric::dot_scalar`.
#[allow(clippy::assign_op_pattern, clippy::needless_range_loop)]
fn dot_scalar(a: &[f32], b: &[f32]) -> f32 {
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

/// Mirror of `metric::sq_euclid_scalar`.
#[allow(clippy::assign_op_pattern, clippy::needless_range_loop)]
fn sq_euclid_scalar(a: &[f32], b: &[f32]) -> f32 {
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

// ---------------------------------------------------------------------------
// Kernel group
// ---------------------------------------------------------------------------

/// Sums `dist(i, j)` over every unordered pair, through the dispatched kernel.
fn pairwise_dispatched(pts: &Points<'_>) -> f32 {
    let mut total = 0.0f32;
    for i in 0..pts.n() {
        for j in (i + 1)..pts.n() {
            total += pts.dist(i, j);
        }
    }
    total
}

/// The same loop over the same rows, through the scalar Euclidean kernel.
fn pairwise_scalar_euclidean(pts: &Points<'_>) -> f32 {
    let mut total = 0.0f32;
    for i in 0..pts.n() {
        for j in (i + 1)..pts.n() {
            total += sq_euclid_scalar(pts.row(i), pts.row(j)).sqrt();
        }
    }
    total
}

/// The same loop over the same rows, through the scalar cosine kernel.
fn pairwise_scalar_cosine(pts: &Points<'_>) -> f32 {
    let mut total = 0.0f32;
    for i in 0..pts.n() {
        for j in (i + 1)..pts.n() {
            total += (1.0 - dot_scalar(pts.row(i), pts.row(j))).clamp(0.0, 2.0);
        }
    }
    total
}

/// `kernel/sq_euclid` and `kernel/dot`: dispatched versus scalar at each width.
fn kernels(c: &mut Criterion) {
    for &dim in &DIMS {
        let mut rng = SplitMix64(0x5eed_0000_0000_0001 ^ dim as u64);
        let data = rng.gaussian_points(KERNEL_ROWS, dim);
        let euclidean = Points::new(data.clone(), dim, Metric::Euclidean)
            .expect("gaussian points are a valid point set");
        let cosine =
            Points::new(data, dim, Metric::Cosine).expect("gaussian rows have a usable norm");

        let mut group = c.benchmark_group("kernel/sq_euclid");
        group.bench_function(BenchmarkId::new("dispatched", dim), |b| {
            b.iter(|| pairwise_dispatched(black_box(&euclidean)))
        });
        group.bench_function(BenchmarkId::new("scalar", dim), |b| {
            b.iter(|| pairwise_scalar_euclidean(black_box(&euclidean)))
        });
        group.finish();

        let mut group = c.benchmark_group("kernel/dot");
        group.bench_function(BenchmarkId::new("dispatched", dim), |b| {
            b.iter(|| pairwise_dispatched(black_box(&cosine)))
        });
        group.bench_function(BenchmarkId::new("scalar", dim), |b| {
            b.iter(|| pairwise_scalar_cosine(black_box(&cosine)))
        });
        group.finish();
    }
}

/// `dispatch/arch_new`: what one `pulp::Arch::new()` costs.
///
/// This is the measurement behind caching the detected instruction set in a
/// `OnceLock` instead of re-detecting inside every `dist` call. Compare it
/// against one `kernel/sq_euclid/dispatched/64` iteration divided by 2016 pairs.
fn dispatch(c: &mut Criterion) {
    let mut group = c.benchmark_group("dispatch");
    group.bench_function("arch_new", |b| b.iter(|| black_box(pulp::Arch::new())));
    group.finish();
}

// ---------------------------------------------------------------------------
// GIST groups
// ---------------------------------------------------------------------------

/// The configuration every GIST group shares, apart from `k` and the diameter
/// mode: the brief's `eps = 0.1`, the paper's geometric threshold set.
fn config(k: usize, diameter: DiameterMode) -> GistConfig {
    GistConfig {
        k,
        lambda: 1.0,
        eps: 0.1,
        exhaustive_thresholds: false,
        diameter,
    }
}

/// Benches `gist` over one `(n, dim)` fixture for both budgets.
///
/// `util` is rebuilt per budget but reused across iterations: `gist` returns it
/// reset, so a second call sees exactly the state the first one did.
fn bench_gist(
    c: &mut Criterion,
    group_name: &str,
    pts: &Points<'_>,
    dim: usize,
    diameter: DiameterMode,
    mut make_util: impl FnMut() -> Box<dyn Utility>,
) {
    let mut group = c.benchmark_group(group_name);
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    for &k in &BUDGETS {
        let cfg = config(k, diameter);
        let mut util = make_util();
        group.bench_function(
            BenchmarkId::new(format!("dim={dim}"), format!("k={k}")),
            |b| b.iter(|| gist(black_box(pts), util.as_mut(), &cfg).expect("valid configuration")),
        );
    }
    group.finish();
}

/// Builds `n` Gaussian points in `dim` dimensions under the Euclidean metric.
fn fixture(n: usize, dim: usize, seed: u64) -> (Points<'static>, Vec<f64>) {
    let mut rng = SplitMix64(seed);
    let pts = Points::new(rng.gaussian_points(n, dim), dim, Metric::Euclidean)
        .expect("gaussian points are a valid point set");
    let weights = rng.uniform_weights(n);
    (pts, weights)
}

/// `gist/linear/n=10000`: the modular utility, exact diameter.
fn gist_linear(c: &mut Criterion) {
    for &dim in &DIMS {
        let (pts, weights) = fixture(N_DEFAULT, dim, 0x9e37_79b9_0000_0001 ^ dim as u64);
        bench_gist(
            c,
            "gist/linear/n=10000",
            &pts,
            dim,
            DiameterMode::Exact,
            || Box::new(Linear::new(weights.clone())),
        );
    }
}

/// `gist/facility_location/n=10000`: the paper's own utility.
///
/// Per R-G21 this runs at `n = 10_000` only. Its marginal is `O(n * dim)`, so a
/// single greedy round costs `O(n^2 * dim)` -- the same order as a full diameter
/// scan, once per threshold.
///
/// The similarity scale is taken from the exact diameter here rather than by
/// `FacilityLocation::new`, so the setup pays for that scan once instead of once
/// per constructed utility.
fn gist_facility_location(c: &mut Criterion) {
    for &dim in &DIMS {
        let (pts, _) = fixture(N_DEFAULT, dim, 0x1234_5678_0000_0002 ^ dim as u64);
        let scale = pts.diameter().0;
        bench_gist(
            c,
            "gist/facility_location/n=10000",
            &pts,
            dim,
            DiameterMode::Exact,
            || Box::new(FacilityLocation::with_scale(pts.n(), scale)),
        );
    }
}

/// `gist/linear/n=100000` and `gist/linear/n=1000000`, behind `bench-large`.
///
/// Both use `DiameterMode::Approx { sweeps: 3 }`: an exact diameter is
/// `O(n^2 * dim)` pair distances, which at `n = 100_000` is already 5e9 pairs and
/// at `n = 1_000_000` is 5e11 -- minutes to hours of setup for a number the sweep
/// only needs to within a factor of two.
#[cfg(feature = "bench-large")]
fn gist_large(c: &mut Criterion) {
    for &n in &[100_000usize, 1_000_000] {
        for &dim in &DIMS {
            let (pts, weights) = fixture(n, dim, 0x0bad_c0de_0000_0003 ^ (n as u64) ^ dim as u64);
            bench_gist(
                c,
                &format!("gist/linear/n={n}"),
                &pts,
                dim,
                DiameterMode::Approx { sweeps: 3 },
                || Box::new(Linear::new(weights.clone())),
            );
        }
    }
}

/// Placeholder so the group list is the same with and without `bench-large`.
#[cfg(not(feature = "bench-large"))]
fn gist_large(_c: &mut Criterion) {}

// `criterion_group!`'s named form calls `configure_from_args()` on the config it
// is given, so `--quick`, `--sample-size` and friends still apply.
criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = kernels, dispatch, gist_linear, gist_facility_location, gist_large
}
criterion_main!(benches);
