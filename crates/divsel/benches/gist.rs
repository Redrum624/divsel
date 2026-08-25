//! `divsel`'s benchmark harness.
//!
//! Two things are measured here:
//!
//! 1. **The distance kernels.** `Points::dist` goes through the `pulp`
//!    runtime-dispatched kernel; the `scalar` arm of the same group runs the
//!    identical loop over the identical rows through `metric.rs`'s scalar body
//!    itself, re-exported by `divsel::testutil`. The ratio is the number that
//!    justifies taking a SIMD dependency at all. The two are bit-identical by
//!    contract (R-G22), so this is a pure speed comparison -- see
//!    `metric::tests::the_dispatched_kernels_are_bit_identical_to_the_scalar_ones`.
//! 2. **GIST end to end**, over the paper's shape of problem: `n` points in
//!    `dim` dimensions, budget `k`, `eps = 0.1`, Euclidean, with the
//!    `Linear` and `FacilityLocation` utilities.
//!
//! Everything is generated from `divsel::testutil::SplitMix64`, so a run is
//! reproducible from this repository alone -- no downloads, no fixtures, no
//! private dataset.
//!
//! # What "scalar" means here
//!
//! The baseline arm is **not** one-element-at-a-time code. `divsel` is built at
//! the default `x86-64` target baseline (no `.cargo/config.toml`, no
//! `RUSTFLAGS`, so `sse`, `sse2` and `sse3` are on and nothing wider is), and at
//! `-O` LLVM auto-vectorises the scalar body's 16-accumulator inner loop into
//! four SSE registers of four lanes each: `rustc -O --emit asm` on that body
//! emits **4 `mulps` + 4 `addps`** in the group loop, with `mulss`/`addss` only
//! in the tail and the final left-to-right reduction, and **zero** `vfmadd`.
//! Bit-identity survives that because the vectorisation is lane-wise -- element
//! `i` still lands in accumulator `i % 16` -- and because Rust emits no
//! fast-math flags, so LLVM may neither reassociate the sum nor contract the
//! multiply and the add into an FMA.
//!
//! So the ratio below is **dispatched (AVX2, 8 lanes) against an auto-vectorised
//! SSE2 baseline (4 lanes)**, not against truly scalar code. A 1.8x-4.4x
//! measured speedup over a 4-lane baseline is a different -- and more honest --
//! claim than the same number over a 1-lane one.
//!
//! # Tiers
//!
//! Exactly two, and every cell is in exactly one of them.
//!
//! ## DEFAULT -- `cargo bench -p divsel`
//!
//! Must finish in single-digit minutes. Measured end to end at **4 min 13 s**
//! on the reference machine below, every cell, no filter.
//!
//! | cell | samples | per iteration (measured) |
//! |---|---|---|
//! | `kernel/sq_euclid/{dispatched,scalar}/{64,384,768}` | 100 | 39.7 us - 656 us |
//! | `kernel/dot/{dispatched,scalar}/{64,384,768}` | 100 | 40.4 us - 732 us |
//! | `dispatch/arch_new` | 100 | 565 ps |
//! | `gist/linear/n=10000/dim={64,384,768}/k={10,100}` | 10 | 0.20 s - 3.20 s |
//!
//! ## `--features bench-large` -- `cargo bench -p divsel --features bench-large`
//!
//! Everything whose per-iteration cost is minutes, or whose fixture is
//! gigabytes. This tier is hours long; run it one filter at a time.
//!
//! | cell | samples | per iteration |
//! |---|---|---|
//! | `gist/facility_location/n=10000/dim=768/k=10` | 10 | UNMEASURED, est. 345 s (see note) |
//! | `gist/facility_location/n=10000/dim=64/k={10,100}` | 10 | 84 s / 122 s (`--quick`, see below) |
//! | `gist/facility_location/n=10000/dim=384/k={10,100}` | 10 | 239 s / 245 s (`--quick`, see below) |
//! | `gist/facility_location/n=10000/dim=768/k=100` | 10 | UNMEASURED, est. 355-660 s |
//! | `gist/linear/n=100000/dim={64,384,768}/k={10,100}`, exact diameter | 10 | UNMEASURED, est. 15-90 s |
//! | `gist/linear/n=1000000/dim={64,384,768}/k={10,100}`, `Approx { sweeps: 3 }` | 10 | UNMEASURED, est. 10-60 s + minutes of setup |
//!
//! The `dim=768` rows are estimates, not measurements. A full ten-sample run of
//! `k=10` was attempted and interrupted after 2905 s of the 3449 s criterion
//! projected from its own warm-up iteration (345 s/iteration); nothing was
//! recorded. The `k=100` range scales that by the 1.03x-1.9x that k=10 -> k=100
//! costs in every pair measured here. To produce either number:
//!
//! ```text
//! cargo bench -p divsel --features bench-large -- --exact "gist/facility_location/n=10000/dim=768/k=10"
//! cargo bench -p divsel --features bench-large -- --exact "gist/facility_location/n=10000/dim=768/k=100"
//! ```
//!
//! `--exact` matters: a criterion filter is a substring match, so a bare
//! `k=10` selects `k=100` as well.
//!
//! The two `--quick` rows are the only figures here that did not come from a
//! full ten-sample run, and they read **high**. `--quick` shortens criterion's
//! measurement, and on this harness its per-iteration estimates came out about
//! twice the full-run ones: `gist/linear/n=10000/dim=768/k=100` measured 7.65 s
//! under `--quick` and 3.20 s in a full run of the *current* code -- which does
//! one greedy run *more* per `gist` call than the code `--quick` measured. The
//! gap is the measurement protocol, not the algorithm, so treat those two rows
//! as upper bounds and re-measure before quoting them:
//!
//! ```text
//! cargo bench -p divsel --features bench-large -- --exact "gist/facility_location/n=10000/dim=64/k=10"
//! cargo bench -p divsel --features bench-large -- --exact "gist/facility_location/n=10000/dim=64/k=100"
//! cargo bench -p divsel --features bench-large -- --exact "gist/facility_location/n=10000/dim=384/k=10"
//! cargo bench -p divsel --features bench-large -- --exact "gist/facility_location/n=10000/dim=384/k=100"
//! ```
//!
//! The `n = 100_000` cells keep the **exact** diameter the brief pins for them.
//! Their estimate is arithmetic, not a measurement: `5e9 pairs * the measured
//! per-pair dispatched kernel cost (19.7 ns at dim 64, 78.8 ns at dim 768) / 16
//! threads` is 6-25 s of diameter scan at perfect scaling, and the sweep adds
//! more on top. Only `n = 1_000_000` uses
//! `DiameterMode::Approx { sweeps: 3 }`, as pinned, because an exact `O(n^2)`
//! diameter there is 5e11 pairs. Its fixture is 3 GiB of `f32` at `dim = 768`
//! and takes minutes to generate before any measurement starts.
//!
//! Commands for the unmeasured cells, one per line:
//!
//! ```text
//! cargo bench -p divsel --features bench-large -- "gist/facility_location/n=10000/dim=768/k=100"
//! cargo bench -p divsel --features bench-large -- "gist/linear/n=100000"
//! cargo bench -p divsel --features bench-large -- "gist/linear/n=1000000"
//! ```
//!
//! A filter is a *substring* match, so `k=10` also selects `k=100`; pass
//! `--exact <full benchmark name>` when you want one cell and only that cell.
//! Fixtures are built lazily inside the benchmark routine (see `LazyFixture`),
//! so a filtered run does not pay for the point sets of the groups it skips --
//! which is what makes the commands above usable one at a time.
//!
//! # Running
//!
//! ```text
//! cargo bench -p divsel                             # the DEFAULT tier
//! cargo bench -p divsel -- --quick kernel           # just the kernels, seconds
//! cargo bench -p divsel -- "gist/linear"            # the modular utility
//! cargo bench -p divsel --features bench-large -- <filter>   # one heavy cell
//! ```
//!
//! The argument after `--` is a name filter, and in the `bench-large` tier you
//! always want one: `facility_location`'s marginal is `O(n * dim)`, which makes
//! CELF's first greedy round `O(n^2 * dim)` -- a full diameter scan's worth of
//! work, once per threshold.
//!
//! # Reference machine
//!
//! The numbers recorded here and in `task-6-report.md` were taken on:
//!
//! ```text
//! CPU:    Intel(R) Core(TM) i7-10875H @ 2.30GHz (8 cores / 16 threads, AVX2, no AVX-512)
//! ISA:    pulp selected V3 -> 8 f32 lanes per register, 2 registers per 16-element group
//! OS:     Windows 11 Pro 26200
//! Rust:   rustc 1.92.0 (ded5c06cf 2025-12-08), release profile, default x86-64 target
//! Date:   2026-08-21
//! ```

use std::hint::black_box;
use std::sync::OnceLock;
use std::time::Duration;

use criterion::{criterion_group, BenchmarkId, Criterion};

// `dot_scalar` / `sq_euclid_scalar` are `metric.rs`'s own bodies, re-exported
// through the doc-hidden `testutil` module. This file used to carry a verbatim
// copy of both -- and a copy is what the crate's headline speedup would then
// have been divided by, with nothing pinning it to the real thing.
use divsel::testutil::{dot_scalar, sq_euclid_scalar, SplitMix64};
use divsel::{gist, DiameterMode, GistConfig, Linear, Metric, Points, Utility};

// Only the `bench-large` tier runs the paper's utility.
#[cfg(feature = "bench-large")]
use divsel::FacilityLocation;

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
///
/// It is the **steady-state** cost, not the cost of feature detection.
/// `pulp::Arch::new()` goes through `is_x86_feature_detected!`, which caches the
/// CPUID result in a process-global after its first call; criterion's warm-up
/// pays that once and every timed iteration then reads the cache. Cold detection
/// happens once per process and is not something a timing loop can measure.
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

/// `n` Gaussian points in `dim` dimensions under the Euclidean metric, plus one
/// uniform weight per point -- built **at most once, and only if criterion
/// actually runs a benchmark that needs them**.
///
/// That laziness is the difference between a usable `bench-large` tier and an
/// unusable one. Criterion invokes a `bench_function` routine only when the name
/// filter matches it (`benchmark_group.rs`: `let do_run =
/// self.criterion.filter_matches(id.id())`), but it cannot skip work this file
/// does *outside* a routine. Building the fixture eagerly would mean that
/// `cargo bench --features bench-large -- <one small filter>` still generated
/// every other group's points first -- 3 GiB of `f32` for `n = 1_000_000` at
/// `dim = 768`. Building it inside the routine, behind a `OnceLock`, skips it
/// entirely for a filtered-out group and still builds it exactly once for a
/// group that runs, without this file re-implementing criterion's filter.
struct LazyFixture {
    n: usize,
    dim: usize,
    seed: u64,
    built: OnceLock<(Points<'static>, Vec<f64>)>,
}

impl LazyFixture {
    /// Declares a fixture. Nothing is generated until [`LazyFixture::get`].
    fn new(n: usize, dim: usize, seed: u64) -> Self {
        Self {
            n,
            dim,
            seed,
            built: OnceLock::new(),
        }
    }

    /// The points and weights, generating them on the first call.
    fn get(&self) -> &(Points<'static>, Vec<f64>) {
        self.built.get_or_init(|| {
            let mut rng = SplitMix64(self.seed);
            let pts = Points::new(
                rng.gaussian_points(self.n, self.dim),
                self.dim,
                Metric::Euclidean,
            )
            .expect("gaussian points are a valid point set");
            let weights = rng.uniform_weights(self.n);
            (pts, weights)
        })
    }
}

/// Benches `gist` over one fixture for both budgets.
///
/// The fixture and the utility are both materialised inside the routine, so a
/// filtered-out cell costs nothing; see [`LazyFixture`]. The utility is built
/// once per budget and reused across iterations -- `gist` returns it reset, so a
/// second call sees exactly the state the first one did -- and only the `gist`
/// call itself is inside `b.iter`.
fn bench_gist(
    c: &mut Criterion,
    group_name: &str,
    fixture: &LazyFixture,
    diameter: DiameterMode,
    make_util: impl Fn(&Points<'_>, &[f64]) -> Box<dyn Utility>,
) {
    let mut group = c.benchmark_group(group_name);
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    for &k in &BUDGETS {
        let cfg = config(k, diameter);
        let mut util: Option<Box<dyn Utility>> = None;
        group.bench_function(
            BenchmarkId::new(format!("dim={}", fixture.dim), format!("k={k}")),
            |b| {
                let (pts, weights) = fixture.get();
                let util = util.get_or_insert_with(|| make_util(pts, weights));
                b.iter(|| gist(black_box(pts), util.as_mut(), &cfg).expect("valid configuration"));
            },
        );
    }
    group.finish();
}

/// `gist/linear/n=10000`: the modular utility, exact diameter. **DEFAULT tier.**
fn gist_linear(c: &mut Criterion) {
    for &dim in &DIMS {
        let fixture = LazyFixture::new(N_DEFAULT, dim, 0x9e37_79b9_0000_0001 ^ dim as u64);
        bench_gist(
            c,
            "gist/linear/n=10000",
            &fixture,
            DiameterMode::Exact,
            |_, weights| Box::new(Linear::new(weights.to_vec())),
        );
    }
}

/// `gist/facility_location/n=10000`: the paper's own utility. **`bench-large`.**
///
/// Per R-G21 this runs at `n = 10_000` only. Its marginal is `O(n * dim)`, so a
/// single greedy round costs `O(n^2 * dim)` -- the same order as a full diameter
/// scan, once per threshold. That is minutes per *iteration* on the reference
/// machine -- see the tier table in the module header -- and criterion takes at
/// least ten samples, so this group alone is a multi-hour run: it cannot sit in
/// the tier `cargo bench -p divsel` runs unfiltered.
///
/// `FacilityLocation::new` takes its similarity scale from the exact Euclidean
/// diameter, which is one extra `O(n^2 * dim)` scan per budget -- seconds
/// against iterations measured in minutes, and it happens outside `b.iter`.
#[cfg(feature = "bench-large")]
fn gist_facility_location(c: &mut Criterion) {
    for &dim in &DIMS {
        let fixture = LazyFixture::new(N_DEFAULT, dim, 0x1234_5678_0000_0002 ^ dim as u64);
        bench_gist(
            c,
            "gist/facility_location/n=10000",
            &fixture,
            DiameterMode::Exact,
            |pts, _| Box::new(FacilityLocation::new(pts)),
        );
    }
}

/// `gist/linear/n=100000`, behind `bench-large`, with the **exact** diameter the
/// brief pins for it.
///
/// Exact here means 5e9 pair distances before the sweep starts -- tens of
/// seconds on sixteen threads at `dim = 64` and rather more at `dim = 768`,
/// which is why it is in this tier and not the default one. It is the only cell
/// that measures the exact-diameter path above `n = 10_000`.
#[cfg(feature = "bench-large")]
fn gist_linear_100k(c: &mut Criterion) {
    const N: usize = 100_000;
    for &dim in &DIMS {
        let fixture = LazyFixture::new(N, dim, 0x0bad_c0de_0000_0003 ^ (N as u64) ^ dim as u64);
        bench_gist(
            c,
            "gist/linear/n=100000",
            &fixture,
            DiameterMode::Exact,
            |_, weights| Box::new(Linear::new(weights.to_vec())),
        );
    }
}

/// `gist/linear/n=1000000`, behind `bench-large`, with
/// `DiameterMode::Approx { sweeps: 3 }` as pinned.
///
/// An exact diameter here would be 5e11 pair distances -- hours of setup for a
/// number the sweep only needs to within a factor of two. The fixture alone is
/// about 3 GiB of `f32` at `dim = 768` and takes minutes to generate, which is
/// why [`LazyFixture`] exists.
#[cfg(feature = "bench-large")]
fn gist_linear_1m(c: &mut Criterion) {
    const N: usize = 1_000_000;
    for &dim in &DIMS {
        let fixture = LazyFixture::new(N, dim, 0x0bad_c0de_0000_0003 ^ (N as u64) ^ dim as u64);
        bench_gist(
            c,
            "gist/linear/n=1000000",
            &fixture,
            DiameterMode::Approx { sweeps: 3 },
            |_, weights| Box::new(Linear::new(weights.to_vec())),
        );
    }
}

/// Everything in the `bench-large` tier, in ascending order of cost.
#[cfg(feature = "bench-large")]
fn gist_large(c: &mut Criterion) {
    gist_facility_location(c);
    gist_linear_100k(c);
    gist_linear_1m(c);
}

/// Placeholder so the target list is the same with and without `bench-large`.
#[cfg(not(feature = "bench-large"))]
fn gist_large(_c: &mut Criterion) {}

// `criterion_group!`'s named form calls `configure_from_args()` on the config it
// is given, so `--quick`, `--sample-size` and friends still apply.
criterion_group! {
    name = benches;
    config = Criterion::default();
    targets = kernels, dispatch, gist_linear, gist_large
}

/// The body [`criterion_main!`](criterion::criterion_main) would generate, with
/// one guard in front of it.
///
/// `Cargo.toml`'s `test = false` keeps this target out of a plain `cargo test`,
/// but **not** out of `cargo test --all-targets`: that form selects every target
/// regardless of the manifest's `test`/`bench` flags, and criterion then sees a
/// command line without `--bench`, which is exactly how it decides to run in its
/// libtest-compatible `Mode::Test` -- every benchmark once, fixtures and all. On
/// this file that means the whole DEFAULT tier in an unoptimised build, and with
/// `--all-features` the `bench-large` tier's ~3 GiB `n = 1000000` fixture too.
///
/// So: no `--bench` and no `--list` means we were run as a test, and a benchmark
/// is not a test. `cargo bench -- --test` still reaches criterion's own test mode
/// (cargo passes `--bench` there), which is the supported way to run every cell
/// once as a smoke check.
fn main() {
    let mut is_bench = false;
    let mut is_list = false;
    for arg in std::env::args() {
        is_bench |= arg == "--bench";
        is_list |= arg == "--list";
    }
    if !is_bench && !is_list {
        println!("benches/gist.rs is a benchmark, not a test: skipping the run");
        println!("that `cargo test --all-targets` asks for, which would build every");
        println!("fixture -- gigabytes of them under --all-features -- in a debug build.");
        println!("Run `cargo bench -p divsel`, or `cargo bench -p divsel -- --test`");
        println!("to execute every cell once.");
        return;
    }

    benches();
    Criterion::default().configure_from_args().final_summary();
}
