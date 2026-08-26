//! Golden-fixture conformance reader — the Rust half of "divsel reproduces
//! `test-assets/golden-selection.json`".
//!
//! The fixture file is the cross-language contract (docs/CONFORMANCE.md): the
//! Aura (Python) and limbic (TypeScript) ports prove conformance against the
//! same file with the same rules this test applies:
//!
//! * `expected_selected` matches EXACTLY (list equality, order included);
//! * `expected_stage` matches exactly;
//! * `expected_g`, `expected_div`, `expected_threshold` and `expected_d_max`
//!   agree within `tol(expected) = f_rel * max(1, |expected|)`, with `f_rel`
//!   read from the file's own `tolerance` block;
//! * `expected_f` agrees within `tol(expected_g) + lam * tol(expected_div)`.
//!   `f = g + lam * div` is derived, not primitive, so its bound is derived
//!   too: an error in `div` reaches `f` multiplied by `lam`, which a bound
//!   relative to `|f|` misses whenever `g` dominates `f` while `div` is small
//!   -- the near-duplicate cosine regime, where `1 - a.b` cancels and the
//!   absolute error stays at the `ulp(1)` scale. See "Why `f`'s bound is
//!   derived instead of relative to `f`" in docs/CONFORMANCE.md.
//!
//! One caveat on that table, which the rules above do **not** encode because no
//! tolerance can: `expected_threshold` under `stage == "sweep"` is a *selected
//! grid entry*, not a measured quantity, so its error is quantized to a factor
//! `1 + eps` and its bound is never the thing that decides. What decides is
//! rule 2's fold, and a tie there can in principle be broken differently by an
//! f32 kernel and a float64 port. It cannot be on these 22 --
//! [`the_reported_threshold_is_never_decided_by_a_breakable_tie`] proves that,
//! and "`expected_threshold` is a selected grid entry" in docs/CONFORMANCE.md
//! says what a port's own harness does about it off them.
//!
//! The generator is `python/tools/gen_golden.py`; the Python-side reader is
//! `python/tests/test_golden.py`.

use divsel::{
    div, eval_g, gist, greedy_independent_set, thresholds, Coverage, DiameterMode,
    FacilityLocation, GistConfig, Linear, Metric, Points, Stage, Utility,
};
use serde::Deserialize;

#[derive(Deserialize)]
struct Golden {
    generator: String,
    paper: String,
    schema: u32,
    tolerance: Tolerance,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Tolerance {
    f_rel: f64,
    selected: String,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    note: String,
    metric: String,
    utility: String,
    vectors: Vec<Vec<f32>>,
    /// `null` (uniform linear / facility location), a flat array of f64
    /// weights (linear), or an array of int arrays (coverage). Dispatched on
    /// the `utility` string, so it is kept as raw JSON here.
    utilities: Option<serde_json::Value>,
    k: usize,
    lam: f64,
    eps: f32,
    exhaustive_thresholds: bool,
    diameter: String,
    diameter_sweeps: usize,
    expected_selected: Vec<usize>,
    expected_f: f64,
    expected_g: f64,
    expected_div: f64,
    expected_threshold: f64,
    expected_stage: String,
    expected_d_max: f64,
}

/// The per-field tolerance budget: `tol(x) = f_rel * max(1, |x|)`.
///
/// The `max(1, .)` floor is the load-bearing half for a distance: a cosine
/// distance near zero has no relative accuracy left -- it is `1 - a.b`, a
/// cancellation -- but its absolute error stays bounded by a few `ulp(1)`.
fn tol(expected: f64, rel: f64) -> f64 {
    rel * expected.abs().max(1.0)
}

/// The conformance float rule: `|actual - expected| <= bound`, where `bound` is
/// [`tol`] of the field itself for every primitive field, and
/// `tol(g) + lam * tol(div)` for the derived `f`.
fn close(actual: f64, expected: f64, bound: f64) -> bool {
    (actual - expected).abs() <= bound
}

/// Rebuilds one case's instance: the point set and a fresh utility for it.
///
/// Shared by [`check_case`] and
/// [`the_reported_threshold_is_never_decided_by_a_breakable_tie`], which needs
/// the same instance to re-run the sweep one threshold at a time. A malformed
/// fixture entry (unknown metric, missing utilities, an input the library
/// rejects) panics: that is a file error, not a conformance failure, and it
/// stops the run.
fn build_instance(case: &Case, ctx: &str) -> (Points<'static>, Box<dyn Utility>) {
    let dim = case.vectors[0].len();
    let flat: Vec<f32> = case.vectors.iter().flatten().copied().collect();
    let metric = match case.metric.as_str() {
        "euclidean" => Metric::Euclidean,
        "cosine" => Metric::Cosine,
        other => panic!("{ctx}: unknown metric {other:?}"),
    };
    let pts = Points::new(flat, dim, metric).unwrap_or_else(|e| panic!("{ctx}: {e}"));

    let util: Box<dyn Utility> = match case.utility.as_str() {
        "linear" => match &case.utilities {
            None => Box::new(Linear::uniform(pts.n())),
            Some(value) => {
                let weights: Vec<f64> = serde_json::from_value(value.clone())
                    .unwrap_or_else(|e| panic!("{ctx}: linear utilities: {e}"));
                Box::new(Linear::new(weights))
            }
        },
        "coverage" => {
            let value = case
                .utilities
                .clone()
                .unwrap_or_else(|| panic!("{ctx}: coverage requires utilities"));
            let sets: Vec<Vec<u32>> = serde_json::from_value(value)
                .unwrap_or_else(|e| panic!("{ctx}: coverage utilities: {e}"));
            // Same universe inference as the Python binding: largest id + 1.
            let universe = sets
                .iter()
                .flatten()
                .max()
                .map_or(0, |&item| item as usize + 1);
            Box::new(Coverage::new(sets, universe).unwrap_or_else(|e| panic!("{ctx}: {e}")))
        }
        "facility_location" => {
            assert!(
                case.utilities.is_none(),
                "{ctx}: facility_location takes no utilities"
            );
            Box::new(FacilityLocation::new(&pts))
        }
        other => panic!("{ctx}: unknown utility {other:?}"),
    };
    (pts, util)
}

/// Runs one case and returns every conformance mismatch it produced -- empty
/// when the case passes.
fn check_case(case: &Case, rel: f64) -> Vec<String> {
    // `note` carries the case's one-line rationale; surface it on failure.
    let ctx = format!("case {:?} — {}", case.name, case.note);
    let (pts, mut util) = build_instance(case, &ctx);

    let diameter = match case.diameter.as_str() {
        "exact" => DiameterMode::Exact,
        "approx" => DiameterMode::Approx {
            sweeps: case.diameter_sweeps,
        },
        other => panic!("{ctx}: unknown diameter mode {other:?}"),
    };
    let cfg = GistConfig {
        k: case.k,
        lambda: case.lam,
        eps: case.eps,
        exhaustive_thresholds: case.exhaustive_thresholds,
        diameter,
    };

    let out = gist(&pts, util.as_mut(), &cfg).unwrap_or_else(|e| panic!("{ctx}: {e}"));

    let mut problems = Vec::new();
    if out.selected != case.expected_selected {
        problems.push(format!(
            "selected {:?} != expected {:?}",
            out.selected, case.expected_selected
        ));
    }
    let stage = match out.stage {
        Stage::Greedy => "greedy",
        Stage::DiameterPair => "diameter_pair",
        Stage::Sweep => "sweep",
    };
    if stage != case.expected_stage {
        problems.push(format!(
            "stage {stage:?} != expected {:?}",
            case.expected_stage
        ));
    }
    // `f` is derived from `g` and `div` (`f = g + lam * div`), so its budget is
    // the sum of theirs with `lam` applied to `div`'s -- the same `lam` the
    // objective applies. At `lam == 0` it collapses to `tol(g)`, which is what
    // rule 18 promises: `f == g` exactly, even for an infinite `div`.
    let floats = [
        (
            "f_value",
            out.f_value,
            case.expected_f,
            tol(case.expected_g, rel) + case.lam * tol(case.expected_div, rel),
        ),
        (
            "g_value",
            out.g_value,
            case.expected_g,
            tol(case.expected_g, rel),
        ),
        (
            "div",
            f64::from(out.div),
            case.expected_div,
            tol(case.expected_div, rel),
        ),
        (
            "threshold",
            f64::from(out.threshold),
            case.expected_threshold,
            tol(case.expected_threshold, rel),
        ),
        (
            "d_max",
            f64::from(out.d_max),
            case.expected_d_max,
            tol(case.expected_d_max, rel),
        ),
    ];
    for (field, actual, expected, bound) in floats {
        if !close(actual, expected, bound) {
            problems.push(format!(
                "{field} = {actual} differs from expected {expected} \
                 by {} beyond the conformance bound {bound}",
                (actual - expected).abs()
            ));
        }
    }
    problems
}

/// The fixture lives at the workspace root, which is **outside** this package:
/// `cargo package` cannot reach it (`include` paths may not escape the package
/// root), so the published `.crate` ships this reader without its contract file.
///
/// Returns the fixture text, or `None` when the file is genuinely unreachable —
/// a registry checkout, a vendored copy, an unpacked `target/package` tree. In
/// **this workspace** it is never `None`; see [`missing_fixture_policy`] for how
/// that is decided, and note that CI additionally sets `DIVSEL_REQUIRE_GOLDEN`,
/// so the gate there cannot skip for any reason at all.
fn fixture_text() -> Option<String> {
    const PATH: &str = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-assets/golden-selection.json"
    );
    const ROOT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");
    match std::fs::read_to_string(PATH) {
        Ok(text) => Some(text),
        Err(e) => match missing_fixture_policy(required_by_env(), std::path::Path::new(ROOT)) {
            Missing::Fail => panic!("cannot read the golden fixture file {PATH}: {e}"),
            Missing::Skip => None,
        },
    }
}

/// What a missing fixture means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Missing {
    /// The checkout owns the fixture, so its absence is a broken checkout.
    Fail,
    /// The fixture was never shipped here; there is nothing to verify.
    Skip,
}

/// `DIVSEL_REQUIRE_GOLDEN` set to anything but empty or `0`.
fn required_by_env() -> bool {
    required_from(std::env::var_os("DIVSEL_REQUIRE_GOLDEN").as_deref())
}

/// The parsing half of [`required_by_env`], separated so it is testable without
/// mutating the process environment.
fn required_from(value: Option<&std::ffi::OsStr>) -> bool {
    match value {
        None => false,
        Some(value) => !value.is_empty() && value != "0",
    }
}

/// Decides whether a missing fixture is a hard failure or a legitimate skip,
/// from the environment plus what sits at the workspace root `root`.
///
/// The skip exists for exactly one situation: this reader compiled **outside**
/// the repository that owns the fixture, because the file lives at the workspace
/// root and `cargo package` cannot reach past a package root. Anything that says
/// "you are in that repository" therefore turns the skip back into a failure.
///
/// Several markers, not one, and none of them may be the only coupling: an
/// earlier version keyed the whole decision on `python/tools/gen_golden.py`, an
/// unrelated file that a refactor is free to move — and moving it while the
/// fixture was also absent turned the 22-case contract into `1 passed` having
/// checked nothing. `DIVSEL_REQUIRE_GOLDEN` short-circuits all of it, which is
/// what CI sets: a gate that can decide it has nothing to gate is not a gate.
fn missing_fixture_policy(required: bool, root: &std::path::Path) -> Missing {
    if required {
        return Missing::Fail;
    }
    let owns_the_fixture = root.join(".git").exists()
        || root.join("test-assets").is_dir()
        || root
            .join("python")
            .join("tools")
            .join("gen_golden.py")
            .exists()
        || std::fs::read_to_string(root.join("Cargo.toml"))
            .is_ok_and(|manifest| manifest.contains("[workspace]"));
    if owns_the_fixture {
        Missing::Fail
    } else {
        Missing::Skip
    }
}

/// Every case is checked; the failures are collected and reported together,
/// so one broken case never hides the others.
#[test]
fn divsel_reproduces_the_golden_fixtures() {
    let Some(text) = fixture_text() else {
        // Not a silent pass: say which file is missing and where it lives.
        println!(
            "skipped: test-assets/golden-selection.json is not part of the published crate \
             (it sits at the workspace root, outside this package). Run this test from a \
             checkout of {}, or fetch the fixture from there.",
            env!("CARGO_PKG_REPOSITORY")
        );
        return;
    };
    let golden: Golden = serde_json::from_str(&text).expect("golden-selection.json parses");

    assert_eq!(golden.schema, 1, "unknown golden schema version");
    assert!(
        golden.generator.starts_with("divsel "),
        "unexpected generator {:?}",
        golden.generator
    );
    assert_eq!(golden.paper, "arXiv:2405.18754v3");
    assert_eq!(golden.tolerance.selected, "exact");
    let rel = golden.tolerance.f_rel;
    assert_eq!(golden.cases.len(), 22, "the contract is exactly 22 cases");

    let mut failures = Vec::new();
    for case in &golden.cases {
        let problems = check_case(case, rel);
        if !problems.is_empty() {
            failures.push(format!(
                "case {:?} — {}\n    {}",
                case.name,
                case.note,
                problems.join("\n    ")
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} golden cases failed:\n{}",
        failures.len(),
        golden.cases.len(),
        failures.join("\n")
    );
}

/// The `f` bound carries `lam` times `div`'s budget, and that is load-bearing.
///
/// This pins the fix for the tolerance defect the 2026-08-26 differential found:
/// `f = g + lam * div`, so an error in `div` reaches `f` multiplied by `lam`,
/// and a bound relative to `|f|` misses it whenever `g` dominates `f` while
/// `div` is small. The numbers are the worked case in docs/CONFORMANCE.md --
/// two near-duplicate cosine rows at `lam = 64`, where `1 - a.b` cancels and
/// the f32 distance carries an absolute error at the `ulp(1)` scale.
#[test]
fn the_f_bound_carries_lam_times_the_div_budget() {
    const REL: f64 = 1e-6;
    let lam = 64.0;
    // What divsel reports (f32 distance kernel).
    let expected_f = 2.0066680908203125;
    let expected_g = 2.0;
    let expected_div = 1.0418891906738281e-4;
    // What a float64 port reports for the SAME selection and stage: the same
    // algorithm, a wider distance arithmetic -- and the more accurate side.
    let port_f = 2.0066829063008953;

    // The old, lam-independent rule rejected it. That was the contract defect:
    // it failed a correct port at high lam.
    assert!(
        !close(port_f, expected_f, tol(expected_f, REL)),
        "the old rule was supposed to reject this case"
    );

    // The derived rule accepts it, and uses under a quarter of the budget.
    let bound = tol(expected_g, REL) + lam * tol(expected_div, REL);
    assert!(close(port_f, expected_f, bound));
    assert!((port_f - expected_f).abs() < 0.25 * bound);

    // It still rejects an error a real bug would produce. The smallest discrete
    // step this fixture family's objective can take is 1/64 (dyadic weights),
    // 237x the bound; the generator's own robustness margin (1e-4 relative) is
    // 3x it even in this worst regime.
    assert!(!close(expected_f + 1.0 / 64.0, expected_f, bound));
    assert!(!close(
        expected_f + 1e-4 * expected_f.abs().max(1.0),
        expected_f,
        bound
    ));

    // At lam == 0 the bound is exactly g's: rule 18 makes `f` equal `g` there.
    assert_eq!(
        tol(expected_g, REL) + 0.0 * tol(expected_div, REL),
        tol(expected_g, REL)
    );
}

/// No committed case reports a `threshold` that a last ulp could move.
///
/// `expected_threshold` under `stage == "sweep"` is not a measured quantity: it
/// is the grid entry rule 2's non-strict fold kept, so its error is quantized —
/// a port either picks the same entry (agreeing to the last ulp) or a
/// neighbour, a factor `1 + eps` away and five orders outside the bound. Which
/// entry the fold keeps is decided by `f`, and where two entries attain the
/// **same** `f` with **different** selections, an ulp of difference between
/// divsel's f32 kernel and a float64 port can break that tie the other way:
/// `threshold` then moves a whole grid gap while `selected`, `stage`, `g`,
/// `div` and `f` all still agree. That is a false failure, and no tolerance can
/// cover it — see "`expected_threshold` is a selected grid entry, so its bound
/// is not a tolerance" in docs/CONFORMANCE.md.
///
/// The contract's answer is that the 22 are immune, and this is what makes that
/// true rather than hoped for: on 12 of the 14 geometric-grid sweep cases every
/// entry tied at the best `f` yields the *same* selection, so the fold's answer
/// is the largest of them whatever the arithmetic. The two exceptions tie
/// *distinct* selections, and both are margin-exempt tie cases whose ties are
/// **exact** dyadic arithmetic (1-D dyadic coordinates give dyadic distances),
/// so no width can break them either.
///
/// A future fixture that introduced a breakable tie fails here rather than
/// shipping a case whose `threshold` a correct port could legitimately miss.
#[test]
fn the_reported_threshold_is_never_decided_by_a_breakable_tie() {
    let Some(text) = fixture_text() else {
        println!("skipped: no fixture; see divsel_reproduces_the_golden_fixtures");
        return;
    };
    let golden: Golden = serde_json::from_str(&text).expect("golden-selection.json parses");

    /// The cases whose best-`f` tie spans distinct selections, with the reason
    /// that is safe. Both are in the fixture's own margin-exempt set (see
    /// "Robustness margin" in docs/CONFORMANCE.md): their ties are exact dyadic
    /// arithmetic, built to pin the tie rules, and identical on every platform.
    const EXACT_DYADIC_TIES: [&str; 2] = [
        "weighted_line_middle_threshold",  // case 2, rule 2
        "coverage_exact_tie_lowest_index", // case 18, rules 1 and 2
    ];

    let mut geometric_sweep_cases = 0;
    let mut unique_selection = 0;
    let mut exact_dyadic = 0;
    for case in &golden.cases {
        // Only the geometric grid under an exact diameter is reachable through
        // the public API (`thresholds` carries the paper's `2/eps` ceiling), and
        // only a `"sweep"` case reports a grid entry at all: `"greedy"` reports
        // `0` and `"diameter_pair"` reports `d_max`, both measured quantities.
        if case.expected_stage != "sweep" || case.exhaustive_thresholds || case.diameter != "exact"
        {
            continue;
        }
        geometric_sweep_cases += 1;
        let ctx = format!("case {:?} — {}", case.name, case.note);
        let (pts, mut util) = build_instance(case, &ctx);
        let k = case.k.min(pts.n());
        let d_max = pts.diameter().0;

        // (f, selection) at every entry, folded exactly as the driver folds it.
        let mut best_f = f64::NEG_INFINITY;
        let mut tied: Vec<Vec<usize>> = Vec::new();
        for d in thresholds(d_max, case.eps) {
            let selection = greedy_independent_set(&pts, util.as_mut(), d, k);
            let g_value = eval_g(util.as_mut(), &selection, &pts);
            let f_value = if case.lam == 0.0 {
                g_value
            } else {
                g_value + case.lam * f64::from(div(&pts, &selection))
            };
            if f_value > best_f {
                best_f = f_value;
                tied.clear();
            }
            if f_value == best_f {
                tied.push(selection);
            }
        }
        assert!(!tied.is_empty(), "{ctx}: the sweep produced no entry");

        let distinct = tied.iter().any(|selection| selection != &tied[0]);
        if distinct {
            exact_dyadic += 1;
            assert!(
                EXACT_DYADIC_TIES.contains(&case.name.as_str()),
                "{ctx}: entries with distinct selections tie at f = {best_f}, so the reported \
                 threshold is decided by that tie -- a float64 port may legitimately keep a \
                 different entry. Either the case is not safe to compare on `threshold`, or it \
                 belongs in EXACT_DYADIC_TIES with the argument for why its tie is exact."
            );
        } else {
            unique_selection += 1;
        }
    }

    // The counts docs/CONFORMANCE.md quotes, pinned so the prose cannot drift.
    assert_eq!(geometric_sweep_cases, 14);
    assert_eq!(unique_selection, 12);
    assert_eq!(exact_dyadic, 2);
}

/// `DIVSEL_REQUIRE_GOLDEN` is read the way CI sets it, and an unset or disabled
/// value leaves the decision to the markers.
#[test]
fn the_require_golden_override_is_read_the_way_ci_sets_it() {
    use std::ffi::OsStr;
    assert!(!required_from(None));
    assert!(!required_from(Some(OsStr::new(""))));
    assert!(!required_from(Some(OsStr::new("0"))));
    assert!(required_from(Some(OsStr::new("1"))));
    assert!(required_from(Some(OsStr::new("true"))));
}

/// A missing fixture is a **failure** in any tree that looks like this
/// repository, and a skip only where the fixture was never shipped.
///
/// The regression this pins: keying the decision on `python/tools/gen_golden.py`
/// alone meant that moving the generator — a refactor nothing forbids — while
/// the fixture was also absent made `cargo test -p divsel --test golden` report
/// `1 passed` with zero of the 22 cases checked, which is the CI gate for the
/// whole cross-language contract.
#[test]
fn a_missing_fixture_only_skips_outside_the_repository_that_owns_it() {
    /// Removes the scratch tree on the way out, panic or not.
    ///
    /// Every directory below is created under `std::env::temp_dir()` with a
    /// pid-and-thread name, so a failing assertion used to leave the whole tree
    /// behind and a red CI loop accumulated one copy per run.
    struct ScratchDir(std::path::PathBuf);

    impl Drop for ScratchDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    let scratch = ScratchDir(std::env::temp_dir().join(format!(
        "divsel-golden-policy-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    )));
    let root = scratch.0.clone();
    let _ = std::fs::remove_dir_all(&root);
    std::fs::create_dir_all(&root).expect("scratch root");

    // Nothing at all: an unpacked `.crate`, a registry checkout, a vendor tree.
    assert_eq!(missing_fixture_policy(false, &root), Missing::Skip);
    // ... unless CI asked for the gate explicitly.
    assert_eq!(missing_fixture_policy(true, &root), Missing::Fail);

    // Each marker on its own is enough, and each is independent of the others.
    for marker in [
        "test-assets",
        ".git",
        "python/tools/gen_golden.py",
        "Cargo.toml",
    ] {
        let case = root.join(marker.replace(['/', '.'], "_"));
        let _ = std::fs::remove_dir_all(&case);
        std::fs::create_dir_all(&case).expect("case root");
        assert_eq!(
            missing_fixture_policy(false, &case),
            Missing::Skip,
            "{marker}: empty case root is not a skip"
        );
        match marker {
            "test-assets" | ".git" => {
                std::fs::create_dir_all(case.join(marker)).expect("marker dir");
            }
            "Cargo.toml" => {
                std::fs::write(case.join(marker), "[workspace]\nmembers = []\n")
                    .expect("marker file");
            }
            _ => {
                std::fs::create_dir_all(case.join("python").join("tools")).expect("marker dirs");
                std::fs::write(case.join(marker), "# generator\n").expect("marker file");
            }
        }
        assert_eq!(
            missing_fixture_policy(false, &case),
            Missing::Fail,
            "{marker} did not mark the tree as owning the fixture"
        );
        let _ = std::fs::remove_dir_all(&case);
    }

    // A package-root manifest is not a workspace manifest.
    let package = root.join("package");
    std::fs::create_dir_all(&package).expect("package root");
    std::fs::write(package.join("Cargo.toml"), "[package]\nname = \"divsel\"\n")
        .expect("package manifest");
    assert_eq!(missing_fixture_policy(false, &package), Missing::Skip);

    // And the tree this test is compiled in owns the fixture.
    assert_eq!(
        missing_fixture_policy(
            false,
            std::path::Path::new(concat!(env!("CARGO_MANIFEST_DIR"), "/../.."))
        ),
        Missing::Fail
    );

    // `scratch` removes the tree here, and just as reliably on the way out of a
    // failing assertion above.
    drop(scratch);
    assert!(!root.exists(), "the scratch tree outlived the test");
}
