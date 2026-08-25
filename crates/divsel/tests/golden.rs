//! Golden-fixture conformance reader — the Rust half of "divsel reproduces
//! `test-assets/golden-selection.json`".
//!
//! The fixture file is the cross-language contract (docs/CONFORMANCE.md): the
//! Aura (Python) and limbic (TypeScript) ports prove conformance against the
//! same file with the same rules this test applies:
//!
//! * `expected_selected` matches EXACTLY (list equality, order included);
//! * every float field agrees within `f_rel * max(1, |expected|)`, with
//!   `f_rel` read from the file's own `tolerance` block;
//! * `expected_stage` matches exactly.
//!
//! The generator is `python/tools/gen_golden.py`; the Python-side reader is
//! `python/tests/test_golden.py`.

use divsel::{
    gist, Coverage, DiameterMode, FacilityLocation, GistConfig, Linear, Metric, Points, Stage,
    Utility,
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

/// The conformance float rule: `|actual - expected| <= rel * max(1, |expected|)`.
fn close(actual: f64, expected: f64, rel: f64) -> bool {
    (actual - expected).abs() <= rel * expected.abs().max(1.0)
}

/// Runs one case and returns every conformance mismatch it produced -- empty
/// when the case passes. A malformed fixture entry (unknown metric, missing
/// utilities, an input the library rejects) still panics: that is a file
/// error, not a conformance failure, and it stops the run.
fn check_case(case: &Case, rel: f64) -> Vec<String> {
    // `note` carries the case's one-line rationale; surface it on failure.
    let ctx = format!("case {:?} — {}", case.name, case.note);

    let dim = case.vectors[0].len();
    let flat: Vec<f32> = case.vectors.iter().flatten().copied().collect();
    let metric = match case.metric.as_str() {
        "euclidean" => Metric::Euclidean,
        "cosine" => Metric::Cosine,
        other => panic!("{ctx}: unknown metric {other:?}"),
    };
    let pts = Points::new(flat, dim, metric).unwrap_or_else(|e| panic!("{ctx}: {e}"));

    let mut util: Box<dyn Utility> = match case.utility.as_str() {
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
    let floats = [
        ("f_value", out.f_value, case.expected_f),
        ("g_value", out.g_value, case.expected_g),
        ("div", f64::from(out.div), case.expected_div),
        (
            "threshold",
            f64::from(out.threshold),
            case.expected_threshold,
        ),
        ("d_max", f64::from(out.d_max), case.expected_d_max),
    ];
    for (field, actual, expected) in floats {
        if !close(actual, expected, rel) {
            problems.push(format!(
                "{field} = {actual} differs from expected {expected} \
                 beyond {rel} * max(1, |expected|)"
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
    let root = std::env::temp_dir().join(format!(
        "divsel-golden-policy-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
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

    let _ = std::fs::remove_dir_all(&root);
}
