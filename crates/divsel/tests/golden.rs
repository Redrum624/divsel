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

fn assert_close(actual: f64, expected: f64, rel: f64, field: &str, ctx: &str) {
    assert!(
        close(actual, expected, rel),
        "{ctx}: {field} = {actual} differs from expected {expected} \
         beyond {rel} * max(1, |expected|)"
    );
}

#[test]
fn divsel_reproduces_the_golden_fixtures() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../test-assets/golden-selection.json"
    );
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("cannot read the golden fixture file {path}: {e}"));
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
    assert_eq!(golden.cases.len(), 20, "the contract is exactly 20 cases");

    for case in &golden.cases {
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

        assert_eq!(
            out.selected, case.expected_selected,
            "{ctx}: selected differs"
        );
        let stage = match out.stage {
            Stage::Greedy => "greedy",
            Stage::DiameterPair => "diameter_pair",
            Stage::Sweep => "sweep",
        };
        assert_eq!(stage, case.expected_stage, "{ctx}: stage differs");
        assert_close(out.f_value, case.expected_f, rel, "f_value", &ctx);
        assert_close(out.g_value, case.expected_g, rel, "g_value", &ctx);
        assert_close(f64::from(out.div), case.expected_div, rel, "div", &ctx);
        assert_close(
            f64::from(out.threshold),
            case.expected_threshold,
            rel,
            "threshold",
            &ctx,
        );
        assert_close(
            f64::from(out.d_max),
            case.expected_d_max,
            rel,
            "d_max",
            &ctx,
        );
    }
}
