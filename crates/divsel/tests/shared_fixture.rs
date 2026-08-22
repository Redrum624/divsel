//! The Rust half of the shared hand-computed fixture (R-G25).
//!
//! `python/tests/fixtures.py` carries the same three cases with the same
//! expected values, and `python/tests/test_api.py` asserts them through the
//! PyO3 bindings. This file asserts them against the core directly, so the two
//! language surfaces are pinned to one hand-derived answer. The arithmetic is
//! reproduced here in full so that neither file has to be read to check the
//! other.
//!
//! Rules the arithmetic relies on (all documented `[divsel choice]`s):
//! Euclidean metric; greedy `argmax` ties go to the lowest index; the
//! diametrical pair displaces greedy only on a strict `>`; the sweep compares
//! `>=` in ascending threshold order, so the largest threshold attaining the
//! best `f` wins; `div(S)` is the minimum pairwise distance for `|S| >= 2`; the
//! `eps = 0.1` threshold set is `{0.05 * d_max * 1.1^i : i = 0..31}` and only
//! which interval between consecutive pairwise distances a threshold lands in
//! matters (`1.1^i` for `i = 10, 13, 17, 22, 25, 28, 30, 31` is
//! `2.594, 3.452, 5.054, 8.140, 10.83, 14.42, 17.45, 19.19`).

use divsel::{gist, GistConfig, GistResult, Linear, Metric, Points, Stage};

/// Runs one fixture case through the public API with the Python defaults
/// (`eps = 0.1`, exact diameter, geometric thresholds).
fn run(data: &[f32], dim: usize, k: usize, lambda: f64, weights: Option<&[f64]>) -> GistResult {
    let pts = Points::borrowed(data, dim, Metric::Euclidean).expect("fixture points");
    let mut util = match weights {
        Some(w) => Linear::new(w.to_vec()),
        None => Linear::uniform(pts.n()),
    };
    let cfg = GistConfig {
        k,
        lambda,
        eps: 0.1,
        ..Default::default()
    };
    gist(&pts, &mut util, &cfg).expect("fixture solve")
}

#[track_caller]
fn assert_case(
    got: &GistResult,
    selected: &[usize],
    f_value: f64,
    g_value: f64,
    div: f32,
    threshold_range: (f32, f32),
) {
    assert_eq!(got.selected, selected, "selected");
    assert_eq!(
        got.f_value.to_bits(),
        f_value.to_bits(),
        "f_value {}",
        got.f_value
    );
    assert_eq!(
        got.g_value.to_bits(),
        g_value.to_bits(),
        "g_value {}",
        got.g_value
    );
    assert_eq!(got.div.to_bits(), div.to_bits(), "div {}", got.div);
    assert_eq!(got.stage, Stage::Sweep);
    let (lo, hi) = threshold_range;
    assert!(
        lo < got.threshold && got.threshold <= hi,
        "threshold {} not in ({lo}, {hi}]",
        got.threshold
    );
}

/// Case A, "line_pick_widest": `x = [0, 1, 5, 6]`, uniform weights, `k = 2`,
/// `lambda = 1`.
///
/// ```text
/// d01 = 1, d02 = 5, d03 = 6, d12 = 4, d13 = 5, d23 = 1; d_max = 6 at (0, 3).
/// greedy d=0: [0, 1], g = 2, div = 1, f = 3.
/// pair [0, 3]: g = 2, div = 6, f = 8 > 3 -> incumbent f = 8.
/// thresholds 0.3 * 1.1^i:
///   (0, 1]  i <= 12:  [0, 1], f = 3.
///   (1, 5]  i = 13..29: 0, then {2, 3} -> 2, then nothing -> [0, 2], f = 7.
///   (5, 6]  i = 30, 31 (5.235, 5.758): 0, then only 3 -> [0, 3], f = 8 >= 8.
/// => [0, 3], f = 8, g = 2, div = 6, Sweep, threshold in (5, 6].
/// ```
#[test]
fn case_a_line_pick_widest() {
    let got = run(&[0.0, 1.0, 5.0, 6.0], 1, 2, 1.0, None);
    assert_case(&got, &[0, 3], 8.0, 2.0, 6.0, (5.0, 6.0));
}

/// Case B, "weighted_line_middle_threshold": `x = [0, 1, 2, 3, 4, 8]`,
/// `weights = [4, 1, 4, 1, 4, 3]`, `k = 3`, `lambda = 0.5`.
///
/// ```text
/// d_max = 8 at (0, 5).
/// greedy d=0: weights 4 at 0, 2, 4 -> [0, 2, 4], g = 12, div = min(2, 4, 2) = 2,
///             f = 12 + 0.5*2 = 13.
/// pair [0, 5]: g = 4 + 3 = 7, div = 8, f = 7 + 4 = 11 < 13 -> greedy survives.
/// thresholds 0.4 * 1.1^i:
///   (0, 1]  i <= 9:      [0, 2, 4], f = 13 >= 13.
///   (1, 2]  i = 10..16:  0; {2,3,4,5} -> 2; {4 (d24 = 2), 5} -> 4 -> [0, 2, 4], f = 13.
///   (2, 3]  i = 17..21:  0; {3,4,5} -> 4 (w = 4); {5 (d05 = 8, d45 = 4)} -> 5
///                        -> [0, 4, 5], g = 11, div = min(4, 8, 4) = 4,
///                        f = 11 + 0.5*4 = 13 >= 13 -> selected = [0, 4, 5].
///   (3, 4]  i = 22..24 (3.256 .. 3.940): 0; {4, 5} -> 4; 5 -> [0, 4, 5], f = 13.
///   (4, 8]  i = 25..31:  0; only 5; nothing else -> [0, 5], f = 11 < 13.
/// => [0, 4, 5], f = 13, g = 11, div = 4, Sweep, threshold in (3, 4].
/// ```
#[test]
fn case_b_weighted_line_middle_threshold() {
    let got = run(
        &[0.0, 1.0, 2.0, 3.0, 4.0, 8.0],
        1,
        3,
        0.5,
        Some(&[4.0, 1.0, 4.0, 1.0, 4.0, 3.0]),
    );
    assert_case(&got, &[0, 4, 5], 13.0, 11.0, 4.0, (3.0, 4.0));
}

/// Case C, "rectangle_short_return": the 3-4-5 rectangle `(0,0) (3,0) (0,4)
/// (3,4)`, uniform weights, `k = 3`, `lambda = 1`.
///
/// ```text
/// d01 = 3, d02 = 4, d03 = 5, d12 = 5, d13 = 4, d23 = 3.
/// d_max = 5 at both (0, 3) and (1, 2); the tie rule picks u = 0 -> T = [0, 3].
/// greedy d=0: [0, 1, 2], g = 3, div = min(3, 4, 5) = 3, f = 6.
/// pair [0, 3]: g = 2, div = 5, f = 7 > 6 -> incumbent f = 7.
/// thresholds 0.25 * 1.1^i:
///   (0, 3]  i <= 26 (2.98): [0, 1, 2], f = 6.
///   (3, 4]  i = 27..29:     0; {2, 3} -> 2; 1 (d01 = 3) no, 3 (d23 = 3) no
///                           -> [0, 2], f = 2 + 4 = 6 < 7.
///   (4, 5]  i = 30, 31 (4.362, 4.799): 0; only 3; nothing -> [0, 3], f = 7 >= 7.
/// => [0, 3], f = 7, g = 2, div = 5, Sweep, threshold in (4, 5]; |S| = 2 < k.
/// ```
#[test]
fn case_c_rectangle_short_return() {
    let got = run(&[0.0, 0.0, 3.0, 0.0, 0.0, 4.0, 3.0, 4.0], 2, 3, 1.0, None);
    assert_case(&got, &[0, 3], 7.0, 2.0, 5.0, (4.0, 5.0));
    assert_eq!(got.d_max, 5.0);
}
