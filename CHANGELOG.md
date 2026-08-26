# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- **Conformance contract** (`docs/CONFORMANCE.md`), after an independent
  90,000-instance differential against Aura's pure-Python port — a port written
  from that document alone (2026-08-26, 0 algorithmic disagreements). No library
  behaviour changed and `test-assets/golden-selection.json` is byte-identical;
  what changed is the contract a port is held to.
  - The float tolerance for `expected_f` is now **derived from the primitives**:
    `tol(expected_g) + lam * tol(expected_div)` with
    `tol(x) = f_rel * max(1, abs(x))`, instead of `tol(expected_f)`. `f` is
    `g + lam * div`, so a distance ulp reaches `f` multiplied by `lam`; the old
    rule failed a correct port at high `lam` (147 of the 90,000 instances). The
    other float fields keep the unchanged `tol(expected)` rule and the
    `tolerance` block still carries `f_rel` as its single knob. Both readers
    (`crates/divsel/tests/golden.rs`, `python/tests/test_golden.py`) implement
    the new rule and pin it with a regression test.
  - **Corrected**, after an adversarial check of the above: the section claimed
    the `f` check "adds a consistency check on the combination ... and every one
    of those is gross, not marginal". Measured, the first half understates it
    and the second half is false. The `f` check is frequently the **sole**
    detector of a combination error (a doubled `lam` is caught on 20 of the 22
    cases and on 16 of those by `f` alone) — but at the f32 noise floor, where
    `lam * div` is a rounding-scale contribution, a wrong `lam` or a flipped
    sign is **unobservable** rather than caught: on a verified exact-duplicate
    cosine instance at `lam = 64` the derived bound accepts a sign flip at 11.6%
    of budget where the old bound rejected it at 3.81x, and the whole contract
    then passes. The section now states the escape band
    (`div ∈ (1.6e-08, 5.2e-07)` at `lam = 64`, 21.7% of 60,000 exact-duplicate
    draws), states that no committed fixture is anywhere near it, and argues
    from unobservability — measured, the diversity term is at most `1.7e-05` of
    `f` there and never changed `selected` or `stage` in 1,500 out-of-fixture
    instances — instead of claiming a strength the `f` check does not have.

### Added

- **`expected_threshold` is a selected grid entry** section in
  `docs/CONFORMANCE.md`, addressing the same width sensitivity in the one field
  the tolerance fix left on its old bound. Because `threshold` is a *choice*
  among grid entries and not a measured quantity, its error is quantized to a
  factor `1 + eps` (9.09% at the default `eps = 0.1`) — a correct float64 port
  reproduces all 22 to within `3.35e-08` relative, 3.4% of the bound, while a
  neighbouring entry is 90,909x the bound away — so no tolerance can separate a
  false failure from a real one. The **rule is deliberately unchanged**: a port
  sweeping the wrong grid is caught on 8 of the 22 through this field alone
  (one entry too few: cases 1, 3, 4, 5, 7, 9, 10, 18), and both candidate
  relaxations were rejected on measurement. The mode is documented instead, the
  way Finding B is, with the provoking families and the diagnostic a harness
  runs. New Rust test
  `the_reported_threshold_is_never_decided_by_a_breakable_tie` proves the 22 are
  immune: on 12 of the 14 geometric-grid sweep cases every entry tied at the
  best `f` yields the same selection, and the other two (cases 2 and 18) tie by
  exact dyadic arithmetic no width can break.

- **Degenerate geometry** section in `docs/CONFORMANCE.md`: on exact duplicates,
  signed-axis vectors and antipodal pairs, divsel's f32 distance kernel is the
  *less* accurate side, so a float64 port legitimately disagrees about
  `selected` and `stage` there — with the measured rates, what is safe to
  compare, and how to classify a difference. The optional bit-identity section
  does not cover this and now says so.
- **Independent verification** section in `docs/CONFORMANCE.md` and a matching
  README design commitment, recording the differential's date, parameter ranges
  and its qualifier.

## [0.1.0] - 2026-08-22

First release.

### Added

- **GIST core** (`divsel` crate): Algorithm 1 of arXiv:2405.18754v3 (NeurIPS 2025) —
  greedy independent set thresholding for `max f(S) = g(S) + λ · min-pairwise-distance(S)`
  subject to `|S| ≤ k`, with the paper's geometric threshold set
  (`1 + ⌊log(2/ε)/log(1+ε)⌋` thresholds), an optional exhaustive threshold mode,
  and exact / approximate (double-sweep) diameter modes.
- **Utilities**: monotone submodular `g` implementations — `Linear`, `Coverage`,
  `FacilityLocation` — behind the `Utility` trait.
- **CELF**: lazy-evaluation greedy (`greedy_independent_set`) for submodular `g`,
  with a linear fast path where CELF cannot skip evaluations.
- **Brute-force oracle** and guarantee tests: enumerates `OPT` on small instances and
  asserts the `(1/2 − ε)·OPT` (submodular) and `(2/3 − ε)·OPT` (linear) bounds actually
  hold on 500 random instances.
- **SIMD kernels**: pulp-dispatched cosine/euclidean distance kernels that are
  **bit-identical** to the scalar reference (fixed association order), tested per-ISA;
  CI runs the parity tests on both x86_64 and aarch64.
- **Parallel threshold sweep** via rayon (deterministic result independent of thread count).
- **Python bindings** (`divsel` on PyPI): PyO3 `abi3-py311` — one wheel per platform covers
  CPython 3.11–3.14 (free-threaded 3.14t gets a version-specific `cp314t` wheel);
  zero-copy for `metric="euclidean"` input, exactly one L2-normalised copy for
  `metric="cosine"`; `gist_select` / `gist_select_full` with type stubs (`py.typed`).
- **Adapters** (optional extras): `divsel[langchain]` — `DivselRetriever` (drop-in for
  `search_type="mmr"`); `divsel[llamaindex]` — `DivselNodePostprocessor`.
- **Benchmarks**: reproducible comparison (`bench/compare.py`) against `gist-select`,
  `gist-sampling` and a numpy MMR baseline, plus criterion benches for the native
  kernels; results and the installability matrix in `docs/benchmarks/README.md`.
- **Golden fixtures** (`test-assets/golden-selection.json`, schema 1): 22 cases —
  dyadic-rational inputs, hand-checked arithmetic, both metrics, all three
  utilities, exhaustive and approx-diameter modes, and the tie-breaking rules a
  fixture can reach: greedy's lowest-index `argmax`, the sweep's non-strict fold,
  the diametrical pair's ascending index order. One is **not** reachable — the
  strictness of line 5's `f_pair > f_value` comparison, since no case
  distinguishes `>` from `>=` (`docs/CONFORMANCE.md` rule 3 says so, and
  `gist.rs`'s own
  `the_diametrical_pair_does_not_displace_a_greedy_it_only_ties` is what pins it
  in-repo) — 17 of
  them protected by a brute-force robustness margin (best vs second-best `f` at
  least 1e-4 apart relative), and 5 deliberately exempt because their ties are
  exact dyadic arithmetic and exist to pin the tie-breaking rules (cases 2, 3,
  10, 18, 22; see `docs/CONFORMANCE.md`). The cross-implementation
  conformance contract (`docs/CONFORMANCE.md`); reproduced by both the Rust
  (`cargo test --test golden`) and Python (`pytest python/tests/test_golden.py`)
  readers, regenerated byte-identically by `python/tools/gen_golden.py --check`.

[0.1.0]: https://github.com/Redrum624/divsel/releases/tag/v0.1.0
