# PROGRESS — divsel v0.1

Plan: `docs/superpowers/plans/2026-08-21-divsel-v0.1.md` · Execution: Subagent-Driven Development on branch `feat/v0.1`.

CURRENT LOT: COMPLETE — awaiting the user's publish decisions
NEXT ACTION: MERGED. `main` == `feat/v0.1` == c95a128, pushed. Everything green locally. Remaining work is the user's, in `docs/RELEASE.md` steps 2-7: flip the repo public (optional), `cargo login` + `cargo publish -p divsel`, configure PyPI trusted publishing, `git tag -a v0.1.0` + `git push --follow-tags` (that tag is what triggers release.yml and builds the wheels), then the GitHub Release. NOTE: GitHub Actions minutes appear exhausted on this private repo — runs fail in ~5 s with zero steps; release.yml has never been triggered, so check billing first.

| Lot | Goal (one line) | Status | Commit SHA | Files touched | Next concrete action |
|-----|-----------------|--------|-----------|---------------|----------------------|
| 0 | Workspace scaffold (publish/push deferred to user checklist) | done | b639630 | Cargo.toml, crates/*, pyproject.toml, LICENSE, python/divsel | — |
| 1 | Metric + `Points` storage | done | dacb363 | crates/divsel/src/{error,metric,points,lib}.rs | — |
| 2 | Utility functions (Linear, Coverage, FacilityLocation) | done | 8233c90 | crates/divsel/src/{utility,error,lib}.rs | — |
| 3 | `GreedyIndependentSet` + CELF lazy greedy | done | 06366e5 | crates/divsel/src/{greedy,testutil,lib,utility}.rs | — |
| 4 | GIST driver + public API | done | ff3b1da | crates/divsel/src/{gist,lib}.rs | — |
| 5 | Brute-force oracle + approximation-ratio property test | done | ccaa83d | crates/divsel/tests/exact_oracle.rs | — |
| 6 | SIMD distance kernels + parallel thresholds + criterion benches | done | 7ece44e | crates/divsel/{src/metric,src/gist,src/testutil}.rs, benches/gist.rs, Cargo.toml | — |
| 7 | PyO3 bindings + Python package | done | 39566f1 | crates/divsel-py/src/lib.rs, python/divsel/{__init__.py,_divsel.pyi}, python/tests/{test_api.py,fixtures.py}, crates/divsel/tests/shared_fixture.rs, pyproject.toml, .gitignore | — |
| 8 | Benchmarks vs incumbents + installability matrix | done | 818ef9c | bench/*, docs/benchmarks/*, .github/workflows/install-matrix.yml, .github/scripts/* | — |
| 9 | LangChain / LlamaIndex adapters | done | 6fed702 | python/divsel/adapters/*, python/tests/test_adapters.py, README.md | — |
| 10 | CI, wheels, 0.1.0 release prep (publishes deferred per R-PUB) | done | 4d93311 | .github/workflows/{ci,release}.yml, CHANGELOG.md, docs/RELEASE.md, Cargo.toml, pyproject.toml, README.md | — |
| 11 | Golden fixtures + CONFORMANCE.md (22 cases) | done | b5eb01b | test-assets/golden-selection.json, .gitattributes, python/tools/gen_golden.py, crates/divsel/tests/golden.rs, python/tests/test_golden.py, docs/CONFORMANCE.md, CHANGELOG.md, ci.yml | — |

## Blockers known at start (2026-08-21)
- No crates.io token (`~/.cargo/credentials.toml` absent) and no PyPI token on this machine → `cargo publish` / `maturin publish` steps cannot run unattended; everything up to the publish command is prepared and the exact command is recorded here when reached.
