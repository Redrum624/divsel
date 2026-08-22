# PROGRESS — divsel v0.1

Plan: `docs/superpowers/plans/2026-08-21-divsel-v0.1.md` · Execution: Subagent-Driven Development on branch `feat/v0.1`.

CURRENT LOT: 10
NEXT ACTION: Task 10 implementer running from `.superpowers/sdd/2026-08-21-divsel-v0.1/task-10-brief.md` + `task-10-context.md` (generate release.yml via maturin generate-ci with trusted-publishing, hand-write ci.yml incl. MSRV-check and aarch64 parity jobs, bump 0.0.1→0.1.0, CHANGELOG, cargo publish --dry-run, local wheel matrix 3.11-3.14+3.14t, docs/RELEASE.md user checklist; NO publish/tag/push of tags per R-PUB). On resume: if commits exist past 6fed702, run the task review; else re-dispatch.

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
| 10 | CI, wheels, 0.1.0 release prep (publishes deferred per R-PUB) | in-progress | — | .github/workflows/{ci,release}.yml, CHANGELOG.md, docs/RELEASE.md, version bump | see NEXT ACTION |
| 11 | Golden fixtures + CONFORMANCE.md | todo | — | — | after lot 10 |

## Blockers known at start (2026-08-21)
- No crates.io token (`~/.cargo/credentials.toml` absent) and no PyPI token on this machine → `cargo publish` / `maturin publish` steps cannot run unattended; everything up to the publish command is prepared and the exact command is recorded here when reached.
