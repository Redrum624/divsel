# PROGRESS — divsel v0.1

Plan: `docs/superpowers/plans/2026-08-21-divsel-v0.1.md` · Execution: Subagent-Driven Development on branch `feat/v0.1`.

CURRENT LOT: 9
NEXT ACTION: Task 9 implementer running from `.superpowers/sdd/2026-08-21-divsel-v0.1/task-9-brief.md` + `task-9-context.md` (verify installed langchain-core / llama-index-core signatures BEFORE implementing; DivselRetriever + DivselNodePostprocessor with the R-G13 fallback; lazy framework imports; adapters venv). On resume: if commits exist past the lot-8 checkpoint, run the task review; else re-dispatch.

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
| 9 | LangChain / LlamaIndex adapters | in-progress | — | python/divsel/adapters/*, python/tests/test_adapters.py | see NEXT ACTION |
| 10 | CI, wheels, 0.1.0 release | todo | — | — | after lot 9 |
| 11 | Golden fixtures + CONFORMANCE.md | todo | — | — | after lot 10 |

## Blockers known at start (2026-08-21)
- No crates.io token (`~/.cargo/credentials.toml` absent) and no PyPI token on this machine → `cargo publish` / `maturin publish` steps cannot run unattended; everything up to the publish command is prepared and the exact command is recorded here when reached.
