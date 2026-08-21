# PROGRESS — divsel v0.1

Plan: `docs/superpowers/plans/2026-08-21-divsel-v0.1.md` · Execution: Subagent-Driven Development on branch `feat/v0.1`.

CURRENT LOT: 5
NEXT ACTION: Task 5 implementer running from `.superpowers/sdd/2026-08-21-divsel-v0.1/task-5-brief.md` + `task-5-context.md` (tests/exact_oracle.rs: enumerate subsets of size 0..=k on 500 instances, assert (1/2−ε), (2/3−ε), exact 2/3 exhaustive, 0.387 regression, sanity). On resume: if commits exist past ff3b1da, run the task review; else re-dispatch. If the oracle FAILS, the library is wrong — route to a library fix, never weaken the bound.

| Lot | Goal (one line) | Status | Commit SHA | Files touched | Next concrete action |
|-----|-----------------|--------|-----------|---------------|----------------------|
| 0 | Workspace scaffold (publish/push deferred to user checklist) | done | b639630 | Cargo.toml, crates/*, pyproject.toml, LICENSE, python/divsel | — |
| 1 | Metric + `Points` storage | done | dacb363 | crates/divsel/src/{error,metric,points,lib}.rs | — |
| 2 | Utility functions (Linear, Coverage, FacilityLocation) | done | 8233c90 | crates/divsel/src/{utility,error,lib}.rs | — |
| 3 | `GreedyIndependentSet` + CELF lazy greedy | done | 06366e5 | crates/divsel/src/{greedy,testutil,lib,utility}.rs | — |
| 4 | GIST driver + public API | done | ff3b1da | crates/divsel/src/{gist,lib}.rs | — |
| 5 | Brute-force oracle + approximation-ratio property test | in-progress | — | crates/divsel/tests/exact_oracle.rs | see NEXT ACTION |
| 6 | SIMD distance kernels + parallel thresholds + criterion benches | todo | — | — | after lot 5 |
| 7 | PyO3 bindings + Python package | todo | — | — | after lot 5 |
| 8 | Benchmarks vs incumbents + installability matrix | todo | — | — | after lots 6, 7 |
| 9 | LangChain / LlamaIndex adapters | todo | — | — | after lot 8 |
| 10 | CI, wheels, 0.1.0 release | todo | — | — | after lot 9 |
| 11 | Golden fixtures + CONFORMANCE.md | todo | — | — | after lot 10 |

## Blockers known at start (2026-08-21)
- No crates.io token (`~/.cargo/credentials.toml` absent) and no PyPI token on this machine → `cargo publish` / `maturin publish` steps cannot run unattended; everything up to the publish command is prepared and the exact command is recorded here when reached.
