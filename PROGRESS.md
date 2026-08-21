# PROGRESS — divsel v0.1

Plan: `docs/superpowers/plans/2026-08-21-divsel-v0.1.md` · Execution: Subagent-Driven Development on branch `feat/v0.1`.

CURRENT LOT: 0
NEXT ACTION: Task 0 Step 1 — write workspace `Cargo.toml` + `crates/divsel/Cargo.toml` exactly as in the plan, `cargo build` green.

| Lot | Goal (one line) | Status | Commit SHA | Files touched | Next concrete action |
|-----|-----------------|--------|-----------|---------------|----------------------|
| 0 | Workspace scaffold + claim both names | todo | — | — | Plan Task 0 Step 1 |
| 1 | Metric + `Points` storage | todo | — | — | after lot 0 |
| 2 | Utility functions (Linear, Coverage, FacilityLocation) | todo | — | — | after lot 1 |
| 3 | `GreedyIndependentSet` + CELF lazy greedy | todo | — | — | after lot 2 |
| 4 | GIST driver + public API | todo | — | — | after lot 3 |
| 5 | Brute-force oracle + approximation-ratio property test | todo | — | — | after lot 4 |
| 6 | SIMD distance kernels + parallel thresholds + criterion benches | todo | — | — | after lot 5 |
| 7 | PyO3 bindings + Python package | todo | — | — | after lot 5 |
| 8 | Benchmarks vs incumbents + installability matrix | todo | — | — | after lots 6, 7 |
| 9 | LangChain / LlamaIndex adapters | todo | — | — | after lot 8 |
| 10 | CI, wheels, 0.1.0 release | todo | — | — | after lot 9 |
| 11 | Golden fixtures + CONFORMANCE.md | todo | — | — | after lot 10 |

## Blockers known at start (2026-08-21)
- No crates.io token (`~/.cargo/credentials.toml` absent) and no PyPI token on this machine → `cargo publish` / `maturin publish` steps cannot run unattended; everything up to the publish command is prepared and the exact command is recorded here when reached.
