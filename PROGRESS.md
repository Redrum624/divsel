# PROGRESS — divsel v0.1

Plan: `docs/superpowers/plans/2026-08-21-divsel-v0.1.md` · Execution: Subagent-Driven Development on branch `feat/v0.1`.

CURRENT LOT: COMPLETE — code, review and CI are done; only the two registry credentials remain.
NEXT ACTION: `main` == `origin/main` == `9262375`, working tree clean. **Three claims in the previous version of this line were stale and are corrected here:** the recorded head was `c95a128` (five commits behind); the repo was described as private (it is **PUBLIC** since 2026-08-26); and GitHub Actions minutes were reported exhausted (**CI is green on HEAD** — runs `33009873991` `checks` and `33009873968` `install-matrix`, both success at 2026-08-26T20:20:09Z). Tests at HEAD: **134 Rust / 155 Python (74 skipped)**, CI-verified on this exact tree. All 39 deferred minors were dispositioned in `ea7aaac..65fc69a` — see `.superpowers/sdd/2026-08-21-divsel-v0.1/final-review-report.md`.

Remaining work, in order — the first three need no credentials:

1. **CHANGELOG** — fold the `## [Unreleased]` conformance-contract section into `## [0.1.0]` and re-date it. Tagging as-is ships a 0.1.0 section that does not describe the tagged tree, and `[Unreleased]` has no link definition.
2. **`docs/RELEASE.md`** — rewrite; four statements are now false. Step 1 tells you to run `gh repo edit --visibility public` (already public); its note says `release.yml` has never run and quotes a `0 18` divergence (main is 0/0 and CI is green); the tail says the workflow is generated and must never be hand-edited, contradicting `release.yml:14-24` which lists three deliberate hand edits that must be re-applied after any regeneration; and the verification log stops at 2026-08-25, covering neither `e9447ac` nor `9262375`.
3. **Create the `release` GitHub environment** — `release.yml:217` requires `environment: release` and `gh api repos/Redrum624/divsel/environments` returns `total_count: 0`. PyPI trusted publishing cannot work without it.
4. **Publish** (needs credentials — see Blockers): `cargo login` + `cargo publish -p divsel`; configure the PyPI pending publisher; `git tag -a v0.1.0` + `git push --follow-tags` (the tag is the only trigger for the wheel matrix); then the GitHub Release. **Check first that the name `divsel` is actually free on crates.io and PyPI** — nobody has, and README, CONFORMANCE.md and `CHANGELOG.md:122` all already assume it.

Also open, non-blocking: the adapters pass the *query* vector straight to numpy while document vectors go through `_vectors_or_reason` (`python/divsel/adapters/langchain.py:191`, `llamaindex.py:217`), so a wrong-rank query embedding escapes both the `DivselFallbackWarning` path and the `strict=True` ValueError contract.

Honest note for any write-up: the final adversarial review (session `divsel-55`) hit its **4-round cap without producing a clean round**; its stated exit condition was two consecutive clean rounds. The fixes it produced are real and merged and CI is green — but describe the process accurately.

Downstream consumers are **not** blocked by any of the above: Aura holds `golden-selection.json` copied verbatim with `upstream_sha256` recorded, and that hash still matches this tree. Publishing gates divsel's own distribution and nothing else.

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
