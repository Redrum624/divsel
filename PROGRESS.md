# PROGRESS — divsel v0.1

Plan: `docs/superpowers/plans/2026-08-21-divsel-v0.1.md` · Execution: Subagent-Driven Development on branch `feat/v0.1`.

CURRENT LOT: COMPLETE — code, review, CI, the CHANGELOG and the release checklist are all done. What remains needs a registry credential and nothing else.
NEXT ACTION: **Publish 0.1.0.** `main` == `origin/main`, working tree clean, **no tags exist yet** (`git tag -l` empty, `gh api .../tags` → `[]`). `docs/RELEASE.md` is now accurate against the tree and is the authority for the sequence; every claim in it is dated and names the command that proves it. The order:

1. **Re-run the local gates at the commit you tag** (RELEASE.md step 1) and **rebuild `wheels/`** — its newest artifact is `divsel-0.1.0.tar.gz` from 2026-08-22 14:18, and `git log --oneline --since="2026-08-22 14:18:51" | wc -l` counts what has landed since (49 on 2026-08-27, and only growing — the count matters only in that it is not zero). Last full green pass: 2026-08-27 at `60159b8` — `cargo test --workspace` 134 passed (117+5+5+3+4), clippy/fmt/`cargo doc` clean, bench guard held, `gen_golden.py --check` byte-identical (19365 bytes), `cargo publish -p divsel --dry-run` green (18 files, 376.3KiB), `pytest python/tests -q` **155 passed / 74 skipped** in `.venv` and **227 passed / 2 skipped** in `.venv-adapters`. CI `checks` run **33020266950** success on this exact tree.
2. **crates.io**: `cargo login` + `cargo publish -p divsel`. Needs a token — `~/.cargo/credentials.toml` is absent and `CARGO_REGISTRY_TOKEN` is unset on this machine. Irreversible.
3. **PyPI**: add the pending publisher by hand at <https://pypi.org/manage/account/publishing/> — project `divsel`, owner `Redrum624`, repo `divsel`, workflow `release.yml`, environment `release`. Cannot be scripted without a PyPI token.
4. **Tag**: `git tag -a v0.1.0 -m "divsel 0.1.0"` + `git push --follow-tags` — the tag is the only trigger for the wheel matrix — **then go and approve the deployment**: the `release` environment has a required reviewer, so the `Release` job waits in `waiting` until you click.
5. GitHub Release from the tag, then the post-release `install-matrix` re-run and the README status lines that RELEASE.md step 9 lists.

Closed this session (2026-08-27), all committed, nothing pushed:

- **CHANGELOG** — `[Unreleased]` folded into `## [0.1.0] - 2026-08-27`. Since nothing preceded 0.1.0 the folded material sits under `### Added`, not `### Changed`; the docs-only / `golden-selection.json`-byte-identical fact is restated in release tense. Link references now resolve: `[0.1.0]` is the only shortcut reference in the file and it has a definition; the definition-less `[Unreleased]` is gone.
- **`docs/RELEASE.md`** — rewritten, all four false statements gone. The visibility flip is dropped (verified PUBLIC). The "`release.yml` has never run" note is replaced by the truth: it ran **three times and failed all three** — runs `32927028215`, `32927043835`, `32964560023`, all `event: push`, all with the `Release` job skipped, from when the generated workflow was still named `CI` with the generator's triggers. The stale `0 18` divergence is replaced by measured sync (`0 0`, and `feat/v0.1` is `0 6`, fully merged). The "generated, never hand-edit" tail is replaced by the three hand edits at `release.yml:13-23` (name, tag-only triggers, concurrency) and an instruction to re-apply them after every regeneration. A 2026-08-27 verification-log entry now covers `60159b8`.
- **GitHub `release` environment created** — id `20688404160`, 2026-08-27. `gh api repos/Redrum624/divsel/environments` now returns `total_count: 1`. Protection rules deliberately on: required reviewer `Redrum624` (`prevent_self_review: false` — one self-approve click between a mistyped tag and an irreversible PyPI upload, on a public repo whose release trigger is `tags: '*'`) and a `v*` deployment tag policy.
- **Name availability checked for the first time**, 2026-08-27: `https://crates.io/api/v1/crates/divsel` → HTTP 404 `{"errors":[{"detail":"crate \`divsel\` does not exist"}]}`; `https://pypi.org/pypi/divsel/json` → HTTP 404. Recorded in `docs/RELEASE.md` with the date and an instruction to re-check immediately before publishing — availability is not a reservation.
- **Version consistency swept — no drift.** `Cargo.toml:6` `version = "0.1.0"` is the single source; `crates/divsel/Cargo.toml:4` and `crates/divsel-py/Cargo.toml:3` both inherit `version.workspace = true`; `Cargo.lock:201` and `:212` both `0.1.0`; `pyproject.toml:7` is `dynamic = ["version"]` (maturin reads the crate); `python/tests/test_import.py:10` asserts `divsel.__version__ == "0.1.0"` and passes; the CHANGELOG heading is `0.1.0`. Not a version string but pre-publish wording that must be retired after step 5: `README.md:5`, `:6`, `:16`.

Also open, non-blocking: the adapters pass the *query* vector straight to numpy while document vectors go through `_vectors_or_reason` (`python/divsel/adapters/langchain.py:191`, `llamaindex.py:217`), so a wrong-rank query embedding escapes both the `DivselFallbackWarning` path and the `strict=True` ValueError contract.

Honest note for any write-up: the final adversarial review (session `divsel-55`) hit its **4-round cap without producing a clean round**; its stated exit condition was two consecutive clean rounds. The fixes it produced are real and merged and CI is green — but describe the process accurately.

Downstream consumers are **not** blocked by any of the above: Aura holds `golden-selection.json` copied verbatim with `upstream_sha256` recorded, and that hash still matches this tree. Publishing gates divsel's own distribution and nothing else.

| Lot | Goal (one line) | Status | Commit SHA | Files touched | Next concrete action |
|-----|-----------------|--------|-----------|---------------|----------------------|
| 0 | Workspace scaffold (publish/push deferred to user checklist) | done | 16b99ab | Cargo.toml, crates/*, pyproject.toml, LICENSE, python/divsel | — |
| 1 | Metric + `Points` storage | done | 3fa5f24 | crates/divsel/src/{error,metric,points,lib}.rs | — |
| 2 | Utility functions (Linear, Coverage, FacilityLocation) | done | f94a660 | crates/divsel/src/{utility,error,lib}.rs | — |
| 3 | `GreedyIndependentSet` + CELF lazy greedy | done | ddf6ff0 | crates/divsel/src/{greedy,testutil,lib,utility}.rs | — |
| 4 | GIST driver + public API | done | 34e8e54 | crates/divsel/src/{gist,lib}.rs | — |
| 5 | Brute-force oracle + approximation-ratio property test | done | a96409f | crates/divsel/tests/exact_oracle.rs | — |
| 6 | SIMD distance kernels + parallel thresholds + criterion benches | done | 1b2789e | crates/divsel/{src/metric,src/gist,src/testutil}.rs, benches/gist.rs, Cargo.toml | — |
| 7 | PyO3 bindings + Python package | done | 16bf5fb | crates/divsel-py/src/lib.rs, python/divsel/{__init__.py,_divsel.pyi}, python/tests/{test_api.py,fixtures.py}, crates/divsel/tests/shared_fixture.rs, pyproject.toml, .gitignore | — |
| 8 | Benchmarks vs incumbents + installability matrix | done | 9b3523c | bench/*, docs/benchmarks/*, .github/workflows/install-matrix.yml, .github/scripts/* | — |
| 9 | LangChain / LlamaIndex adapters | done | 7dcfa1e | python/divsel/adapters/*, python/tests/test_adapters.py, README.md | — |
| 10 | CI, wheels, 0.1.0 release prep (publishes deferred per R-PUB) | done | 53559ed | .github/workflows/{ci,release}.yml, CHANGELOG.md, docs/RELEASE.md, Cargo.toml, pyproject.toml, README.md | — |
| 11 | Golden fixtures + CONFORMANCE.md (22 cases) | done | 4c304e0 | test-assets/golden-selection.json, .gitattributes, python/tools/gen_golden.py, crates/divsel/tests/golden.rs, python/tests/test_golden.py, docs/CONFORMANCE.md, CHANGELOG.md, ci.yml | — |

## Blockers known at start (2026-08-21)
- No crates.io token (`~/.cargo/credentials.toml` absent) and no PyPI token on this machine → `cargo publish` / `maturin publish` steps cannot run unattended; everything up to the publish command is prepared and the exact command is recorded here when reached.
