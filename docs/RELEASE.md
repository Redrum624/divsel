# Releasing divsel 0.1.0 — the remaining (user-run) steps

> **Done 2026-09-03.** Tag `v0.1.0` = `5860457`. PyPI: release run
> [33782944986](https://github.com/Redrum624/divsel/actions/runs/33782944986) built 12 wheels + the sdist
> and published them over trusted publishing after the owner approved the `release` deployment (the first
> attempt failed with `invalid-publisher` because the pending publisher had not been registered yet; the
> Release job was re-run alone once it was). crates.io: `cargo publish -p divsel` by the owner at
> 17:41:56Z; the published `.crate` (sha256 `ebbff30c…`) is byte-identical to `cargo package` at `5860457`.
> GitHub Release v0.1.0 carries all 13 files plus the installability matrix; `install-matrix.yml` re-ran
> green against the tag ([33785213423](https://github.com/Redrum624/divsel/actions/runs/33785213423)) —
> but its 12 divsel cells still installed `spec: "."` (the checkout), so it does **not** yet measure the
> published wheel; the workflow needs a `divsel==0.1.0` spec for that. Still open from step 8 + step 9:
> revoke the `publish-new` token (crates.io Trusted Publishing was configured 2026-09-03 and the
> `crates` job now uses it), the matrix was re-run against
> `divsel==0.1.0` (run 33789059581: 48/48 recorded, divsel 12/12 with no toolchain on the runner) and
> `docs/benchmarks/README.md` refreshed from its artifact. Nothing from step 9 remains open.

Everything that could be done without a registry credential has been done. What
remains is the credentialed and outward-facing part: publishing to two
registries, one manual PyPI configuration, the tag, and the GitHub Release.
Run the steps **in order**.

## State of the tree (verified 2026-08-27)

| Fact | Command that proves it | Value |
|---|---|---|
| Repository visibility | `gh repo view Redrum624/divsel --json visibility` | **PUBLIC** (`isPrivate: false`) |
| Branch sync | `git rev-list --left-right --count origin/main...HEAD` | **run it.** Left must be `0` — anything behind means fetch and rebase before tagging. Right is however many local commits you have not pushed yet, and is expected to be non-zero while this document is being edited. |
| `feat/v0.1` | `git rev-list --left-right --count origin/feat/v0.1...main` | **run it.** Left must be `0` — that is what "fully merged into `main`" means. The right-hand count only grows and carries no information. |
| Working tree | `git status --porcelain` | clean |
| CI on this tree | run **33020266950** (`checks`), 2026-08-26T22:37:19Z, head `60159b8` | **success**, 17 jobs |
| `install-matrix` | run **33009873968**, 2026-08-26T20:20:09Z, head `02c546f` | **success** — `60159b8` touched only `PROGRESS.md`, which the workflow's `paths:` filter excludes |
| Tags | `gh api repos/Redrum624/divsel/tags` → `[]`; `git tag -l` → empty | **none yet** |
| `release` environment | `gh api repos/Redrum624/divsel/environments` | exists, id `20688404160`, created 2026-08-27 |

Two rows above say "run it" rather than carrying a number. That is deliberate,
and it is the second time this document has been wrong the same way. The
previous version asserted `0 0 — main == origin/main == 60159b8` as the first
thing a tagger reads; it was already `0 2` when it was committed, because
writing this document created commits of its own. A snapshot of a count that
changes every time you touch the repo is a claim with a shelf life measured in
minutes, and the fix is not a fresher number — it is a criterion. The rows that
*do* carry values below are ones that change only when someone deliberately
changes them.

There is **no** step here for creating the repo, pushing `main`, or flipping
visibility. All three are already done — `gh repo edit --visibility public` in
particular would now be a no-op, and it used to be step 1 of this document.

> **`release.yml` has run — three times, and all three failed.** Not on a tag:
> on plain pushes to `main`, back when the generated workflow was still named
> `CI` and still carried the generator's `on: push` triggers. Runs
> **32927028215** (`84b61fd`), **32927043835** (`aa53ba0`) and **32964560023**
> (`11c4437`), all `conclusion: failure`, all `event: push`,
> all `path: .github/workflows/release.yml`. In each the wheel jobs failed or
> were cancelled and the `Release` job was **skipped**, so nothing was ever
> uploaded anywhere — but do not read "the workflow has never run" anywhere and
> believe it. `20e2b19` is the commit that narrowed the triggers to tags +
> `workflow_dispatch`; see the hand-edit list at the bottom of this file.

## 1. Re-run the local gates at the commit you are tagging

These are not optional and they are not covered by CI. Re-run them **at the exact
commit you tag** — the last full pass is in the verification log below.

```
cargo test --workspace
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo fmt --all --check
RUSTDOCFLAGS="-D warnings" cargo doc --no-deps
cargo test -p divsel --benches      # must print "is a benchmark, not a test", not run fixtures
python python/tools/gen_golden.py --check
python -m pytest python/tests -q    # in .venv and in .venv-adapters
cargo publish -p divsel --dry-run
```

Then rebuild the wheels, because the ones in `wheels/` are **stale**: the newest
is `divsel-0.1.0.tar.gz` at 2026-08-22 14:18:51. Count what has landed since with
`git log --oneline --since="2026-08-22 14:18:51" | wc -l` — it was 49 on
2026-08-27 and only grows. A build artifact is never evidence about a later
commit, so the count matters only in that it is not zero; rebuild regardless.

```
python -m maturin build --release -o wheels
python -m maturin sdist -o wheels
```

Check `git log <sha>..HEAD` before reusing **any** dated entry in this file.

## 2. Confirm the CHANGELOG date

`CHANGELOG.md`'s heading reads `## [0.1.0] - 2026-08-27`. If you tag on a later
date, change it and amend before tagging — the released section must carry the
release date.

## 3. Publish the crate

```
cargo login          # a crates.io token with publish-new scope (0.1.1 and later need publish-update)
cargo publish -p divsel
```

This publishes **0.1.0** directly; there is no 0.0.1 squat on crates.io to skip.
The name is free — see "Registry name availability" below, and re-check it
immediately before you run this. `cargo publish -p divsel --dry-run` passes
locally (18 files, 376.3 KiB / 107.1 KiB compressed), including the
`categories`/`keywords` slug validation.

**Irreversible.** crates.io has no unpublish; a bad 0.1.0 can only be yanked,
and the version number can never be reused.

## 4. Configure PyPI Trusted Publishing

The GitHub half is **already done** (step 5 below explains what was created).
The PyPI half is manual and cannot be scripted without a PyPI token:

- <https://pypi.org/manage/account/publishing/> → add a **pending publisher** for
  a new project `divsel`:
  - PyPI project name: `divsel`
  - Owner: `Redrum624`
  - Repository: `divsel`
  - Workflow: `release.yml`
  - Environment: `release`

The environment name must match `[tool.maturin.generate-ci.github]
publishing-environment = "release"` in `pyproject.toml` and `environment:
release` at `.github/workflows/release.yml:217`. No token is stored anywhere:
`release.yml` publishes with `uv publish --trusted-publishing always` over OIDC
(`permissions: id-token: write`).

## 5. The `release` environment — already created, and what it will do to you

Created 2026-08-27 (`gh api --method PUT repos/Redrum624/divsel/environments/release`),
id `20688404160`, with two protection rules **deliberately** enabled:

- **Required reviewer: `Redrum624`** (`prevent_self_review: false`). The
  `Release` job — the only job that touches PyPI — will sit in
  `waiting` until it is approved. You approve your own deployment in one click
  from the run page. On a public repo with `on: push: tags: '*'`, where *any*
  tag starts a six-target wheel matrix and an irreversible upload, one click is
  the cheapest possible interlock. Without it there is nothing at all between a
  mistyped `git push --tags` and a permanent PyPI release.
- **Deployment branch/tag policy: `v*` only** (`custom_branch_policies: true`).
  A tag that does not match `v*` builds wheels but cannot reach the `Release`
  job.

If you decide you want neither, remove them consciously — a release environment
with no gate on a public repo is a choice, not a default.

## 6. Tag and push

```
git tag -a v0.1.0 -m "divsel 0.1.0"
git push --follow-tags
```

The tag is the **only** trigger for the wheel matrix. It builds linux/windows/macos
× (x86_64|x64, aarch64), plus the free-threaded `cp314t` wheels and the sdist,
attests them, and publishes to PyPI from the `release` environment.

Then **go and approve the deployment** (step 5). The run will not finish on its
own.

## 7. Verify the published package

On a clean Windows machine, or any machine that took no part in the build:

```
py -3.14 -m pip install divsel
py -3.14 -c "import divsel, numpy as np; print(divsel.gist_select(np.random.default_rng(0).standard_normal((50,8),dtype=np.float32), k=5))"
```

Also check the PyPI project page renders the README, and that the Homepage /
Source / Changelog / Issues links from `[project.urls]` are present.

## 8. GitHub Release

Create release `v0.1.0` from the tag. Body: the `CHANGELOG.md` 0.1.0 section, the
benchmark tables from `docs/benchmarks/README.md` (comparison + Windows
installability), and the installability matrix artifact from the
`install-matrix.yml` run.

## 9. Post-release

Re-run `.github/workflows/install-matrix.yml` (`workflow_dispatch`) against the
release tag, so the matrix reflects the published wheel rather than a
`pip install .` of the checkout, and refresh the assembled table in
`docs/benchmarks/README.md`. All 48 cells are already measured for the branch;
this re-run is about the published artifact, not about filling gaps.

Then retire the pre-publish wording, which is wrong the moment step 3 succeeds:

- `README.md:5` — `Status: **0.1.0 — release candidate; publish pending.**`
- `README.md:6` — the pointer to this file as "the crates.io / PyPI publish steps"
- `README.md:16` — the "Until 0.1.0 lands on PyPI/crates.io, install from a
  checkout instead" install fallback

---

## Registry name availability

`divsel` is unclaimed on both registries. Checked **2026-08-27**:

| Registry | Request | Response |
|---|---|---|
| crates.io | `curl https://crates.io/api/v1/crates/divsel` | HTTP 404, `{"errors":[{"detail":"crate \`divsel\` does not exist"}]}` |
| PyPI | `curl -o /dev/null -w "%{http_code}" https://pypi.org/pypi/divsel/json` | HTTP 404 |

**Availability is not permanent and this is not a reservation.** `README.md`,
`docs/CONFORMANCE.md` and the `CHANGELOG.md` 0.1.0 section all already write the
name as if it were ours. Re-run both requests immediately before step 3; if
either returns 200, stop — a rename touches the crate name in
`crates/divsel/Cargo.toml`, `name` in `pyproject.toml`, the Python package under
`python/`, and every one of those documents, and none of it can be done after a
publish.

---

## Verification log

> **Every entry below is dated and pinned to a commit. It is evidence for *that*
> tree and for nothing later.** Anything built from source — everything under
> `wheels/` in particular — is a build artifact, never proof about a later
> commit. Before reusing an entry, run `git log <sha>..HEAD`; if it prints
> anything, re-run the step-1 gates at the commit you are actually tagging.

### 2026-08-27, Windows 11 x64, at `60159b8` (`main`, current HEAD)

Local, on this machine:

- `cargo test --workspace`: **134 passed, 0 failed** — 117 lib + 5 `exact_oracle`
  + 5 `golden` + 3 `shared_fixture` + 4 doc-tests.
- `cargo clippy --workspace --all-targets --all-features -- -D warnings`: clean.
- `cargo fmt --all --check`: clean (exit 0).
- `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`: clean.
- `cargo test -p divsel --benches`: printed `benches/gist.rs is a benchmark, not
  a test: skipping the run` — the guard held.
- `python python/tools/gen_golden.py --check`: byte-identical, 19365 bytes.
- `pytest python/tests -q` in `.venv` (CPython 3.14.2, GIL): **155 passed, 74
  skipped**. In `.venv-adapters` (3.14.2 + langchain-core + llama-index-core):
  **227 passed, 2 skipped** — the adapter tests that the base venv skips.
- `cargo publish -p divsel --dry-run`: green. `Packaged 18 files, 376.3KiB
  (107.1KiB compressed)`, verify step compiled, upload aborted for the dry run.
- Registry name check: both 404 (see above).

On CI, same tree:

- `checks` run **33020266950**, 2026-08-26T22:37:19Z, head `60159b8`: success,
  17 jobs — `rust (fmt, clippy, test, MSRV check)`, `simd parity (aarch64)`,
  `golden conformance (rust reader)` × 3 OSes, and 12 python cells including
  `python 3.12 / ubuntu-latest + adapters`.
- `install-matrix` run **33009873968**, head `02c546f`: success. Not re-run at
  `60159b8` because that commit touches only `PROGRESS.md`.

**Not** verified at this commit: the per-interpreter wheel installs (3.11–3.14
and 3.14t) and the wheel/sdist builds themselves. The artifacts in `wheels/` are
from 2026-08-22 and do not describe this tree. Rebuild them in step 1.

### 2026-08-25, Windows 11 x64, final-review fixes

- `cargo test --workspace`, `cargo clippy --workspace --all-targets --all-features
  -- -D warnings`, `cargo fmt --all --check`, `RUSTDOCFLAGS="-D warnings" cargo doc
  --no-deps`: green.
- `pytest python/tests -q` in both local venvs (3.14.2 GIL, and the adapter venv
  with langchain-core + llama-index-core): green.
- `python python/tools/gen_golden.py --check`: byte-identical.
- `cargo package -p divsel` + `cargo test --test golden` from the unpacked
  `target/package/divsel-0.1.0/`: the crate now carries `LICENSE`, and the golden
  reader reports a named skip there instead of panicking (the fixture lives at the
  workspace root, which `cargo package` cannot reach).
- `python -m maturin sdist`: the sdist now carries `test-assets/golden-selection.json`
  and `python/tests/**`, so both readers run from an unpacked source distribution.

### 2026-08-22, Windows 11 x64, at the 0.1.0 tree of that date

- `cargo test` (workspace), `cargo clippy --all-targets --all-features -- -D warnings`,
  `cargo fmt --check`, `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`: green.
- `cargo publish -p divsel --dry-run`: green (packaging + categories/keywords slugs).
- `python -m maturin build --release -o wheels` → `divsel-0.1.0-cp311-abi3-win_amd64.whl`;
  `python -m maturin sdist -o wheels` → `divsel-0.1.0.tar.gz`.
- The one abi3 wheel installed into fresh `uv` venvs for CPython 3.11, 3.12, 3.13 and
  3.14 (GIL); the smoke selection ran identically in all four.
- Free-threaded CPython 3.14t: the abi3 wheel is (correctly) refused; the
  version-specific `divsel-0.1.0-cp314-cp314t-win_amd64.whl` was built and smoke-tested
  in a 3.14t venv.
- `.github/workflows/ci.yml` and `release.yml` parse as valid YAML; `release.yml`
  publishes via trusted publishing with **no token env**, and contains exactly the
  intended target matrix plus the `python3.14t` build steps.

**These wheel and sdist results are the ones that have not been reproduced
since.** They are the reason step 1 tells you to rebuild.

What could **not** be verified locally at any date: Linux/macOS wheel builds
(CI-only), the actual PyPI/crates.io uploads, and the Trusted Publishing
round-trip. Those are what steps 3–7 close.

---

## Regenerating release.yml — and the four hand edits you must re-apply

`release.yml` is generated by maturin:

```
python -m maturin generate-ci github -o .github/workflows/release.yml
```

Configuration lives in `[tool.maturin.generate-ci.github]` in `pyproject.toml`,
and regeneration **rewrites the whole file**. Prefer editing the TOML.

But the file is **not** purely generated, and the difference is
release-breaking. `.github/workflows/release.yml:13-23` lists four local
changes the generator has no configuration key for. Regeneration silently
discards all four; **re-apply them every time**, and diff the result before
committing:

1. **`name: release`**, not the generator's `name: CI` — which collided with
   `ci.yml`.
2. **`on:` is tags + `workflow_dispatch` only.** The generator also fires this
   on every push to `main`/`master` and on every pull request, building the full
   six-target wheel matrix — two of them macOS, billed at 10× — for changes that
   are not releases. That is what produced the three failed `release.yml` runs
   noted at the top of this file, and together with the other two workflows it
   exhausted a 2,000-minute monthly allowance in about four pushes.
3. **`concurrency:`** cancels a superseded run of the same ref.
4. **The `crates` job** (last in the file) publishes the core crate to crates.io over
   Trusted Publishing after the PyPI publish, from the same `release` environment; it is
   not generated at all.

The explanatory header comment (`release.yml:7-23`) is itself clobbered by
regeneration — re-add it too, since it is what tells the next person these four
edits exist.
