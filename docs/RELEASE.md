# Releasing divsel 0.1.0 — the remaining (user-run) steps

Everything buildable was built and verified locally (see "Already verified" below).
What remains is exactly the outward-facing actions: pushing, publishing, and the
one-time PyPI/GitHub configuration. Run these **in order**.

## 1. Merge and create the public repository

Merge `feat/v0.1` into `main`, then create and push the GitHub repo:

```
git checkout main
git merge --ff-only feat/v0.1     # falls back: git merge --no-ff feat/v0.1
gh repo create Redrum624/divsel --public --source . --push
```

(or create the repo first, then `git remote add origin git@github.com:Redrum624/divsel.git`
and `git push -u origin main`).

> Note: taking the repo public was your earlier decision (private until release);
> `--public` here is that flip. If you want a last look first, use `--private` and
> run `gh repo edit Redrum624/divsel --visibility public --accept-visibility-change-consequences` when ready.

## 2. Publish the crate

```
cargo login          # paste a crates.io token with publish-new scope (0.1.1 and later need publish-update)
cargo publish -p divsel
```

This publishes **0.1.0** directly (skip any 0.0.1 squat — the `divsel` name was
verified free on 2026-08-21). `cargo publish -p divsel --dry-run` already passes
locally, including the `categories`/`keywords` validation.

## 3. Configure PyPI Trusted Publishing + the GitHub environment

- On PyPI (https://pypi.org/manage/account/publishing/): add a **pending publisher**
  for a new project `divsel`: owner `Redrum624`, repository `divsel`, workflow
  `release.yml`, environment `release`.
- On GitHub (repo Settings → Environments): create an environment named **`release`**
  (optionally add yourself as required reviewer — that gates every PyPI upload).

No token is stored anywhere: `release.yml` publishes with
`uv publish --trusted-publishing always` via OIDC (`permissions: id-token: write`).

## 4. Tag and push

```
git tag -a v0.1.0 -m "divsel 0.1.0"
git push --follow-tags
```

The tag triggers `release.yml`: wheels for linux/windows/macos × (x86_64|x64, aarch64)
plus the free-threaded `cp314t` wheels and the sdist are built, attested, and published
to PyPI from the `release` environment.

## 5. Verify the published package

On a clean Windows machine (or any machine that took no part in the build):

```
py -3.14 -m pip install divsel
py -3.14 -c "import divsel, numpy as np; print(divsel.gist_select(np.random.default_rng(0).standard_normal((50,8),dtype=np.float32), k=5))"
```

Also check the PyPI project page renders the README.

## 6. GitHub Release

Create release `v0.1.0` from the tag. Body: the `CHANGELOG.md` 0.1.0 section, the
benchmark tables from `docs/benchmarks/README.md` (comparison + Windows installability),
and the installability matrix artifact from the `install-matrix.yml` run.

## 7. Post-release

Re-run `.github/workflows/install-matrix.yml` (workflow_dispatch) so the divsel column
fills for Linux/macOS, and paste the assembled table into `docs/benchmarks/README.md`
(the 32 pending cells noted there).

---

## Already verified locally (2026-08-22, Windows 11 x64, this checkout at 0.1.0)

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
- `.github/workflows/ci.yml` and the generated `release.yml` parse as valid YAML;
  `release.yml` was verified to publish via trusted publishing with **no token env**,
  and to contain exactly the intended target matrix + the `python3.14t` build steps.

What could **not** be verified locally: Linux/macOS wheel builds (CI-only), the actual
PyPI/crates.io uploads, and the Trusted Publishing round-trip — that is what steps 1–7 close.

## Regenerating release.yml

`release.yml` is generated — never hand-edit it:

```
python -m maturin generate-ci github -o .github/workflows/release.yml
```

Configuration lives in `[tool.maturin.generate-ci.github]` in `pyproject.toml`.
Regeneration **clobbers** the extra header comment (verified) — re-add it after.
