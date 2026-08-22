# divsel benchmarks

Everything in this file was produced by a command written in this file, on the machine named below, on
2026-08-22, with the library versions and commit SHAs listed. Nothing is quoted from another project's
README as a measurement; where another project's claim is discussed, the command that was run against it
and what happened are shown.

Two things are measured:

1. **Installability** of divsel and the three incumbents on Windows x64 for CPython 3.11, 3.12, 3.13 and
   3.14 (16 cells, measured here), plus a workflow that measures the same 4 libraries on Linux and macOS
   (`.github/workflows/install-matrix.yml`, not run yet -- this repository has not been pushed to CI).
2. **Wall-clock, peak RSS and achieved `f(S)`** of divsel against PyPI `gist-select`, git `gist-sampling`
   and a naive numpy MMR baseline on identical inputs, scored by one shared evaluator.

## Machine, toolchain, versions

| | |
|---|---|
| CPU | Intel(R) Core(TM) i7-10875H @ 2.30GHz, 8 cores / 16 logical, AVX2 (no AVX-512) |
| RAM | 63.9 GiB |
| OS | Microsoft Windows 11 Pro 10.0.26200 (build 26200) |
| Rust | rustc 1.92.0 (ded5c06cf 2025-12-08), cargo 1.92.0; release profile, default x86-64 target |
| Python (comparison venv `.venv-3.14`) | CPython 3.14.2 (GIL build), `C:\Python314\python.exe` |
| Pythons (installability) | 3.11.9, 3.12.11 (uv-managed), 3.13.11, 3.14.2 -- see the matrix for the interpreter paths |
| numpy / scipy / numba / psutil | 2.5.2 / 1.18.1 / 0.67.0 / 7.2.2 (in `.venv-3.14`) |
| uv / git | uv 0.11.26, git 2.55.0.windows.4 |
| divsel | 0.0.1, wheel `divsel-0.0.1-cp311-abi3-win_amd64.whl` built from git `d379953` with `python -m maturin build --release -o wheels` (sha256 `5d7a9578c7bf…248c0`) |
| gist-select | 0.1.0 from PyPI (`gist_select-0.1.0-py3-none-any.whl`, uploaded 2026-02-19, sha256 `12267b8cc544239d…`); the GitHub repository `kclaka/gist-select` is at `f6281f3611c1881b613ec6d8de22459ff89f161a` (its only commit, 2026-02-19) |
| gist-sampling | 0.1.0 from `git+https://github.com/musubi-labs/gist-sampling` at `ab97eb5acc619ec1527acac91accee8c9dfa0b8d` (HEAD on 2026-08-22); pulls numba 0.67.0, llvmlite 0.49.0, scikit-learn 1.9.0, pandas 3.0.5 |
| submodlib-py | 0.0.3 on PyPI (uploaded 2025-05-14); see the matrix -- it does not install here |
| Seed / objective | `np.random.default_rng(0)`; `lam = 1.0`, `eps = 0.1`, metric `cosine`; MMR `lambda = 0.5` |
| Date | 2026-08-22 |

## Installability matrix

### Windows x64, measured here

One fresh venv per interpreter, four installs per venv. The divsel wheel is the **same file** in all four
cells -- the point of the `abi3` wheel is that one build covers 3.11 through 3.14. The line shown is the
first success or failure line of the command's output (full logs were kept for the run).

```
uv venv --python 3.11 .venv-3.11           # CPython 3.11.9  C:\Users\Razer\AppData\Local\Programs\Python\Python311\python.exe
uv venv --python 3.12 .venv-3.12           # CPython 3.12.11 (uv-managed)
uv venv --python 3.13 .venv-3.13           # CPython 3.13.11 C:\Users\Razer\AppData\Local\Programs\Python\Python313\python.exe
uv venv --python C:/Python314/python.exe .venv-3.14   # CPython 3.14.2 (GIL build; a bare `3.14` resolved to the free-threaded 3.14.6, see below)
python -m maturin build --release -o wheels            # once; system Python 3.14.2
for each venv:
  uv pip install --python .venv-<ver>/Scripts/python.exe gist-select
  uv pip install --python .venv-<ver>/Scripts/python.exe submodlib-py
  uv pip install --python .venv-<ver>/Scripts/python.exe git+https://github.com/musubi-labs/gist-sampling
  uv pip install --python .venv-<ver>/Scripts/python.exe wheels/divsel-0.0.1-cp311-abi3-win_amd64.whl numpy
  .venv-<ver>/Scripts/python.exe -c "import numpy as np, divsel; print(divsel.gist_select(np.eye(3, dtype=np.float32), k=2))"
```

| library | 3.11.9 | 3.12.11 | 3.13.11 | 3.14.2 |
|---|---|---|---|---|
| `gist-select` (PyPI) | ok: `+ gist-select==0.1.0` (+ numpy 2.4.6, scipy 1.17.1) | ok: `+ gist-select==0.1.0` (+ numpy 2.5.2, scipy 1.18.1) | ok: `+ gist-select==0.1.0` (+ numpy 2.5.2, scipy 1.18.1) | ok: `+ gist-select==0.1.0` (+ numpy 2.5.2, scipy 1.18.1) |
| `submodlib-py` (PyPI) | **fail**: `Because all versions of submodlib-py have no wheels with a matching platform tag (e.g., win_amd64)` | **fail**: `Because all versions of submodlib-py have no wheels with a matching platform tag (e.g., win_amd64)` | **fail**: `Because all versions of submodlib-py have no wheels with a matching Python ABI tag (e.g., cp313)` | **fail**: `Because all versions of submodlib-py have no wheels with a matching Python ABI tag (e.g., cp314)` |
| `gist-sampling` (git) | ok: `+ gist-sampling==0.1.0 (from git+…@ab97eb5…)`, 11 packages | ok: same, 11 packages | ok: same, 11 packages | ok: same, 11 packages |
| `divsel` (one abi3 wheel) | ok: `+ divsel==0.0.1 (from file:///C:/Dev/divsel/wheels/divsel-0.0.1-cp311-abi3-win_amd64.whl)`; import ok, `[0, 1]` | ok: same wheel; import ok, `[0, 1]` | ok: same wheel; import ok, `[0, 1]` | ok: same wheel; import ok, `[0, 1]` |

The `submodlib-py` hints uv printed alongside the failures: on 3.11/3.12 "Wheels are available for
`submodlib-py` (v0.0.3) on the following platforms: `manylinux_2_17_x86_64`, `manylinux2014_x86_64`,
`musllinux_1_2_x86_64`, `macosx_10_9_x86_64` (3.12: `macosx_10_13_x86_64`), `macosx_11_0_arm64`"; on
3.13/3.14 "we only found wheels for `submodlib-py` (v0.0.3) with the following Python ABI tags: `cp38`,
`cp39`, `cp310`, `cp311`, `cp312`". Cross-checked with pip itself (a `python -m venv` on 3.14.2,
`python -m pip install submodlib-py`, via `.github/scripts/install_cell.sh` run locally):
`ERROR: Could not find a version that satisfies the requirement submodlib-py (from versions: none)`.

What PyPI serves for `submodlib-py` 0.0.3 (`curl https://pypi.org/pypi/submodlib-py/json`, 2026-08-22):
20 files, all `bdist_wheel`, for `cp38`/`cp39`/`cp310`/`cp311`/`cp312` × `manylinux_2_17_x86_64`,
`musllinux_1_2_x86_64`, `macosx_10_9_x86_64` (or `10_13`), `macosx_11_0_arm64`. No `win_*` wheel, no
`cp313`/`cp314` wheel, and no sdist, so there is nothing pip could build from source. `requires_python`
is `>=3.8`; the metadata declares `numba>=0.43.0`, `scipy`, `scikit-learn`, `matplotlib`, `tqdm`, `pandas`,
`joblib` as dependencies.

`gist-sampling` is not on PyPI: `curl https://pypi.org/pypi/gist-sampling/json` returned HTTP 404 on
2026-08-22; its own README says "This project isn't published on PyPI. Install from a local checkout."
The git install works on all four interpreters because numba 0.67.0 ships cp311-cp314 Windows wheels.

Extra cell, outside the matrix: `uv venv --python 3.14` on this machine resolved to the **free-threaded**
CPython 3.14.6 (`cp314t`). There, `gist-select` and `gist-sampling` install; `submodlib-py` fails with
"no wheels with a free-threading compatible ABI tag"; and the divsel abi3 wheel is refused --
`A path (wheels\divsel-0.0.1-cp311-abi3-win_amd64.whl) dependency is incompatible with the current
platform … the wheel was built for the stable ABI (abi3), which requires a GIL-enabled interpreter`.
That is expected (`python/divsel/_divsel.pyi` says free-threaded CPython needs a separate `cp314t` build),
and it is why the 3.14 column above was re-created from the GIL interpreter explicitly.

### Linux and macOS: pending CI run

`.github/workflows/install-matrix.yml` runs the same four installs for `ubuntu-latest`, `windows-latest`
and `macos-latest` × Python 3.11/3.12/3.13/3.14 with `pip` in a fresh `python -m venv` per library
(`.github/scripts/install_cell.sh`), each as a `continue-on-error` step that writes an ok/fail JSON record
with pip's output tail; the `assemble` job merges them into `docs/benchmarks/install-matrix.json` and prints
the table into the job summary (`.github/scripts/assemble_matrix.py`). The divsel cell is `pip install .`
on the checkout, so it needs the Rust toolchain the workflow installs with `dtolnay/rust-toolchain@stable`.
The YAML was checked with `python -c "import yaml; yaml.safe_load(open('.github/workflows/install-matrix.yml'))"`
and the cell script + assembler were exercised locally on two real cells (see the pip cross-check above);
the workflow itself has not run, so those 32 cells are **not measured**.

## Comparison against the incumbents

### What is compared, and how each library is called

All calls are exactly what each package's installed source documents; the harness is `bench/compare.py`.

| method | call |
|---|---|
| `divsel` | `divsel.gist_select_full(X, w_or_None, k=k, lam=1.0, eps=0.1, metric="cosine", utility="linear" \| "facility_location", diameter="exact")` |
| `divsel[diameter=approx]` | the same call with `diameter="approx"` (farthest-point double sweeps, 3 of them, estimate in `[d_max/2, d_max]`) -- the documented option for large n and the one `--large` uses at n = 1M. |
| `gist-select` | `gist.gist(points=X, utility=gist.LinearUtility(w), distance=gist.CosineDistance(), k=k, lam=1.0, eps=0.1, n_jobs=1, seed=0)` -- the README's documented call (`src/gist/algorithm.py`). Its diameter is a seeded 5-start double scan (`approximate_diameter`), its sweep stops at the first threshold that yields `\|S\| <= 1`. |
| `gist-select[n_jobs=-1]` | the same call with `n_jobs=-1` (joblib threads over the threshold sweep -- the README's "parallel threshold sweep"; joblib is present in the venv as a scikit-learn dependency). |
| `gist-sampling` | `gist_sampling.GISTSelector(n_samples=k, utility="facility_location", metric="cosine", epsilon=0.1, lambda_diversity=1.0, random_state=0).fit(X)` -- documented defaults, so `mode="auto"`, which is `"exact"` for n <= 2000 and the approximate kNN/lazy-distance mode above that (`selectors/gist_selector.py`, `DEFAULT_APPROXIMATE_THRESHOLD = 2000`). `n_jobs=-1` by default. Its `g` is facility location over an RBF similarity `exp(-gamma d^2)` with a median-heuristic gamma. |
| `gist-sampling[mode=exact]` | the same selector with `mode="exact"`: a dense `scipy.spatial.distance.cdist` distance matrix and a dense similarity matrix, both `n x n` float64. |
| `mmr` | numpy, in `bench/compare.py`: greedy `argmax 0.5 * rel(v) - 0.5 * max_{s in S} sim(v, s)`; `rel` = the same Linear weights (uniform in [0, 1)), `sim` = cosine similarity of the normalised rows, `max sim = 0` for the empty selection. The same `rel` is used in the facility-location cells (MMR has no `g`). |

Not comparable, and reported as `unavailable` in the tables with the reason the harness prints:

* `gist-select` has no facility-location utility (`gist/objectives.py` exports `LinearUtility`,
  `CoverageFunction`, `SubmodularFunction`). Its documented custom-`SubmodularFunction` extension point
  was not used: a facility location written in this harness would time the harness's own numpy, not the
  library.
* `gist-sampling` has no linear utility (`UtilityType = Literal["facility_location"]`).
* `submodlib-py` is not in the comparison: it cannot be installed on this machine (above), and it
  implements no `g(S) + lambda * div(S)` objective to compare against.

### Inputs, evaluator, protocol

* **Inputs**, regenerated inside every worker from `np.random.default_rng(0)`: `X = rng.standard_normal((n, dim),
  dtype=np.float32)` L2-normalised row-wise; `w = rng.random(n)` float64. Every library receives the
  identical arrays (gist-sampling converts to float64 internally; gist-select re-normalises; divsel
  makes one normalised float32 copy for `metric="cosine"`).
* **Shared evaluator** (`evaluate` in `bench/compare.py`), the same function for every library's
  selection, float64 from the float32 rows: `f = g(S) + 1.0 * div(S)`; `dist(i, j) = clamp(1 - x_i . x_j, 0, 2)`;
  `div(S) = min_{u != v in S} dist(u, v)`, or the exact diameter `d_max` when `|S| <= 1`;
  Linear `g(S) = sum_{v in S} w[v]`; FacilityLocation `g(S) = sum_{i in V} max_{j in S} max(0, 1 - dist(i, j) / 1.0)`
  (divsel's definition from `crates/divsel/src/utility.rs`: `sim = max(0, 1 - dist/scale)`, `scale = 1.0`
  under cosine). Each library's own reported objective is shown next to it as "library f"; for divsel the
  two are compared as a self-check (`abs(f_eval - f_divsel)`).
* **Wall-clock**: every (cell, method) runs in a fresh subprocess; one warm-up call (absorbing numba JIT,
  BLAS and rayon thread start-up), then 3 timed calls with `time.perf_counter`; the median is reported with
  min-max. A warm-up longer than 600 s marks the cell `timeout`.
* **Peak RSS**: `psutil.Process().memory_info().peak_wset` at the end of the worker (Windows;
  `resource.getrusage(...).ru_maxrss` elsewhere). It is process-wide, so it includes the interpreter,
  numpy, the generated data and the library's imports (importing gist-sampling alone pulls pandas,
  scikit-learn and numba); the value taken after data generation and before the first call is shown as
  "baseline" next to it.
* **Threads**: divsel's sweep and diameter scan use rayon's default pool (16 threads here);
  `gist-sampling` runs with its default `n_jobs=-1`; `gist-select` with `n_jobs=1` as documented and again
  with `n_jobs=-1`; numpy's BLAS is left at its default thread count for everyone.
* All numbers below are from these commands, run one after another on an otherwise idle machine (the two `divsel[diameter=approx]` runs were added after the first three finished):

```
.venv-3.14/Scripts/python.exe bench/compare.py --n 2000 --dim 64 --k 5                                   # smoke
.venv-3.14/Scripts/python.exe bench/compare.py --n 10000 --dim 384 --k 10,50,100 --out docs/benchmarks/results-2026-08-22.json
.venv-3.14/Scripts/python.exe bench/compare.py --n 100000 --dim 384 --k 10 --utility linear --out docs/benchmarks/results-2026-08-22.json
.venv-3.14/Scripts/python.exe bench/compare.py --n 10000 --dim 384 --k 10,50,100 --utility linear --methods "divsel[diameter=approx]" --out docs/benchmarks/results-2026-08-22.json
.venv-3.14/Scripts/python.exe bench/compare.py --n 100000 --dim 384 --k 10 --utility linear --methods "divsel[diameter=approx]" --out docs/benchmarks/results-2026-08-22.json
```

### n = 10 000, dim = 384, Linear utility

| n | dim | k | utility | method | wall median s (min-max) | peak RSS MiB (baseline) | f(S) | g(S) | div(S) | \|S\| | library f | note |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 10000 | 384 | 10 | linear | divsel | 0.527 (0.518-0.538) | 72 (65) | 10.9196 | 9.9927 | 0.9270 | 10 | 10.9196 | self-check abs(f_eval - f_divsel) = 1.23e-08; stage=sweep, diameter=exact |
| 10000 | 384 | 10 | linear | divsel[diameter=approx] | 0.065 (0.061-0.079) | 72 (65) | 10.9549 | 9.9824 | 0.9725 | 10 | 10.9549 | self-check abs(f_eval - f_divsel) = 1.91e-08; stage=sweep, diameter=approx |
| 10000 | 384 | 10 | linear | gist-select | 1.677 (1.668-1.700) | 86 (65) | 10.9516 | 9.9755 | 0.9761 | 10 | 10.9516 |  |
| 10000 | 384 | 10 | linear | gist-select[n_jobs=-1] | 0.951 (0.949-0.974) | 334 (65) | 10.9516 | 9.9755 | 0.9761 | 10 | 10.9516 |  |
| 10000 | 384 | 10 | linear | gist-sampling | unavailable | 113 (65) | - | - | - | - | - | gist-sampling supports utility="facility_location" only (selectors/gist_selector.py: UtilityType = Literal["facility_location"]) |
| 10000 | 384 | 10 | linear | gist-sampling[mode=exact] | unavailable | 113 (65) | - | - | - | - | - | gist-sampling supports utility="facility_location" only (selectors/gist_selector.py: UtilityType = Literal["facility_location"]) |
| 10000 | 384 | 10 | linear | mmr | 0.006 (0.003-0.006) | 65 (65) | 10.9505 | 9.9710 | 0.9796 | 10 | - |  |
| 10000 | 384 | 50 | linear | divsel | 0.729 (0.729-0.738) | 72 (65) | 50.7116 | 49.8760 | 0.8355 | 50 | 50.7116 | self-check abs(f_eval - f_divsel) = 2.09e-08; stage=sweep, diameter=exact |
| 10000 | 384 | 50 | linear | divsel[diameter=approx] | 0.297 (0.292-0.314) | 72 (65) | 50.7390 | 49.8654 | 0.8736 | 50 | 50.7390 | self-check abs(f_eval - f_divsel) = 2.18e-08; stage=sweep, diameter=approx |
| 10000 | 384 | 50 | linear | gist-select | 7.695 (7.371-8.021) | 86 (65) | 50.7362 | 49.8500 | 0.8862 | 50 | 50.7362 |  |
| 10000 | 384 | 50 | linear | gist-select[n_jobs=-1] | 4.020 (3.854-4.173) | 346 (65) | 50.7362 | 49.8500 | 0.8862 | 50 | 50.7362 |  |
| 10000 | 384 | 50 | linear | gist-sampling | unavailable | 113 (65) | - | - | - | - | - | gist-sampling supports utility="facility_location" only (selectors/gist_selector.py: UtilityType = Literal["facility_location"]) |
| 10000 | 384 | 50 | linear | gist-sampling[mode=exact] | unavailable | 113 (65) | - | - | - | - | - | gist-sampling supports utility="facility_location" only (selectors/gist_selector.py: UtilityType = Literal["facility_location"]) |
| 10000 | 384 | 50 | linear | mmr | 0.014 (0.013-0.017) | 65 (65) | 50.5043 | 49.5898 | 0.9144 | 50 | - |  |
| 10000 | 384 | 100 | linear | divsel | 0.944 (0.942-0.984) | 72 (65) | 100.3434 | 99.5065 | 0.8370 | 100 | 100.3434 | self-check abs(f_eval - f_divsel) = 4.09e-08; stage=sweep, diameter=exact |
| 10000 | 384 | 100 | linear | divsel[diameter=approx] | 0.653 (0.652-0.716) | 72 (65) | 100.3345 | 99.4676 | 0.8669 | 100 | 100.3345 | self-check abs(f_eval - f_divsel) = 1.06e-09; stage=sweep, diameter=approx |
| 10000 | 384 | 100 | linear | gist-select | 15.978 (14.640-18.390) | 86 (65) | 100.3384 | 99.5192 | 0.8192 | 100 | 100.3384 |  |
| 10000 | 384 | 100 | linear | gist-select[n_jobs=-1] | 7.706 (7.692-8.074) | 343 (65) | 100.3384 | 99.5192 | 0.8192 | 100 | 100.3384 |  |
| 10000 | 384 | 100 | linear | gist-sampling | unavailable | 113 (65) | - | - | - | - | - | gist-sampling supports utility="facility_location" only (selectors/gist_selector.py: UtilityType = Literal["facility_location"]) |
| 10000 | 384 | 100 | linear | gist-sampling[mode=exact] | unavailable | 113 (65) | - | - | - | - | - | gist-sampling supports utility="facility_location" only (selectors/gist_selector.py: UtilityType = Literal["facility_location"]) |
| 10000 | 384 | 100 | linear | mmr | 0.038 (0.036-0.039) | 65 (65) | 99.9665 | 99.0810 | 0.8855 | 100 | - |  |

### n = 10 000, dim = 384, FacilityLocation utility

| n | dim | k | utility | method | wall median s (min-max) | peak RSS MiB (baseline) | f(S) | g(S) | div(S) | \|S\| | library f | note |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 10000 | 384 | 10 | facility_location | divsel | 383.732 (381.109-390.204) | 97 (65) | 837.1464 | 836.2170 | 0.9294 | 10 | 837.1464 | self-check abs(f_eval - f_divsel) = 5.75e-05; stage=sweep, diameter=exact |
| 10000 | 384 | 10 | facility_location | gist-select | unavailable | 65 (65) | - | - | - | - | - | gist-select ships no facility-location utility (gist/objectives.py exports LinearUtility, CoverageFunction, SubmodularFunction only) |
| 10000 | 384 | 10 | facility_location | gist-select[n_jobs=-1] | unavailable | 65 (65) | - | - | - | - | - | gist-select ships no facility-location utility (gist/objectives.py exports LinearUtility, CoverageFunction, SubmodularFunction only) |
| 10000 | 384 | 10 | facility_location | gist-sampling | 3.277 (3.259-3.307) | 2177 (65) | 815.8093 | 814.9006 | 0.9087 | 10 | 617.8829 | mode_used=approximate |
| 10000 | 384 | 10 | facility_location | gist-sampling[mode=exact] | 96.680 (95.056-99.076) | 3959 (65) | 830.2765 | 829.3671 | 0.9094 | 10 | 4314.4685 | mode_used=exact |
| 10000 | 384 | 10 | facility_location | mmr | 0.005 (0.005-0.005) | 95 (65) | 805.1585 | 804.1789 | 0.9796 | 10 | - |  |
| 10000 | 384 | 50 | facility_location | divsel | timeout | 100 (65) | 1239.6868 | 1238.8541 | 0.8328 | 50 | 1239.6869 | warm-up call took 791.0 s > 600 s; self-check abs(f_eval - f_divsel) = 7.65e-05; stage=sweep, diameter=exact |
| 10000 | 384 | 50 | facility_location | gist-select | unavailable | 65 (65) | - | - | - | - | - | gist-select ships no facility-location utility (gist/objectives.py exports LinearUtility, CoverageFunction, SubmodularFunction only) |
| 10000 | 384 | 50 | facility_location | gist-select[n_jobs=-1] | unavailable | 65 (65) | - | - | - | - | - | gist-select ships no facility-location utility (gist/objectives.py exports LinearUtility, CoverageFunction, SubmodularFunction only) |
| 10000 | 384 | 50 | facility_location | gist-sampling | 8.159 (8.152-8.730) | 2143 (65) | 1235.8592 | 1235.0026 | 0.8566 | 50 | 2448.3867 | mode_used=approximate |
| 10000 | 384 | 50 | facility_location | gist-sampling[mode=exact] | 162.891 (159.785-163.179) | 3959 (65) | 1236.7193 | 1235.8891 | 0.8303 | 50 | 4632.4141 | mode_used=exact |
| 10000 | 384 | 50 | facility_location | mmr | 0.012 (0.012-0.013) | 99 (65) | 1193.7784 | 1192.8640 | 0.9144 | 50 | - |  |
| 10000 | 384 | 100 | facility_location | divsel | timeout | 105 (65) | 1422.4231 | 1421.5815 | 0.8416 | 100 | 1422.4231 | warm-up call took 959.7 s > 600 s; self-check abs(f_eval - f_divsel) = 7.88e-05; stage=sweep, diameter=exact |
| 10000 | 384 | 100 | facility_location | gist-select | unavailable | 65 (65) | - | - | - | - | - | gist-select ships no facility-location utility (gist/objectives.py exports LinearUtility, CoverageFunction, SubmodularFunction only) |
| 10000 | 384 | 100 | facility_location | gist-select[n_jobs=-1] | unavailable | 65 (65) | - | - | - | - | - | gist-select ships no facility-location utility (gist/objectives.py exports LinearUtility, CoverageFunction, SubmodularFunction only) |
| 10000 | 384 | 100 | facility_location | gist-sampling | 14.215 (13.586-14.294) | 2147 (65) | 1411.0389 | 1410.2270 | 0.8119 | 100 | 3772.6245 | mode_used=approximate |
| 10000 | 384 | 100 | facility_location | gist-sampling[mode=exact] | 193.172 (188.284-199.769) | 3958 (65) | 1420.8014 | 1419.9781 | 0.8233 | 100 | 4773.2076 | mode_used=exact |
| 10000 | 384 | 100 | facility_location | mmr | 0.025 (0.025-0.025) | 104 (65) | 1366.7582 | 1365.8727 | 0.8855 | 100 | - |  |

### n = 100 000, dim = 384, k = 10, Linear utility

| n | dim | k | utility | method | wall median s (min-max) | peak RSS MiB (baseline) | f(S) | g(S) | div(S) | \|S\| | library f | note |
|---:|---:|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| 100000 | 384 | 10 | linear | divsel | 209.331 (206.243-212.915) | 355 (330) | 10.9423 | 9.9988 | 0.9434 | 10 | 10.9423 | self-check abs(f_eval - f_divsel) = 1.02e-09; stage=sweep, diameter=exact |
| 100000 | 384 | 10 | linear | divsel[diameter=approx] | 1.552 (1.487-1.615) | 354 (330) | 10.9706 | 9.9983 | 0.9723 | 10 | 10.9706 | self-check abs(f_eval - f_divsel) = 3.07e-08; stage=sweep, diameter=approx |
| 100000 | 384 | 10 | linear | gist-select | 17.451 (16.993-17.919) | 503 (330) | 10.9758 | 9.9973 | 0.9784 | 10 | 10.9758 |  |
| 100000 | 384 | 10 | linear | gist-select[n_jobs=-1] | 10.623 (10.263-11.039) | 2801 (330) | 10.9758 | 9.9973 | 0.9784 | 10 | 10.9758 |  |
| 100000 | 384 | 10 | linear | gist-sampling | unavailable | 330 (330) | - | - | - | - | - | gist-sampling supports utility="facility_location" only (selectors/gist_selector.py: UtilityType = Literal["facility_location"]) |
| 100000 | 384 | 10 | linear | gist-sampling[mode=exact] | unavailable | 330 (330) | - | - | - | - | - | gist-sampling supports utility="facility_location" only (selectors/gist_selector.py: UtilityType = Literal["facility_location"]) |
| 100000 | 384 | 10 | linear | mmr | 0.066 (0.065-0.067) | 329 (329) | 10.9766 | 9.9794 | 0.9972 | 10 | - |  |

### Reading the tables

* **Achieved `f(S)`.** In every Linear cell the methods that ran land within 0.5 % of each other.
  The measured max-min spreads, from `results-2026-08-22.json`: n = 10k k = 10 **0.322 %** (10.9196
  divsel ... 10.9549 divsel approx), k = 50 **0.463 %** (50.5043 MMR ... 50.7390 divsel approx),
  k = 100 **0.376 %** (99.9665 MMR ... 100.3434 divsel), and n = 100k k = 10 **0.313 %** (10.9423
  divsel ... 10.9766 MMR). The three GIST implementations apply the same threshold formula
  `(1+eps)^i * eps * d_max / 2` to different diameters -- divsel's exact `d_max = 1.2749`, the approximate
  estimates of `divsel[diameter=approx]` and of gist-select's 5-start double scan -- so their threshold grids,
  selections and `div` values differ; at `eps = 0.1` the grid step is 10 %, and the spread between them is
  inside that. On these inputs (isotropic Gaussian rows in 384 dimensions, so almost every pair is at a
  cosine distance near 1) the diversity term separates the methods very little.
* **FacilityLocation.** divsel's selection scores highest under the shared evaluator in all three cells
  (837.15 / 1239.69 / 1422.42 against gist-sampling exact 830.28 / 1236.72 / 1420.80, gist-sampling
  approximate 815.81 / 1235.86 / 1411.04, MMR 805.16 / 1193.78 / 1366.76), and it is by far the slowest:
  383.7 s per call at k = 10, and single calls of 791.0 s (k = 50) and 959.7 s (k = 100) that exceeded the
  600 s per-call cap, so those two rows are `timeout` and show that one call's time and selection rather than
  a median of three. gist-sampling's exact mode took 96.7 / 162.9 / 193.2 s and its approximate mode 3.3 /
  8.2 / 14.2 s. divsel's facility-location marginal is an `O(n * dim)` scan per candidate with no kNN
  sparsification, which makes the first greedy round of each of the 32 threshold runs (plus the `d = 0`
  greedy and the replay of the winning threshold) `O(n^2 * dim)`.
* **Peak RSS.** divsel 72 MiB (Linear) and 97-105 MiB (FacilityLocation) at n = 10k, 354-355 MiB at
  n = 100k (330 MiB of which is the worker's baseline: interpreter, numpy and the 153 MB input);
  gist-select 86 MiB at 10k and 503 MiB at 100k with `n_jobs=1`, 334-346 MiB and 2801 MiB with
  `n_jobs=-1`; gist-sampling 2.1 GiB in approximate mode and 3.9 GiB in exact mode at n = 10k (two dense
  10 000 x 10 000 float64 matrices); MMR 65-104 MiB.
* **The exact diameter at n = 100k.** divsel's pinned `diameter="exact"` call takes 209.3 s per call at
  n = 100k; the same call with `diameter="approx"` takes 1.552 s (f 10.9706 against 10.9423). At n = 10k,
  k = 10 the exact call is 0.527 s and the approximate one 0.065 s. The 209.3 s figure is a full cosine
  `gist_select_full` call from Python (diameter scan + threshold sweep). The closest header estimate in
  `crates/divsel/benches/gist.rs` is its whole-cell "UNMEASURED, est. 15-90 s" for
  `gist/linear/n=100000`, exact diameter -- but that cell is Euclidean and Rust-only (its 6-25 s
  arithmetic, from per-pair kernel cost over 2016 cache-resident pairs "at perfect scaling", covers the
  diameter scan alone, "and the sweep adds more on top"). Against the 15-90 s whole-cell estimate the
  measured 209.3 s is roughly 2.3x-14x, with metric and call path also differing. The header estimate
  (Rust, out of scope here) should still be re-derived from measurement in a later task, at this cell's
  actual metric and shape: the 100k x 384 float32 set is 153 MB and the scan streams it once per row.
* **Self-check (divsel's own `f_value` against the evaluator).** Linear cells agree to at most 4.1e-8
  absolute. FacilityLocation cells differ by 5.8e-5 to 7.9e-5 absolute -- 4e-8 to 7e-8 relative on
  f = 837-1422 -- because divsel derives each similarity from a float32 distance (its SIMD kernel) and the
  evaluator works in float64, and the per-point rounding is summed over n = 10 000 terms of `g`. The brief's
  1e-6 agreement holds in relative terms everywhere and in absolute terms for Linear; it does not hold in
  absolute terms for FacilityLocation at this n, and cannot without a bit-identical float32 evaluator.
* **Determinism.** In every `ok` cell the three timed calls of every method returned the identical selection
  (`selection_stable` in the JSON).
* **"library f".** gist-sampling's own objective (617.88 at k = 10, 4314.47 in exact mode) is its RBF
  facility location and is not comparable across libraries; the `f(S)` column is the shared evaluator for
  everyone. gist-select's and divsel's own objectives coincide with the evaluator's because they use the
  same definitions.
* **Threads.** divsel's 16-thread sweep and gist-sampling's `n_jobs=-1` are compared against gist-select
  both single-threaded (as its README benchmarks it) and with `n_jobs=-1`; numpy's BLAS (scipy-openblas
  0.3.34) is at its default thread count in every process, including inside `n_jobs=1` gist-select.

### Cells of the brief's matrix that were not run here

The brief's matrix is n in {10k, 100k, 1M} × dim in {384, 768} × k in {10, 50, 100} × {Linear,
FacilityLocation}, with the ruling that FacilityLocation runs only at n = 10k and n = 1M only for divsel
Linear with `diameter="approx"`. Measured above: n = 10k × dim 384 × all k × both utilities, and
n = 100k × dim 384 × k = 10 Linear. Everything else is unmeasured; the commands that produce it:

```
.venv-3.14/Scripts/python.exe bench/compare.py --all --out docs/benchmarks/results-<date>.json            # dim 768, k 50/100 at 100k, ...
.venv-3.14/Scripts/python.exe bench/compare.py --all --large --out docs/benchmarks/results-<date>.json    # + n = 1M, divsel Linear, diameter="approx"
```

`--large` passes `diameter="approx"` to divsel (a farthest-point double sweep, estimate in `[d_max/2, d_max]`,
3 sweeps) because an exact diameter at n = 1M is 5e11 pairs; the n = 1M rows of the other methods are
reported as `not run`. The n = 1M fixture is 1.5 GiB (dim 384) / 3 GiB (dim 768) of float32 per worker.

## Kernel and end-to-end numbers from the Rust benches (criterion)

From `task-6-report.md` § "Fix round 1", commit `7ece44e`, `cargo bench -p divsel` (the DEFAULT tier,
253 s end to end), 2026-08-21, same machine and toolchain as above. The header of
`crates/divsel/benches/gist.rs` carries the same numbers and the tier table.

What "scalar" means there: the baseline arm is the same 16-accumulator loop compiled at the default
`x86-64` target, which LLVM auto-vectorises to SSE2 (4 `mulps` + 4 `addps` in the group loop, no FMA, no
reassociation). The ratio is therefore **runtime-dispatched AVX2 (8 lanes) against an auto-vectorised SSE2
baseline (4 lanes)**, not against one-element-at-a-time code, and the two are bit-identical by test.

`kernel/sq_euclid`, 2016 unordered pairs per iteration, 100 samples:

| dim | dispatched | scalar (SSE2 baseline) | ratio |
|---:|---:|---:|---:|
| 64  | 39.714 us | 79.610 us | 2.01x |
| 384 | 103.64 us | 337.60 us | 3.26x |
| 768 | 158.87 us | 656.43 us | 4.13x |

`kernel/dot`:

| dim | dispatched | scalar (SSE2 baseline) | ratio |
|---:|---:|---:|---:|
| 64  | 40.423 us | 80.444 us | 1.99x |
| 384 | 91.457 us | 368.10 us | 4.02x |
| 768 | 147.09 us | 731.58 us | 4.97x |

`gist/linear/n=10000`, Euclidean, `eps = 0.1`, exact diameter, 10 samples (Rust core, no Python):

| dim | k = 10 | k = 100 |
|---:|---:|---:|
| 64  | 195.47 ms | 368.40 ms |
| 384 | 626.64 ms | 1.1842 s  |
| 768 | 2.0474 s  | 3.2038 s  |

The `bench-large` tier (`facility_location` at n = 10k, `linear` at n = 100k with the exact diameter, `linear`
at n = 1M with `Approx { sweeps: 3 }`) is unmeasured apart from two `--quick` upper bounds; the bench header
lists each cell's estimate and the exact `cargo bench -p divsel --features bench-large -- --exact <cell>`
command. A full run of `gist/facility_location/n=10000/dim=768/k=10` was attempted and interrupted after
2905 s of criterion's projected 3449 s (345 s per iteration); no sample was recorded.

## Incumbents' own claims: what was run and what happened

### gist-select README, "Performance"

The README shipped in the 0.1.0 wheel (and at `f6281f3`) says: "Benchmarked on Apple M-series,
single-threaded, `eps=0.1`" and lists

| Points | Dimensions | k | Time |
|---|---|---|---|
| 10K | 64 | 50 | 0.3s |
| 100K | 128 | 100 | 6s |
| 500K | 128 | 100 | ~30s |
| 2M | 128 | 100 | ~2 min |

The repository (`git clone --depth 1 https://github.com/kclaka/gist-select`, HEAD `f6281f3`, 13 tracked
files) contains no script that produces this table. The only scale test is `tests/test_scale.py`:
1.5M points, 64-d, k = 100, `EuclideanDistance`, `n_jobs=4`, `pytest.mark.slow`, which prints its duration and
asserts only that a non-empty selection with a positive objective is returned. The README's inputs for the
table are not stated; `bench/gist_select_readme_table.py` uses the README's Quick Start recipe
(`rng = default_rng(42)`, `standard_normal((n, d)).astype(np.float32)`, `weights = rng.random(n)`,
`LinearUtility`, `EuclideanDistance()`, `lam=1.0`, `seed=42`) with the table's `eps=0.1` and `n_jobs=1`, one
call per row, timed with `perf_counter`, on the machine above:

```
.venv-3.14/Scripts/python.exe bench/gist_select_readme_table.py --rows 1,2,3
timeout 1200 .venv-3.14/Scripts/python.exe bench/gist_select_readme_table.py --rows 4
```

| row | n | d | k | README | measured here (`n_jobs=1`, `eps=0.1`, float32, seed 42) | \|S\| | objective | diversity |
|---:|---:|---:|---:|---|---|---:|---:|---:|
| 1 | 10,000 | 64 | 50 | 0.3s | 1.4 s (data generation 0.0 s, not counted) | 50 | 59.9654 | 10.2633 |
| 2 | 100,000 | 128 | 100 | 6s | 47.6 s (data generation 0.2 s, not counted) | 100 | 114.4452 | 14.5697 |
| 3 | 500,000 | 128 | 100 | ~30s | 251.9 s (data generation 0.8 s, not counted) | 100 | 115.0809 | 15.1313 |
| 4 | 2,000,000 | 128 | 100 | ~2 min | 1015.6 s (data generation 3.0 s, not counted) | 100 | 116.1992 | 16.5513 |

Row 4's process exit status: `exit=0` (124 means the 1200 s `timeout` killed it).

This machine is not an Apple M-series and the README's inputs are unknown, so these are the README's rows
re-run with the README's own example data on different hardware, nothing more.

### Other facts checked while installing (recorded because the plan's prior-art table asserts them)

* `kclaka/gist-select` at `f6281f3` has **no `LICENSE` file** in git (`git ls-files` lists 13 files, none of
  them a license) and no `license` key in its `pyproject.toml`; the PyPI wheel, however, **does** ship
  `gist_select-0.1.0.dist-info/licenses/LICENSE` (MIT text) and declares `License-Expression: MIT`. The
  released artifact is licensed; the committed tree is not.
* `gist-select` on PyPI has exactly one release, `0.1.0` (wheel + sdist, 2026-02-19); `requires_python >= 3.10`;
  dependencies `numpy>=1.24`, `scipy>=1.10` (`joblib` only under the `parallel` extra).
* `gist-sampling`'s classifiers stop at Python 3.12, but it installs and runs on 3.13.11 and 3.14.2 here.

## Reproducing

```
python -m maturin build --release -o wheels                          # system Python; one abi3 wheel
uv venv --python C:/Python314/python.exe .venv-3.14
uv pip install --python .venv-3.14/Scripts/python.exe wheels/divsel-0.0.1-cp311-abi3-win_amd64.whl numpy psutil gist-select git+https://github.com/musubi-labs/gist-sampling
.venv-3.14/Scripts/python.exe bench/compare.py --n 2000 --dim 64 --k 5
.venv-3.14/Scripts/python.exe bench/compare.py --n 10000 --dim 384 --k 10,50,100 --out docs/benchmarks/results-<date>.json
```

`docs/benchmarks/results-2026-08-22.json` holds every cell of the tables above with the per-run timings,
the selections (truncated to 100 indices), each library's own reported values and the environment record.
