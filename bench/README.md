# bench/

`compare.py` runs divsel, PyPI `gist-select`, git `gist-sampling` and a naive numpy MMR on identical
seeded inputs, times each in its own subprocess (one warm-up, then the median of `--runs` calls), records
peak RSS, and scores every library's selection with one shared evaluator using divsel's definition of
`f(S) = g(S) + λ·div(S)`. `python bench/compare.py --n 2000 --dim 64 --k 5` is the smoke test;
`--n 10000 --dim 384 --k 10,50,100 --out docs/benchmarks/results-<date>.json` is a real run (the JSON is
merged cell by cell, so several invocations can fill one file); `--all` is the small/medium matrix and
`--all --large` adds the n = 1M divsel-only cells. It needs a venv with `numpy`, `psutil`, the divsel wheel
and whichever incumbents you want compared — a missing incumbent is reported as `unavailable`, not a crash.
`gist_select_readme_table.py` re-runs the four rows of gist-select's README performance table as the README
states them. Results, method and the installability matrix are written up in
[`docs/benchmarks/README.md`](../docs/benchmarks/README.md); the cross-OS matrix itself is produced by
`.github/workflows/install-matrix.yml`.
