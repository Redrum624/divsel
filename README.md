# divsel

**Diverse subset selection that actually has a guarantee.** A native (Rust) implementation of **GIST** — max-min diversification with submodular utility — with zero-copy Python bindings.

> Status: **0.0.1 scaffold — core implementation in progress**. Implementation plan:
> [`docs/superpowers/plans/2026-08-21-divsel-v0.1.md`](docs/superpowers/plans/2026-08-21-divsel-v0.1.md)

## The problem

You retrieved 50 candidates and need the best 5 to put in a prompt, a training batch, or a recommendation slate. Take the top 5 by score and you get five near-duplicates. The standard fix is **MMR** (Maximal Marginal Relevance, 1998) — a greedy heuristic with **no approximation guarantee** and an uninterpretable λ.

`divsel` solves the same shape of problem with a proof attached:

```
maximize   f(S) = g(S) + λ · min-pairwise-distance(S)     subject to |S| ≤ k
```

where `g` is any monotone submodular utility (relevance, coverage, facility location). GIST achieves **(1/2 − ε)·OPT** for submodular `g` and **(2/3 − ε)·OPT** for linear `g` — the latter provably tight, since no polynomial-time `(2/3 + ε)` algorithm exists unless P = NP. The problem is NP-hard to approximate beyond ≈0.5584.

## Why this exists

**GIST** — *Greedy Independent Set Thresholding for Max-Min Diversification with Submodular Utility*, Fahrbach, Ramalingam, Zadimoghaddam, Ahmadian, Citovsky & DeSalvo (Google Research), **NeurIPS 2025**, [arXiv:2405.18754](https://arxiv.org/abs/2405.18754) — is a strong result with no production-grade implementation. As of August 2026:

- crates.io returns **one** result for `submodular`, and it is unrelated. No maintained Rust crate does submodular maximization, facility location, or GIST.
- The two Python implementations are a 1-commit release with **no LICENSE file** and a 6-commit repo that was never published to PyPI.
- `submodlib` (131★) implements no combined `g(S) + λ·div(S)` objective at all, has been unmaintained since April 2025, and ships **no Windows wheels and no sdist** — you cannot install it on Windows, or on Python 3.13/3.14, at any price.

So `divsel` aims at three things nobody currently offers together: **a real license, wheels that install everywhere, and benchmarks you can reproduce from the repo.**

## Design commitments

- **The paper is the spec.** Every constant transcribed from arXiv:2405.18754v3. Where the paper is silent (argmax tie-breaking), `divsel` documents its choice as a choice rather than inventing a citation.
- **Proven, not asserted.** A brute-force oracle enumerates `OPT` on small instances and asserts the approximation ratio actually holds — the test no incumbent ships.
- **Installs everywhere.** abi3 wheels for Python 3.11 → 3.14, Linux · macOS · **Windows**, x86_64 and aarch64.
- **Reference implementation.** Exports `golden-selection.json`; the ports in [Aura](https://github.com/Redrum624/Aura) (Python) and `limbic` (TypeScript) conform to it.
- **Apache-2.0.**

## Name

`divsel` = *diverse selection*. Verified free on both crates.io and PyPI, 2026-08-21.
