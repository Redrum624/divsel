# divsel

**Diverse subset selection that actually has a guarantee.** A native (Rust) implementation of **GIST** — max-min diversification with submodular utility — with Python bindings that are zero-copy for euclidean input (cosine makes exactly one L2-normalised copy).

> Status: **0.1.0 — release candidate; publish pending.** Everything below is built and
> tested; the crates.io / PyPI publish steps are listed in [`docs/RELEASE.md`](docs/RELEASE.md).
> Changes: [`CHANGELOG.md`](CHANGELOG.md).

## Install

```
pip install divsel      # Python: one abi3 wheel covers CPython 3.11-3.14 per platform
cargo add divsel        # Rust: the core crate
```

Until 0.1.0 lands on PyPI/crates.io, install from a checkout instead: `pip install .`
(needs a Rust toolchain, 1.83+). Free-threaded CPython 3.14t is served by a separate
version-specific `cp314t` wheel, not the abi3 one. Adapter extras:
`pip install "divsel[langchain]"` / `"divsel[llamaindex]"`.

## Quick start

```python
import numpy as np
import divsel

vectors = np.random.default_rng(0).standard_normal((50, 8), dtype=np.float32)
picked = divsel.gist_select(vectors, k=5, lam=1.0)      # diverse-but-relevant row indices
full = divsel.gist_select_full(vectors, k=5, lam=1.0)   # + objective, diversity, threshold, stage
print(picked, full["objective"])
```

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
- The two Python implementations are a 1-commit release with **no LICENSE file in its repository** (the PyPI wheel does carry one) and a 6-commit repo that was never published to PyPI.
- `submodlib` (131★) implements no combined `g(S) + λ·div(S)` objective at all, has been unmaintained since April 2025, and ships **no Windows wheels and no sdist** — you cannot install it on Windows, or on Python 3.13/3.14, at any price.

So `divsel` aims at three things nobody currently offers together: **a real license, wheels that install everywhere, and benchmarks you can reproduce from the repo.** The measurements behind these claims — the installability matrix (Windows measured; the 32 Linux/macOS cells pending the CI run; the abi3 wheel does not cover free-threaded 3.14t), the comparison against `gist-select`, `gist-sampling` and MMR, and the incumbents' own README numbers re-run — are in [`docs/benchmarks/README.md`](docs/benchmarks/README.md), produced by `bench/compare.py`.

## Design commitments

- **The paper is the spec.** Every constant transcribed from arXiv:2405.18754v3. Where the paper is silent (argmax tie-breaking), `divsel` documents its choice as a choice rather than inventing a citation.
- **Proven, not asserted.** A brute-force oracle enumerates `OPT` on small instances and asserts the approximation ratio actually holds — the test no incumbent ships.
- **Installs everywhere.** abi3 wheels for Python 3.11 → 3.14, Linux · macOS · **Windows**, x86_64 and aarch64.
- **Reference implementation.** Exports `golden-selection.json`; the ports in [Aura](https://github.com/Redrum624/Aura) (Python) and `limbic` (TypeScript) conform to it.
- **Apache-2.0.**

## Drop-in for MMR

Both adapters live behind optional extras; plain `import divsel` never imports either framework.

### LangChain — `pip install "divsel[langchain]"`

```python
from divsel.adapters.langchain import DivselRetriever

retriever = DivselRetriever(vectorstore=vs, k=5, fetch_k=20, lam=1.0)
docs = retriever.invoke("your query")   # replaces vs.as_retriever(search_type="mmr")
```

Like MMR it fetches `fetch_k` candidates by query similarity, then returns the `k` of them maximizing `g(S) + λ·min-distance(S)` — with the guarantee instead of the greedy heuristic. Honest caveats: candidate texts are **re-embedded** through `vectorstore.embeddings` (stores do not expose their stored vectors uniformly), and when the store exposes no embeddings at all the retriever emits `DivselFallbackWarning` and returns plain, undiversified top-k (`strict=True` raises instead).

### LlamaIndex — `pip install "divsel[llamaindex]"`

```python
from divsel.adapters.llamaindex import DivselNodePostprocessor

engine = index.as_query_engine(
    similarity_top_k=20,   # this is the fetch_k — the candidate pool
    node_postprocessors=[DivselNodePostprocessor(k=5, lam=1.0)],
)
```

There is no `fetch_k` parameter here: the retriever's `similarity_top_k` already fixes the candidate pool, and the postprocessor diversifies it down to `k`. Vectors come from `node.embedding` when every node carries one, else from an optional `embed_model`; with neither, it warns (`DivselFallbackWarning`) and returns top-k by score (`strict=True` raises).

## Name

`divsel` = *diverse selection*. Verified free on both crates.io and PyPI, 2026-08-21.
