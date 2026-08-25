"""divsel: GIST max-min diversification with submodular utility (arXiv:2405.18754, NeurIPS 2025).

Given a candidate set and a monotone submodular relevance/coverage utility ``g``, GIST
selects a subset ``S`` of size at most ``k`` maximizing ``g(S) + lambda * min-pairwise-distance(S)``
with a provable approximation guarantee, as a drop-in replacement for heuristics like MMR
(Maximal Marginal Relevance). This package wraps a native Rust core with numpy bindings
that are zero-copy for euclidean input (cosine makes exactly one L2-normalised copy):
:func:`gist_select` returns the selected row indices, :func:`gist_select_full` the full
result dict (objective value, diversity, winning threshold and stage).
"""

try:
    from ._divsel import __version__, gist_select, gist_select_full
except ImportError as exc:
    raise ImportError(
        "divsel's compiled extension module (_divsel) is not built. "
        "If you are working from a source checkout, run `maturin develop` "
        "(or `python -m maturin develop`) to build and install it into your environment."
    ) from exc

__all__ = ["__version__", "gist_select", "gist_select_full"]
