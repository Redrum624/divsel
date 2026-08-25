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
    # "Not built" is one cause of this ImportError, not the only one: a present
    # extension can fail to load on a missing C runtime, an ABI-mismatched numpy,
    # or any broken transitive DLL, and telling that user to run `maturin
    # develop` sends them to rebuild something they already have. So look, and
    # carry the original message either way.
    def _extension_files() -> list[str]:
        import os

        try:
            here = os.path.dirname(os.path.abspath(__file__))
            return sorted(
                name
                for name in os.listdir(here)
                if name.startswith("_divsel.")
                and name.endswith((".pyd", ".so", ".dylib"))
            )
        except OSError:  # pragma: no cover - unreadable package directory
            return []

    _found = _extension_files()
    if _found:
        raise ImportError(
            f"divsel found its compiled extension module ({', '.join(_found)}) but could "
            f"not load it: {exc}. The module itself is built, so this is a load failure "
            "of it or of something it depends on -- a missing C runtime, a numpy built "
            "for a different ABI, a mixed-up virtual environment. The message above names "
            "what failed; rebuilding with `maturin develop` only helps if it does not."
        ) from exc
    raise ImportError(
        "divsel's compiled extension module (_divsel) is not built: no _divsel.pyd/.so "
        f"next to this file, and importing it said: {exc}. "
        "If you are working from a source checkout, run `maturin develop` "
        "(or `python -m maturin develop`) to build and install it into your environment."
    ) from exc

__all__ = ["__version__", "gist_select", "gist_select_full"]
