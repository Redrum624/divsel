"""Framework adapters: drop-in MMR replacements backed by :func:`divsel.gist_select`.

Importing this package never imports LangChain or LlamaIndex; each adapter
module imports its framework lazily and raises an ``ImportError`` pointing at
the matching extra (``pip install "divsel[langchain]"`` /
``pip install "divsel[llamaindex]"``) when it is missing. The adapter classes
are re-exported lazily here for convenience::

    from divsel.adapters import DivselRetriever          # needs langchain-core
    from divsel.adapters import DivselNodePostprocessor  # needs llama-index-core
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "MIN_EPS",
    "DivselFallbackWarning",
    "DivselNodePostprocessor",
    "DivselRetriever",
]

#: The smallest ``eps`` :func:`divsel.gist_select` accepts: ``2 ** -23``, which
#: is ``float(np.finfo(np.float32).eps)`` exactly. Both adapters constrain their
#: ``eps`` field to ``[MIN_EPS, 1.0]``, so the range a field advertises is the
#: range the library serves -- an adapter that accepted ``(0, 1]`` took
#: ``eps=1e-9`` at construction and then raised a Rust-worded ``ValueError``
#: from inside the first query, after a full fetch-and-embed round trip.
MIN_EPS = 2.0**-23


class DivselFallbackWarning(UserWarning):
    """An adapter could not obtain embeddings and fell back to plain top-k.

    Emitted (unless ``strict=True``, which raises ``ValueError`` instead) when
    a ``DivselRetriever``'s vector store exposes no embeddings, or a
    ``DivselNodePostprocessor`` gets nodes without embeddings and has no
    ``embed_model``. The fallback result is NOT diversified.
    """


def __getattr__(name: str) -> Any:
    # Lazy re-exports so `import divsel.adapters` stays framework-free.
    if name == "DivselRetriever":
        from divsel.adapters.langchain import DivselRetriever

        return DivselRetriever
    if name == "DivselNodePostprocessor":
        from divsel.adapters.llamaindex import DivselNodePostprocessor

        return DivselNodePostprocessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
