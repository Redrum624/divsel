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

__all__ = ["DivselFallbackWarning", "DivselNodePostprocessor", "DivselRetriever"]


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
