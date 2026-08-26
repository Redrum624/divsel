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

import numpy as np

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
    """An adapter could not obtain usable embeddings and fell back to plain top-k.

    Emitted (unless ``strict=True``, which raises ``ValueError`` instead)
    whenever an adapter cannot get one usable vector per candidate. Every
    reason, so that this list is not shorter than the code's:

    * no embeddings to ask -- a ``DivselRetriever``'s ``vectorstore.embeddings``
      is ``None`` or itself raises ``NotImplementedError``, or a
      ``DivselNodePostprocessor`` gets nodes without embeddings and has no
      ``embed_model``;
    * an embedding hook raises ``NotImplementedError`` (``embed_query``,
      ``embed_documents``, ``similarity_search_by_vector``,
      ``get_text_embedding_batch``);
    * a hook returns the wrong number of vectors -- **how many**, not what they
      look like: three vectors for twelve documents used to diversify over a
      prefix, and thirteen used to index past the end;
    * a hook returns vectors of an unusable **shape**: not a rectangular
      ``(n, d)`` block of numbers with ``d >= 1``. One flat list of the right
      length, ragged rows, or non-numeric entries all land here rather than as
      a raw numpy ``AxisError``/``ValueError`` or a binding ``TypeError``.

    The fallback result is NOT diversified.
    """


def _vectors_or_reason(raw: Any, count: int, source: str, noun: str):
    """``(vectors, None)`` if ``raw`` is a usable ``(count, d)`` float32 matrix.

    Otherwise ``(None, reason)``, where ``reason`` is a sentence naming
    ``source`` and the ``count`` ``noun`` it had to cover -- for the adapters'
    shared warn / ``strict=True`` fallback.

    Both adapters used to check only ``len(raw) == count`` and hand the rest to
    ``np.ascontiguousarray``. That leaves three ways for an embedding hook to
    escape the fallback with a raw exception from inside numpy or the binding:
    a flat list of the right length (rank 1 -- ``np.linalg.norm(v, axis=1)``
    raises ``AxisError``, and ``gist_select`` raises ``TypeError`` about a
    C-contiguous ``(n, d)`` array), ragged rows (``ValueError: setting an array
    element with a sequence``), and non-numeric entries. The check lives here
    so the two twins cannot answer differently.
    """
    if len(raw) != count:
        return None, f"{source} returned {len(raw)} vectors for {count} {noun}"
    try:
        vectors = np.ascontiguousarray(raw, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        return None, f"{source} did not return a rectangular block of numbers: {exc}"
    if vectors.ndim != 2 or vectors.shape[0] != count or vectors.shape[1] < 1:
        return (
            None,
            f"{source} returned shape {vectors.shape}, not ({count}, d) with d >= 1",
        )
    return vectors, None


def __getattr__(name: str) -> Any:
    # Lazy re-exports so `import divsel.adapters` stays framework-free.
    if name == "DivselRetriever":
        from divsel.adapters.langchain import DivselRetriever

        return DivselRetriever
    if name == "DivselNodePostprocessor":
        from divsel.adapters.llamaindex import DivselNodePostprocessor

        return DivselNodePostprocessor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
