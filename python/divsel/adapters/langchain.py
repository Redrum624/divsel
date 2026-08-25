"""``DivselRetriever`` — a LangChain retriever that replaces MMR with GIST.

Verified against the installed ``langchain-core`` 1.6.0 source
(``langchain_core/retrievers.py``, ``langchain_core/vectorstores/base.py``):

* ``BaseRetriever._get_relevant_documents(self, query: str, *,
  run_manager: CallbackManagerForRetrieverRun) -> list[Document]`` is the
  abstract hook; ``BaseRetriever`` is a pydantic model, so configuration lives
  in declared fields.
* Only the sync hook is implemented here: ``BaseRetriever._aget_relevant_documents``
  already delegates to it via ``run_in_executor`` (retrievers.py line 311).
* ``VectorStore.similarity_search_with_score_by_vector`` is NOT part of the
  1.6.0 base ``VectorStore`` API (some concrete stores add it), so candidates
  are fetched with the base-API ``similarity_search_by_vector``.
"""

from __future__ import annotations

import warnings
from typing import Any, Literal

import numpy as np

from divsel import gist_select
from divsel.adapters import DivselFallbackWarning

try:
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.documents import Document
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.vectorstores import VectorStore
    from pydantic import ConfigDict, Field
except ImportError as exc:  # pragma: no cover - exercised in framework-free venvs
    raise ImportError(
        "divsel.adapters.langchain requires langchain-core. "
        'Install the extra: pip install "divsel[langchain]"'
    ) from exc

__all__ = ["DivselRetriever"]


class DivselRetriever(BaseRetriever):
    """Retrieve ``fetch_k`` candidates, then GIST-diversify down to ``k``.

    A drop-in replacement for ``vectorstore.as_retriever(search_type="mmr")``:
    like MMR it fetches ``fetch_k`` candidates by query similarity and returns
    ``k`` of them, but the subset maximizes ``g(S) + lam * min-pairwise-distance(S)``
    with GIST's approximation guarantee instead of MMR's greedy heuristic.

    Candidate vectors are re-embedded with ``vectorstore.embeddings.embed_documents``
    (stores do not expose their stored vectors uniformly, so the documents'
    ``page_content`` is embedded again; with a deterministic embedding model
    this equals the stored vectors). For ``utility="linear"`` the relevance
    weight of each candidate is its cosine similarity to the query embedding
    shifted to be non-negative (``w = 1 + cos`` in ``[0, 2]``); for
    ``utility="facility_location"`` no weights are passed.

    When embeddings are unavailable (``vectorstore.embeddings`` is ``None`` or
    an ``embed_*`` call raises ``NotImplementedError``) the retriever warns
    with :class:`~divsel.adapters.DivselFallbackWarning` and returns the plain
    top-``k`` ``similarity_search`` result — not diversified. Set
    ``strict=True`` to get a ``ValueError`` instead.
    """

    # Every field carries its constraint, so an unusable configuration is a
    # pydantic ValidationError at construction instead of a Rust-worded
    # ValueError from inside the first query, after a full fetch+embed round
    # trip. The runtime `utility` branch below stays as a guard: pydantic does
    # not revalidate assignment by default, so a field can still be changed to
    # something invalid after the model is built.
    vectorstore: VectorStore
    """The wrapped vector store; must implement ``similarity_search_by_vector``."""
    k: int = Field(default=5, gt=0)
    """Number of documents to return (``|S| <= k``); must be ``> 0``."""
    fetch_k: int = Field(default=20, gt=0)
    """Number of candidates fetched by query similarity before diversifying."""
    lam: float = Field(default=1.0, ge=0.0)
    """Weight of the min-pairwise-distance diversity term; must be ``>= 0``."""
    eps: float = Field(default=0.1, gt=0.0, le=1.0)
    """GIST threshold-sweep accuracy, in ``(0, 1]``."""
    metric: Literal["cosine", "euclidean"] = "cosine"
    """``"cosine"`` or ``"euclidean"``."""
    utility: Literal["linear", "facility_location"] = "linear"
    """``"linear"`` (cosine-relevance weights) or ``"facility_location"``."""
    strict: bool = False
    """Raise ``ValueError`` instead of falling back to plain top-k."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def from_vectorstore(cls, vectorstore: VectorStore, **kwargs: Any) -> DivselRetriever:
        """Convenience constructor: ``DivselRetriever.from_vectorstore(vs, k=4)``."""
        return cls(vectorstore=vectorstore, **kwargs)

    # -- internals ---------------------------------------------------------- #

    def _fallback(self, query: str, reason: str) -> list[Document]:
        if self.strict:
            raise ValueError(
                f"DivselRetriever(strict=True) cannot diversify: {reason}"
            )
        warnings.warn(
            DivselFallbackWarning(
                f"DivselRetriever falling back to plain top-{self.k} similarity "
                f"search (no diversification): {reason}"
            ),
            stacklevel=2,
        )
        return self.vectorstore.similarity_search(query, k=self.k)

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> list[Document]:
        embeddings = self.vectorstore.embeddings
        if embeddings is None:
            return self._fallback(query, "vectorstore.embeddings is None")

        try:
            query_vec = embeddings.embed_query(query)
        except NotImplementedError:
            return self._fallback(query, "embeddings.embed_query is not implemented")

        try:
            docs = self.vectorstore.similarity_search_by_vector(
                query_vec, k=self.fetch_k
            )
        except NotImplementedError:
            return self._fallback(
                query, "vectorstore.similarity_search_by_vector is not implemented"
            )
        if not docs:
            return []

        try:
            doc_vecs = embeddings.embed_documents([d.page_content for d in docs])
        except NotImplementedError:
            return self._fallback(
                query, "embeddings.embed_documents is not implemented"
            )

        vectors = np.ascontiguousarray(doc_vecs, dtype=np.float32)

        if self.utility == "linear":
            v = vectors.astype(np.float64)
            norms = np.linalg.norm(v, axis=1)
            norms[norms == 0.0] = 1.0
            q = np.asarray(query_vec, dtype=np.float64)
            q_norm = np.linalg.norm(q)
            if q_norm == 0.0:
                q_norm = 1.0
            cos = (v @ q) / (norms * q_norm)
            utilities = np.ascontiguousarray(np.clip(1.0 + cos, 0.0, None))
        elif self.utility == "facility_location":
            utilities = None
        else:
            raise ValueError(
                f"DivselRetriever supports utility='linear' or "
                f"'facility_location', got {self.utility!r}"
            )

        selected = gist_select(
            vectors,
            utilities,
            k=min(self.k, len(docs)),
            lam=self.lam,
            eps=self.eps,
            metric=self.metric,
            utility=self.utility,
        )
        return [docs[i] for i in selected]
