"""``DivselNodePostprocessor`` — a LlamaIndex node postprocessor running GIST.

Verified against the installed ``llama-index-core`` 0.14.24 source
(``llama_index/core/postprocessor/types.py``, ``llama_index/core/schema.py``):

* ``BaseNodePostprocessor`` is a pydantic model (``BaseComponent``) with
  ``model_config = ConfigDict(arbitrary_types_allowed=True)``; the abstract
  hook is ``_postprocess_nodes(self, nodes: List[NodeWithScore],
  query_bundle: Optional[QueryBundle] = None) -> List[NodeWithScore]``.
* ``NodeWithScore`` has ``.node`` (with ``.embedding: Optional[List[float]]``
  and ``.get_content()``) and ``.score: Optional[float]``.
* ``QueryBundle`` has ``.embedding: Optional[List[float]]``.
"""

from __future__ import annotations

import warnings
from typing import List, Literal, Optional

import numpy as np

from divsel import gist_select
from divsel.adapters import MIN_EPS, DivselFallbackWarning, _vectors_or_reason

try:
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.postprocessor.types import BaseNodePostprocessor
    from llama_index.core.schema import NodeWithScore, QueryBundle
    from pydantic import Field, field_validator
except ImportError as exc:  # pragma: no cover - exercised in framework-free venvs
    raise ImportError(
        "divsel.adapters.llamaindex requires llama-index-core. "
        'Install the extra: pip install "divsel[llamaindex]"'
    ) from exc

__all__ = ["DivselNodePostprocessor"]


class DivselNodePostprocessor(BaseNodePostprocessor):
    """GIST-diversify retrieved nodes down to ``k``.

    A drop-in replacement for LlamaIndex's MMR-style reranking: the retriever
    has already fetched the candidate set (so there is no ``fetch_k`` here —
    the caller's ``similarity_top_k`` plays that role), and this postprocessor
    returns the subset of at most ``k`` nodes maximizing
    ``g(S) + lam * min-pairwise-distance(S)`` with GIST's guarantee.

    Candidate vectors come from ``node.embedding`` when EVERY node has one;
    otherwise, when ``embed_model`` is set, each node's ``get_content()`` is
    embedded with it. Relevance weights for ``utility="linear"``: the nodes'
    ``score`` values shifted to be non-negative when all scores exist, else
    cosine similarity to ``query_bundle.embedding`` (shifted, ``1 + cos``)
    when that exists, else uniform.

    When no usable vectors can be obtained it warns with
    :class:`~divsel.adapters.DivselFallbackWarning` and returns the plain
    top-``k`` by existing ``score`` (or the first ``k`` nodes when scores are
    missing) — not diversified. Set ``strict=True`` for a ``ValueError``. That
    covers nodes with no embeddings and no ``embed_model``,
    ``get_text_embedding_batch`` raising ``NotImplementedError``, and either
    source — the nodes' own ``embedding`` values or the model's return —
    yielding the wrong number of vectors or vectors of an unusable shape (not a
    rectangular ``(n, d)`` block of numbers).
    :class:`~divsel.adapters.DivselFallbackWarning` lists them all.
    """

    # Constraints live on the fields, so an unusable configuration fails at
    # construction rather than at the first query. The runtime `utility` branch
    # below stays as a guard: pydantic does not revalidate assignment by default.
    k: int = Field(default=5, gt=0)
    """Number of nodes to return (``|S| <= k``); must be ``> 0``."""
    lam: float = Field(default=1.0, ge=0.0, allow_inf_nan=False)
    """Weight of the diversity term; must be finite and ``>= 0``.

    ``allow_inf_nan=False`` because pydantic allows both by default and
    ``inf >= 0.0`` is true, so ``+inf`` would pass the bound here and fail the
    core's finite check at the first query. ``nan`` fails ``>= 0`` either way.
    """
    eps: float = Field(default=0.1, ge=MIN_EPS, le=1.0)
    """GIST threshold-sweep accuracy, in ``[MIN_EPS, 1]`` -- the range the core
    accepts, ``float(np.finfo(np.float32).eps)`` (about ``1.19e-7``) to ``1.0``.
    Below that floor the ``float32`` threshold grid cannot separate two
    consecutive entries."""
    metric: Literal["cosine", "euclidean"] = "cosine"
    """``"cosine"`` or ``"euclidean"``."""
    utility: Literal["linear", "facility_location"] = "linear"
    """``"linear"`` (score/cosine weights) or ``"facility_location"``."""
    strict: bool = False
    """Raise ``ValueError`` instead of falling back to plain top-k."""
    embed_model: Optional[BaseEmbedding] = None
    """Used to embed node content when nodes carry no embeddings."""

    @field_validator("k", mode="before")
    @classmethod
    def _budget_is_not_a_bool(cls, value: object) -> object:
        # `bool` is a subclass of `int` and pydantic's default lax mode coerces
        # it, so `k=True` would arrive here as `1` and the postprocessor would
        # quietly return one node. That is the same input divsel's own binding
        # rejects (`k must be an int, not bool`); `gt=0` catches `False` but
        # nothing catches `True`.
        if isinstance(value, bool):
            raise ValueError("must be an int, not bool")
        return value

    @classmethod
    def class_name(cls) -> str:
        return "DivselNodePostprocessor"

    # -- internals ---------------------------------------------------------- #

    def _fallback(
        self, nodes: List[NodeWithScore], reason: str
    ) -> List[NodeWithScore]:
        if self.strict:
            raise ValueError(
                f"DivselNodePostprocessor(strict=True) cannot diversify: {reason}"
            )
        warnings.warn(
            DivselFallbackWarning(
                f"DivselNodePostprocessor falling back to plain top-{self.k} "
                f"(no diversification): {reason}"
            ),
            stacklevel=2,
        )
        if all(n.score is not None for n in nodes):
            ranked = sorted(nodes, key=lambda n: -n.score)
        else:
            ranked = list(nodes)
        return ranked[: self.k]

    def _postprocess_nodes(
        self,
        nodes: List[NodeWithScore],
        query_bundle: Optional[QueryBundle] = None,
    ) -> List[NodeWithScore]:
        if not nodes:
            return []

        node_embeddings = [n.node.embedding for n in nodes]
        if all(e is not None for e in node_embeddings):
            # The nodes' own embeddings had no guard at all: one node carrying a
            # dimension-2 embedding while the rest carry dimension 4 went
            # straight into `np.ascontiguousarray` and raised a raw numpy
            # `ValueError` past both the warn path and the strict path.
            vectors, reason = _vectors_or_reason(
                node_embeddings, len(nodes), "the nodes' own node.embedding", "nodes"
            )
            if reason is not None:
                return self._fallback(nodes, reason)
        elif self.embed_model is not None:
            # An embed model is free to implement only the single-text hook, or
            # none at all: unguarded, that `NotImplementedError` escaped past
            # both the warn path and the `strict=True` ValueError path.
            try:
                raw_vectors = self.embed_model.get_text_embedding_batch(
                    [n.node.get_content() for n in nodes]
                )
            except NotImplementedError:
                return self._fallback(
                    nodes, "embed_model.get_text_embedding_batch is not implemented"
                )
            # And a return that is not one usable vector per node is a failure
            # too, not a shape to be passed on: a short one silently diversified
            # over a prefix of `nodes`, a long one let `selected` index past
            # them, and a right-length return of the wrong rank or with ragged
            # rows escaped as a raw numpy error or the binding's `TypeError`.
            # `_vectors_or_reason` is shared with the LangChain twin.
            vectors, reason = _vectors_or_reason(
                raw_vectors,
                len(nodes),
                "embed_model.get_text_embedding_batch",
                "nodes",
            )
            if reason is not None:
                return self._fallback(nodes, reason)
        else:
            return self._fallback(
                nodes, "nodes carry no embeddings and no embed_model is set"
            )

        if self.utility == "linear":
            utilities = self._linear_weights(nodes, vectors, query_bundle)
        elif self.utility == "facility_location":
            utilities = None
        else:
            raise ValueError(
                f"DivselNodePostprocessor supports utility='linear' or "
                f"'facility_location', got {self.utility!r}"
            )

        selected = gist_select(
            vectors,
            utilities,
            k=min(self.k, len(nodes)),
            lam=self.lam,
            eps=self.eps,
            metric=self.metric,
            utility=self.utility,
        )
        return [nodes[i] for i in selected]

    @staticmethod
    def _linear_weights(
        nodes: List[NodeWithScore],
        vectors: np.ndarray,
        query_bundle: Optional[QueryBundle],
    ) -> Optional[np.ndarray]:
        scores = [n.score for n in nodes]
        if all(s is not None for s in scores):
            w = np.asarray(scores, dtype=np.float64)
            floor = min(float(w.min()), 0.0)
            return np.ascontiguousarray(w - floor)
        if query_bundle is not None and query_bundle.embedding is not None:
            v = vectors.astype(np.float64)
            norms = np.linalg.norm(v, axis=1)
            norms[norms == 0.0] = 1.0
            q = np.asarray(query_bundle.embedding, dtype=np.float64)
            q_norm = np.linalg.norm(q)
            if q_norm == 0.0:
                q_norm = 1.0
            cos = (v @ q) / (norms * q_norm)
            return np.ascontiguousarray(np.clip(1.0 + cos, 0.0, None))
        return None  # uniform weights
