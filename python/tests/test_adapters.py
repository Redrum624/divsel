"""Tests for ``divsel.adapters`` (Task 9): the LangChain and LlamaIndex MMR drop-ins.

The framework-specific tests skip cleanly (``pytest.importorskip``) in a venv
without ``langchain-core`` / ``llama-index-core``; the subprocess isolation test
and the missing-extra ImportError tests run everywhere.

Fixture geometry (shared by both adapter suites): 12 unit vectors in R^4 around
the query direction e1 —

* cluster A (5 vectors): tiny jitters of e1, relevance ~1.0 — a plain top-k
  collapses entirely into this cluster (min pairwise cosine distance ~5e-5);
* cluster B (4 vectors): a tight cluster ~0.75 cosine-distance away from A;
* 3 outliers: mutually spread, ~0.6-0.95 from everything else.

With ``lam`` high enough, GIST must leave cluster A, so the min pairwise cosine
distance of its selection is strictly (and hugely) larger than top-k's.
"""

from __future__ import annotations

import functools
import importlib.util
import subprocess
import sys
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# shared fixture data (numpy only — safe to build without any framework)      #
# --------------------------------------------------------------------------- #

TEXTS = ["a0", "a1", "a2", "a3", "a4", "b0", "b1", "b2", "b3", "o0", "o1", "o2"]

QUERY_TEXT = "query"
QUERY_VEC = [1.0, 0.0, 0.0, 0.0]


def _unit(row):
    r = np.asarray(row, dtype=np.float64)
    return r / np.linalg.norm(r)


def _fixture_vectors() -> np.ndarray:
    rows = []
    # cluster A: five near-copies of e1
    for d in (0.00, 0.01, 0.02, 0.03, 0.04):
        rows.append(_unit([1.0, d, 0.0, 0.0]))
    # cluster B: tight, ~41 degrees away
    for d in (0.00, 0.01, 0.02, 0.03):
        rows.append(_unit([0.25, 0.97, d, 0.0]))
    # outliers: mutually spread
    rows.append(_unit([0.30, 0.0, 0.95, 0.0]))
    rows.append(_unit([0.20, 0.0, 0.0, 0.98]))
    rows.append(_unit([0.10, 0.40, 0.40, 0.82]))
    return np.asarray(rows, dtype=np.float32)


VECS = _fixture_vectors()
TABLE = {t: [float(x) for x in VECS[i]] for i, t in enumerate(TEXTS)}
TABLE[QUERY_TEXT] = QUERY_VEC

# Query-relevance order: cluster A first, then B, then outliers.
SIMS = VECS.astype(np.float64) @ np.asarray(QUERY_VEC, dtype=np.float64)
RELEVANCE_ORDER = list(np.argsort(-SIMS, kind="stable"))


def min_pairwise_cos_dist(vectors) -> float:
    v = np.asarray(vectors, dtype=np.float64)
    v = v / np.linalg.norm(v, axis=1, keepdims=True)
    n = len(v)
    assert n >= 2
    best = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            best = min(best, 1.0 - float(v[i] @ v[j]))
    return best


def topk_diversity(k: int) -> float:
    """Min pairwise cosine distance of the plain top-k by query similarity."""
    return min_pairwise_cos_dist(VECS[RELEVANCE_ORDER[:k]])


# --------------------------------------------------------------------------- #
# isolation: importing divsel must never import a framework                   #
# --------------------------------------------------------------------------- #


def test_import_divsel_imports_no_framework():
    code = (
        "import sys\n"
        "import divsel\n"
        "import divsel.adapters\n"
        "bad = [m for m in sys.modules"
        " if m.startswith(('langchain', 'llama_index'))]\n"
        "print(','.join(bad))\n"
    )
    # A child that hangs (a half-written .pyd, an import hook that blocks, an
    # antivirus stall) must fail this test, not the whole pytest run: nothing in
    # pyproject.toml sets a per-test timeout.
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == ""


@pytest.mark.skipif(
    importlib.util.find_spec("langchain_core") is not None,
    reason="langchain-core installed; the missing-extra hint cannot fire",
)
def test_langchain_missing_extra_hint():
    with pytest.raises(ImportError, match=r"divsel\[langchain\]"):
        import divsel.adapters.langchain  # noqa: F401


@pytest.mark.skipif(
    importlib.util.find_spec("llama_index") is not None,
    reason="llama-index-core installed; the missing-extra hint cannot fire",
)
def test_llamaindex_missing_extra_hint():
    with pytest.raises(ImportError, match=r"divsel\[llamaindex\]"):
        import divsel.adapters.llamaindex  # noqa: F401


# --------------------------------------------------------------------------- #
# LangChain                                                                   #
# --------------------------------------------------------------------------- #


@functools.lru_cache(maxsize=1)
def _lc() -> SimpleNamespace:
    """Fake LangChain vector store + embeddings over the shared fixture."""
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore

    class TinyEmbeddings(Embeddings):
        def embed_documents(self, texts):
            return [TABLE[t] for t in texts]

        def embed_query(self, text):
            return TABLE[text]

    class TinyStore(VectorStore):
        """In-memory store over the 12 fixture docs; exposes embeddings."""

        def __init__(self):
            self._docs = [Document(page_content=t) for t in TEXTS]
            self._embeddings = TinyEmbeddings()

        @property
        def embeddings(self):
            return self._embeddings

        @classmethod
        def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
            raise NotImplementedError

        def similarity_search_by_vector(self, embedding, k=4, **kwargs):
            q = _unit(embedding)
            sims = VECS.astype(np.float64) @ q
            order = np.argsort(-sims, kind="stable")[:k]
            return [self._docs[i] for i in order]

        def similarity_search(self, query, k=4, **kwargs):
            return self.similarity_search_by_vector(
                self._embeddings.embed_query(query), k=k
            )

    class NoEmbeddingsStore(VectorStore):
        """A store whose ``embeddings`` is None (the base default)."""

        def __init__(self):
            self._docs = [Document(page_content=t) for t in TEXTS]

        @classmethod
        def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
            raise NotImplementedError

        def similarity_search(self, query, k=4, **kwargs):
            # "relevance order" for the fallback: the fixture's true order
            return [self._docs[i] for i in RELEVANCE_ORDER[:k]]

    return SimpleNamespace(
        Document=Document, TinyStore=TinyStore, NoEmbeddingsStore=NoEmbeddingsStore
    )


def _make_retriever(**kwargs):
    from divsel.adapters.langchain import DivselRetriever

    return DivselRetriever(vectorstore=_lc().TinyStore(), **kwargs)


def test_langchain_returns_exactly_k():
    pytest.importorskip("langchain_core")
    docs = _make_retriever(k=4, fetch_k=12, lam=4.0).invoke(QUERY_TEXT)
    assert len(docs) == 4
    assert len({d.page_content for d in docs}) == 4


def test_langchain_more_diverse_than_topk():
    pytest.importorskip("langchain_core")
    docs = _make_retriever(k=4, fetch_k=12, lam=4.0).invoke(QUERY_TEXT)
    got = min_pairwise_cos_dist([TABLE[d.page_content] for d in docs])
    assert got > topk_diversity(4)


def test_langchain_deterministic():
    pytest.importorskip("langchain_core")
    retriever = _make_retriever(k=4, fetch_k=12, lam=4.0)
    first = [d.page_content for d in retriever.invoke(QUERY_TEXT)]
    for _ in range(3):
        assert [d.page_content for d in retriever.invoke(QUERY_TEXT)] == first


def test_langchain_fetch_k_smaller_than_k():
    pytest.importorskip("langchain_core")
    docs = _make_retriever(k=5, fetch_k=3, lam=4.0).invoke(QUERY_TEXT)
    # k is clamped to the 3 fetched candidates, and on this fixture all 3 come
    # back: they are the near-duplicate cluster A, so no threshold can beat the
    # weight of a third point with a diversity gain of ~5e-5.
    assert len(docs) == 3
    assert {d.page_content for d in docs} == {TEXTS[i] for i in RELEVANCE_ORDER[:3]}


def test_langchain_fallback_warns_and_returns_topk():
    pytest.importorskip("langchain_core")
    from divsel.adapters import DivselFallbackWarning
    from divsel.adapters.langchain import DivselRetriever

    store = _lc().NoEmbeddingsStore()
    retriever = DivselRetriever(vectorstore=store, k=4)
    with pytest.warns(DivselFallbackWarning):
        docs = retriever.invoke(QUERY_TEXT)
    assert [d.page_content for d in docs] == [TEXTS[i] for i in RELEVANCE_ORDER[:4]]


def test_langchain_strict_raises():
    pytest.importorskip("langchain_core")
    from divsel.adapters.langchain import DivselRetriever

    retriever = DivselRetriever(vectorstore=_lc().NoEmbeddingsStore(), k=4, strict=True)
    with pytest.raises(ValueError):
        retriever.invoke(QUERY_TEXT)


def test_langchain_from_vectorstore():
    pytest.importorskip("langchain_core")
    from divsel.adapters.langchain import DivselRetriever

    retriever = DivselRetriever.from_vectorstore(_lc().TinyStore(), k=3, lam=2.0)
    assert retriever.k == 3
    assert retriever.lam == 2.0


# --------------------------------------------------------------------------- #
# LlamaIndex                                                                  #
# --------------------------------------------------------------------------- #

# Retriever-style scores: A ranks first, but B and the outliers are close, so a
# small diversity term already justifies leaving cluster A.
SCORES = [1.0, 0.99, 0.98, 0.97, 0.96, 0.90, 0.895, 0.89, 0.885, 0.87, 0.86, 0.85]


@functools.lru_cache(maxsize=1)
def _li() -> SimpleNamespace:
    from llama_index.core.base.embeddings.base import BaseEmbedding
    from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode

    class TinyEmbedModel(BaseEmbedding):
        def _get_query_embedding(self, query):
            return TABLE[query]

        async def _aget_query_embedding(self, query):
            return TABLE[query]

        def _get_text_embedding(self, text):
            return TABLE[text]

    def make_nodes(with_embeddings=True, with_scores=True):
        nodes = []
        for i, t in enumerate(TEXTS):
            node = TextNode(text=t)
            if with_embeddings:
                node.embedding = TABLE[t]
            score = SCORES[i] if with_scores else None
            nodes.append(NodeWithScore(node=node, score=score))
        return nodes

    return SimpleNamespace(
        NodeWithScore=NodeWithScore,
        QueryBundle=QueryBundle,
        TextNode=TextNode,
        TinyEmbedModel=TinyEmbedModel,
        make_nodes=make_nodes,
    )


def _topk_by_score(nodes, k):
    return sorted(nodes, key=lambda n: -n.score)[:k]


def test_llamaindex_returns_exactly_k():
    pytest.importorskip("llama_index.core")
    from divsel.adapters.llamaindex import DivselNodePostprocessor

    ns = _li()
    out = DivselNodePostprocessor(k=4, lam=1.0).postprocess_nodes(
        ns.make_nodes(), query_str=QUERY_TEXT
    )
    assert len(out) == 4
    assert len({n.node.get_content() for n in out}) == 4


def test_llamaindex_more_diverse_than_topk():
    pytest.importorskip("llama_index.core")
    from divsel.adapters.llamaindex import DivselNodePostprocessor

    ns = _li()
    nodes = ns.make_nodes()
    out = DivselNodePostprocessor(k=4, lam=1.0).postprocess_nodes(
        nodes, query_str=QUERY_TEXT
    )
    got = min_pairwise_cos_dist([n.node.embedding for n in out])
    baseline = min_pairwise_cos_dist(
        [n.node.embedding for n in _topk_by_score(nodes, 4)]
    )
    assert got > baseline


def test_llamaindex_deterministic():
    pytest.importorskip("llama_index.core")
    from divsel.adapters.llamaindex import DivselNodePostprocessor

    ns = _li()
    pp = DivselNodePostprocessor(k=4, lam=1.0)
    first = [n.node.get_content() for n in pp.postprocess_nodes(ns.make_nodes(), query_str=QUERY_TEXT)]
    for _ in range(3):
        again = [
            n.node.get_content()
            for n in pp.postprocess_nodes(ns.make_nodes(), query_str=QUERY_TEXT)
        ]
        assert again == first


def test_llamaindex_fallback_warns_and_returns_topk_by_score():
    pytest.importorskip("llama_index.core")
    from divsel.adapters import DivselFallbackWarning
    from divsel.adapters.llamaindex import DivselNodePostprocessor

    ns = _li()
    nodes = ns.make_nodes(with_embeddings=False)
    with pytest.warns(DivselFallbackWarning):
        out = DivselNodePostprocessor(k=4).postprocess_nodes(
            nodes, query_str=QUERY_TEXT
        )
    assert [n.node.get_content() for n in out] == [
        n.node.get_content() for n in _topk_by_score(nodes, 4)
    ]


def test_llamaindex_fallback_no_scores_returns_first_k():
    pytest.importorskip("llama_index.core")
    from divsel.adapters import DivselFallbackWarning
    from divsel.adapters.llamaindex import DivselNodePostprocessor

    ns = _li()
    nodes = ns.make_nodes(with_embeddings=False, with_scores=False)
    with pytest.warns(DivselFallbackWarning):
        out = DivselNodePostprocessor(k=4).postprocess_nodes(
            nodes, query_str=QUERY_TEXT
        )
    assert [n.node.get_content() for n in out] == TEXTS[:4]


def test_llamaindex_strict_raises():
    pytest.importorskip("llama_index.core")
    from divsel.adapters.llamaindex import DivselNodePostprocessor

    ns = _li()
    with pytest.raises(ValueError):
        DivselNodePostprocessor(k=4, strict=True).postprocess_nodes(
            ns.make_nodes(with_embeddings=False), query_str=QUERY_TEXT
        )


def test_llamaindex_query_embedding_weights_when_scores_are_missing():
    """Second weight rung: no scores, but the query bundle carries an embedding."""
    pytest.importorskip("llama_index.core")
    from divsel.adapters.llamaindex import DivselNodePostprocessor

    ns = _li()
    nodes = ns.make_nodes(with_scores=False)
    vectors = np.ascontiguousarray([n.node.embedding for n in nodes], dtype=np.float32)
    bundle = ns.QueryBundle(query_str=QUERY_TEXT, embedding=QUERY_VEC)

    w = DivselNodePostprocessor._linear_weights(nodes, vectors, bundle)
    assert w is not None and w.shape == (len(nodes),) and w.dtype == np.float64
    # 1 + cos(node, query), clipped at 0; the fixture rows are unit vectors and
    # the query is e1, so cos is simply the first coordinate.
    expected = 1.0 + vectors.astype(np.float64) @ np.asarray(QUERY_VEC, dtype=np.float64)
    assert np.allclose(w, expected)
    assert (w >= 0.0).all()

    # These weights spread wider than SCORES (cluster A ~2.0 against ~1.1-1.3
    # elsewhere), so leaving A needs the same lam the LangChain suite uses.
    out = DivselNodePostprocessor(k=4, lam=4.0).postprocess_nodes(nodes, query_bundle=bundle)
    assert len(out) == 4
    got = min_pairwise_cos_dist([n.node.embedding for n in out])
    assert got > topk_diversity(4)


def test_llamaindex_uniform_weights_when_no_scores_and_no_query_embedding():
    """Third weight rung: nothing to weight by, so ``utilities`` is None (uniform)."""
    pytest.importorskip("llama_index.core")
    from divsel.adapters.llamaindex import DivselNodePostprocessor

    ns = _li()
    nodes = ns.make_nodes(with_scores=False)
    vectors = np.ascontiguousarray([n.node.embedding for n in nodes], dtype=np.float32)
    assert DivselNodePostprocessor._linear_weights(nodes, vectors, None) is None
    no_embedding = ns.QueryBundle(query_str=QUERY_TEXT)
    assert no_embedding.embedding is None
    assert DivselNodePostprocessor._linear_weights(nodes, vectors, no_embedding) is None

    # End to end through the public API: query_str alone builds a bundle
    # without an embedding, so this is the uniform rung. g(S) = |S| for every
    # 4-subset, so f is decided by diversity alone.
    out = DivselNodePostprocessor(k=4, lam=1.0).postprocess_nodes(nodes, query_str=QUERY_TEXT)
    assert len(out) == 4
    got = min_pairwise_cos_dist([n.node.embedding for n in out])
    assert got > topk_diversity(4)


def test_llamaindex_embed_model_diversifies():
    pytest.importorskip("llama_index.core")
    from divsel.adapters.llamaindex import DivselNodePostprocessor

    ns = _li()
    nodes = ns.make_nodes(with_embeddings=False)
    pp = DivselNodePostprocessor(k=4, lam=1.0, embed_model=ns.TinyEmbedModel())
    out = pp.postprocess_nodes(nodes, query_str=QUERY_TEXT)
    assert len(out) == 4
    got = min_pairwise_cos_dist([TABLE[n.node.get_content()] for n in out])
    baseline = min_pairwise_cos_dist(
        [TABLE[n.node.get_content()] for n in _topk_by_score(nodes, 4)]
    )
    assert got > baseline


# --------------------------------------------------------------------------- #
# adapters package: the lazy re-exports named in __all__                      #
# --------------------------------------------------------------------------- #


def test_adapters_unknown_attribute_raises_attributeerror():
    import divsel.adapters as adapters

    with pytest.raises(AttributeError, match="has no attribute 'nope'"):
        adapters.nope  # noqa: B018


def test_adapters_lazy_reexport_of_the_langchain_class():
    pytest.importorskip("langchain_core")
    import divsel.adapters as adapters
    from divsel.adapters.langchain import DivselRetriever

    assert "DivselRetriever" in adapters.__all__
    assert getattr(adapters, "DivselRetriever") is DivselRetriever


def test_adapters_lazy_reexport_of_the_llamaindex_class():
    pytest.importorskip("llama_index.core")
    import divsel.adapters as adapters
    from divsel.adapters.llamaindex import DivselNodePostprocessor

    assert "DivselNodePostprocessor" in adapters.__all__
    assert getattr(adapters, "DivselNodePostprocessor") is DivselNodePostprocessor


# --------------------------------------------------------------------------- #
# LangChain: construction-time validation and every fallback branch           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"k": -1},
        {"k": True},
        {"fetch_k": 0},
        {"fetch_k": True},
        {"lam": -1.0},
        {"lam": float("inf")},
        {"lam": float("nan")},
        {"eps": 0.0},
        {"eps": 1.5},
        {"eps": 1e-9},
        {"eps": 1e-30},
        {"metric": "manhattan"},
        {"utility": "coverage"},
    ],
    ids=[
        "k0",
        "k_neg",
        "k_true",
        "fetch_k0",
        "fetch_k_true",
        "lam_neg",
        "lam_inf",
        "lam_nan",
        "eps0",
        "eps_high",
        "eps_below_f32_epsilon",
        "eps_tiny",
        "metric",
        "utility",
    ],
)
def test_langchain_invalid_configuration_fails_at_construction(kwargs):
    """Every constraint is a field constraint, so nothing waits for a query.

    Before this, ``DivselRetriever(vectorstore=vs, k=0).invoke("q")`` raised a
    Rust-worded ``ValueError`` from inside ``_get_relevant_documents``, after a
    full fetch-and-embed round trip.
    """
    pytest.importorskip("langchain_core")
    from divsel.adapters.langchain import DivselRetriever

    # pydantic's ValidationError is a ValueError subclass.
    with pytest.raises(ValueError):
        DivselRetriever(vectorstore=_lc().TinyStore(), **kwargs)


@functools.lru_cache(maxsize=1)
def _lc_edge() -> SimpleNamespace:
    """Vector stores for the LangChain adapter's fallback and edge branches."""
    from langchain_core.documents import Document
    from langchain_core.embeddings import Embeddings
    from langchain_core.vectorstores import VectorStore

    class Partial(Embeddings):
        """Embeddings whose two halves can each be turned off."""

        def __init__(self, query=True, documents=True):
            self._query = query
            self._documents = documents

        def embed_query(self, text):
            if not self._query:
                raise NotImplementedError
            return TABLE[text]

        def embed_documents(self, texts):
            if not self._documents:
                raise NotImplementedError
            return [TABLE[t] for t in texts]

    class ZeroEmbeddings(Embeddings):
        """A zero query vector and one zero document vector."""

        def embed_query(self, text):
            return [0.0, 0.0, 0.0, 0.0]

        def embed_documents(self, texts):
            return [[0.0] * 4 if t == TEXTS[0] else TABLE[t] for t in texts]

    class Store(VectorStore):
        def __init__(self, embeddings, *, by_vector=True, candidates=None):
            self._docs = [Document(page_content=t) for t in TEXTS]
            self._embeddings = embeddings
            self._by_vector = by_vector
            self._candidates = candidates

        @property
        def embeddings(self):
            return self._embeddings

        @classmethod
        def from_texts(cls, texts, embedding, metadatas=None, **kwargs):
            raise NotImplementedError

        def similarity_search_by_vector(self, embedding, k=4, **kwargs):
            if not self._by_vector:
                raise NotImplementedError
            if self._candidates is not None:
                return list(self._candidates)
            return [self._docs[i] for i in RELEVANCE_ORDER[:k]]

        def similarity_search(self, query, k=4, **kwargs):
            return [self._docs[i] for i in RELEVANCE_ORDER[:k]]

    return SimpleNamespace(Partial=Partial, ZeroEmbeddings=ZeroEmbeddings, Store=Store)


@pytest.mark.parametrize(
    "broken,reason",
    [
        ("embed_query", "embed_query is not implemented"),
        ("similarity_search_by_vector", "similarity_search_by_vector is not implemented"),
        ("embed_documents", "embed_documents is not implemented"),
    ],
)
def test_langchain_falls_back_on_each_notimplemented_hook(broken, reason):
    pytest.importorskip("langchain_core")
    from divsel.adapters import DivselFallbackWarning
    from divsel.adapters.langchain import DivselRetriever

    ns = _lc_edge()
    if broken == "embed_query":
        store = ns.Store(ns.Partial(query=False))
    elif broken == "embed_documents":
        store = ns.Store(ns.Partial(documents=False))
    else:
        store = ns.Store(ns.Partial(), by_vector=False)

    retriever = DivselRetriever(vectorstore=store, k=4)
    with pytest.warns(DivselFallbackWarning, match=reason):
        docs = retriever.invoke(QUERY_TEXT)
    assert [d.page_content for d in docs] == [TEXTS[i] for i in RELEVANCE_ORDER[:4]]

    # ... and strict=True turns each of them into a ValueError instead.
    strict = DivselRetriever(vectorstore=store, k=4, strict=True)
    with pytest.raises(ValueError, match=reason):
        strict.invoke(QUERY_TEXT)


def test_langchain_falls_back_when_the_embeddings_property_itself_raises():
    """``vectorstore.embeddings`` is a property, and a property can raise.

    The adapter guarded ``embeddings is None`` and then guarded
    ``NotImplementedError`` around each ``embed_*`` call and around
    ``similarity_search_by_vector`` — but not around the attribute access that
    reaches them. langchain-core's base ``VectorStore.embeddings`` returns
    ``None``; a concrete store is free to raise ``NotImplementedError`` there
    instead, and that escaped past both the warn path and the strict path as a
    bare ``NotImplementedError``.
    """
    pytest.importorskip("langchain_core")
    from divsel.adapters import DivselFallbackWarning
    from divsel.adapters.langchain import DivselRetriever

    ns = _lc_edge()

    class NoEmbeddings(ns.Store):
        @property
        def embeddings(self):
            raise NotImplementedError

    store = NoEmbeddings(ns.Partial())
    retriever = DivselRetriever(vectorstore=store, k=4)
    with pytest.warns(DivselFallbackWarning, match="embeddings"):
        docs = retriever.invoke(QUERY_TEXT)
    assert [d.page_content for d in docs] == [TEXTS[i] for i in RELEVANCE_ORDER[:4]]

    strict = DivselRetriever(vectorstore=store, k=4, strict=True)
    with pytest.raises(ValueError, match="embeddings"):
        strict.invoke(QUERY_TEXT)


def test_langchain_no_candidates_returns_empty_without_warning():
    pytest.importorskip("langchain_core")
    from divsel.adapters.langchain import DivselRetriever

    ns = _lc_edge()
    store = ns.Store(ns.Partial(), candidates=[])
    retriever = DivselRetriever(vectorstore=store, k=4)
    # An empty candidate set is not a fallback: there is nothing to diversify
    # and nothing to warn about.
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert retriever.invoke(QUERY_TEXT) == []


def test_langchain_zero_norm_vectors_do_not_divide_by_zero():
    pytest.importorskip("langchain_core")
    from divsel.adapters.langchain import DivselRetriever

    ns = _lc_edge()
    # Euclidean: a zero row is a legal point (cosine would reject it as
    # unnormalisable, which is a core-level error, not this guard).
    retriever = DivselRetriever(
        vectorstore=ns.Store(ns.ZeroEmbeddings()), k=3, fetch_k=6, metric="euclidean"
    )
    docs = retriever.invoke(QUERY_TEXT)
    assert 0 < len(docs) <= 3
    assert len({d.page_content for d in docs}) == len(docs)


def test_langchain_facility_location_diversifies():
    pytest.importorskip("langchain_core")
    docs = _make_retriever(
        k=4, fetch_k=12, lam=1.0, utility="facility_location"
    ).invoke(QUERY_TEXT)
    assert len(docs) == 4
    got = min_pairwise_cos_dist([TABLE[d.page_content] for d in docs])
    assert got > topk_diversity(4)


def test_langchain_unknown_utility_after_construction_raises():
    pytest.importorskip("langchain_core")

    retriever = _make_retriever(k=3, fetch_k=6)
    # pydantic does not revalidate assignment, so the runtime branch is still
    # reachable and still has to say what it supports.
    object.__setattr__(retriever, "utility", "coverage")
    with pytest.raises(ValueError, match="utility='linear' or 'facility_location'"):
        retriever.invoke(QUERY_TEXT)


# --------------------------------------------------------------------------- #
# LlamaIndex: construction-time validation and the remaining branches         #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 0},
        {"k": -1},
        {"k": True},
        {"lam": -1.0},
        {"lam": float("inf")},
        {"lam": float("nan")},
        {"eps": 0.0},
        {"eps": 1.5},
        {"eps": 1e-9},
        {"eps": 1e-30},
        {"metric": "manhattan"},
        {"utility": "coverage"},
    ],
    ids=[
        "k0",
        "k_neg",
        "k_true",
        "lam_neg",
        "lam_inf",
        "lam_nan",
        "eps0",
        "eps_high",
        "eps_below_f32_epsilon",
        "eps_tiny",
        "metric",
        "utility",
    ],
)
def test_llamaindex_invalid_configuration_fails_at_construction(kwargs):
    pytest.importorskip("llama_index.core")
    from divsel.adapters.llamaindex import DivselNodePostprocessor

    with pytest.raises(ValueError):
        DivselNodePostprocessor(**kwargs)


def test_llamaindex_no_nodes_returns_empty_without_warning():
    """The empty-input guard, whose LangChain twin is already covered.

    Nothing to diversify is not a fallback: no warning, no ``ValueError`` under
    ``strict``, and no call into the core.
    """
    pytest.importorskip("llama_index.core")
    from divsel.adapters.llamaindex import DivselNodePostprocessor

    for post in (
        DivselNodePostprocessor(k=3),
        DivselNodePostprocessor(k=3, strict=True),
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            assert post.postprocess_nodes([], query_str=QUERY_TEXT) == []


def test_llamaindex_facility_location_diversifies():
    pytest.importorskip("llama_index.core")
    from divsel.adapters.llamaindex import DivselNodePostprocessor

    ns = _li()
    nodes = ns.make_nodes()
    out = DivselNodePostprocessor(k=4, lam=1.0, utility="facility_location").postprocess_nodes(
        nodes, query_str=QUERY_TEXT
    )
    assert len(out) == 4
    got = min_pairwise_cos_dist([n.node.embedding for n in out])
    baseline = min_pairwise_cos_dist(
        [n.node.embedding for n in _topk_by_score(nodes, 4)]
    )
    assert got > baseline


def test_llamaindex_unknown_utility_after_construction_raises():
    pytest.importorskip("llama_index.core")
    from divsel.adapters.llamaindex import DivselNodePostprocessor

    ns = _li()
    pp = DivselNodePostprocessor(k=3)
    object.__setattr__(pp, "utility", "coverage")
    with pytest.raises(ValueError, match="utility='linear' or 'facility_location'"):
        pp.postprocess_nodes(ns.make_nodes(), query_str=QUERY_TEXT)


def test_llamaindex_negative_scores_are_shifted_non_negative():
    """Retrievers may report negative relevance; divsel's weights may not be."""
    pytest.importorskip("llama_index.core")
    from divsel.adapters.llamaindex import DivselNodePostprocessor

    ns = _li()
    nodes = ns.make_nodes()
    for i, node in enumerate(nodes):
        node.score = SCORES[i] - 1.5  # every score now in [-0.65, -0.5]
    vectors = np.ascontiguousarray([n.node.embedding for n in nodes], dtype=np.float32)

    w = DivselNodePostprocessor._linear_weights(nodes, vectors, None)
    assert (w >= 0.0).all()
    assert w.min() == 0.0
    # The shift is a translation: differences survive it exactly.
    assert np.allclose(np.diff(w), np.diff([n.score for n in nodes]))

    # The shifted weights are tiny next to lam, so GIST legitimately returns
    # fewer than k here (a singleton's div is d_max); what matters is that the
    # negative scores went through the core at all instead of being rejected as
    # negative weights.
    out = DivselNodePostprocessor(k=4, lam=1.0).postprocess_nodes(nodes, query_str=QUERY_TEXT)
    assert 0 < len(out) <= 4
    assert all(n in nodes for n in out)


def test_llamaindex_zero_norm_vectors_do_not_divide_by_zero():
    pytest.importorskip("llama_index.core")
    from divsel.adapters.llamaindex import DivselNodePostprocessor

    ns = _li()
    nodes = ns.make_nodes(with_scores=False)
    vectors = np.zeros((len(nodes), 4), dtype=np.float32)
    bundle = ns.QueryBundle(query_str=QUERY_TEXT, embedding=[0.0, 0.0, 0.0, 0.0])

    w = DivselNodePostprocessor._linear_weights(nodes, vectors, bundle)
    assert w is not None
    assert np.isfinite(w).all()
    # cos is 0 against a zero query, so every weight is the shift itself.
    assert np.allclose(w, 1.0)


# --------------------------------------------------------------------------- #
# Round-3 gaps: the embedding hooks that can fail without raising              #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "returned,reason",
    [
        (None, "get_text_embedding_batch is not implemented"),
        ([], "returned 0 vectors for 12 nodes"),
        ("short", "returned 3 vectors for 12 nodes"),
        ("long", "returned 13 vectors for 12 nodes"),
    ],
)
def test_llamaindex_falls_back_when_the_embed_model_misbehaves(returned, reason):
    """The LlamaIndex twin of the LangChain hook guards.

    ``get_text_embedding_batch`` was called unguarded, so a model that does not
    implement it raised ``NotImplementedError`` past BOTH the warn path and the
    ``strict=True`` ``ValueError`` path -- the defect round 2 repaired on the
    LangChain side. A model that returns the wrong number of vectors is the
    same class of failure: an empty list made numpy raise ``AxisError`` deep
    inside the core, and a short one silently diversified over a prefix of the
    nodes.
    """
    pytest.importorskip("llama_index.core")
    from divsel.adapters import DivselFallbackWarning
    from divsel.adapters.llamaindex import DivselNodePostprocessor

    ns = _li()
    nodes = ns.make_nodes(with_embeddings=False)

    class Misbehaving(ns.TinyEmbedModel):
        def get_text_embedding_batch(self, texts, **kwargs):
            if returned is None:
                raise NotImplementedError
            if returned == "short":
                return [TABLE[t] for t in texts[:3]]
            if returned == "long":
                return [TABLE[t] for t in texts] + [TABLE[texts[0]]]
            return list(returned)

    pp = DivselNodePostprocessor(k=4, embed_model=Misbehaving())
    with pytest.warns(DivselFallbackWarning, match=reason):
        out = pp.postprocess_nodes(nodes, query_str=QUERY_TEXT)
    assert out == _topk_by_score(nodes, 4)

    strict = DivselNodePostprocessor(k=4, embed_model=Misbehaving(), strict=True)
    with pytest.raises(ValueError, match=reason):
        strict.postprocess_nodes(nodes, query_str=QUERY_TEXT)


@pytest.mark.parametrize(
    "count,reason",
    [
        (0, "returned 0 vectors for 12 documents"),
        (3, "returned 3 vectors for 12 documents"),
        (13, "returned 13 vectors for 12 documents"),
    ],
)
def test_langchain_falls_back_when_embed_documents_returns_the_wrong_count(count, reason):
    """A return whose length does not match ``docs`` is a failure, not a shape.

    Measured before this guard: an empty list reached
    ``np.linalg.norm(v, axis=1)`` on a 1-D array and raised a raw
    ``numpy.AxisError`` (and, under ``facility_location``, the binding's
    ``TypeError`` about C-contiguous float32 -- two different errors for one
    input); three vectors for six documents diversified over the first three
    with no warning at all; and a longer return let ``selected`` index past the
    end of ``docs`` with an ``IndexError``.
    """
    pytest.importorskip("langchain_core")
    from divsel.adapters import DivselFallbackWarning
    from divsel.adapters.langchain import DivselRetriever

    ns = _lc_edge()

    class WrongCount(ns.Partial):
        def embed_documents(self, texts):
            vectors = [TABLE[t] for t in texts]
            if count <= len(vectors):
                return vectors[:count]
            return vectors + [vectors[0]] * (count - len(vectors))

    store = ns.Store(WrongCount())
    retriever = DivselRetriever(vectorstore=store, k=4)
    with pytest.warns(DivselFallbackWarning, match=reason):
        docs = retriever.invoke(QUERY_TEXT)
    assert [d.page_content for d in docs] == [TEXTS[i] for i in RELEVANCE_ORDER[:4]]

    strict = DivselRetriever(vectorstore=store, k=4, strict=True)
    with pytest.raises(ValueError, match=reason):
        strict.invoke(QUERY_TEXT)


def test_min_eps_is_the_cores_own_floor_and_is_accepted():
    """``MIN_EPS`` is a hand-copied constant; nothing pinned it to its source.

    Both adapters advertise ``eps`` in ``[MIN_EPS, 1.0]`` because that is the
    range :func:`divsel.gist_select` serves. The existing parametrisations only
    assert rejection *below* the floor, so a typo that narrowed (or widened)
    MIN_EPS would leave every test passing while legitimate configurations
    started failing at construction.
    """
    from divsel import gist_select
    from divsel.adapters import MIN_EPS

    assert MIN_EPS == float(np.finfo(np.float32).eps) == 2.0**-23
    # The library itself accepts exactly that value, and nothing below it.
    x = np.ascontiguousarray(_fixture_vectors(), dtype=np.float32)
    assert gist_select(x, None, k=2, eps=MIN_EPS, metric="cosine")
    with pytest.raises(ValueError):
        gist_select(x, None, k=2, eps=MIN_EPS * 0.5, metric="cosine")

    def installed(module: str) -> bool:
        # `find_spec("llama_index.core")` raises when the parent package is
        # missing, which is exactly the venv this has to be silent in.
        try:
            return importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            return False

    if installed("langchain_core"):
        from divsel.adapters.langchain import DivselRetriever

        ns = _lc_edge()
        docs = DivselRetriever(
            vectorstore=ns.Store(ns.Partial()), k=4, eps=MIN_EPS
        ).invoke(QUERY_TEXT)
        assert len(docs) <= 4
    if installed("llama_index.core"):
        from divsel.adapters.llamaindex import DivselNodePostprocessor

        ns = _li()
        out = DivselNodePostprocessor(k=4, eps=MIN_EPS).postprocess_nodes(
            ns.make_nodes(), query_str=QUERY_TEXT
        )
        assert len(out) <= 4
