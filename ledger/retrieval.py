"""
ledger/retrieval.py — Stage 6: find the chunks that answer a question.

Four search methods, each built on the previous one:
  "vector"         meaning-based search (embeddings + Qdrant)
  "bm25"           keyword search (exact words and numbers)
  "hybrid"         both, combined with Reciprocal Rank Fusion (RRF)
  "hybrid_rerank"  hybrid, then re-sorted by a cross-encoder reranker

Plus metadata FILTERS (ticker, fiscal year) applied before searching,
which is the fix for the "2023 revenue vs 2024 revenue" problem.
"""
import json
import re
from functools import lru_cache

from ledger import config
from ledger.chunk import context_header

# Very common words that carry no search signal.
STOPWORDS = {
    "a", "an", "the", "of", "in", "on", "for", "to", "and", "or", "is", "was",
    "were", "what", "how", "did", "does", "do", "by", "with", "as", "at", "its",
    "it", "be", "are", "from", "that", "this", "which", "their", "during",
}
# A "token" for keyword search: letters/digits, keeping numbers like
# 391,035 and 7.8 in one piece so they can be matched exactly.
TOKEN = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")


def tokenize(text: str) -> list[str]:
    return [t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS]


def reciprocal_rank_fusion(rankings: list[list[str]], k: int = 60) -> list[str]:
    """
    Merge several ranked lists into one.

    Each item earns 1 / (k + rank) from every list it appears in
    (rank 0 = top). Items ranked high in BOTH lists win. k=60 is the
    standard value; it stops the #1 spot from dominating everything.
    """
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores, key=scores.get, reverse=True)


def load_chunks() -> list[dict]:
    if not config.CHUNKS_PATH.exists():
        raise SystemExit("No index found. Run `uv run python -m ledger.index` first.")
    with open(config.CHUNKS_PATH, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class Retriever:
    """
    Holds everything search needs. Loading models is slow (seconds), so
    create ONE Retriever and reuse it — see get_retriever() at the bottom.
    """

    def __init__(self, chunks: list[dict] | None = None, embedder=None, qdrant=None):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks if chunks is not None else load_chunks()
        self.by_id = {c["chunk_id"]: c for c in self.chunks}
        # Keyword index. We include the context header so a chunk's company
        # and year are searchable words too.
        self.bm25 = BM25Okapi([tokenize(context_header(c) + " " + c["text"]) for c in self.chunks])
        # The heavy models load lazily: only when first used.
        self._embedder = embedder
        self._qdrant = qdrant
        self._reranker = None

    # ---- lazy-loaded models ------------------------------------------------
    @property
    def embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(config.EMBED_MODEL)
        return self._embedder

    @property
    def qdrant(self):
        if self._qdrant is None:
            from ledger.index import make_qdrant_client
            self._qdrant = make_qdrant_client()
        return self._qdrant

    @property
    def reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(config.RERANK_MODEL)
        return self._reranker

    # ---- information about what's indexed --------------------------------
    def available_filings(self) -> dict[str, list[int]]:
        """{"AAPL": [2022, 2023, 2024], ...}"""
        found: dict[str, set[int]] = {}
        for c in self.chunks:
            found.setdefault(c["ticker"], set()).add(c["fiscal_year"])
        return {t: sorted(years) for t, years in sorted(found.items())}

    def company_names(self) -> dict[str, str]:
        return {c["ticker"]: c["company"] for c in self.chunks}

    # ---- the individual search methods -----------------------------------
    @staticmethod
    def _passes(chunk: dict, tickers, years) -> bool:
        return ((not tickers or chunk["ticker"] in tickers)
                and (not years or chunk["fiscal_year"] in years))

    def vector_search(self, query: str, tickers=None, years=None,
                      limit: int = config.RETRIEVE_CANDIDATES) -> list[str]:
        from qdrant_client import models

        vector = self.embedder.encode(config.EMBED_QUERY_PREFIX + query,
                                      normalize_embeddings=True)
        # Filters run INSIDE Qdrant, before similarity ranking, so a 2023
        # chunk can never outrank a 2024 chunk when we asked for 2024.
        conditions = []
        if tickers:
            conditions.append(models.FieldCondition(
                key="ticker", match=models.MatchAny(any=list(tickers))))
        if years:
            conditions.append(models.FieldCondition(
                key="fiscal_year", match=models.MatchAny(any=[int(y) for y in years])))
        response = self.qdrant.query_points(
            collection_name=config.QDRANT_COLLECTION,
            query=vector.tolist(),
            query_filter=models.Filter(must=conditions) if conditions else None,
            limit=limit,
            with_payload=True,
        )
        return [point.payload["chunk_id"] for point in response.points]

    def bm25_search(self, query: str, tickers=None, years=None,
                    limit: int = config.RETRIEVE_CANDIDATES) -> list[str]:
        scores = self.bm25.get_scores(tokenize(query))   # one score per chunk
        results = []
        for i in scores.argsort()[::-1]:                  # highest score first
            if self._passes(self.chunks[i], tickers, years):
                results.append(self.chunks[i]["chunk_id"])
                if len(results) >= limit:
                    break
        return results

    def rerank(self, query: str, chunk_ids: list[str], top_n: int) -> list[str]:
        if not chunk_ids:
            return []
        # The cross-encoder reads the question and the passage TOGETHER,
        # which is slower but much sharper than comparing two vectors.
        pairs = [(query, context_header(self.by_id[i]) + "\n" + self.by_id[i]["text"])
                 for i in chunk_ids]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(chunk_ids, scores), key=lambda pair: pair[1], reverse=True)
        return [chunk_id for chunk_id, _ in ranked[:top_n]]

    # ---- the one method the rest of the project calls ---------------------
    def search(self, query: str, tickers=None, years=None,
               k: int = config.EVIDENCE_PER_QUERY,
               mode: str = "hybrid", use_filters: bool = True) -> list[dict]:
        if not use_filters:
            tickers = years = None
        if mode == "vector":
            ids = self.vector_search(query, tickers, years)[:k]
        elif mode == "bm25":
            ids = self.bm25_search(query, tickers, years)[:k]
        elif mode in ("hybrid", "hybrid_rerank"):
            fused = reciprocal_rank_fusion([
                self.vector_search(query, tickers, years),
                self.bm25_search(query, tickers, years),
            ])
            if mode == "hybrid":
                ids = fused[:k]
            else:
                ids = self.rerank(query, fused[:config.RETRIEVE_CANDIDATES], k)
        else:
            raise ValueError(f"unknown mode: {mode}")
        return [self.by_id[i] for i in ids]


@lru_cache(maxsize=1)
def get_retriever() -> Retriever:
    """
    Return the one shared Retriever. @lru_cache means: the first call builds
    it, every later call returns that same object instantly.
    """
    return Retriever()
