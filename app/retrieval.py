"""Retrieval: dense (Chroma), sparse (BM25), hybrid (reciprocal rank fusion) and cross-encoder reranking."""
import re
from typing import Protocol

import chromadb
from rank_bm25 import BM25Okapi
from sqlalchemy import select

from .db import Chunk

MODES = ("vector", "bm25", "hybrid", "hybrid_rerank")


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class Reranker(Protocol):
    def score(self, query: str, texts: list[str]) -> list[float]: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, normalize_embeddings=True).tolist()


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(s) for s in self.model.predict([(query, t) for t in texts])]


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def rrf(rankings: list[list[int]], k: int = 60) -> list[int]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank)."""
    scores: dict[int, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))


class Retriever:
    def __init__(self, session_factory, embedder: Embedder, chroma_dir: str | None,
                 reranker: Reranker | None = None, reranker_factory=None):
        self.session_factory = session_factory
        self.embedder = embedder
        self._reranker = reranker
        self._reranker_factory = reranker_factory  # loaded lazily on first rerank request
        client = chromadb.PersistentClient(chroma_dir) if chroma_dir else chromadb.EphemeralClient()
        self.collection = client.get_or_create_collection("chunks", metadata={"hnsw:space": "cosine"})
        self._bm25: tuple[BM25Okapi, list[int], list[int]] | None = None

    # ---- indexing ----
    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        self.collection.add(
            ids=[str(c.id) for c in chunks],
            embeddings=self.embedder.embed([c.text for c in chunks]),
            metadatas=[{"document_id": c.document_id} for c in chunks],
        )
        self._bm25 = None

    def delete_document(self, document_id: int) -> None:
        self.collection.delete(where={"document_id": document_id})
        self._bm25 = None

    def _bm25_index(self):
        if self._bm25 is None:
            with self.session_factory() as s:
                rows = s.execute(select(Chunk.id, Chunk.document_id, Chunk.text).order_by(Chunk.id)).all()
            corpus = [tokenize(r.text) for r in rows]
            index = BM25Okapi(corpus) if corpus else None
            self._bm25 = (index, [r.id for r in rows], [r.document_id for r in rows])
        return self._bm25

    # ---- search ----
    def _vector(self, query: str, n: int, document_ids: list[int] | None) -> list[int]:
        total = self.collection.count()
        if total == 0:
            return []
        where = {"document_id": {"$in": document_ids}} if document_ids else None
        res = self.collection.query(
            query_embeddings=self.embedder.embed([query]), n_results=min(n, total), where=where, include=[]
        )
        return [int(i) for i in res["ids"][0]]

    def _sparse(self, query: str, n: int, document_ids: list[int] | None) -> list[int]:
        index, ids, doc_ids = self._bm25_index()
        if index is None:
            return []
        scores = index.get_scores(tokenize(query))
        allowed = set(document_ids) if document_ids else None
        ranked = sorted(
            (i for i in range(len(ids)) if scores[i] > 0 and (allowed is None or doc_ids[i] in allowed)),
            key=lambda i: -scores[i],
        )
        return [ids[i] for i in ranked[:n]]

    def search(self, query: str, mode: str = "hybrid_rerank", top_k: int = 5, candidate_k: int = 20,
               document_ids: list[int] | None = None) -> list[Chunk]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if mode == "vector":
            ids = self._vector(query, top_k, document_ids)
        elif mode == "bm25":
            ids = self._sparse(query, top_k, document_ids)
        else:
            ids = rrf([self._vector(query, candidate_k, document_ids),
                       self._sparse(query, candidate_k, document_ids)])[:candidate_k]
        if not ids:
            return []
        with self.session_factory() as s:
            by_id = {c.id: c for c in s.scalars(select(Chunk).where(Chunk.id.in_(ids)))}
            for c in by_id.values():
                _ = c.document.filename  # load while the session is open
        chunks = [by_id[i] for i in ids if i in by_id]
        if mode == "hybrid_rerank":
            if self._reranker is None and self._reranker_factory:
                self._reranker = self._reranker_factory()
            if self._reranker is not None:
                scores = self._reranker.score(query, [c.text for c in chunks])
                chunks = [c for _, c in sorted(zip(scores, chunks), key=lambda p: -p[0])]
        return chunks[:top_k]
