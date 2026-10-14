"""Chroma vector store (replaces Spring AI's ChromaVectorStore)."""

from __future__ import annotations

from typing import Any

from .domain import Document
from .llm import EmbeddingModel, embed_one


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Chroma only accepts str/int/float/bool metadata values."""
    return {
        k: (v if isinstance(v, (str, int, float, bool)) else str(v))
        for k, v in metadata.items()
        if v is not None
    }


class ChromaVectorStore:
    def __init__(self, host: str, port: int, collection: str, embedding_model: EmbeddingModel) -> None:
        import chromadb

        self._client = chromadb.HttpClient(host=host, port=port)
        # Cosine space so that similarity = 1 - distance, matching Spring AI's threshold semantics.
        self._collection = self._client.get_or_create_collection(
            name=collection, metadata={"hnsw:space": "cosine"}
        )
        self._embedding_model = embedding_model

    def add(self, documents: list[Document]) -> None:
        if not documents:
            return
        self._collection.add(
            ids=[d.id for d in documents],
            documents=[d.text for d in documents],
            metadatas=[_clean_metadata(d.metadata) for d in documents],
            embeddings=self._embedding_model.embed_many([d.text for d in documents]),
        )

    def similarity_search(self, query: str, top_k: int, similarity_threshold: float) -> list[Document]:
        result = self._collection.query(
            query_embeddings=[embed_one(self._embedding_model, query)],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs: list[Document] = []
        for doc_id, text, meta, distance in zip(
            result["ids"][0], result["documents"][0], result["metadatas"][0], result["distances"][0]
        ):
            score = 1.0 - distance
            if score >= similarity_threshold:
                docs.append(Document(id=doc_id, text=text, metadata={**(meta or {}), "score": score}))
        return docs
