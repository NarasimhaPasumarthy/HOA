"""Shared in-memory fakes so tests run without OpenAI/Ollama, Chroma or Redis."""

from __future__ import annotations

import hashlib
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hoa_support.domain import Document  # noqa: E402
from hoa_support.llm import TokenUsage  # noqa: E402
from hoa_support.splitter import cosine_similarity  # noqa: E402


class FakeEmbeddingModel:
    """Hashed bag-of-words vectors: sentences sharing words are similar."""

    dims = 64

    def embed_many(self, texts):
        vectors = []
        for text in texts:
            vec = [0.0] * self.dims
            for word in re.findall(r"\w+", text.lower()):
                vec[int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dims] += 1.0
            vectors.append(vec)
        return vectors


class InMemoryVectorStore:
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
        self.items: list[tuple[Document, list[float]]] = []

    def add(self, documents):
        vectors = self.embedding_model.embed_many([d.text for d in documents])
        self.items.extend(zip(documents, vectors))

    def similarity_search(self, query, top_k, similarity_threshold):
        q = self.embedding_model.embed_many([query])[0]
        scored = [(cosine_similarity(q, v), d) for d, v in self.items]
        scored = [s for s in scored if s[0] >= similarity_threshold]
        scored.sort(key=lambda s: s[0], reverse=True)
        return [d for _, d in scored[:top_k]]


class FakeChatModel:
    def __init__(self):
        self.calls = []

    def chat(self, system, user):
        self.calls.append((system, user))
        return f"ANSWER for: {user}", TokenUsage(10, 5)


class DictChecksumStore:
    def __init__(self):
        self.data = {}

    def has(self, checksum):
        return checksum in self.data

    def set(self, checksum, file_name):
        self.data[checksum] = file_name


@pytest.fixture
def fakes():
    emb = FakeEmbeddingModel()
    return {
        "embedding": emb,
        "vector_store": InMemoryVectorStore(emb),
        "chat": FakeChatModel(),
        "checksums": DictChecksumStore(),
    }
