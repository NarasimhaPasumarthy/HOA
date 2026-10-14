"""Semantic chunking (port of SemanticDocumentSplitter.java).

Splits text into sentences and starts a new chunk whenever two neighbouring
sentences have cosine similarity below the threshold (0.72 by default).
"""

from __future__ import annotations

import math
import re

from .domain import Document
from .llm import EmbeddingModel

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+")


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class SemanticDocumentSplitter:
    def __init__(self, embedding_model: EmbeddingModel, threshold: float = 0.72) -> None:
        self._embedding_model = embedding_model
        self._threshold = threshold

    def split(self, document: Document) -> list[str]:
        sentences = [s for s in _SENTENCE_BOUNDARY.split(document.text or "") if s.strip()]
        if not sentences:
            return []
        if len(sentences) == 1:
            return sentences

        # The Java version embeds each sentence twice (once per neighbour pair);
        # embedding every sentence once in a single batch gives the same result.
        vectors = self._embedding_model.embed_many(sentences)

        chunks: list[str] = []
        current = [sentences[0]]
        for i in range(1, len(sentences)):
            if cosine_similarity(vectors[i - 1], vectors[i]) < self._threshold:
                chunks.append(" ".join(current))
                current = [sentences[i]]
            else:
                current.append(sentences[i])
        chunks.append(" ".join(current))
        return chunks
