"""BM25 keyword index (replaces LucerneDocumentWriter + LucerneSearch).

Uses SQLite FTS5, which ships with Python and ranks with BM25 — no extra
service or dependency, and new documents are searchable immediately.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from pathlib import Path

from .domain import Document

logger = logging.getLogger(__name__)

_TOKEN = re.compile(r"\w+", re.UNICODE)


class BM25Index:
    def __init__(self, index_path: str | Path) -> None:
        path = Path(index_path)
        if str(path) != ":memory:":
            path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.execute(
            "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
            "id UNINDEXED, content, metadata UNINDEXED, tokenize='unicode61')"
        )
        self._conn.commit()

    def add(self, documents: list[Document]) -> None:
        rows = [(d.id, d.text, json.dumps(d.metadata, default=str)) for d in documents]
        with self._lock:
            self._conn.executemany("INSERT INTO chunks(id, content, metadata) VALUES (?, ?, ?)", rows)
            self._conn.commit()
        for d in documents:
            logger.debug("Added document to BM25 index: %s", d.metadata)

    def search(self, query_text: str, top_k: int) -> list[Document]:
        logger.info("Searching BM25 index for query: %s", query_text)
        # Like Lucene's QueryParser.escape + default OR operator: match any term.
        terms = {t.lower() for t in _TOKEN.findall(query_text or "")}
        if not terms:
            return []
        fts_query = " OR ".join(f'"{t}"' for t in sorted(terms))
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, content, metadata FROM chunks WHERE chunks MATCH ? "
                "ORDER BY bm25(chunks) LIMIT ?",
                (fts_query, top_k),
            ).fetchall()
        return [Document(id=r[0], text=r[1], metadata=json.loads(r[2])) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
