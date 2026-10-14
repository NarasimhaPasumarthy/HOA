"""Core HOA support logic (port of HoaSupportService.java + HoaDocumentLoader.java)."""

from __future__ import annotations

import hashlib
import logging
import time
from pathlib import Path
from typing import Protocol

from .domain import Document
from .llm import ChatModel
from .pdf_utils import OcrExtractor, PdfOcrDetector, read_pdf_pages
from .splitter import SemanticDocumentSplitter

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """Use the following context to answer the question.

Context:
{context}
"""

ERROR_MESSAGE = "I could not process that HOA question right now. Please try again later."


class VectorStore(Protocol):
    def add(self, documents: list[Document]) -> None: ...
    def similarity_search(self, query: str, top_k: int, similarity_threshold: float) -> list[Document]: ...


class KeywordIndex(Protocol):
    def add(self, documents: list[Document]) -> None: ...
    def search(self, query_text: str, top_k: int) -> list[Document]: ...


class ChecksumStore(Protocol):
    def has(self, checksum: str) -> bool: ...
    def set(self, checksum: str, file_name: str) -> None: ...


class RedisChecksumStore:
    """Redis-backed ingestion de-duplication (replaces RedisTemplate usage)."""

    def __init__(self, host: str, port: int, password: str = "", db: int = 0) -> None:
        import redis

        self._redis = redis.Redis(host=host, port=port, password=password or None, db=db, decode_responses=True)

    def has(self, checksum: str) -> bool:
        return bool(self._redis.exists(checksum))

    def set(self, checksum: str, file_name: str) -> None:
        self._redis.set(checksum, file_name)


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reciprocal_rank_fusion(result_lists: list[list[Document]], limit: int, k: int = 60) -> list[Document]:
    """score(doc) = sum over lists of 1 / (k + rank). Documents are matched by id."""
    scores: dict[str, float] = {}
    documents: dict[str, Document] = {}
    for results in result_lists:
        for rank, document in enumerate(results, start=1):
            documents.setdefault(document.id, document)
            scores[document.id] = scores.get(document.id, 0.0) + 1.0 / (k + rank)
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    return [documents[doc_id] for doc_id, _ in ranked]


class HoaSupportService:
    def __init__(
        self,
        splitter: SemanticDocumentSplitter,
        vector_store: VectorStore,
        keyword_index: KeywordIndex,
        chat_model: ChatModel,
        checksum_store: ChecksumStore,
        ocr_detector: PdfOcrDetector | None = None,
        ocr_extractor: OcrExtractor | None = None,
        ocr_enabled: bool = True,
        top_k: int = 10,
        similarity_threshold: float = 0.3,
        rrf_k: int = 60,
        metrics=None,
    ) -> None:
        self.splitter = splitter
        self.vector_store = vector_store
        self.keyword_index = keyword_index
        self.chat_model = chat_model
        self.checksum_store = checksum_store
        self.ocr_detector = ocr_detector or PdfOcrDetector()
        self.ocr_extractor = ocr_extractor or OcrExtractor()
        self.ocr_enabled = ocr_enabled
        self.top_k = top_k
        self.similarity_threshold = similarity_threshold
        self.rrf_k = rrf_k
        self.metrics = metrics

    # ------------------------------------------------------------------ search (MCP: hoa-document-search)

    def retrieve(self, query: str) -> list[Document]:
        vector_results = self.vector_store.similarity_search(query, self.top_k, self.similarity_threshold)
        logger.debug("Vector search results: %s", vector_results)
        keyword_results = self.keyword_index.search(query, self.top_k)
        logger.debug("BM25 search results: %s", keyword_results)
        fused = reciprocal_rank_fusion([vector_results, keyword_results], self.top_k, self.rrf_k)
        logger.debug("Fused search results: %s", fused)
        return fused

    def get_hoa_support_info(self, query: str) -> str:
        started = time.perf_counter()
        try:
            logger.info("HOA search request: %s", query)
            fused = self.retrieve(query)
            context = "\n\n".join(d.text for d in fused)
            answer, _usage = self.chat_model.chat(SYSTEM_PROMPT_TEMPLATE.format(context=context), query)
            self._observe("search", "success", started)
            return answer
        except Exception:  # noqa: BLE001 - return a friendly message to the agent
            logger.exception("Error processing HOA search request for query: %s", query)
            self._observe("search", "error", started)
            return ERROR_MESSAGE

    # ------------------------------------------------------------------ ingestion

    def split_and_store(self, text: str, metadata: dict) -> int:
        chunks = [c for c in self.splitter.split(Document(text=text, metadata=metadata)) if c.strip()]
        if not chunks:
            return 0
        # Same Document objects (same ids) go to both stores so RRF can fuse them.
        documents = [Document(text=chunk, metadata=dict(metadata)) for chunk in chunks]
        self.vector_store.add(documents)
        self.keyword_index.add(documents)
        return len(documents)

    def ingest_pdf(self, path: Path) -> int:
        """Extract, chunk and index one PDF. Returns the number of chunks stored."""
        total = 0
        for page in read_pdf_pages(path):
            if not page.text:
                continue
            metadata = {**page.metadata, "source": path.name, "type": "pdf"}
            try:
                total += self.split_and_store(page.text, metadata)
            except Exception:  # noqa: BLE001
                logger.exception("Error splitting and storing page %s of %s", page.metadata.get("page_number"), path.name)

        if self.ocr_enabled and self.ocr_detector.has_ocr_content(path):
            ocr_text = self.ocr_extractor.extract_text(path)
            if ocr_text.strip():
                total += self.split_and_store(ocr_text, {"source": path.name, "type": "pdf-ocr", "ocr": True})
                logger.info("OCR fallback used for %s", path.name)
            else:
                logger.warning("No text extracted for %s, even after OCR fallback.", path.name)

        logger.info("Ingested %d chunks from %s", total, path.name)
        return total

    def add_document(self, path_str: str) -> str:
        """MCP: hoa-document-add."""
        path = Path(path_str).expanduser()
        if not path.is_file():
            return "Document path is invalid or the file does not exist."
        if path.suffix.lower() != ".pdf":
            return "Only PDF documents are supported."

        checksum = sha256_of_file(path)
        logger.info("Checksum for file %s: %s", path.name, checksum)
        if self.checksum_store.has(checksum):
            logger.info("Skipping HOA PDF file (already processed): %s", path.name)
            return "Document has already been processed."

        logger.info("Processing HOA PDF file: %s", path.name)
        if self.ingest_pdf(path) > 0:
            self.checksum_store.set(checksum, path.name)
            return "Document added to vector store successfully."
        return "Document could not be added to vector store. No valid chunks were created."

    def load_folder(self, folder: str) -> None:
        """Startup ingestion of every PDF in a folder (HoaDocumentLoader.init)."""
        if not folder:
            logger.warning("Skipping HOA document ingestion because HOA_DOCUMENT_FOLDER_PATH is not configured.")
            return
        folder_path = Path(folder).expanduser()
        if not folder_path.is_dir():
            raise RuntimeError(f"HOA document folder path is invalid: {folder}")

        for path in sorted(p for p in folder_path.iterdir() if p.is_file() and p.suffix.lower() == ".pdf"):
            checksum = sha256_of_file(path)
            logger.info("Checksum for file %s: %s", path.name, checksum)
            if self.checksum_store.has(checksum):
                logger.info("Skipping HOA PDF file (already processed): %s", path.name)
                continue
            logger.info("Processing HOA PDF file: %s", path.name)
            # Mark as processed only after a successful ingest, so a failed file is retried next start.
            if self.ingest_pdf(path) > 0:
                self.checksum_store.set(checksum, path.name)
            logger.info("Finished processing HOA PDF file: %s", path.name)

    # ------------------------------------------------------------------ metrics

    def _observe(self, operation: str, status: str, started: float) -> None:
        if self.metrics is not None:
            self.metrics.record_request(operation, status, time.perf_counter() - started)
