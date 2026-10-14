"""MCP server entry point (replaces AiHoaSupportApplication + @McpTool annotations).

Run with:  python -m hoa_support.server
Endpoint:  http://localhost:8083/mcp  (streamable HTTP)
"""

from __future__ import annotations

import logging
from pathlib import Path

import anyio
from mcp.server.mcpserver import MCPServer
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, Response

from .bm25_index import BM25Index
from .config import Settings
from .llm import build_models
from .metrics import Metrics
from .pdf_utils import OcrExtractor, PdfOcrDetector
from .service import HoaSupportService, RedisChecksumStore
from .splitter import SemanticDocumentSplitter
from .vector_store import ChromaVectorStore

logger = logging.getLogger("hoa_support")

STATIC_DIR = Path(__file__).parent / "static"
MAX_QUESTION_LENGTH = 2000


def build_service(settings: Settings, metrics: Metrics | None = None) -> HoaSupportService:
    embedding_model, chat_model = build_models(settings, metrics)
    return HoaSupportService(
        splitter=SemanticDocumentSplitter(embedding_model, settings.split_similarity_threshold),
        vector_store=ChromaVectorStore(
            settings.chroma_host, settings.chroma_port, settings.chroma_collection, embedding_model
        ),
        keyword_index=BM25Index(settings.bm25_index_path),
        chat_model=chat_model,
        checksum_store=RedisChecksumStore(
            settings.redis_host, settings.redis_port, settings.redis_password, settings.redis_db
        ),
        ocr_detector=PdfOcrDetector(),
        ocr_extractor=OcrExtractor(settings.ocr_language, settings.ocr_tesseract_path, settings.ocr_tessdata_path),
        ocr_enabled=settings.ocr_enabled,
        top_k=settings.top_k,
        similarity_threshold=settings.similarity_threshold,
        rrf_k=settings.rrf_k,
        metrics=metrics,
    )


def create_server(service: HoaSupportService, settings: Settings, metrics: Metrics | None = None) -> MCPServer:
    server = MCPServer(name=settings.server_name, version=settings.server_version)

    @server.tool(name="hoa-document-search", description="Provides support information for HOA-related queries")
    async def hoa_document_search(query: str) -> str:
        return await anyio.to_thread.run_sync(service.get_hoa_support_info, query)

    @server.tool(name="hoa-document-add", description="Adds a new document to the HOA store")
    async def hoa_document_add(hoa_document_path: str) -> str:
        return await anyio.to_thread.run_sync(service.add_document, hoa_document_path)

    @server.custom_route("/", methods=["GET"], include_in_schema=False)
    async def chat_page(_: Request) -> Response:
        return FileResponse(STATIC_DIR / "index.html")

    @server.custom_route("/api/chat", methods=["POST"])
    async def chat(request: Request) -> Response:
        try:
            body = await request.json()
        except ValueError:
            return JSONResponse({"error": "Request body must be JSON."}, status_code=400)
        question = body.get("question") if isinstance(body, dict) else None
        if not isinstance(question, str) or not question.strip():
            return JSONResponse({"error": "Please enter a question."}, status_code=400)
        if len(question) > MAX_QUESTION_LENGTH:
            return JSONResponse({"error": f"Questions are limited to {MAX_QUESTION_LENGTH} characters."}, status_code=400)
        answer = await anyio.to_thread.run_sync(service.get_hoa_support_info, question.strip())
        return JSONResponse({"answer": answer})

    @server.custom_route("/health", methods=["GET"])
    async def health(_: Request) -> Response:
        return JSONResponse({"status": "UP"})

    @server.custom_route("/metrics", methods=["GET"])
    async def prometheus(_: Request) -> Response:
        body, content_type = (metrics or Metrics()).render()
        return Response(body, media_type=content_type)

    return server


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    settings = Settings.from_env()
    metrics = Metrics()
    service = build_service(settings, metrics)

    if settings.document_loading_enabled:
        service.load_folder(settings.document_folder_path)
    else:
        logger.info("Skipping HOA document ingestion because HOA_DOCUMENT_LOADING_ENABLED is false.")

    server = create_server(service, settings, metrics)
    logger.info("Starting MCP server on http://%s:%d%s", settings.host, settings.port, settings.mcp_path)
    logger.info("Chat UI available at http://localhost:%d/", settings.port)
    server.run(
        transport="streamable-http",
        host=settings.host,
        port=settings.port,
        streamable_http_path=settings.mcp_path,
    )


if __name__ == "__main__":
    main()
