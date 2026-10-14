"""Application settings, read from environment variables.

Mirrors src/main/resources/application.yaml from the Java project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str = "") -> str:
    value = os.getenv(name)
    return default if value is None or value.strip() == "" else value.strip()


def _env_bool(name: str, default: bool) -> bool:
    return _env(name, str(default)).lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    return int(_env(name, str(default)))


def _load_dotenv(path: Path = Path(".env")) -> None:
    """Minimal .env loader so python-dotenv isn't required."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


@dataclass(frozen=True)
class Settings:
    # Server (server.port / spring.ai.mcp.server.*)
    server_name: str = "ai-hoa-support"
    server_version: str = "0.0.1"
    host: str = "0.0.0.0"
    port: int = 8083
    mcp_path: str = "/mcp"

    # Model provider (app.embedding.llm) — selects BOTH chat and embedding models
    llm_provider: str = "openai"  # "openai" | "ollama"
    chat_model: str = ""
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-large"
    ollama_base_url: str = "http://localhost:11434"
    ollama_embedding_model: str = "mxbai-embed-large"
    ollama_pull_models: bool = True  # spring.ai.ollama.init.pull-model-strategy: always
    log_prompts: bool = False  # spring.ai.chat.client.observations.log-prompt

    # Chroma (spring.ai.vectorstore.chroma.*)
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "hoa-documents"

    # Redis (spring.data.redis.*)
    redis_host: str = "localhost"
    redis_port: int = 6379
    redis_password: str = ""
    redis_db: int = 0

    # BM25 keyword index (lucene.index.path)
    bm25_index_path: str = "data/bm25-index.sqlite3"

    # HOA ingestion + OCR (hoa.*)
    document_folder_path: str = ""
    document_loading_enabled: bool = False
    ocr_enabled: bool = True
    ocr_language: str = "eng"
    ocr_tesseract_path: str = ""
    ocr_tessdata_path: str = ""

    # Retrieval tuning (hard-coded in the Java service)
    top_k: int = 10
    similarity_threshold: float = 0.3
    rrf_k: int = 60
    split_similarity_threshold: float = 0.72

    @property
    def resolved_chat_model(self) -> str:
        if self.chat_model:
            return self.chat_model
        return "gpt-4o-mini" if self.llm_provider == "openai" else "llama3.1"

    @property
    def embedding_model(self) -> str:
        return (
            self.openai_embedding_model
            if self.llm_provider == "openai"
            else self.ollama_embedding_model
        )

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        return cls(
            host=_env("HOST", cls.host),
            port=_env_int("PORT", cls.port),
            llm_provider=_env("LLM_PROVIDER", cls.llm_provider).lower(),
            chat_model=_env("CHAT_MODEL_NAME"),
            openai_api_key=_env("OPENAI_API_KEY"),
            openai_embedding_model=_env("OPENAI_EMBEDDING_MODEL", cls.openai_embedding_model),
            ollama_base_url=_env("OLLAMA_BASE_URL", cls.ollama_base_url),
            ollama_embedding_model=_env("OLLAMA_EMBEDDING_MODEL", cls.ollama_embedding_model),
            ollama_pull_models=_env_bool("OLLAMA_PULL_MODELS", cls.ollama_pull_models),
            log_prompts=_env_bool("LOG_PROMPTS", cls.log_prompts),
            chroma_host=_env("CHROMA_HOST", cls.chroma_host),
            chroma_port=_env_int("CHROMA_PORT", cls.chroma_port),
            chroma_collection=_env("CHROMA_COLLECTION", cls.chroma_collection),
            redis_host=_env("REDIS_HOST", cls.redis_host),
            redis_port=_env_int("REDIS_PORT", cls.redis_port),
            redis_password=_env("REDIS_PASSWORD"),
            redis_db=_env_int("REDIS_DB", cls.redis_db),
            bm25_index_path=_env("BM25_INDEX_PATH", cls.bm25_index_path),
            document_folder_path=_env("HOA_DOCUMENT_FOLDER_PATH"),
            document_loading_enabled=_env_bool("HOA_DOCUMENT_LOADING_ENABLED", False),
            ocr_enabled=_env_bool("HOA_OCR_ENABLED", True),
            ocr_language=_env("HOA_OCR_LANGUAGE", cls.ocr_language),
            ocr_tesseract_path=_env("HOA_OCR_TESSERACT_PATH"),
            ocr_tessdata_path=_env("HOA_OCR_TESSDATA_PATH"),
        )
