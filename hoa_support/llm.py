"""Chat and embedding model clients for OpenAI and Ollama.

Replaces AppConfig's EmbeddingModel/ChatModel beans and TokenUsageAdvisor.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from .config import Settings

logger = logging.getLogger(__name__)


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class EmbeddingModel(Protocol):
    def embed_many(self, texts: list[str]) -> list[list[float]]: ...


class ChatModel(Protocol):
    def chat(self, system: str, user: str) -> tuple[str, TokenUsage]: ...


def embed_one(model: EmbeddingModel, text: str) -> list[float]:
    return model.embed_many([text])[0]


# --------------------------------------------------------------------------- OpenAI


class OpenAIEmbeddingModel:
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or None)
        self._model = model

    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        out: list[list[float]] = []
        for start in range(0, len(texts), 256):  # stay well under the request limit
            resp = self._client.embeddings.create(model=self._model, input=texts[start : start + 256])
            out.extend(item.embedding for item in sorted(resp.data, key=lambda d: d.index))
        return out


class OpenAIChatModel:
    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key or None)
        self._model = model

    def chat(self, system: str, user: str) -> tuple[str, TokenUsage]:
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        usage = TokenUsage(
            prompt_tokens=getattr(resp.usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(resp.usage, "completion_tokens", 0) or 0,
        )
        return resp.choices[0].message.content or "", usage


# --------------------------------------------------------------------------- Ollama


class _OllamaBase:
    def __init__(self, base_url: str, model: str, pull: bool) -> None:
        import httpx

        self._http = httpx.Client(base_url=base_url.rstrip("/"), timeout=300)
        self._model = model
        if pull:
            logger.info("Pulling Ollama model %s", model)
            self._http.post("/api/pull", json={"model": model, "stream": False}).raise_for_status()


class OllamaEmbeddingModel(_OllamaBase):
    def embed_many(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self._http.post("/api/embed", json={"model": self._model, "input": texts})
        resp.raise_for_status()
        return resp.json()["embeddings"]


class OllamaChatModel(_OllamaBase):
    def chat(self, system: str, user: str) -> tuple[str, TokenUsage]:
        resp = self._http.post(
            "/api/chat",
            json={
                "model": self._model,
                "stream": False,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            },
        )
        resp.raise_for_status()
        body = resp.json()
        usage = TokenUsage(body.get("prompt_eval_count", 0), body.get("eval_count", 0))
        return body["message"]["content"], usage


# --------------------------------------------------------------------------- Factory + advisor


class TokenUsageLoggingChatModel:
    """Equivalent of TokenUsageAdvisor: wraps a chat model and logs token usage per call."""

    def __init__(self, inner: ChatModel, log_prompts: bool = False, metrics=None) -> None:
        self._inner = inner
        self._log_prompts = log_prompts
        self._metrics = metrics

    def chat(self, system: str, user: str) -> tuple[str, TokenUsage]:
        if self._log_prompts:
            logger.info("Prompt system=%r user=%r", system, user)
        text, usage = self._inner.chat(system, user)
        logger.info(
            "Token usage for the call: %d prompt tokens, %d completion tokens, %d total tokens.",
            usage.prompt_tokens,
            usage.completion_tokens,
            usage.total_tokens,
        )
        if self._metrics is not None:
            self._metrics.record_tokens(usage)
        if self._log_prompts:
            logger.info("Completion: %r", text)
        return text, usage


def build_models(settings: Settings, metrics=None) -> tuple[EmbeddingModel, ChatModel]:
    if settings.llm_provider == "openai":
        embedding: EmbeddingModel = OpenAIEmbeddingModel(settings.openai_api_key, settings.openai_embedding_model)
        chat: ChatModel = OpenAIChatModel(settings.openai_api_key, settings.resolved_chat_model)
    elif settings.llm_provider == "ollama":
        embedding = OllamaEmbeddingModel(
            settings.ollama_base_url, settings.ollama_embedding_model, settings.ollama_pull_models
        )
        chat = OllamaChatModel(settings.ollama_base_url, settings.resolved_chat_model, settings.ollama_pull_models)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER '{settings.llm_provider}' (expected 'openai' or 'ollama')")
    return embedding, TokenUsageLoggingChatModel(chat, settings.log_prompts, metrics)
