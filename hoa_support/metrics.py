"""Optional Prometheus metrics (replaces Actuator + Micrometer).

If prometheus-client isn't installed, metrics are silently disabled.
"""

from __future__ import annotations

from .llm import TokenUsage

try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    _ENABLED = True
except ImportError:  # pragma: no cover
    _ENABLED = False


class Metrics:
    def __init__(self) -> None:
        self.enabled = _ENABLED
        if not self.enabled:
            return
        self._requests = Counter("hoa_tool_requests_total", "MCP tool calls", ["operation", "status"])
        self._latency = Histogram("hoa_tool_latency_seconds", "MCP tool latency", ["operation"])
        self._tokens = Counter("hoa_llm_tokens_total", "LLM tokens used", ["kind"])

    def record_request(self, operation: str, status: str, seconds: float) -> None:
        if self.enabled:
            self._requests.labels(operation, status).inc()
            self._latency.labels(operation).observe(seconds)

    def record_tokens(self, usage: TokenUsage) -> None:
        if self.enabled:
            self._tokens.labels("prompt").inc(usage.prompt_tokens)
            self._tokens.labels("completion").inc(usage.completion_tokens)

    def render(self) -> tuple[bytes, str]:
        if not self.enabled:
            return b"# prometheus-client not installed\n", "text/plain"
        return generate_latest(), CONTENT_TYPE_LATEST
