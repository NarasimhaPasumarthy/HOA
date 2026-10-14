"""Domain types (replaces Spring AI's Document and HoaDocumentPayload record)."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Document:
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(frozen=True)
class HoaDocumentPayload:
    title: str
    content: str
    mime_type: str
