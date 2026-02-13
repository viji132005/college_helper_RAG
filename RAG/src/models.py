from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class DocumentChunk:
    id: str
    text: str
    source_file: str
    page_number: int | None
    chunk_index: int
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class RetrievalResult:
    chunk: DocumentChunk
    score: float


@dataclass(slots=True)
class RAGAnswer:
    answer_text: str
    sources: list[RetrievalResult]
    used_model: str
    warnings: list[str] = field(default_factory=list)
