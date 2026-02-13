from __future__ import annotations

from src.models import RetrievalResult


def format_source_reference(index: int, result: RetrievalResult, preview_chars: int = 220) -> str:
    page = result.chunk.page_number if result.chunk.page_number is not None else "n/a"
    snippet = result.chunk.text.strip().replace("\n", " ")[:preview_chars]
    return (
        f"S{index} | {result.chunk.source_file} | page={page} | "
        f"chunk={result.chunk.chunk_index} | score={result.score:.3f}\n{snippet}"
    )
