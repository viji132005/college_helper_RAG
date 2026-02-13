from __future__ import annotations

from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.models import DocumentChunk
from src.utils import sha1_text


def chunk_records(
    records: list[dict[str, Any]],
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 50,
) -> list[DocumentChunk]:
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size_tokens,
        chunk_overlap=overlap_tokens,
    )

    output: list[DocumentChunk] = []

    for record in records:
        text = (record.get("text") or "").strip()
        if not text:
            continue

        source_file = record.get("source_file", "unknown")
        page_number = record.get("page_number")

        pieces = splitter.split_text(text)
        for idx, piece in enumerate(pieces):
            metadata = {
                "source_file": source_file,
                "page_number": page_number,
                "chunk_index": idx,
            }
            chunk_id = sha1_text(f"{source_file}|{page_number}|{idx}|{piece}")
            output.append(
                DocumentChunk(
                    id=chunk_id,
                    text=piece,
                    source_file=source_file,
                    page_number=page_number,
                    chunk_index=idx,
                    metadata=metadata,
                )
            )

    return output
