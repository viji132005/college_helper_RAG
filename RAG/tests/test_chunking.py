from __future__ import annotations

import pytest

from src.chunking import chunk_records


def test_chunk_records_preserves_metadata() -> None:
    records = [{"text": "A " * 1200, "source_file": "sample.txt", "page_number": None}]
    chunks = chunk_records(records, chunk_size_tokens=100, overlap_tokens=10)
    assert chunks
    assert chunks[0].source_file == "sample.txt"
    assert chunks[0].metadata["chunk_index"] == 0
    assert all(c.text for c in chunks)


def test_chunk_records_ignores_empty() -> None:
    chunks = chunk_records([{"text": "", "source_file": "x.txt", "page_number": None}])
    assert chunks == []
