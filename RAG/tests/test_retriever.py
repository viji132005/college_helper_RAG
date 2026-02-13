from __future__ import annotations

from langchain_chroma import Chroma
from langchain_community.embeddings import FakeEmbeddings

from src.chunking import chunk_records
from src.retriever import retrieve
from src.vector_store import build_or_update_vectorstore


def test_retrieve_top_match(tmp_path) -> None:
    records = [
        {"text": "Calculus studies derivatives and integrals.", "source_file": "math.txt", "page_number": None},
        {"text": "Biology studies living organisms and cells.", "source_file": "bio.txt", "page_number": None},
    ]
    chunks = chunk_records(records, chunk_size_tokens=200, overlap_tokens=0)
    embedding = FakeEmbeddings(size=32)
    vs = build_or_update_vectorstore(chunks, embedding, tmp_path / "db")
    results = retrieve("What are derivatives?", vs, k=1)
    assert len(results) == 1
    assert results[0].chunk.source_file in {"math.txt", "bio.txt"}
