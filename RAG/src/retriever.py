from __future__ import annotations

from langchain_core.documents import Document

from src.models import DocumentChunk, RetrievalResult


def retrieve(query: str, vectorstore, k: int = 5) -> list[RetrievalResult]:
    raw_results: list[tuple[Document, float]] = vectorstore.similarity_search_with_relevance_scores(query, k=k)
    output: list[RetrievalResult] = []

    for doc, score in raw_results:
        metadata = doc.metadata or {}
        chunk = DocumentChunk(
            id=str(metadata.get("id", "")),
            text=doc.page_content,
            source_file=str(metadata.get("source_file", "unknown")),
            page_number=metadata.get("page_number"),
            chunk_index=int(metadata.get("chunk_index", 0)),
            metadata=metadata,
        )
        output.append(RetrievalResult(chunk=chunk, score=float(score)))

    return output
