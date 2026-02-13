from __future__ import annotations

from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from src.models import DocumentChunk


def build_or_update_vectorstore(
    chunks: list[DocumentChunk],
    embedding_model,
    persist_dir: str | Path,
) -> Chroma:
    vectorstore = Chroma(
        embedding_function=embedding_model,
        persist_directory=str(persist_dir),
    )

    if not chunks:
        return vectorstore

    docs = [
        Document(page_content=chunk.text, metadata={**chunk.metadata, "id": chunk.id})
        for chunk in chunks
    ]
    ids = [chunk.id for chunk in chunks]

    # Chroma handles upsert-like behavior by id.
    vectorstore.add_documents(documents=docs, ids=ids)
    return vectorstore


def load_vectorstore(persist_dir: str | Path, embedding_model) -> Chroma:
    return Chroma(
        embedding_function=embedding_model,
        persist_directory=str(persist_dir),
    )


def clear_vectorstore(persist_dir: str | Path) -> None:
    path = Path(persist_dir)
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_file():
            child.unlink()
        elif child.is_dir():
            for nested in child.rglob("*"):
                if nested.is_file():
                    nested.unlink()
            for nested_dir in sorted([p for p in child.rglob("*") if p.is_dir()], reverse=True):
                nested_dir.rmdir()
            child.rmdir()
