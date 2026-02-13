from __future__ import annotations

from dataclasses import replace

from langchain_community.embeddings import FakeEmbeddings

from src.chunking import chunk_records
from src.config import AppConfig
from src.rag_pipeline import answer_query
from src.vector_store import build_or_update_vectorstore


def test_pipeline_smoke(tmp_path, monkeypatch) -> None:
    config = AppConfig(
        openai_api_key="x",
        groq_api_key="x",
        gemini_api_key="x",
        embedding_provider="gemini",
        chroma_persist_dir=tmp_path / "db",
        upload_dir=tmp_path / "uploads",
        chunk_size_tokens=500,
        chunk_overlap_tokens=50,
        retriever_top_k=5,
        retrieval_score_threshold=0.0,
        openai_embedding_model="text-embedding-3-small",
        gemini_embedding_model="models/text-embedding-004",
        groq_model="fake",
        gemini_model="fake",
    )

    records = [{"text": "Scholarships are financial aid for students.", "source_file": "aid.txt", "page_number": None}]
    chunks = chunk_records(records)
    vs = build_or_update_vectorstore(chunks, FakeEmbeddings(size=16), config.chroma_persist_dir)

    monkeypatch.setattr("src.rag_pipeline.generate_with_groq", lambda q, c, cfg: "Scholarships help pay for college. [S1]")

    result = answer_query("What is a scholarship?", vs, config)
    assert "Scholarships" in result.answer_text
    assert result.sources
    assert result.used_model == "groq"
