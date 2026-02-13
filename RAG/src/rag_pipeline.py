from __future__ import annotations

from typing import Sequence

from src.config import AppConfig
from src.llm import generate_with_gemini_multimodal, generate_with_groq
from src.models import RAGAnswer
from src.retriever import retrieve


def answer_query(
    query: str,
    vectorstore,
    config: AppConfig,
    use_multimodal: bool = False,
    images: Sequence[str] | None = None,
) -> RAGAnswer:
    warnings: list[str] = []
    results = retrieve(query=query, vectorstore=vectorstore, k=config.retriever_top_k)

    filtered = [r for r in results if r.score >= config.retrieval_score_threshold]
    if not filtered:
        filtered = results
        warnings.append("No chunks met the score threshold; using best available matches.")

    if not filtered:
        return RAGAnswer(
            answer_text="I could not find relevant context in the indexed documents.",
            sources=[],
            used_model="none",
            warnings=["Vector store returned no results."],
        )

    if use_multimodal and images:
        text = generate_with_gemini_multimodal(query, filtered, images, config)
        used_model = "gemini"
    else:
        text = generate_with_groq(query, filtered, config)
        used_model = "groq"

    return RAGAnswer(answer_text=text, sources=filtered, used_model=used_model, warnings=warnings)
