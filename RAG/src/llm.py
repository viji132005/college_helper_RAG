from __future__ import annotations

from typing import Sequence

import google.generativeai as genai
from groq import Groq

from src.config import AppConfig
from src.models import RetrievalResult


def _build_context(chunks: Sequence[RetrievalResult]) -> str:
    lines: list[str] = []
    for idx, item in enumerate(chunks, start=1):
        page_label = f"p.{item.chunk.page_number}" if item.chunk.page_number else "p.n/a"
        lines.append(
            f"[S{idx}] File={item.chunk.source_file} {page_label} score={item.score:.3f}\n{item.chunk.text}"
        )
    return "\n\n".join(lines)


def generate_with_groq(query: str, context_chunks: Sequence[RetrievalResult], config: AppConfig) -> str:
    client = Groq(api_key=config.groq_api_key)
    context = _build_context(context_chunks)
    prompt = (
        "You are a college helper assistant. Use only the provided context. "
        "If context is insufficient, clearly say so. Cite supporting chunks as [S1], [S2], etc.\n\n"
        f"Question: {query}\n\nContext:\n{context}"
    )

    response = client.chat.completions.create(
        model=config.groq_model,
        messages=[
            {"role": "system", "content": "Answer with concise, factual responses and explicit citations."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )

    return response.choices[0].message.content or "No response generated."


def generate_with_gemini_multimodal(
    query: str,
    context_chunks: Sequence[RetrievalResult],
    images: Sequence[str] | None,
    config: AppConfig,
) -> str:
    genai.configure(api_key=config.gemini_api_key)
    model = genai.GenerativeModel(config.gemini_model)

    context = _build_context(context_chunks)
    prompt = (
        "Use only the provided context and attached images. "
        "If insufficient evidence, say so. Include [S#] citations where possible.\n\n"
        f"Question: {query}\n\nContext:\n{context}"
    )

    parts: list = [prompt]
    for image_path in images or []:
        parts.append(genai.upload_file(image_path))

    response = model.generate_content(parts)
    text = getattr(response, "text", None)
    return text or "No response generated."
