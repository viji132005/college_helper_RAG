from __future__ import annotations

from typing import Any

import google.generativeai as genai
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.config import AppConfig


class GeminiEmbeddings:
    def __init__(self, api_key: str, model: str) -> None:
        self.model = model
        genai.configure(api_key=api_key)

    def _embed(self, text: str, task_type: str) -> list[float]:
        models_to_try = [self.model]
        if self.model != "models/embedding-001":
            models_to_try.append("models/embedding-001")

        last_error: Exception | None = None
        for model_name in models_to_try:
            try:
                response: Any = genai.embed_content(
                    model=model_name,
                    content=text,
                    task_type=task_type,
                )
                return response["embedding"]
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                continue

        raise RuntimeError(
            "Gemini embedding failed for all configured fallback models. "
            f"Last error: {last_error}"
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text, "retrieval_document") for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text, "retrieval_query")


def get_embedding_model(config: AppConfig) -> Any:
    if config.embedding_provider == "openai":
        if not config.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai")
        return OpenAIEmbeddings(
            api_key=config.openai_api_key,
            model=config.openai_embedding_model,
        )
    if config.embedding_provider == "gemini":
        if not config.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when EMBEDDING_PROVIDER=gemini")
        return GeminiEmbeddings(
            api_key=config.gemini_api_key,
            model=config.gemini_embedding_model,
        )
    if config.embedding_provider == "local":
        return HuggingFaceEmbeddings(model_name=config.local_embedding_model)
    raise ValueError(
        "Invalid EMBEDDING_PROVIDER. Use one of: local, gemini, openai."
    )
