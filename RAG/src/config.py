from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(slots=True)
class AppConfig:
    openai_api_key: str
    groq_api_key: str
    gemini_api_key: str
    embedding_provider: str
    local_embedding_model: str
    chroma_persist_dir: Path
    upload_dir: Path
    chunk_size_tokens: int
    chunk_overlap_tokens: int
    retriever_top_k: int
    retrieval_score_threshold: float
    openai_embedding_model: str
    gemini_embedding_model: str
    groq_model: str
    gemini_model: str

    @classmethod
    def from_env(cls) -> "AppConfig":
        load_dotenv()

        openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
        groq_api_key = os.getenv("GROQ_API_KEY", "").strip()
        gemini_api_key = os.getenv("GEMINI_API_KEY", "").strip()

        missing = [
            name
            for name, value in {
                "GROQ_API_KEY": groq_api_key,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

        chroma_persist_dir = Path(os.getenv("CHROMA_PERSIST_DIR", "data/vectordb"))
        upload_dir = Path(os.getenv("UPLOAD_DIR", "data/uploads"))
        chroma_persist_dir.mkdir(parents=True, exist_ok=True)
        upload_dir.mkdir(parents=True, exist_ok=True)

        return cls(
            openai_api_key=openai_api_key,
            groq_api_key=groq_api_key,
            gemini_api_key=gemini_api_key,
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local").strip().lower(),
            local_embedding_model=os.getenv(
                "LOCAL_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            chroma_persist_dir=chroma_persist_dir,
            upload_dir=upload_dir,
            chunk_size_tokens=int(os.getenv("CHUNK_SIZE_TOKENS", "500")),
            chunk_overlap_tokens=int(os.getenv("CHUNK_OVERLAP_TOKENS", "50")),
            retriever_top_k=int(os.getenv("RETRIEVER_TOP_K", "5")),
            retrieval_score_threshold=float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.2")),
            openai_embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
            gemini_embedding_model=os.getenv("GEMINI_EMBEDDING_MODEL", "models/embedding-001"),
            groq_model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
            gemini_model=os.getenv("GEMINI_MODEL", "gemini-1.5-flash"),
        )
