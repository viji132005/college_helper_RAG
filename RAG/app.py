from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.chunking import chunk_records
from src.citations import format_source_reference
from src.config import AppConfig
from src.embeddings import get_embedding_model
from src.ingestion import extract_text_from_file
from src.rag_pipeline import answer_query
from src.vector_store import build_or_update_vectorstore, clear_vectorstore, load_vectorstore

st.set_page_config(page_title="College Helper RAG", layout="wide")
st.title("College Helper RAG")


def _init_session() -> None:
    st.session_state.setdefault("vectorstore", None)
    st.session_state.setdefault("uploaded_paths", [])
    st.session_state.setdefault("image_paths", [])


def _save_uploaded_files(upload_dir: Path, uploaded_files) -> tuple[list[Path], list[Path]]:
    saved: list[Path] = []
    images: list[Path] = []
    for file in uploaded_files:
        path = upload_dir / file.name
        path.write_bytes(file.getbuffer())
        saved.append(path)
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            images.append(path)
    return saved, images


def main() -> None:
    _init_session()

    try:
        config = AppConfig.from_env()
    except Exception as exc:  # noqa: BLE001
        st.error(f"Configuration error: {exc}")
        st.stop()

    st.subheader("Configuration")
    st.write(f"Chroma persist dir: `{config.chroma_persist_dir}`")
    st.write(f"Upload dir: `{config.upload_dir}`")

    embedding_model = get_embedding_model(config)

    col1, col2 = st.columns(2)
    with col1:
        uploaded_files = st.file_uploader(
            "Upload PDFs, text files, or images",
            type=["pdf", "txt", "png", "jpg", "jpeg", "webp"],
            accept_multiple_files=True,
        )
    with col2:
        clear_and_rebuild = st.checkbox("Clear and rebuild vector store", value=False)

    if st.button("Process Documents", type="primary"):
        if not uploaded_files:
            st.warning("Please upload at least one file.")
        else:
            try:
                saved_paths, image_paths = _save_uploaded_files(config.upload_dir, uploaded_files)
                st.session_state.uploaded_paths = [str(p) for p in saved_paths]
                st.session_state.image_paths = [str(p) for p in image_paths]

                all_records = []
                warnings = []
                for p in saved_paths:
                    try:
                        records = extract_text_from_file(p)
                        if not records:
                            warnings.append(f"No text extracted from {p.name}.")
                        all_records.extend(records)
                    except Exception as exc:  # noqa: BLE001
                        warnings.append(f"{p.name}: {exc}")

                if clear_and_rebuild:
                    clear_vectorstore(config.chroma_persist_dir)

                chunks = chunk_records(
                    all_records,
                    chunk_size_tokens=config.chunk_size_tokens,
                    overlap_tokens=config.chunk_overlap_tokens,
                )

                if not chunks:
                    st.warning("No chunks were created. Check document content/OCR.")
                else:
                    st.session_state.vectorstore = build_or_update_vectorstore(
                        chunks=chunks,
                        embedding_model=embedding_model,
                        persist_dir=config.chroma_persist_dir,
                    )
                    st.success(f"Indexed {len(chunks)} chunks from {len(saved_paths)} files.")

                for warning in warnings:
                    st.warning(warning)
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to process documents: {exc}")

    if st.session_state.vectorstore is None:
        try:
            st.session_state.vectorstore = load_vectorstore(config.chroma_persist_dir, embedding_model)
        except Exception:  # noqa: BLE001
            st.session_state.vectorstore = None

    st.subheader("Ask a Question")
    query = st.text_input("Enter your question")
    use_multimodal = st.checkbox("Use multimodal reasoning (Gemini with uploaded images)", value=False)

    if st.button("Ask"):
        if not query.strip():
            st.warning("Please enter a question.")
        elif st.session_state.vectorstore is None:
            st.warning("Vector store is not available. Please process documents first.")
        else:
            try:
                result = answer_query(
                    query=query.strip(),
                    vectorstore=st.session_state.vectorstore,
                    config=config,
                    use_multimodal=use_multimodal,
                    images=st.session_state.image_paths,
                )
                st.markdown("### Answer")
                st.write(result.answer_text)
                st.caption(f"Model used: {result.used_model}")

                for warning in result.warnings:
                    st.warning(warning)

                st.markdown("### Sources")
                if not result.sources:
                    st.info("No sources available.")
                else:
                    for idx, item in enumerate(result.sources, start=1):
                        with st.expander(f"Source S{idx}"):
                            st.text(format_source_reference(idx, item))
            except Exception as exc:  # noqa: BLE001
                st.error(f"Query failed: {exc}")


if __name__ == "__main__":
    main()
