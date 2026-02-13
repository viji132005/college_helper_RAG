from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import pytesseract
from PIL import Image
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".png", ".jpg", ".jpeg", ".webp"}


def _configure_tesseract() -> None:
    # Prefer PATH if available; otherwise try common Windows install locations.
    path_cmd = shutil.which("tesseract")
    if path_cmd:
        pytesseract.pytesseract.tesseract_cmd = path_cmd
        return

    candidates = [
        Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe"),
        Path(r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe"),
    ]
    for candidate in candidates:
        if candidate.exists():
            pytesseract.pytesseract.tesseract_cmd = str(candidate)
            return


def extract_text_from_file(path: str | Path) -> list[dict[str, Any]]:
    file_path = Path(path)
    ext = file_path.suffix.lower()

    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: {ext}")

    if ext == ".pdf":
        return _extract_pdf(file_path)
    if ext == ".txt":
        return _extract_txt(file_path)
    return _extract_image(file_path)


def _extract_pdf(file_path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    try:
        reader = PdfReader(str(file_path))
        for i, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                records.append({"text": text, "source_file": file_path.name, "page_number": i})
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to parse PDF: {file_path.name}") from exc
    return records


def _extract_txt(file_path: Path) -> list[dict[str, Any]]:
    try:
        text = file_path.read_text(encoding="utf-8", errors="replace").strip()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f"Failed to read text file: {file_path.name}") from exc
    if not text:
        return []
    return [{"text": text, "source_file": file_path.name, "page_number": None}]


def _extract_image(file_path: Path) -> list[dict[str, Any]]:
    _configure_tesseract()
    try:
        with Image.open(file_path) as image:
            text = pytesseract.image_to_string(image).strip()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to OCR image: {file_path.name}. "
            "Ensure Tesseract is installed and reachable via PATH or common install directory. "
            f"Original error: {exc}"
        ) from exc
    if not text:
        return []
    return [{"text": text, "source_file": file_path.name, "page_number": None}]

