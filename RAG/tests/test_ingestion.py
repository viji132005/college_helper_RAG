from __future__ import annotations

from pathlib import Path

import pytest

from src import ingestion


def test_extract_text_from_txt(tmp_path: Path) -> None:
    file_path = tmp_path / "a.txt"
    file_path.write_text("hello world", encoding="utf-8")
    records = ingestion.extract_text_from_file(file_path)
    assert len(records) == 1
    assert records[0]["text"] == "hello world"


def test_extract_text_from_pdf_monkeypatched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    file_path = tmp_path / "a.pdf"
    file_path.write_bytes(b"%PDF-1.4")

    class FakePage:
        def extract_text(self):
            return "Page text"

    class FakeReader:
        def __init__(self, _):
            self.pages = [FakePage()]

    monkeypatch.setattr(ingestion, "PdfReader", FakeReader)
    records = ingestion.extract_text_from_file(file_path)
    assert len(records) == 1
    assert records[0]["page_number"] == 1


def test_extract_text_from_image_monkeypatched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from PIL import Image

    image_path = tmp_path / "a.png"
    Image.new("RGB", (20, 20), color="white").save(image_path)

    monkeypatch.setattr(ingestion.pytesseract, "image_to_string", lambda _img: "ocr text")
    records = ingestion.extract_text_from_file(image_path)
    assert len(records) == 1
    assert records[0]["text"] == "ocr text"


def test_unsupported_extension(tmp_path: Path) -> None:
    file_path = tmp_path / "a.md"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError):
        ingestion.extract_text_from_file(file_path)
