# College Helper RAG

A Streamlit-based retrieval-augmented generation (RAG) app for college-study help. It ingests PDFs, text files, and images, extracts text (OCR for images), builds embeddings in Chroma, and answers questions with cited sources.

## Prerequisites

- Python 3.10+
- Tesseract OCR installed and available in PATH
  - Windows example install path: `C:\\Program Files\\Tesseract-OCR\\tesseract.exe`
  - If not on PATH, set it in code before OCR calls:
    - `pytesseract.pytesseract.tesseract_cmd = r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe"`

## Setup

1. Create and activate virtual environment.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies.

```powershell
pip install -r requirements.txt
```

3. Create `.env` from template.

```powershell
Copy-Item .env.example .env
```

4. Fill API keys in `.env`:

```env
GROQ_API_KEY=your_groq_key
GEMINI_API_KEY=your_gemini_key
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
GEMINI_EMBEDDING_MODEL=models/embedding-001
# Optional only when EMBEDDING_PROVIDER=openai:
OPENAI_API_KEY=your_openai_key
```

## Run

```powershell
streamlit run app.py
```

## Usage

1. Upload one or more `.pdf`, `.txt`, `.png`, `.jpg`, `.jpeg`, or `.webp` files.
2. Click **Process Documents** to extract, chunk, and index content.
3. Enter a question and click **Ask**.
4. Review answer and source references (`S1`, `S2`, ...), including filename, page/chunk, score, and snippet.

## Troubleshooting

- `Configuration error` on startup:
  - Ensure `.env` exists and required keys are set.
- OCR errors:
  - Confirm Tesseract is installed and available in PATH.
- Empty retrieval results:
  - Upload richer content, re-index, or lower `RETRIEVAL_SCORE_THRESHOLD` in `.env`.

## Tests

```powershell
pytest -q
```
