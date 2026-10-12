from pathlib import Path
from dotenv import find_dotenv, load_dotenv
from src.prepareData.prepareKnowledge import ingest_text, DATA_DIR   # main.py imports ingest_text from here

load_dotenv(find_dotenv(usecwd=True))
ALLOWED_EXT = {".txt", ".md"}
MAX_BYTES = 2 * 1024 * 1024    # 2 MB


def handle_file_upload(filename: str, raw: bytes) -> tuple[str, str]:
    """Validate and save an uploaded file. Returns (safe_filename, text)."""
    name = Path(filename or "").name                       # strips any path components
    if Path(name).suffix.lower() not in ALLOWED_EXT:
        raise ValueError("Only .txt and .md files are supported.")
    if len(raw) > MAX_BYTES:
        raise ValueError("File is larger than 2 MB.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ValueError("File is not valid UTF-8 text.")
    if not text.strip():
        raise ValueError("File is empty.")

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DATA_DIR / name).write_bytes(raw)                     # keep the original in Data/
    return name, text