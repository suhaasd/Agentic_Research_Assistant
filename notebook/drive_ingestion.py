import re
import uuid
import tempfile
from pathlib import Path

import fitz
import gdown
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = str(_PROJECT_ROOT / "data" / "vector_store")
COLLECTION_NAME = "pdf_documents"


def _chunk_pdf(pdf_path: str, source_name: str) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )

    doc = fitz.open(pdf_path)
    chunks = []

    for page_num in range(len(doc)):
        page_text = doc[page_num].get_text()
        if not page_text.strip():
            continue
        for split in splitter.split_text(page_text):
            chunks.append({
                "text": split,
                "metadata": {
                    "source_file": source_name,
                    "file_type": "pdf",
                    "page": page_num + 1,
                },
            })

    doc.close()
    return chunks


def _already_ingested(collection, source_name: str) -> bool:
    results = collection.get(where={"source_file": source_name}, limit=1)
    return len(results["ids"]) > 0


def _store_chunks(chunks: list[dict], collection, model) -> int:
    if not chunks:
        return 0
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    ids = [f"drive_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(chunks))]
    metadatas = [c["metadata"] for c in chunks]
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(chunks)


def _resolve_pdf_url(url: str) -> str:
    arxiv_abs = re.match(r"https?://arxiv\.org/abs/(.+)", url)
    if arxiv_abs:
        return f"https://arxiv.org/pdf/{arxiv_abs.group(1)}"
    return url


def _download_pdf_from_url(url: str) -> tuple[bytes, str]:
    url = _resolve_pdf_url(url)
    headers = {"User-Agent": "Mozilla/5.0 (Research Assistant Bot)"}
    response = requests.get(url, headers=headers, allow_redirects=True, timeout=30)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    if "pdf" not in content_type and not url.lower().endswith(".pdf"):
        raise ValueError(f"URL did not return a PDF (Content-Type: {content_type})")

    content_disposition = response.headers.get("Content-Disposition", "")
    filename_match = re.search(r'filename[^;=\n]*=(["\']?)([^"\'\n;]+)\1', content_disposition)
    if filename_match:
        filename = filename_match.group(2).strip()
    else:
        filename = url.split("/")[-1].split("?")[0]
        if not filename.lower().endswith(".pdf"):
            filename += ".pdf"

    return response.content, filename


def ingest_folder(folder_id: str) -> dict:
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"  Downloading from Google Drive folder: {folder_id}")
        gdown.download_folder(id=folder_id, output=tmp_dir, quiet=False, use_cookies=False)

        pdf_files = list(Path(tmp_dir).glob("**/*.pdf"))
        if not pdf_files:
            return {"ingested": 0, "total_chunks": 0, "files": []}

        summary = []
        for pdf_path in pdf_files:
            if _already_ingested(collection, pdf_path.name):
                print(f"  Skipped {pdf_path.name} — already in vector store")
                summary.append({"name": pdf_path.name, "chunks": 0, "skipped": True})
                continue
            print(f"  Processing: {pdf_path.name}")
            try:
                chunks = _chunk_pdf(str(pdf_path), pdf_path.name)
                count = _store_chunks(chunks, collection, model)
                summary.append({"name": pdf_path.name, "chunks": count, "skipped": False})
                print(f"  Stored {count} chunks for {pdf_path.name}")
            except Exception as e:
                print(f"  Failed {pdf_path.name}: {e}")
                continue

    return {
        "ingested": len(summary),
        "total_chunks": sum(f["chunks"] for f in summary),
        "files": summary,
    }


def fetch_and_ingest(url: str) -> dict:
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL)

    print(f"  Fetching PDF from: {url}")
    pdf_bytes, filename = _download_pdf_from_url(url)

    if _already_ingested(collection, filename):
        print(f"  Skipped {filename} — already in vector store")
        return {"source": url, "filename": filename, "chunks": 0, "skipped": True}

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(pdf_bytes)
        tmp_path = tmp.name

    try:
        chunks = _chunk_pdf(tmp_path, filename)
        count = _store_chunks(chunks, collection, model)
        print(f"  Stored {count} chunks for {filename}")
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return {
        "source": url,
        "filename": filename,
        "chunks": count,
    }
