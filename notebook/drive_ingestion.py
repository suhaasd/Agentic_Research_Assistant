import io
import os
import re
import uuid
import tempfile
from pathlib import Path

import fitz
import requests
import chromadb
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from dotenv import load_dotenv

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
CREDENTIALS_FILE = str(_PROJECT_ROOT / "credentials.json")
TOKEN_FILE = str(_PROJECT_ROOT / "token.json")

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH = str(_PROJECT_ROOT / "data" / "vector_store")
COLLECTION_NAME = "pdf_documents"


def _authenticate():
    creds = None
    if os.path.exists(TOKEN_FILE):
        creds = Credentials.from_authorized_user_file(TOKEN_FILE, SCOPES)
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0)
        with open(TOKEN_FILE, "w") as f:
            f.write(creds.to_json())
    return creds


def _get_drive_service():
    return build("drive", "v3", credentials=_authenticate())


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


def _store_chunks(chunks: list[dict], collection, model) -> int:
    if not chunks:
        return 0
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    ids = [f"drive_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(chunks))]
    metadatas = [c["metadata"] for c in chunks]
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
    return len(chunks)


def _already_ingested(collection, source_name: str) -> bool:
    results = collection.get(where={"source_file": source_name}, limit=1)
    return len(results["ids"]) > 0


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


def ingest_from_drive() -> dict:
    if not os.path.exists(CREDENTIALS_FILE):
        return {"ingested": 0, "total_chunks": 0, "files": [], "error": "credentials.json not found"}

    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)
    service = _get_drive_service()

    response = service.files().list(
        q="mimeType='application/pdf' and trashed=false",
        fields="files(id, name)",
        pageSize=100,
    ).execute()
    files = response.get("files", [])

    if not files:
        return {"ingested": 0, "total_chunks": 0, "files": []}

    summary = []
    for file in files:
        if _already_ingested(collection, file["name"]):
            print(f"  Skipped {file['name']} — already ingested")
            summary.append({"name": file["name"], "chunks": 0, "skipped": True})
            continue

        print(f"  Downloading: {file['name']}")
        try:
            request = service.files().get_media(fileId=file["id"])
            buffer = io.BytesIO()
            downloader = MediaIoBaseDownload(buffer, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
            pdf_bytes = buffer.getvalue()

            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            try:
                chunks = _chunk_pdf(tmp_path, file["name"])
                count = _store_chunks(chunks, collection, model)
                summary.append({"name": file["name"], "chunks": count, "skipped": False})
                print(f"  Stored {count} chunks for {file['name']}")
            finally:
                Path(tmp_path).unlink(missing_ok=True)

        except Exception as e:
            print(f"  Failed {file['name']}: {e}")
            continue

    return {
        "ingested": len([f for f in summary if not f.get("skipped")]),
        "total_chunks": sum(f["chunks"] for f in summary),
        "files": summary,
    }


def fetch_and_ingest(url: str) -> dict:
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)

    print(f"  Fetching PDF from: {url}")
    pdf_bytes, filename = _download_pdf_from_url(url)

    if _already_ingested(collection, filename):
        print(f"  Skipped {filename} — already ingested")
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

    return {"source": url, "filename": filename, "chunks": count, "skipped": False}
