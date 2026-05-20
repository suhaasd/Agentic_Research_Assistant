import uuid
import tempfile
from pathlib import Path

import fitz
import gdown
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


def ingest_folder(folder_id: str) -> dict:
    client = chromadb.PersistentClient(path=VECTOR_STORE_PATH)
    collection = client.get_or_create_collection(COLLECTION_NAME)
    model = SentenceTransformer(EMBEDDING_MODEL, local_files_only=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        print(f"  Downloading from Google Drive folder: {folder_id}")
        gdown.download_folder(id=folder_id, output=tmp_dir, quiet=False, use_cookies=False)

        pdf_files = list(Path(tmp_dir).glob("**/*.pdf"))
        if not pdf_files:
            return {"ingested": 0, "total_chunks": 0, "files": []}

        summary = []
        for pdf_path in pdf_files:
            print(f"  Processing: {pdf_path.name}")
            try:
                chunks = _chunk_pdf(str(pdf_path), pdf_path.name)
                if not chunks:
                    print(f"  Skipped {pdf_path.name} — no extractable text")
                    continue

                texts = [c["text"] for c in chunks]
                embeddings = model.encode(texts, show_progress_bar=False).tolist()
                ids = [f"drive_{uuid.uuid4().hex[:8]}_{i}" for i in range(len(chunks))]
                metadatas = [c["metadata"] for c in chunks]

                collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
                summary.append({"name": pdf_path.name, "chunks": len(chunks)})
                print(f"  Stored {len(chunks)} chunks for {pdf_path.name}")

            except Exception as e:
                print(f"  Failed {pdf_path.name}: {e}")
                continue

    return {
        "ingested": len(summary),
        "total_chunks": sum(f["chunks"] for f in summary),
        "files": summary,
    }
