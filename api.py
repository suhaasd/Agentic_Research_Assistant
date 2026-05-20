import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "notebook"))

from langchain_groq import ChatGroq
from research_assistant import ResearchRetriever, rag_with_ollama
from workflow_quality_reviewer import review_both_answers
from workflow_web_search import get_groq_llm, search_and_synthesize

GROQ_MODEL        = "llama-3.1-8b-instant"
VECTOR_STORE_PATH = os.path.join(os.path.dirname(__file__), "data", "vector_store")

_retriever: Optional[ResearchRetriever] = None
_llm: Optional[ChatGroq] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _retriever, _llm
    print("[Startup] Loading retriever and LLM...")
    _retriever = ResearchRetriever(persist_directory=VECTOR_STORE_PATH)
    _llm = ChatGroq(model=GROQ_MODEL, temperature=0.1)
    print("[Startup] Ready.")
    yield
    print("[Shutdown] Cleaning up.")


app = FastAPI(title="Research Assistant API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    query: str

class EnhanceRequest(BaseModel):
    query: str
    rag_answer: str

class IngestRequest(BaseModel):
    folder_id: str

@app.get("/api/health")
def health():
    return {"status": "ok", "model": GROQ_MODEL}


@app.post("/api/ask")
def ask(req: AskRequest):
    """Run the RAG pipeline and return the paper answer with sources."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        result = rag_with_ollama(req.query, _retriever, _llm)
        return {
            "answer": result["answer"],
            "sources": result["sources"],
            "confidence": result["confidence"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/enhance")
def enhance(req: EnhanceRequest):
    """Run web search + synthesis + quality review for both answers."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        llm = get_groq_llm(model=GROQ_MODEL)
        web_result = search_and_synthesize(req.query, req.rag_answer, llm)

        review = review_both_answers(
            question=req.query,
            paper_answer=req.rag_answer,
            web_answer=web_result["synthesized"],
            groq_model=GROQ_MODEL,
        )

        paper_rv = review["paper_review"]
        web_rv   = review["web_review"]

        return {
            "web_results":  web_result["web_results"],
            "synthesized":  web_result["synthesized"],
            "paper_review": {
                "relevance":     paper_rv.relevance,
                "completeness":  paper_rv.completeness,
                "clarity":       paper_rv.clarity,
                "average_score": paper_rv.average_score,
                "verdict":       paper_rv.verdict,
                "reviewer_notes":paper_rv.reviewer_notes,
                "suggestions":   paper_rv.suggestions,
            },
            "web_review": {
                "relevance":     web_rv.relevance,
                "completeness":  web_rv.completeness,
                "clarity":       web_rv.clarity,
                "average_score": web_rv.average_score,
                "verdict":       web_rv.verdict,
                "reviewer_notes":web_rv.reviewer_notes,
                "suggestions":   web_rv.suggestions,
            } if web_rv else None,
            "recommendation": review["recommendation"],
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
def ingest(req: IngestRequest):
    """Download PDFs from a Google Drive folder, chunk, embed, and store them."""
    if not req.folder_id.strip():
        raise HTTPException(status_code=400, detail="folder_id cannot be empty.")
    try:
        from drive_ingestion import ingest_folder
        result = ingest_folder(req.folder_id)
        global _retriever
        _retriever = ResearchRetriever(persist_directory=VECTOR_STORE_PATH)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(FRONTEND_DIR):
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

@app.get("/")
def serve_frontend():
    index = os.path.join(FRONTEND_DIR, "index.html")
    if os.path.isfile(index):
        return FileResponse(index)
    return {"message": "Research Assistant API is running. See /docs for API reference."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
