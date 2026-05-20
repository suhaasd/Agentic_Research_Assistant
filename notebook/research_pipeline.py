import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq

sys.path.insert(0, str(Path(__file__).parent))

from research_assistant import ResearchRetriever, rag_with_ollama, GROQ_MODEL
from drive_ingestion import ingest_from_drive, fetch_and_ingest
from workflow_web_search import search_and_synthesize
from workflow_quality_reviewer import review_both_answers

load_dotenv()

SUFFICIENT_SCORE = 0.6
ARXIV_MAX_PAPERS = 5


def _is_sufficient(results: list) -> bool:
    return bool(results) and results[0]["similarity_score"] >= SUFFICIENT_SCORE


_STOP_WORDS = re.compile(
    r"\b(what|how|why|when|where|who|is|are|was|were|does|do|did|the|a|an|"
    r"it|its|and|or|of|to|in|for|with|about|explain|describe|difference|"
    r"between|vs|versus|compare|tell|me|please|can|you)\b",
    re.IGNORECASE,
)

def _search_arxiv(query: str) -> list[str]:
    clean = _STOP_WORDS.sub(" ", query)
    clean = " ".join(clean.split())[:120]
    url = (
        f"http://export.arxiv.org/api/query"
        f"?search_query=all:{requests.utils.quote(clean)}"
        f"&max_results={ARXIV_MAX_PAPERS + 2}&sortBy=relevance"
    )
    response = requests.get(url, timeout=15)
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}
    urls = []
    for entry in root.findall("atom:entry", ns):
        arxiv_id = entry.find("atom:id", ns).text.strip().split("/abs/")[-1]
        urls.append(f"https://arxiv.org/pdf/{arxiv_id}")
    return urls[:ARXIV_MAX_PAPERS]


def run_research_pipeline(query: str, retriever: ResearchRetriever, llm: ChatGroq) -> dict:
    steps = []

    print(f"\n[Pipeline] Step 1 — Searching local knowledge base...")
    results = retriever.retrieve(query)

    if _is_sufficient(results):
        steps.append({"step": "local_search", "status": "sufficient"})
        print(f"[Pipeline] Local results sufficient (top score: {results[0]['similarity_score']:.3f})")
    else:
        steps.append({"step": "local_search", "status": "weak"})
        print(f"[Pipeline] Local results weak — pulling from Google Drive...")

        drive_result = ingest_from_drive()
        new_files = drive_result.get("ingested", 0)
        steps.append({"step": "drive_ingestion", "status": "done", "new_files": new_files})

        results = retriever.retrieve(query)

        if not _is_sufficient(results):
            print(f"[Pipeline] Still weak — searching arXiv...")
            arxiv_urls = _search_arxiv(query)
            fetched = 0
            for url in arxiv_urls:
                result = fetch_and_ingest(url)
                if not result.get("skipped"):
                    fetched += 1
            steps.append({"step": "arxiv_fetch", "status": "done", "papers_fetched": fetched})
            results = retriever.retrieve(query)
        else:
            steps.append({"step": "arxiv_fetch", "status": "skipped"})

    print(f"[Pipeline] Step 4 — Generating paper answer...")
    paper_result = rag_with_ollama(query, retriever, llm)
    steps.append({"step": "paper_answer", "status": "done"})

    print(f"[Pipeline] Step 5 — Running web search...")
    web_result = search_and_synthesize(query, paper_result["answer"], llm)
    steps.append({"step": "web_search", "status": "done"})

    print(f"[Pipeline] Step 6 — Running quality review...")
    review = review_both_answers(
        question=query,
        paper_answer=paper_result["answer"],
        web_answer=web_result["synthesized"],
        groq_model=GROQ_MODEL,
    )
    steps.append({"step": "quality_review", "status": "done"})

    paper_rv = review["paper_review"]
    web_rv = review["web_review"]

    return {
        "paper_answer": paper_result["answer"],
        "sources": paper_result["sources"],
        "confidence": paper_result["confidence"],
        "web_answer": web_result["synthesized"],
        "paper_review": {
            "relevance": paper_rv.relevance,
            "completeness": paper_rv.completeness,
            "clarity": paper_rv.clarity,
            "average_score": paper_rv.average_score,
            "verdict": paper_rv.verdict,
            "reviewer_notes": paper_rv.reviewer_notes,
            "suggestions": paper_rv.suggestions,
        },
        "web_review": {
            "relevance": web_rv.relevance,
            "completeness": web_rv.completeness,
            "clarity": web_rv.clarity,
            "average_score": web_rv.average_score,
            "verdict": web_rv.verdict,
            "reviewer_notes": web_rv.reviewer_notes,
            "suggestions": web_rv.suggestions,
        } if web_rv else None,
        "recommendation": review["recommendation"],
        "pipeline_steps": steps,
    }
