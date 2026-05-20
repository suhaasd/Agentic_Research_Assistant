import os
from pathlib import Path
from dotenv import load_dotenv

import chromadb
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from typing import List, Dict, Any, Optional

from workflow_web_search import run_web_search_workflow
from workflow_quality_reviewer import review_both_answers, review_answer_quality

load_dotenv(override=True)

GROQ_MODEL         = "llama-3.3-70b-versatile"
EMBEDDING_MODEL    = "all-MiniLM-L6-v2"
VECTOR_STORE_PATH  = "../data/vector_store"
COLLECTION_NAME    = "pdf_documents"
TOP_K              = 6
MIN_SCORE          = 0.2


class ResearchRetriever:

    def __init__(
        self,
        persist_directory: str = VECTOR_STORE_PATH,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.embed_model = SentenceTransformer(embedding_model, local_files_only=True)
        client = chromadb.PersistentClient(path=persist_directory)
        self.collection = client.get_collection(collection_name)
        print(
            f"[Retriever] Connected to '{collection_name}' "
            f"({self.collection.count()} chunks)"
        )

    def retrieve(self, query: str, top_k: int = TOP_K, score_threshold: float = MIN_SCORE) -> List[Dict]:
        query_embedding = self.embed_model.encode([query])[0].tolist()
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        docs = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            similarity = 1 / (1 + dist)
            if similarity >= score_threshold:
                docs.append(
                    {"content": doc, "metadata": meta, "similarity_score": round(similarity, 4)}
                )
        return docs


def rag_with_ollama(
    query: str,
    retriever: ResearchRetriever,
    llm: ChatGroq,
    top_k: int = TOP_K,
    min_score: float = MIN_SCORE,
) -> Dict[str, Any]:

    results = retriever.retrieve(query, top_k=top_k, score_threshold=min_score)

    if not results:
        return {
            "answer": "No relevant context found in your papers for this question.",
            "sources": [],
            "confidence": 0.0,
            "context": "",
        }

    context = "\n\n".join([doc["content"] for doc in results])
    sources = [
        {
            "source": doc["metadata"].get("source_file", doc["metadata"].get("source", "unknown")),
            "page": doc["metadata"].get("page", "unknown"),
            "score": doc["similarity_score"],
            "preview": doc["content"][:200] + "...",
        }
        for doc in results
    ]
    confidence = max(doc["similarity_score"] for doc in results)

    prompt = (
        "You are a research assistant. Use the following context from research papers "
        "as your primary source. If the context covers the question well, base your "
        "answer on it and cite which papers support each point. If the context only "
        "partially covers the question, supplement the gaps with your general knowledge "
        "and clearly distinguish what comes from the papers versus general knowledge. "
        "Give a thorough, well-structured answer.\n\n"
        f"Context from papers:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )

    response = llm.invoke(prompt)
    answer = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer,
        "sources": sources,
        "confidence": confidence,
        "context": context,
    }

def _print_sources(sources: List[Dict]):
    if not sources:
        return
    print("\n  Sources used:")
    for i, s in enumerate(sources, 1):
        print(f"    [{i}] {s['source']}  (page {s['page']}, score {s['score']:.3f})")

def main():
    print("\n" + "=" * 70)
    print("  RESEARCH ASSISTANT  –  PDF RAG + Web Search + Quality Review")
    print("=" * 70)
    print(f"  Embedding model : {EMBEDDING_MODEL}")
    print(f"  Groq model      : {GROQ_MODEL}")
    print(f"  Vector store    : {VECTOR_STORE_PATH}")
    print("=" * 70)

    retriever = ResearchRetriever()
    llm = ChatGroq(model=GROQ_MODEL, temperature=0.1)

    while True:
        print("\n" + "-" * 70)
        query = input("Ask a question about your papers (or type 'quit' to exit):\n> ").strip()

        if query.lower() in ("quit", "exit", "q"):
            print("\nExiting Research Assistant. Goodbye!")
            break

        if not query:
            continue

        print(f"\n[RAG] Retrieving answer from your papers ...")
        rag_result = rag_with_ollama(query, retriever, llm)
        _print_sources(rag_result["sources"])
        print(f"[RAG] Confidence: {rag_result['confidence']:.3f}")

        paper_answer = rag_result["answer"]

        web_result = run_web_search_workflow(
            query=query,
            rag_answer=paper_answer,
            groq_model=GROQ_MODEL,
        )
        web_answer = web_result["synthesized"] if web_result else None

        print("\n[Reviewer] Performing quality review ...")
        review_results = review_both_answers(
            question=query,
            paper_answer=paper_answer,
            web_answer=web_answer,
            groq_model=GROQ_MODEL,
        )

        print("\n" + "=" * 70)
        print("FINAL SUMMARY")
        print("=" * 70)

        paper_rv = review_results["paper_review"]
        print(f"  Paper answer quality  : {paper_rv.average_score:.1f}/10  {paper_rv.verdict}")

        if review_results["web_review"]:
            web_rv = review_results["web_review"]
            print(f"  Web answer quality    : {web_rv.average_score:.1f}/10  {web_rv.verdict}")

        print(f"\n  {review_results['recommendation']}")

        show = input("\nPrint the recommended answer in full? (yes/no): ").strip().lower()
        if show in ("yes", "y"):
            if web_answer and review_results["web_review"] and \
               review_results["web_review"].average_score > paper_rv.average_score:
                print(f"\n[Best Answer – Web Synthesis]\n{web_answer}")
            else:
                print(f"\n[Best Answer – Paper]\n{paper_answer}")


if __name__ == "__main__":
    main()
