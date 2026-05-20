import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_groq import ChatGroq
from typing import Optional


def get_groq_llm(model: str = "llama-3.1-8b-instant", temperature: float = 0.2) -> ChatGroq:
    return ChatGroq(model=model, temperature=temperature)


def search_and_synthesize(
    query: str,
    rag_answer: str,
    llm: ChatGroq,
    num_results: int = 5,
) -> dict:
    """
    Search the web via Tavily for `query`, then ask the LLM to synthesize the
    results alongside the original RAG answer into a richer response.

    Returns a dict with keys:
        web_results   – formatted Tavily snippet string
        synthesized   – LLM-synthesized combined answer
        query         – the search query used
    """
    print(f"\n[WebSearch] Searching the web for: '{query}' ...")
    search_tool = TavilySearchResults(max_results=num_results)

    try:
        results = search_tool.invoke(query)
        if isinstance(results, list):
            web_results = "\n\n".join(
                f"Source: {r.get('url', 'unknown')}\n{r.get('content', '')}"
                for r in results
            )
        else:
            web_results = str(results)
    except Exception as e:
        print(f"[WebSearch] Search failed: {e}")
        return {
            "web_results": "",
            "synthesized": f"Web search failed: {e}",
            "query": query,
        }

    print("[WebSearch] Results retrieved. Synthesizing with Groq ...")

    synthesis_prompt = f"""You are a research assistant helping a student understand a topic more deeply.

The student asked: "{query}"

Here is what was found directly in their research paper:
--- PAPER ANSWER ---
{rag_answer}
--- END PAPER ANSWER ---

Here are additional web search results to broaden the context:
--- WEB RESULTS ---
{web_results}
--- END WEB RESULTS ---

Your task:
1. Synthesize both sources into a comprehensive, well-structured answer.
2. Highlight any differences or additional insights the web provides beyond the paper.
3. Keep the response focused and academic in tone.
4. At the end, add a brief note on which source (paper vs web) was more informative for this question.

Synthesized Answer:"""

    response = llm.invoke(synthesis_prompt)
    synthesized = response.content if hasattr(response, "content") else str(response)

    return {
        "web_results": web_results,
        "synthesized": synthesized,
        "query": query,
    }


def prompt_user_for_web_search(rag_answer: str) -> bool:
    print("\n" + "=" * 70)
    print("ANSWER FROM YOUR PAPER:")
    print("=" * 70)
    print(rag_answer)
    print("=" * 70)
    print("\nthis is the answer for your question directly from the paper.")
    print("Would you like to look up the internet for more details?")

    while True:
        choice = input("Enter yes or no: ").strip().lower()
        if choice in ("yes", "y"):
            return True
        elif choice in ("no", "n"):
            return False
        else:
            print("Please enter 'yes' or 'no'.")


def run_web_search_workflow(
    query: str,
    rag_answer: str,
    groq_model: str = "llama-3.1-8b-instant",
) -> Optional[dict]:
    """
    Full Workflow 2:
        1. Display the RAG answer.
        2. Prompt user for yes/no.
        3. If yes  → search web + synthesize → return result dict.
        4. If no   → return None (pipeline stops here).
    """
    wants_web = prompt_user_for_web_search(rag_answer)

    if not wants_web:
        print("\n[WebSearch] Stopping here. Using only the paper answer.")
        return None

    llm = get_groq_llm(model=groq_model)
    result = search_and_synthesize(query, rag_answer, llm)

    print("\n" + "=" * 70)
    print("SYNTHESIZED ANSWER (PAPER + WEB):")
    print("=" * 70)
    print(result["synthesized"])

    return result


if __name__ == "__main__":
    sample_query = "What is FlashFill?"
    sample_rag_answer = (
        "FlashFill is a feature in Microsoft Excel that automatically fills "
        "data in a column based on patterns detected from user-provided examples. "
        "It uses program synthesis to infer the transformation rule."
    )

    result = run_web_search_workflow(
        query=sample_query,
        rag_answer=sample_rag_answer,
    )

    if result:
        print("\n[Done] Web-enhanced answer generated.")
    else:
        print("\n[Done] User chose to keep the paper answer only.")
