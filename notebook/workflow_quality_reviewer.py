import json
import re
from dataclasses import dataclass, field
from typing import Optional

from langchain_groq import ChatGroq

@dataclass
class QualityReview:
    relevance: int          
    completeness: int       
    clarity: int            
    average_score: float
    verdict: str          
    reviewer_notes: str    
    suggestions: list   
    raw_llm_response: str = field(repr=False, default="")

    def display(self):
        bar = "=" * 70
        print(f"\n{bar}")
        print("QUALITY REVIEW REPORT")
        print(bar)
        print(f"  Relevance      : {self.relevance}/10")
        print(f"  Completeness   : {self.completeness}/10")
        print(f"  Clarity        : {self.clarity}/10")
        print(f"  Average Score  : {self.average_score:.1f}/10")
        print(f"  Verdict        : {self.verdict}")
        print(f"\n  Reviewer Notes :\n  {self.reviewer_notes}")
        if self.suggestions:
            print("\n  Suggestions for Improvement:")
            for i, s in enumerate(self.suggestions, 1):
                print(f"    {i}. {s}")
        print(bar)


PASS_THRESHOLD = 7.0
NEEDS_IMPROVEMENT_THRESHOLD = 4.5


def _derive_verdict(avg: float) -> str:
    if avg >= PASS_THRESHOLD:
        return "PASS"
    elif avg >= NEEDS_IMPROVEMENT_THRESHOLD:
        return "NEEDS IMPROVEMENT"
    else:
        return "FAIL"

REVIEW_PROMPT_TEMPLATE = """You are a strict academic quality reviewer evaluating answers to research questions.

Original Question:
"{question}"

Answer to Evaluate:
"{answer}"

Source Type: {source_type}

Evaluate the answer on the following three dimensions. For each, give an integer score from 1 to 10 and a brief justification.

1. Relevance (1-10): Does the answer directly address the question asked? A score of 10 means the answer is perfectly on-topic with no irrelevant content.

2. Completeness (1-10): Does the answer cover the key aspects of the question thoroughly? A score of 10 means nothing important is missing.

3. Clarity (1-10): Is the answer well-structured, logically organized, and easy to understand for a graduate-level reader? A score of 10 means the writing is exemplary.

Then provide:
- reviewer_notes: A 2-3 sentence overall assessment of the answer quality.
- suggestions: A list of 2-4 specific, actionable suggestions to improve the answer (or state "None" if no improvements are needed).

Respond ONLY in the following JSON format with no extra text, preamble, or markdown code fences:

{{
  "relevance": <int>,
  "completeness": <int>,
  "clarity": <int>,
  "reviewer_notes": "<string>",
  "suggestions": ["<suggestion 1>", "<suggestion 2>"]
}}"""

def _parse_review_response(raw: str, question: str, answer: str) -> QualityReview:
   
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().rstrip("```").strip()

    json_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if json_match:
        cleaned = json_match.group(0)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        print(f"[Reviewer] Warning: Could not parse JSON. Raw response:\n{raw}")
        return QualityReview(
            relevance=5,
            completeness=5,
            clarity=5,
            average_score=5.0,
            verdict="NEEDS IMPROVEMENT",
            reviewer_notes="Quality review could not be fully parsed. Manual inspection recommended.",
            suggestions=["Retry the review or inspect the answer manually."],
            raw_llm_response=raw,
        )

    relevance = int(data.get("relevance", 5))
    completeness = int(data.get("completeness", 5))
    clarity = int(data.get("clarity", 5))
    avg = round((relevance + completeness + clarity) / 3, 2)
    verdict = _derive_verdict(avg)
    notes = data.get("reviewer_notes", "")
    suggestions_raw = data.get("suggestions", [])

    if isinstance(suggestions_raw, str):
        suggestions = [] if suggestions_raw.strip().lower() == "none" else [suggestions_raw]
    else:
        suggestions = [s for s in suggestions_raw if s.strip().lower() != "none"]

    return QualityReview(
        relevance=relevance,
        completeness=completeness,
        clarity=clarity,
        average_score=avg,
        verdict=verdict,
        reviewer_notes=notes,
        suggestions=suggestions,
        raw_llm_response=raw,
    )

def review_answer_quality(
    question: str,
    answer: str,
    source_type: str = "RAG (paper)",
    groq_model: str = "llama-3.1-8b-instant",
) -> QualityReview:
    """
    Run the quality review for a single answer.

    Args:
        question   : The original research question.
        answer     : The answer to evaluate (from RAG or web synthesis).
        source_type: Human-readable label e.g. "RAG (paper)" or "Web synthesis".
        groq_model : Groq model to use.

    Returns:
        QualityReview dataclass instance.
    """
    print(f"\n[Reviewer] Running quality review using Groq ({groq_model}) ...")

    llm = ChatGroq(model=groq_model, temperature=0)

    prompt = REVIEW_PROMPT_TEMPLATE.format(
        question=question,
        answer=answer,
        source_type=source_type,
    )

    response = llm.invoke(prompt)
    raw = response.content if hasattr(response, "content") else str(response)

    review = _parse_review_response(raw, question, answer)
    review.display()
    return review

def review_both_answers(
    question: str,
    paper_answer: str,
    web_answer: Optional[str] = None,
    groq_model: str = "llama-3.1-8b-instant",
) -> dict:
    """
    Review the paper answer and (optionally) the web-synthesized answer.

    Returns:
        {
            "paper_review" : QualityReview,
            "web_review"   : QualityReview | None,
            "recommendation": str          # Which answer to use
        }
    """
    print("\n[Reviewer] Reviewing paper answer ...")
    paper_review = review_answer_quality(
        question=question,
        answer=paper_answer,
        source_type="RAG (research paper)",
        groq_model=groq_model,
    )

    web_review = None
    if web_answer:
        print("\n[Reviewer] Reviewing web-synthesized answer ...")
        web_review = review_answer_quality(
            question=question,
            answer=web_answer,
            source_type="Web synthesis (Tavily + Groq)",
            groq_model=groq_model,
        )

    if web_review is None:
        recommendation = "Use the paper answer (no web answer available)."
    elif web_review.average_score > paper_review.average_score + 0.5:
        recommendation = "Recommend using the WEB-SYNTHESIZED answer (higher quality score)."
    elif paper_review.average_score > web_review.average_score + 0.5:
        recommendation = "Recommend using the PAPER answer (higher quality score)."
    else:
        recommendation = (
            "Both answers are comparable. Use your judgement, "
            "or combine them manually."
        )

    print(f"\n[Reviewer] Recommendation: {recommendation}")

    return {
        "paper_review": paper_review,
        "web_review": web_review,
        "recommendation": recommendation,
    }


if __name__ == "__main__":
    sample_question = "What is FlashFill and how does it use program synthesis?"

    sample_paper_answer = (
        "FlashFill is a feature in Excel that fills columns automatically "
        "using patterns from examples."
    ) 

    sample_web_answer = (
        "FlashFill, introduced in Excel 2013, uses a form of inductive program "
        "synthesis to learn string transformation programs from user-provided "
        "input-output examples. The underlying algorithm searches a domain-specific "
        "language (DSL) of string manipulations to find a consistent program. "
        "It has been extended to support conditional logic and multiple transformations, "
        "and the research behind it is rooted in work on synthesis from examples (SyGuS)."
    )

    results = review_both_answers(
        question=sample_question,
        paper_answer=sample_paper_answer,
        web_answer=sample_web_answer,
        groq_model="llama-3.1-8b-instant",
    )

    print("\nFinal Recommendation:", results["recommendation"])
