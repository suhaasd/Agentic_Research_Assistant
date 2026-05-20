# Research Assistant

An intelligent research tool that answers questions from your PDF papers using **Retrieval-Augmented Generation (RAG)**, enriches them with **live web search**, and evaluates answer quality using an **LLM-as-a-judge** reviewer — all running locally with no API keys required.

---

## Features

- **PDF RAG Pipeline** — semantic search over your research papers using ChromaDB and sentence embeddings
- **Google Drive Ingestion** — pull PDFs directly from a public Google Drive folder, chunk and embed them into the same vector store as local papers
- **MCP Server** — exposes ingestion as an MCP tool so any LLM agent can trigger it
- **Web Search Integration** — augments paper answers with DuckDuckGo search results, synthesized by the local LLM
- **LLM Quality Reviewer** — scores every answer on Relevance, Completeness, and Clarity (1–10), gives a verdict (PASS / NEEDS IMPROVEMENT / FAIL), and recommends the best answer
- **Web UI** — clean dark-themed single-page app built on FastAPI + JS
- **Fully local & free** — powered by [Ollama](https://ollama.com) (Llama 3.1), no external API keys needed

---

## How it works

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Question                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   Embed Query   │  all-MiniLM-L6-v2
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  ChromaDB Search│  top-6 chunks, score ≥ 0.2
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Ollama LLM     │  llama3.1 → Paper Answer
                    └────────┬────────┘
                             │
               ┌─────────────▼─────────────┐
               │     Web Search (optional)  │  DuckDuckGo
               └─────────────┬─────────────┘
                             │
                    ┌────────▼────────┐
                    │  Ollama LLM     │  llama3.1 → Synthesized Answer
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Quality Reviewer│  LLM-as-judge scores both answers
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  Recommendation │  Best answer surfaced
                    └─────────────────┘
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Ollama — `llama3.1` (local, free) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (persistent) |
| LLM Framework | LangChain |
| Web Search | DuckDuckGo (via `langchain-community`) |
| Drive Ingestion | `gdown` (public Google Drive) |
| MCP Server | FastMCP |
| API Server | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JS |
| Python | 3.13+ |

---

## Project Structure

```
Research_Assistant/
├── api.py                            # FastAPI server — wraps all pipelines as REST endpoints
├── mcp_server.py                     # FastMCP server — exposes ingestion as an MCP tool
├── main.py                           # Simple entry point
├── requirements.txt                  # Python dependencies
├── pyproject.toml                    # Project metadata
│
├── notebook/
│   ├── research_assistant.py         # Core RAG pipeline + CLI entry point
│   ├── drive_ingestion.py            # Google Drive download + chunking + embedding pipeline
│   ├── workflow_web_search.py        # DuckDuckGo search + LLM synthesis
│   └── workflow_quality_reviewer.py  # LLM-as-judge quality scoring
│
├── frontend/
│   └── index.html                    # Single-page web UI
│
└── data/
    ├── pdf/                          # Local source papers (PDFs)
    ├── vector_store/                 # Persistent ChromaDB embeddings
    ├── text_files/                   # Raw extracted text
    └── json/                         # Structured data
```

---

## Prerequisites

1. **Python 3.13+**
2. **[Ollama](https://ollama.com/download)** installed and running
3. **Llama 3.1** model pulled into Ollama

```bash
# Install Ollama (visit https://ollama.com/download for your OS)
# Then pull the model:
ollama pull llama3.1
```

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Research_Assistant.git
cd Research_Assistant

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Running the App

### Option 1 — Web UI (recommended)

```bash
python api.py
```

Open your browser at **http://localhost:8000**

The UI lets you:
- Ask a question → get the paper answer with confidence score and sources
- Click **Enhance with Web Search** → get a synthesized paper + web answer
- View side-by-side quality review scores for both answers
- See the final recommendation

### Option 2 — CLI

```bash
python notebook/research_assistant.py
```

Interactive terminal session:
```
Ask a question about your papers (or type 'quit' to exit):
> What is FlashFill and how does it use program synthesis?

[RAG] Retrieving answer from your papers ...
  Sources used:
    [1] flashfill.pdf  (page 3, score 0.821)
    [2] flashfill.pdf  (page 7, score 0.774)
[RAG] Confidence: 0.821

Would you like to look up the internet for more details?
Enter yes or no: yes

[WebSearch] Searching the web for: 'What is FlashFill...'
[Reviewer] Running quality review...

FINAL SUMMARY
  Paper answer quality  : 8.3/10  PASS
  Web answer quality    : 9.1/10  PASS

  Recommend using the WEB-SYNTHESIZED answer (higher quality score).
```

---

## API Reference

The FastAPI server exposes the following endpoints (auto-documented at `/docs`):

### `GET /api/health`
Returns server status and loaded model name.

```json
{ "status": "ok", "model": "llama3.1" }
```

### `POST /api/ask`
Runs the RAG pipeline against the PDF vector store.

**Request**
```json
{ "query": "What is program synthesis?" }
```

**Response**
```json
{
  "answer": "Program synthesis is...",
  "sources": [
    {
      "source": "flashfill.pdf",
      "page": 3,
      "score": 0.821,
      "preview": "FlashFill uses a domain-specific language..."
    }
  ],
  "confidence": 0.821
}
```

### `POST /api/enhance`
Runs web search + LLM synthesis + quality review for both answers.

### `POST /api/ingest`
Downloads all PDFs from a public Google Drive folder, chunks and embeds them, and adds them to the vector store. The retriever hot-reloads so new papers are immediately queryable.

**Request**
```json
{ "folder_id": "your_google_drive_folder_id" }
```

The `folder_id` is the last segment of the folder URL:
`https://drive.google.com/drive/folders/`**`THIS_PART`**

The folder must be set to **"Anyone with the link can view"**.

**Response**
```json
{
  "ingested": 4,
  "total_chunks": 312,
  "files": [
    { "name": "paper1.pdf", "chunks": 89 },
    { "name": "paper2.pdf", "chunks": 74 }
  ]
}
```

**Request**
```json
{
  "query": "What is program synthesis?",
  "rag_answer": "Program synthesis is..."
}
```

**Response**
```json
{
  "web_results": "...",
  "synthesized": "Combined answer...",
  "paper_review": {
    "relevance": 8,
    "completeness": 7,
    "clarity": 8,
    "average_score": 7.67,
    "verdict": "PASS",
    "reviewer_notes": "...",
    "suggestions": ["..."]
  },
  "web_review": { "..." },
  "recommendation": "Recommend using the WEB-SYNTHESIZED answer (higher quality score)."
}
```

---

## Research Papers Included

The vector store is pre-built from the following papers:

| Paper | Topic |
|---|---|
| `flashfill.pdf` | FlashFill — automated data transformation in Excel using program synthesis |
| `code generation using LLMs.pdf` | Survey of code generation techniques using large language models |
| `Systematic mapping study of template based code generation.pdf` | Systematic review of template-based code generation approaches |

To add your own PDFs you have two options:
- **Local** — place them in `data/pdf/` and re-run the ingestion notebook (`notebook/pdf_loader.ipynb`)
- **Google Drive** — share a folder publicly and call `POST /api/ingest` with the folder ID

---

## Quality Review System

Every answer is evaluated on three dimensions by the local LLM:

| Dimension | Description |
|---|---|
| **Relevance** | Does the answer directly address the question? |
| **Completeness** | Are all key aspects covered? |
| **Clarity** | Is it well-structured and easy to understand? |

**Verdict thresholds:**

| Score | Verdict |
|---|---|
| ≥ 7.0 | ✅ PASS |
| 4.5 – 6.9 | ⚠️ NEEDS IMPROVEMENT |
| < 4.5 | ❌ FAIL |

---

## Development

### Branch history

| Branch | Feature |
|---|---|
| `data-ingestion` | PDF loading and text extraction |
| `embeddings-vectorstore` | ChromaDB setup and embedding pipeline |
| `typesense` | Typesense cloud vector DB exploration |
| `local-llm` | Switched from Groq API to local Ollama |
| `langgraph` | Agentic AI workflow with LangGraph |
| `frontend` | FastAPI REST API and web UI |
| `mcp` | Google Drive ingestion + MCP server |

---

## Environment Variables (optional)

A `.env` file in `notebook/` is supported. The current setup only requires Ollama running locally — no keys needed. The following are available for optional integrations:

```env
OLLAMA_URL=http://localhost:11434   # default, can be omitted

# Legacy / optional integrations
GOOGLE_API_KEY=...
GROQ_API_KEY=...
TYPESENSE_HOST_NAME=...
TYPESENSE_API_KEY=...
```

---
