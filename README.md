# Research Assistant

An intelligent research tool that answers questions by automatically searching local papers, Google Drive, and arXiv — then enriching the answer with live web search and evaluating both answers with an LLM-as-a-judge quality reviewer. Powered by **Groq** and **Tavily**. One query, fully automated.

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│                        User Question                            │
└────────────────────────────┬────────────────────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  Embed Query    │  all-MiniLM-L6-v2
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  ChromaDB Search│  local vector store
                    └────────┬────────┘
                             │
              ┌──────────────▼──────────────┐
              │  Relevant? (score ≥ 0.6)     │
              └──────┬───────────────┬───────┘
                   Yes               No
                    │                │
                    │      ┌─────────▼─────────┐
                    │      │  Sync Google Drive │  OAuth — scans entire Drive
                    │      └─────────┬──────────┘
                    │                │
                    │      ┌─────────▼─────────┐
                    │      │  Still not enough? │
                    │      └──────┬─────────────┘
                    │           Yes
                    │             │
                    │      ┌──────▼──────────────┐
                    │      │  Fetch from arXiv    │  auto-search + ingest 5 papers
                    │      └──────┬───────────────┘
                    │             │
                    └──────┬──────┘
                           │
                  ┌────────▼────────┐
                  │  Groq LLM       │  llama-3.3-70b-versatile → Paper Answer
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  Tavily Search  │  live web results
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  Groq LLM       │  Synthesized Web Answer
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │ Quality Reviewer │  LLM-as-judge scores both answers
                  └────────┬────────┘
                           │
                  ┌────────▼────────┐
                  │  Recommendation │  Best answer surfaced
                  └─────────────────┘
```

---

## Features

- **Unified Automatic Pipeline** — one query triggers local search → Google Drive sync → arXiv fetch → paper answer → web search → quality review, with no manual steps
- **Groq LLM** — `llama-3.3-70b-versatile` for fast, high-quality answers
- **Tavily Web Search** — real-time web results synthesized alongside paper context
- **Google Drive Integration** — OAuth-based sync of your entire Google Drive for PDFs (no public folder required)
- **arXiv Auto-Fetch** — searches and ingests relevant papers automatically when local knowledge is insufficient
- **PDF RAG Pipeline** — semantic search over research papers using ChromaDB and sentence embeddings
- **LLM Quality Reviewer** — scores both answers on Relevance, Completeness, and Clarity (1–10), gives a verdict, and recommends the best answer
- **Deduplication** — already-ingested papers are skipped automatically
- **MCP Server** — exposes ingestion tools so any MCP-compatible agent (Claude Code, Claude Desktop) can call them
- **Web UI** — clean dark-themed single-page app built on FastAPI + vanilla JS

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq API — `llama-3.3-70b-versatile` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Store | ChromaDB (persistent) |
| LLM Framework | LangChain |
| Web Search | Tavily (via `langchain-tavily`) |
| Drive Ingestion | Google Drive API v3 + OAuth 2.0 |
| arXiv Fetching | arXiv API + PyMuPDF |
| MCP Server | FastMCP |
| API Server | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JS |
| Python | 3.13+ |

> **Optional local alternatives:** Ollama (`llama3.1`) can replace Groq, and DuckDuckGo can replace Tavily if you prefer a fully offline setup. See [Environment Variables](#environment-variables).

---

## Project Structure

```
Research_Assistant/
├── api.py                            # FastAPI server — all pipelines as REST endpoints
├── mcp_server.py                     # FastMCP server — ingestion tools for MCP agents
├── credentials.json                  # Google OAuth credentials (not committed)
├── token.json                        # Auto-generated OAuth token (not committed)
├── requirements.txt
│
├── notebook/
│   ├── research_assistant.py         # Core RAG pipeline (retriever + LLM prompt)
│   ├── research_pipeline.py          # Unified 6-step orchestrator
│   ├── drive_ingestion.py            # Google Drive OAuth sync + arXiv fetch + chunking
│   ├── workflow_web_search.py        # Tavily search + Groq synthesis
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
2. **Groq API key** — free at [console.groq.com](https://console.groq.com)
3. **Tavily API key** — free at [app.tavily.com](https://app.tavily.com)
4. **Google Cloud credentials** — for Drive sync (see [Google Drive Setup](#google-drive-setup))

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

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

Optional (for local alternatives):

```env
# Use Ollama instead of Groq
OLLAMA_URL=http://localhost:11434

# DuckDuckGo requires no key — swap TavilySearchResults for DuckDuckGoSearchRun in workflow_web_search.py
```

---

## Google Drive Setup

The Drive sync uses OAuth 2.0 to access your personal Google Drive — no public folder required.

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a project → enable **Google Drive API**
3. Create **OAuth 2.0 credentials** (Desktop app) → download as `credentials.json`
4. Place `credentials.json` in the project root

On first use, a browser window opens for you to log in. After that, `token.json` is cached automatically and reused on all subsequent calls.

---

## Running the App

```bash
python api.py
```

Open your browser at **http://localhost:8000**

Type any research question and click **Research**. The pipeline runs fully automatically:

1. Searches the local vector store
2. If results are weak → syncs your Google Drive for new PDFs
3. If still weak → fetches relevant papers from arXiv and ingests them
4. Generates a paper-grounded answer using Groq
5. Searches the web with Tavily and synthesizes a second answer
6. Reviews both answers with LLM-as-judge and recommends the best one

The pipeline status bar at the top shows which steps ran (Local → Drive → arXiv → Paper → Web → Review).

---

## API Reference

Full interactive docs at `/docs` when the server is running.

### `GET /api/health`
Returns server status and loaded model.

```json
{ "status": "ok", "model": "llama-3.3-70b-versatile" }
```

### `POST /api/research`
Runs the full unified pipeline — local search → Drive → arXiv → paper answer → web search → quality review.

**Request**
```json
{ "query": "What is the difference between MCP and RAG?" }
```

**Response**
```json
{
  "paper_answer": "...",
  "sources": [{ "source": "paper.pdf", "page": 3, "score": 0.821, "preview": "..." }],
  "confidence": 0.821,
  "web_answer": "...",
  "paper_review": {
    "relevance": 8, "completeness": 7, "clarity": 9,
    "average_score": 8.0, "verdict": "PASS",
    "reviewer_notes": "...", "suggestions": ["..."]
  },
  "web_review": { "..." },
  "recommendation": "Recommend using the WEB-SYNTHESIZED answer (higher quality score).",
  "pipeline_steps": [
    { "step": "local_search", "status": "weak" },
    { "step": "drive_ingestion", "status": "done", "new_files": 0 },
    { "step": "arxiv_fetch", "status": "done", "papers_fetched": 3 },
    { "step": "paper_answer", "status": "done" },
    { "step": "web_search", "status": "done" },
    { "step": "quality_review", "status": "done" }
  ]
}
```

### `POST /api/ask`
Runs only the RAG pipeline (no web search or review).

**Request**
```json
{ "query": "What is program synthesis?" }
```

**Response**
```json
{
  "answer": "Program synthesis is...",
  "sources": [{ "source": "flashfill.pdf", "page": 3, "score": 0.821, "preview": "..." }],
  "confidence": 0.821
}
```

### `POST /api/enhance`
Runs web search + synthesis + quality review for a given query and existing paper answer.

**Request**
```json
{ "query": "What is program synthesis?", "rag_answer": "Program synthesis is..." }
```

### `POST /api/ingest`
Scans your authenticated Google Drive for PDFs and ingests any new ones. Skips already-ingested files. Retriever hot-reloads automatically.

**Response**
```json
{
  "ingested": 4,
  "total_chunks": 312,
  "files": [
    { "name": "paper1.pdf", "chunks": 89, "skipped": false },
    { "name": "paper2.pdf", "chunks": 0, "skipped": true }
  ]
}
```

### `POST /api/fetch`
Fetches a single PDF from an arXiv URL, DOI, or direct PDF link and ingests it.

**Request**
```json
{ "url": "https://arxiv.org/abs/1706.03762" }
```

**Response**
```json
{ "source": "https://arxiv.org/abs/1706.03762", "filename": "1706.03762.pdf", "chunks": 124, "skipped": false }
```

---

## MCP Server

`mcp_server.py` exposes the ingestion pipeline as MCP tools, callable by any MCP-compatible agent (Claude Code, Claude Desktop).

```bash
python mcp_server.py
```

**Available tools:**

| Tool | Description |
|---|---|
| `ingest_papers()` | Scan your authenticated Google Drive for PDFs and ingest any new ones |
| `fetch_paper(url)` | Fetch and ingest a single PDF from arXiv, DOI, or direct URL |

Register with Claude Code:
```bash
claude mcp add research-assistant python mcp_server.py
```

---

## Quality Review System

Every answer is evaluated by the Groq LLM on three dimensions:

| Dimension | Description |
|---|---|
| **Relevance** | Does the answer directly address the question with substantive information? |
| **Completeness** | Are all key aspects covered thoroughly? |
| **Clarity** | Is it well-structured and easy to understand? |

Answers that say "I don't know" or "no information in context" are penalized with scores of 1–2 on Relevance and Completeness regardless of phrasing.

**Verdict thresholds:**

| Score | Verdict |
|---|---|
| ≥ 7.0 | PASS |
| 4.5 – 6.9 | NEEDS IMPROVEMENT |
| < 4.5 | FAIL |

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
| `mcp` | Google Drive ingestion + MCP server + unified pipeline |

---

## Research Papers Included

The vector store is pre-built from the following papers:

| Paper | Topic |
|---|---|
| `flashfill.pdf` | FlashFill — automated data transformation in Excel using program synthesis |
| `code generation using LLMs.pdf` | Survey of code generation techniques using large language models |
| `Systematic mapping study of template based code generation.pdf` | Systematic review of template-based code generation approaches |

To add your own papers, simply ask a question — if the topic isn't in the local store, the pipeline fetches relevant arXiv papers and ingests them automatically. Google Drive PDFs are also synced as part of the pipeline.
