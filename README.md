# Autonomous RAG Agent — Cyber Ireland 2022 Report

An autonomous Retrieval-Augmented Generation (RAG) agent that ingests the **State of the Cyber Security Sector in Ireland 2022 Report**, processes it into a searchable vector store, and answers complex multi-step queries using an LLM-powered reasoning loop.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  FastAPI     │────▶│  LangGraph Agent │────▶│  Tool Execution │
│  /query      │     │  (Groq Llama 3.3)│     │                 │
└─────────────┘     └──────────────────┘     │  search_documents│
                           │  ▲               │  search_tables   │
                           │  │               │  get_page_content│
                           ▼  │               │  calculate       │
                    ┌──────────────┐          │  calculate_cagr  │
                    │  Agent Loop  │          └─────────────────┘
                    │  (max 15     │                   │
                    │   iterations)│                   ▼
                    └──────────────┘          ┌─────────────────┐
                                              │  ChromaDB +     │
                                              │  BM25 Hybrid    │
                                              │  Search         │
                                              └─────────────────┘
```

### Key Components

| Component | File | Description |
|-----------|------|-------------|
| ETL Pipeline | `etl/pdf_processor.py` | Extracts text & tables from PDF with table-aware chunking |
| Vector Store | `etl/vector_store.py` | ChromaDB + BM25 hybrid search (0.6/0.4 weight split) |
| Agent Tools | `agent/tools.py` | 5 tools for retrieval, page lookup, and math |
| Agent Graph | `agent/graph.py` | LangGraph state machine with conditional routing |
| System Prompt | `agent/prompts.py` | Instructions for citation, tool use, and response format |
| API Backend | `api/main.py` | FastAPI with `/query`, `/health`, `/traces` endpoints |
| Observability | `utils/logger.py` | Structured JSON trace logging |
| Tests | `tests/test_scenarios.py` | 3 test scenario runner with validation |

## Technology Stack

- **LLM**: Groq Llama 3.3 70B Versatile (via `langchain-groq`)
- **Agent Framework**: LangGraph (state machine with tool calling loop)
- **Vector Store**: ChromaDB (persistent, local)
- **Embeddings**: `all-MiniLM-L6-v2` (384-dim, local, free via sentence-transformers)
- **Hybrid Search**: Vector similarity + BM25 keyword matching
- **PDF Processing**: pdfplumber (text + tables), PyMuPDF (metadata)
- **Math**: simpleeval (safe expression eval), dedicated CAGR function
- **Backend**: FastAPI + Uvicorn
- **Logging**: Custom structured JSON traces

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # Linux/Mac
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and add your Groq API key:

```
GROQ_API_KEY=your_api_key_here
```

Get your free API key from [Groq Console](https://console.groq.com/keys).

### 4. Place the PDF

Ensure `State-of-the-Cyber-Security-Sector-in-Ireland-2022-Report.pdf` is in the project root.

### 5. Run ETL Pipeline

```bash
python -m etl.run_pipeline
```

This extracts text and tables from the PDF, creates embeddings, and stores them in ChromaDB. Produces **94 chunks** and **28 tables**.

### 6. Start the API Server

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### 7. Run Test Scenarios

```bash
python -m tests.test_scenarios
```

## API Endpoints

### POST /query

```json
{
  "query": "What is the total number of jobs reported?"
}
```

Response:

```json
{
  "answer": "The total number of jobs is 7,351 (Page 19).",
  "citations": [{"page": 19}],
  "trace_id": "abc123",
  "steps": [...],
  "duration_ms": 4099.15
}
```

### GET /health

Returns server status and chunk count.

### GET /traces

Returns recent query trace logs.

## Test Results

All 3 required test scenarios **PASSED**:

| # | Scenario | Duration | Tools Used | Citations |
|---|----------|----------|------------|-----------|
| 1 | **Verification** — Total jobs reported | 4.1s | `search_documents` | Page 19 |
| 2 | **Data Synthesis** — Pure-Play firms concentration | 12.7s | `search_tables`, `search_documents` | Pages 13, 19 |
| 3 | **Forecasting** — CAGR for 2030 job target | 15.4s | `search_documents`, `calculate_cagr` | Page 27 |

### Test 1: Verification Challenge
> **Q**: What is the total number of jobs reported, and where exactly is this stated?  
> **A**: 7,351 total jobs — cited from Table 4.1 on Page 19.

### Test 2: Data Synthesis Challenge
> **Q**: Compare the concentration of 'Pure-Play' cybersecurity firms in the South-West against the National Average.  
> **A**: National average is 33% (160 dedicated/pure-play firms). South-West regional breakdown not available in the document — correctly noted.

### Test 3: Forecasting Challenge
> **Q**: Based on our 2022 baseline and the stated 2030 job target, what is the required CAGR?  
> **A**: CAGR = **10.00%** (from 8,086 → 17,333 over 8 years), with full calculation shown. Cited Page 27.

## Execution Logs

Trace logs are saved to `logs/` as JSON files:

- `logs/test_scenario_results.json` — Aggregated test results with validation status
- `logs/test_scenario_traces.json` — Full execution traces for all 3 tests
- `logs/trace_<id>_<timestamp>.json` — Individual query traces

Each trace contains:
- Step-by-step tool calls and agent reasoning
- Timestamps and duration
- Retrieved document snippets and scores
- Final answer with citations

## Project Structure

```
├── .env                  # API keys (not committed)
├── .env.example          # Template
├── .gitignore
├── requirements.txt
├── README.md
├── agent/
│   ├── __init__.py
│   ├── graph.py          # LangGraph state machine
│   ├── prompts.py        # System prompt
│   └── tools.py          # 5 agent tools
├── api/
│   ├── __init__.py
│   └── main.py           # FastAPI server
├── etl/
│   ├── __init__.py
│   ├── pdf_processor.py  # PDF text + table extraction
│   ├── vector_store.py   # ChromaDB + BM25 hybrid search
│   └── run_pipeline.py   # ETL entry point
├── tests/
│   ├── __init__.py
│   └── test_scenarios.py # 3 test scenario runner
├── utils/
│   ├── __init__.py
│   └── logger.py         # Structured trace logging
├── logs/                 # Execution traces (JSON)
├── chroma_db/            # Persistent vector store
└── data/                 # (optional) processed data
```
