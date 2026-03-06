"""
FastAPI Backend: Exposes the /query endpoint for the agent.
"""

import os
import sys
import time
from contextlib import asynccontextmanager
from typing import Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

load_dotenv()

from etl.vector_store import VectorStore
from agent.graph import CyberIrelandAgent
from utils.logger import AgentTracer


# --- Global instances ---
_agent: Optional[CyberIrelandAgent] = None
_tracer: Optional[AgentTracer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize the agent and vector store on startup."""
    global _agent, _tracer

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    persist_dir = os.path.join(project_root, "chroma_db")
    log_dir = os.path.join(project_root, "logs")

    print("[API] Initializing vector store...")
    vector_store = VectorStore(persist_dir=persist_dir)

    # Check if data exists
    count = vector_store.collection.count()
    if count == 0:
        print("[API] WARNING: Vector store is empty! Run the ETL pipeline first:")
        print("[API]   python -m etl.run_pipeline")
        print("[API] Starting with empty store - queries will return no results.")
    else:
        print(f"[API] Vector store loaded with {count} chunks.")

    print("[API] Initializing agent...")
    _tracer = AgentTracer(log_dir=log_dir)

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("[API] WARNING: GROQ_API_KEY not set. Agent will not function.")
        print("[API] Set it in .env file or as environment variable.")
    else:
        _agent = CyberIrelandAgent(
            vector_store=vector_store,
            tracer=_tracer,
        )
        print("[API] Agent initialized successfully.")

    print("[API] Server ready!")
    yield

    # Cleanup
    if _tracer and _tracer.traces:
        _tracer.save_all_traces("all_traces.json")
        print("[API] Saved all traces.")


# --- API Models ---
class QueryRequest(BaseModel):
    query: str = Field(..., description="The question to ask about the Cyber Ireland 2022 Report.")
    session_id: Optional[str] = Field(None, description="Optional session ID for conversation tracking.")


class Citation(BaseModel):
    page: int


class AgentStep(BaseModel):
    type: str
    tool: Optional[str] = None
    args: Optional[dict] = None
    content_preview: Optional[str] = None
    result_preview: Optional[str] = None


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    trace_id: str
    steps: list[AgentStep]
    duration_ms: float
    query: str


class HealthResponse(BaseModel):
    status: str
    vector_store_count: int
    agent_ready: bool


# --- FastAPI App ---
app = FastAPI(
    title="Cyber Ireland 2022 Report Agent",
    description=(
        "An autonomous AI agent that answers complex queries about Ireland's "
        "cybersecurity sector using the Cyber Ireland 2022 Report. "
        "Powered by LangGraph + Google Gemini + ChromaDB."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check if the system is ready."""
    return HealthResponse(
        status="ok" if _agent else "degraded",
        vector_store_count=_agent.vector_store.collection.count() if _agent else 0,
        agent_ready=_agent is not None,
    )


@app.post("/query", response_model=QueryResponse)
async def query_agent(request: QueryRequest):
    """
    Submit a query to the Cyber Ireland Report agent.

    The agent will autonomously:
    1. Analyze the query
    2. Search relevant documents and tables
    3. Extract data with citations
    4. Perform calculations if needed
    5. Verify citations and return a structured answer
    """
    if not _agent:
        raise HTTPException(
            status_code=503,
            detail="Agent not initialized. Check GROQ_API_KEY is set and ETL pipeline has been run.",
        )

    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    start_time = time.time()

    try:
        result = _agent.query(request.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    duration_ms = (time.time() - start_time) * 1000

    # Parse steps
    steps = []
    for step in result.get("steps", []):
        steps.append(AgentStep(**step))

    # Parse citations
    citations = [Citation(page=c["page"]) for c in result.get("citations", [])]

    return QueryResponse(
        answer=result["answer"],
        citations=citations,
        trace_id=result["trace_id"],
        steps=steps,
        duration_ms=round(duration_ms, 2),
        query=request.query,
    )


@app.get("/traces")
async def get_traces():
    """Get all execution traces (for debugging/observability)."""
    if not _tracer:
        return {"traces": []}

    return {
        "traces": [t.to_dict() for t in _tracer.traces],
        "count": len(_tracer.traces),
    }
